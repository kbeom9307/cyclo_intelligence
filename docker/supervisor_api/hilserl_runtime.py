"""GPU actor and local/external SAC learner, with Mock or remote Isaac NIST environment.

Actor control and learner/network work run on separate threads.
Isaac physics lives in the simulator-owned bridge. Pixel mode trains multi-view
the pinned upstream HIL-SERL JAX agent, or explicitly selected legacy PyTorch SAC.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import sys
import time
import traceback

import numpy as np
from hilserl_profiles import prepare_jax_libraries
prepare_jax_libraries()
import torch
from hilserl_pixels import PixelPolicy, SuccessClassifier
from hilserl_learning import SAC, Policy, Replay, LocalReply, LocalBroadcast
from hilserl_devices import resolve_device, memory_status
from hilserl_learner import AsyncLearner, make_spec
from hilserl_data import Collection, save_rows, task_key
from hilserl_cameras import selected, validate_artifact
from hilserl_profiles import upstream, image_shape
import zmq


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, allow_nan=False))
    temp.replace(path)


class MockPegEnv:
    observation_dim = 4
    action_dim = 3
    schema = 'mock-xyz-state4-delta3-v1'
    info = {}

    def heartbeat(self):
        pass

    def close(self):
        pass

    """XYZ peg tip; plate z=0, hole at origin, insertion towards negative z."""
    def __init__(self, config):
        self.c = config
        self.rng = np.random.default_rng(config['seed'])
        self.reset()

    def reset(self):
        self.position = np.array([*self.rng.uniform(-.025, .025, 2), .03], dtype=np.float32)
        self.steps = self.hold = 0
        return self.observation()

    def observation(self):
        return np.array([self.position[0] / .05, self.position[1] / .05,
                         self.position[2] / max(.05, self.c['success_depth_m']),
                         self.hold / max(1, math.ceil(self.c['success_hold_s'] * self.c['control_hz']))], dtype=np.float32)

    def step(self, action):
        candidate = self.position + np.clip(action, -1, 1) * self.c['translation_limit_m']
        candidate = np.clip(candidate, [-.05, -.05, -self.c['success_depth_m']], [.05, .05, .06])
        if np.linalg.norm(candidate[:2]) > self.c['success_tolerance_m']:
            candidate[2] = max(0., candidate[2])  # toy rigid plate constraint
        self.position = candidate.astype(np.float32)
        self.steps += 1
        aligned = np.linalg.norm(candidate[:2]) <= self.c['success_tolerance_m']
        self.hold = self.hold + 1 if aligned and candidate[2] <= -self.c['success_depth_m'] + 1e-6 else 0
        success = self.hold >= math.ceil(self.c['success_hold_s'] * self.c['control_hz'])
        reward = -float(np.linalg.norm(candidate - [0, 0, -self.c['success_depth_m']])) / .05 + (10. if success else 0.)
        return self.observation(), reward, success, self.steps >= self.c['max_episode_steps']


def run(root, session_id, parent_pid=None):
    torch.set_num_threads(1)
    parent = parent_pid if parent_pid is not None else os.getppid()
    folder = root / session_id
    folder.mkdir(parents=True, exist_ok=True)
    def session():
        with sqlite3.connect(root / 'control.sqlite3', timeout=10) as db:
            return json.loads(db.execute('SELECT body FROM documents WHERE id=?', ('session:' + session_id,)).fetchone()[0])
    config = {'actor_device': 'cuda:0', 'learner_device': 'cuda:0', 'learner_placement': 'local', **session()['config']}
    if upstream(config):
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
        os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.90')
        # Inference autotuning can reserve several GiB for batch 1. Fixed
        # kernels avoid that warmup allocation beside Isaac without changing
        # model parameters or floating-point precision.
        if session()['mode'] != 'train' or config['learner_placement'] == 'remote' or config['learner_device'] == 'cpu':
            if '--xla_gpu_autotune_level' not in os.environ.get('XLA_FLAGS', ''):
                os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') + ' --xla_gpu_autotune_level=0').strip()
    torch.manual_seed(config['seed']); random.seed(config['seed'])
    if config['environment'] == 'isaac_sim':
        from hilserl_isaac import IsaacNistEnv
        env = IsaacNistEnv(config)
    elif config['environment'] == 'mock':
        env = MockPegEnv(config)
    else:
        raise ValueError('Hardware environment is not implemented')
    try:
        train(root, session_id, parent, folder, config, env, session)
    finally:
        env.close()


def train(root, session_id, parent, folder, config, env, session):
    mode = session()['mode']
    pixel = getattr(env, 'pixel_mode', False)
    restored = pixel and upstream(config)
    passive = mode in ('preview', 'demo', 'classifier')
    collecting = mode in ('demo', 'classifier')
    collection = Collection(folder, config, mode, env.schema) if collecting else None
    batch_size = config.get('pixel_batch_size', 32) if pixel else config['batch_size']
    capacity = config.get('pixel_replay_capacity', 2048) if pixel else config['replay_capacity']
    needs_classifier = config['reward_source'] == 'classifier' and mode not in ('preview', 'classifier')
    actor_device = config.get('actor_device', 'cuda:0') if restored else (resolve_device(config.get('actor_device', 'cuda:0')) if not passive or needs_classifier else torch.device('cpu'))
    initial_weights = None
    backend = None
    actor = None
    classifier = None
    if needs_classifier:
        if not pixel:
            raise ValueError('Success classifier requires pixel observations')
        with np.load(config['classifier_path'], allow_pickle=False) as data:
            validate_artifact(data, selected(config, 'classifier'), config)
            if json.loads(str(data['__task__'])) != task_key(config):
                raise ValueError('Classifier task does not match this environment')
        if restored:
            from hilserl_upstream import UpstreamClassifier
            classifier = UpstreamClassifier.load(config['classifier_path'], device=actor_device)
        else:
            classifier = SuccessClassifier.load(config['classifier_path'], device=actor_device)
    classifier_hold = 0
    classifier_probability = None
    classifier_logit = None
    if not passive and config['checkpoint_path']:
        with np.load(config['checkpoint_path'], allow_pickle=False) as data:
            if pixel:
                validate_artifact(data, selected(config, 'observation'), config)
            weights = dict(data)
            weights.pop('__cameras__', None)
            weights.pop('__crops__', None)
            schema = str(weights.pop('__schema__', 'mock-xyz-state4-delta3-v1'))
            if schema != env.schema:
                raise ValueError(f'Checkpoint schema {schema} does not match {env.schema}')
            saved_task = weights.pop('__task__', None)
            if pixel and (saved_task is None or json.loads(str(saved_task)) != task_key(config)):
                raise ValueError('Pixel policy task does not match this environment')
            if not all(np.isfinite(value).all() for value in weights.values()):
                raise ValueError('Policy checkpoint contains non-finite weights')
            initial_weights = weights
    demos = []
    if mode == 'train' and config['demonstration_path']:
        with np.load(config['demonstration_path'], allow_pickle=False) as data:
            schema = str(data['__schema__']) if '__schema__' in data else 'mock-xyz-state4-delta3-v1'
            if schema != env.schema:
                raise ValueError('Dataset schema does not match the selected environment')
            if pixel and ('__task__' not in data or json.loads(str(data['__task__'])) != task_key(config)):
                raise ValueError('Pixel demonstration task does not match this environment')
            if pixel:
                validate_artifact(data, selected(config, 'observation'), config)
            count = len(data['reward'])
            shapes = {'observation': (count, env.observation_dim), 'action': (count, env.action_dim), 'reward': (count,),
                      'next_observation': (count, env.observation_dim), 'terminal': (count,), 'human': (count,)}
            if pixel:
                for key in ('images', 'next_images'):
                    shapes[key] = (count, *image_shape(config))
                    if key not in data or data[key].dtype != np.uint8:
                        raise ValueError('Pixel demonstrations require uint8 RGB observations')
            if 'truncated' in data:
                shapes['truncated'] = (count,)
            if count > capacity:
                raise ValueError('Dataset exceeds configured replay capacity')
            for key, shape in shapes.items():
                if data[key].shape != shape or not np.isfinite(data[key]).all():
                    raise ValueError(f'Invalid dataset field: {key}; expected {shape}, finite values')
            if np.abs(data['action']).max(initial=0) > 1:
                raise ValueError('Dataset actions exceed [-1, 1]')
            arrays = {key: data[key] for key in shapes}
            for i in range(count):
                row = {key: arrays[key][i] for key in ('observation', 'action', 'reward', 'next_observation', 'terminal', 'human')}
                if pixel:
                    row.update(images=arrays['images'][i], next_images=arrays['next_images'][i])
                row['truncated'] = bool(arrays['truncated'][i]) if 'truncated' in arrays else False
                demos.append(row)
    if not passive:
        if restored:
            from hilserl_upstream import UpstreamPolicy
            actor = UpstreamPolicy(config, env.observation_dim, env.action_dim)
        else:
            actor = (PixelPolicy if pixel else Policy)(env.observation_dim, env.action_dim).to(actor_device).eval()
        if initial_weights:
            actor.restore(initial_weights)
    if mode == 'train':
        backend = AsyncLearner(config, make_spec(config, env), folder, initial_weights, demos, env.heartbeat)
        _, weights = backend.snapshot()
        actor.restore(weights)
    del demos
    steps = completed = successes = human_steps = 0
    episode_step = 0
    version = 0
    episode = 1
    last_revision = 0
    obs = env.observation()
    last_row = None
    mode = session()['mode']
    log = (folder / 'transitions.jsonl').open('a')
    status = {}
    def prepare_episode():
        nonlocal last_row, classifier_hold, classifier_probability, classifier_logit
        last_row = None
        classifier_hold, classifier_probability, classifier_logit = 0, None, None
        if collection:
            collection.rows = []
            collection.done = False
            collection.error = None
        status.update(initializing=True, episode_step=0, episode=episode + 1,
                      last_action=[0.] * env.action_dim, last_policy_action=[0.] * env.action_dim,
                      policy_enabled=not passive, last_action_source='hold', last_reward=None)
        atomic_json(folder / 'runtime.json', status)
        return env.reset()
    try:
        while os.getppid() == parent:
            tick = time.monotonic()
            tick_wall = time.time()
            control = session()
            env.heartbeat()
            if control['state'] != 'running':
                if backend:
                    backend.pause(env.heartbeat)
            events = [e for e in control['events'] if e['sequence'] > last_revision]
            for event in events:
                if event['action'] == 'reset':
                    obs = prepare_episode(); episode += 1; episode_step = 0
                elif collection and event['action'] in ('capture_success', 'capture_failure', 'save_demo', 'discard_demo'):
                    try:
                        collection.error = None
                        if event['action'].startswith('capture_'):
                            obs = collection.capture(env, int(event['action'] == 'capture_success'), episode)
                        elif event['action'] == 'save_demo':
                            collection.save_demo(episode)
                        elif event['action'] == 'discard_demo':
                            collection.rows = []
                    except ValueError as exc:
                        collection.error = str(exc)
                elif event['action'] in ('success', 'failure', 'truncate'):
                    if collection:
                        collection.done = True
                        completed += 1; successes += int(event['action'] == 'success')
                        continue
                    # A separate zero-action terminal transition records the human
                    # label without modifying an already transmitted transition.
                    success = event['action'] == 'success'
                    truncated = event['action'] == 'truncate'
                    row = {'observation': obs.copy(), 'action': np.zeros(env.action_dim, dtype=np.float32),
                           'reward': (float(success) if restored else (0. if truncated else (10. if success else -10.))), 'next_observation': obs.copy(),
                           'terminal': not truncated, 'truncated': truncated, 'human': True, 'label_only': True,
                           'episode': episode, 'policy_version': version}
                    if pixel:
                        row.update(images=env.images.copy(), next_images=env.images.copy())
                    if backend:
                        backend.insert(row)
                    log.write(json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in row.items() if k not in ('images', 'next_images')}) + '\n')
                    completed += 1; successes += int(success)
            last_revision = control['revision']
            if control['state'] == 'stopped':
                break
            if backend:
                learner_status, weights = backend.snapshot()
                if learner_status.get('version', 0) > version:
                    actor.restore(weights); version = learner_status['version']
            if control['state'] == 'running' and not (collection and collection.target_reached):
                manual = control.get('manual_action', {})
                human = time.time() < manual.get('expires_at', 0) and any(manual.get('value', []))
                with torch.no_grad():
                    proposed = (np.zeros(env.action_dim, dtype=np.float32) if passive else
                                actor.action(obs, env.images, deterministic=mode == 'evaluate') if restored else
                                actor.sample(torch.as_tensor(obs, dtype=torch.float32, device=actor_device)[None], deterministic=mode == 'evaluate', **({'images': torch.as_tensor(env.images, device=actor_device)[None]} if pixel else {}))[0][0].cpu().numpy())
                if human:
                    action = np.array(manual['value'], dtype=np.float32)
                    if config['environment'] == 'isaac_sim':
                        action = env.tcp_to_action(action)
                else:
                    action = proposed
                images = env.images.copy() if pixel else None
                nxt, reward, terminal, truncated = env.step(action)
                if mode == 'classifier':
                    label_input = control.get('classifier_label', {})
                    label = int(bool(label_input.get('success')) and tick_wall < label_input.get('expires_at', 0))
                    collection.capture(env, label, episode, fresh=False, automatic=True)
                    if len(collection.samples) >= 10000:
                        truncated = True
                        collection.error = 'Collection limit reached (10,000 samples); stop and start a new collection'
                if classifier is not None:
                    classifier_logit, classifier_probability = classifier.predict(env.classifier_images)
                    classifier_hold = classifier_hold + 1 if classifier_probability >= config['classifier_threshold'] else 0
                    detected = classifier_hold >= config['classifier_hold_steps']
                    terminal = detected or bool(env.info.get('is_failure', False))
                    reward = float(detected) if restored else (10. if detected else (-10. if env.info.get('is_failure', False) else 0.))
                    env.info['classifier_success'] = detected
                if config['reward_source'] == 'human':
                    reward, terminal = 0., bool(env.info.get('is_failure', False))
                elif restored and config['reward_source'] == 'environment':
                    reward = float(bool(env.info.get('is_success', False)))
                row = {'observation': obs.copy(), 'action': action.copy(), 'reward': reward,
                       'next_observation': nxt.copy(), 'terminal': bool(terminal), 'truncated': bool(truncated),
                       'human': human, 'label_only': False, 'episode': episode, 'policy_version': version,
                       'step_id': steps, 'timestamp': time.time(), 'proposed_action': proposed.copy(),
                       'environment_info': dict(env.info)}
                if pixel:
                    row.update(images=images, next_images=env.images.copy(), capture_metadata=env.capture_metadata)
                if mode == 'demo':
                    row['human'] = True
                    collection.rows.append(row)
                if backend:
                    backend.insert(row)
                log.write(json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in row.items() if k not in ('images', 'next_images')}) + '\n')
                last_row = row
                obs = nxt; steps += 1; episode_step += 1; human_steps += int(human)
                if terminal or truncated:
                    if collecting:
                        collection_success = bool(env.info.get('classifier_success', False) if classifier is not None else env.info.get('is_success', False))
                        completed += 1; successes += int(collection_success)
                        # Pause at the boundary for capture/episode review. Do not auto-reset demonstrations.
                        collection.done = True
                        with sqlite3.connect(root / 'control.sqlite3', timeout=10) as db:
                            db.execute('BEGIN IMMEDIATE')
                            current = json.loads(db.execute('SELECT body FROM documents WHERE id=?', ('session:' + session_id,)).fetchone()[0])
                            if current['state'] == 'running':
                                current['state'] = 'paused'
                                current['manual_action'] = {}
                                current['classifier_label'] = {}
                                current['episode_outcome'] = 'success' if collection_success else ('failure' if terminal else 'truncated')
                                current['revision'] += 1
                                current['events'].append({'sequence': current['revision'], 'action': 'collection_boundary', 'episode': episode, 'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'preview': False})
                                db.execute('UPDATE documents SET body=? WHERE id=?', (json.dumps(current), 'session:' + session_id))
                        control = session()
                    else:
                        completed += 1; successes += int(env.info.get('classifier_success', False) if classifier is not None else (env.info.get('is_success', terminal) and config['reward_source'] == 'environment'))
                        obs = prepare_episode(); episode += 1; episode_step = 0
            if collection and collection.target_reached:
                collection.done = True
                with sqlite3.connect(root / 'control.sqlite3', timeout=10) as db:
                    db.execute('BEGIN IMMEDIATE')
                    current = json.loads(db.execute('SELECT body FROM documents WHERE id=?', ('session:' + session_id,)).fetchone()[0])
                    if current['state'] != 'stopped':
                        current['state'] = 'stopped'
                        current['manual_action'] = {}
                        current['classifier_label'] = {}
                        current['revision'] += 1
                        current['events'].append({'sequence': current['revision'], 'action': 'success_target_reached', 'episode': episode, 'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'preview': False})
                        db.execute('UPDATE documents SET body=? WHERE id=?', (json.dumps(current), 'session:' + session_id))
                control = session()
                last_revision = control['revision']
            learner_status = {}
            if backend:
                backend.tick(mode == 'train' and control['state'] == 'running')
                learner_status, _ = backend.snapshot()
            metrics = {**learner_status.get('metrics', {'learner_steps': 0, 'replay_size': 0}),
                       'environment_steps': steps, 'policy_version': version, 'human_steps': human_steps,
                       'success_rate': successes / completed if completed else None, 'completed_episodes': completed}
            status = {'heartbeat': time.time(), 'ack_revision': last_revision, 'state': control['state'],
                      'episode': episode, 'episode_step': episode_step, 'max_episode_steps': config['max_episode_steps'], 'metrics': metrics, 'position_m': env.position.tolist(),
                      'last_action': last_row['action'].tolist() if last_row else [0.] * env.action_dim,
                      'last_policy_action': last_row['proposed_action'].tolist() if last_row else [0.] * env.action_dim,
                      'policy_enabled': not passive,
                      'last_action_source': ('human' if last_row['human'] else ('hold' if passive else 'policy')) if last_row else 'hold',
                      'last_reward': float(last_row['reward']) if last_row else None,
                      'environment_info': dict(env.info), 'observation_dim': env.observation_dim,
                      'action_dim': env.action_dim, 'schema': env.schema,
                      'observation_mode': 'pixels' if pixel else 'state', 'image_shape': list(image_shape(config)) if pixel else None,
                      'capture_metadata': env.capture_metadata if pixel else None,
                      'collection': collection.state() if collection else None,
                      'demo_auto_save': False,
                      'classifier_probability': classifier_probability, 'classifier_logit': classifier_logit,
                      'classifier_hold_steps': classifier_hold, 'classifier_enabled': classifier is not None,
                      'effective_batch_size': batch_size,
                      'effective_replay_capacity': capacity,
                      'checkpoint_path': str(folder / 'policy.npz'), 'dataset_path': (str(Path(learner_status['artifact_directory']) / ('replay' if restored else 'replay.npz')) if backend and backend.remote else str(folder / ('demonstrations.npz' if mode == 'demo' else ('replay' if restored else 'replay.npz')))),
                      'transport': backend.transport if backend else 'Policy disabled' if passive else 'GPU policy evaluation',
                      'compute': {'actor_device': str(actor_device) if actor is not None or classifier is not None else 'inactive',
                                  'actor_memory': (actor or classifier).memory() if restored and (actor is not None or classifier is not None) else memory_status(actor_device) if actor is not None or classifier is not None else {},
                                  'learner': learner_status, 'shared_process': bool(backend and not backend.remote)}, 'error': None}
            atomic_json(folder / 'runtime.json', status)
            log.flush()
            time.sleep(max(0, 1 / config['control_hz'] - (time.monotonic() - tick)))
    finally:
        log.close()
        if collection:
            collection.export()
        final_error = None
        if backend:
            try:
                learner_status, weights = backend.close(env.heartbeat)
                actor.restore(weights)
                version = learner_status.get('version', version)
                status.setdefault('metrics', {}).update(learner_status.get('metrics', {}), policy_version=version)
                status.setdefault('compute', {})['learner'] = {**learner_status, 'placement': 'remote' if backend.remote else 'local', 'pending_transitions': 0}
            except Exception as exc:
                final_error = exc
        if actor is not None:
            np.savez_compressed(folder / 'policy.npz', __schema__=np.array(env.schema),
                __task__=np.array(json.dumps(task_key(config), sort_keys=True)),
                __cameras__=np.array(json.dumps(selected(config, 'observation'))),
                __crops__=np.array(json.dumps(config.get('camera_crops', {}), sort_keys=True)), **actor.export())
        status.update(heartbeat=time.time(), ack_revision=last_revision, state='stopped', artifacts_saved=final_error is None)
        if final_error:
            status['error'] = str(final_error)
        atomic_json(folder / 'runtime.json', status)
        if final_error:
            raise final_error



if __name__ == '__main__':
    root, session_id = Path(sys.argv[1]), sys.argv[2]
    try:
        run(root, session_id, int(sys.argv[3]) if len(sys.argv) > 3 else None)
    except Exception:
        detail = traceback.format_exc()
        path = root / session_id / 'runtime.json'
        try:
            status = json.loads(path.read_text())
        except (OSError, ValueError):
            status = {}
        status.update(heartbeat=time.time(), state='stopped', error=detail)
        atomic_json(path, status)
        print(detail, file=sys.stderr)
        sys.exit(1)
