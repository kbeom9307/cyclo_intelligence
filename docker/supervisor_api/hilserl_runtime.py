"""CPU learner/Actor runtime: SAC + Agentlace, with Mock or remote Isaac NIST environment.

Actor and learner run on separate threads with Agentlace loopback transport.
Isaac physics lives in the simulator-owned bridge. Pixel mode trains multi-view
CNN policies/critics; this local SAC is not an upstream HIL-SERL reproduction.
"""
from __future__ import annotations
import copy
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import sys
import threading
import time
import traceback
from collections import deque

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from hilserl_pixels import PixelSAC, PixelPolicy, SuccessClassifier
from hilserl_data import Collection, save_rows, task_key
from hilserl_cameras import selected, validate_artifact
from agentlace import trainer
from agentlace.data.data_store import QueuedDataStore
from agentlace.internal.utils import make_compression_method
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


def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, outputs))


class Policy(nn.Module):
    def __init__(self, observation_dim=4, action_dim=3):
        super().__init__()
        self.net = mlp(observation_dim, action_dim * 2)

    def sample(self, obs, deterministic=False):
        mean, logstd = self.net(obs).chunk(2, dim=-1)
        dist = Normal(mean, logstd.clamp(-5, 2).exp())
        raw = mean if deterministic else dist.rsample()
        action = raw.tanh()
        logp = (dist.log_prob(raw) - torch.log(1 - action.square() + 1e-6)).sum(-1, keepdim=True)
        return action, logp

    def export(self):
        return {key: value.detach().cpu().numpy().copy() for key, value in self.state_dict().items()}

    def restore(self, weights):
        self.load_state_dict({key: torch.as_tensor(value.copy()) for key, value in weights.items()}, strict=True)


class SAC:
    def __init__(self, config, observation_dim=4, action_dim=3):
        self.config = config
        self.policy = Policy(observation_dim, action_dim)
        self.q1, self.q2 = mlp(observation_dim + action_dim, 1), mlp(observation_dim + action_dim, 1)
        self.targets = [copy.deepcopy(self.q1), copy.deepcopy(self.q2)]
        self.qopt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=config['learning_rate'])
        self.popt = torch.optim.Adam(self.policy.parameters(), lr=config['learning_rate'])
        self.steps = 0

    def update(self, rows):
        obs, act, reward, nxt, terminal = [torch.as_tensor(np.stack([r[key] for r in rows]), dtype=torch.float32) for key in ('observation', 'action', 'reward', 'next_observation', 'terminal')]
        reward, terminal = reward[:, None], terminal[:, None]
        with torch.no_grad():
            next_action, logp = self.policy.sample(nxt)
            pair = torch.cat([nxt, next_action], -1)
            target = reward + self.config['discount'] * (1 - terminal) * (torch.minimum(*(q(pair) for q in self.targets)) - .2 * logp)
        pair = torch.cat([obs, act], -1)
        qloss = sum((q(pair) - target).square().mean() for q in (self.q1, self.q2))
        self.qopt.zero_grad(); qloss.backward(); self.qopt.step()
        for q in (self.q1, self.q2):
            q.requires_grad_(False)
        sampled, logp = self.policy.sample(obs)
        pair = torch.cat([obs, sampled], -1)
        ploss = (.2 * logp - torch.minimum(self.q1(pair), self.q2(pair))).mean()
        self.popt.zero_grad(); ploss.backward(); self.popt.step()
        for q, target_q in zip((self.q1, self.q2), self.targets):
            q.requires_grad_(True)
            with torch.no_grad():
                for target_p, param in zip(target_q.parameters(), q.parameters()):
                    target_p.lerp_(param, .005)
        self.steps += 1
        return {'critic_loss': float(qloss.detach()), 'actor_loss': float(ploss.detach())}


class Replay(QueuedDataStore):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.interventions = deque(maxlen=capacity)
        self.sample_lock = threading.Lock()

    def insert(self, row):
        with self.sample_lock:
            super().insert(row)
            if row.get('human', False):
                self.interventions.append(row)

    def sample(self, size):
        with self.sample_lock:
            rows = list(self._data_queue)
            demos = list(self.interventions)
            count = size // 2 if demos else 0
            return random.choices(demos, k=count) + random.choices(rows, k=size-count)


# Upstream 0.1.4 binds all interfaces. Keep its protocol, but bind exclusively
# to ephemeral loopback ports; Agentlace uses pickle and must remain local.
class LocalReply(trainer.ReqRepServer):
    def reset(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.port = self.socket.bind_to_random_port('tcp://127.0.0.1')
        self.socket.setsockopt(zmq.RCVTIMEO, 250)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.is_kill = False


class LocalBroadcast(trainer.BroadcastServer):
    def __init__(self, port, **kwargs):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 3)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.port = self.socket.bind_to_random_port('tcp://127.0.0.1')
        self.compress, _ = make_compression_method('lz4')


def run(root, session_id, parent_pid=None):
    torch.set_num_threads(1)
    parent = parent_pid if parent_pid is not None else os.getppid()
    folder = root / session_id
    folder.mkdir(parents=True, exist_ok=True)
    def session():
        with sqlite3.connect(root / 'control.sqlite3', timeout=10) as db:
            return json.loads(db.execute('SELECT body FROM documents WHERE id=?', ('session:' + session_id,)).fetchone()[0])
    config = session()['config']
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
    passive = mode in ('preview', 'demo', 'classifier')
    collecting = mode in ('demo', 'classifier')
    collection = Collection(folder, config, mode, env.schema) if collecting else None
    batch_size = config.get('pixel_batch_size', 32) if pixel else config['batch_size']
    capacity = config.get('pixel_replay_capacity', 2048) if pixel else config['replay_capacity']
    learner = (PixelSAC if pixel else SAC)(config, env.observation_dim, env.action_dim)
    classifier = None
    if config['reward_source'] == 'classifier' and not passive:
        if not pixel:
            raise ValueError('Success classifier requires pixel observations')
        with np.load(config['classifier_path'], allow_pickle=False) as data:
            validate_artifact(data, selected(config, 'classifier'))
            if json.loads(str(data['__task__'])) != task_key(config):
                raise ValueError('Classifier task does not match this environment')
        classifier = SuccessClassifier.load(config['classifier_path'])
    classifier_hold = 0
    classifier_probability = None
    if not passive and config['checkpoint_path']:
        with np.load(config['checkpoint_path'], allow_pickle=False) as data:
            if pixel:
                validate_artifact(data, selected(config, 'observation'))
            weights = dict(data)
            weights.pop('__cameras__', None)
            schema = str(weights.pop('__schema__', 'mock-xyz-state4-delta3-v1'))
            if schema != env.schema:
                raise ValueError(f'Checkpoint schema {schema} does not match {env.schema}')
            saved_task = weights.pop('__task__', None)
            if pixel and (saved_task is None or json.loads(str(saved_task)) != task_key(config)):
                raise ValueError('Pixel policy task does not match this environment')
            if not all(np.isfinite(value).all() for value in weights.values()):
                raise ValueError('Policy checkpoint contains non-finite weights')
            learner.policy.restore(weights)
    replay = Replay(capacity)
    if not passive and config['demonstration_path']:
        with np.load(config['demonstration_path'], allow_pickle=False) as data:
            schema = str(data['__schema__']) if '__schema__' in data else 'mock-xyz-state4-delta3-v1'
            if schema != env.schema:
                raise ValueError('Dataset schema does not match the selected environment')
            if pixel and ('__task__' not in data or json.loads(str(data['__task__'])) != task_key(config)):
                raise ValueError('Pixel demonstration task does not match this environment')
            if pixel:
                validate_artifact(data, selected(config, 'observation'))
            count = len(data['reward'])
            shapes = {'observation': (count, env.observation_dim), 'action': (count, env.action_dim), 'reward': (count,),
                      'next_observation': (count, env.observation_dim), 'terminal': (count,), 'human': (count,)}
            if pixel:
                for key in ('images', 'next_images'):
                    shapes[key] = (count, 9, 64, 64)
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
            for i in range(count):
                row = {key: data[key][i] for key in ('observation', 'action', 'reward', 'next_observation', 'terminal', 'human')}
                if pixel:
                    row.update(images=data['images'][i], next_images=data['next_images'][i])
                row['truncated'] = bool(data['truncated'][i]) if 'truncated' in data else False
                replay.insert(row)
    trainer.ReqRepServer, trainer.BroadcastServer = LocalReply, LocalBroadcast
    transport = trainer.TrainerConfig()
    server = trainer.TrainerServer(transport)
    transport.port_number, transport.broadcast_port = server.req_rep_server.port, server.broadcast_server.port
    server.register_data_store('actor', replay)
    server.start(threaded=True)
    queue = QueuedDataStore(min(capacity, 256) if pixel else 4096)
    client = trainer.TrainerClient('actor', '127.0.0.1', transport, data_store=queue)
    actor = (PixelPolicy if pixel else Policy)(env.observation_dim, env.action_dim); actor.restore(learner.policy.export())
    mailbox, weight_lock = {}, threading.Lock()
    def receive(message):
        with weight_lock:
            mailbox.update(message)
    client.recv_network_callback(receive)
    stop, learning = threading.Event(), threading.Event()
    stats, stats_lock = {'learner_steps': 0, 'policy_version': 0}, threading.Lock()
    errors = []
    training_lock = threading.Lock()
    def learn():
        try:
            while not stop.wait(.02):
                if not learning.is_set() or len(replay) < batch_size:
                    continue
                with training_lock:
                    if not learning.is_set():
                        continue
                    losses = learner.update(replay.sample(batch_size))
                    if learner.steps % 5 == 0:
                        server.publish_network({'weights': learner.policy.export(), 'version': learner.steps})
                    with stats_lock:
                        stats.update(losses, learner_steps=learner.steps)
                    # Coalesce pending steps to bound CPU use.
                    learning.clear()
        except Exception:
            errors.append(traceback.format_exc())
            stop.set()
    thread = threading.Thread(target=learn, daemon=True); thread.start()
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
        nonlocal last_row, classifier_hold, classifier_probability
        last_row = None
        classifier_hold, classifier_probability = 0, None
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
                learning.clear()
                with training_lock:
                    pass  # Acknowledge pause only after an in-flight optimizer step.
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
                           'reward': 0. if truncated else (10. if success else -10.), 'next_observation': obs.copy(),
                           'terminal': not truncated, 'truncated': truncated, 'human': True, 'label_only': True,
                           'episode': episode, 'policy_version': version}
                    if pixel:
                        row.update(images=env.images.copy(), next_images=env.images.copy())
                    queue.insert(row)
                    log.write(json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in row.items() if k not in ('images', 'next_images')}) + '\n')
                    completed += 1; successes += int(success)
            last_revision = control['revision']
            if control['state'] == 'stopped':
                break
            if errors:
                raise RuntimeError(errors[0])
            with weight_lock:
                if mailbox.get('version', 0) > version:
                    actor.restore(mailbox['weights']); version = mailbox['version']
            if control['state'] == 'running' and not (collection and collection.target_reached):
                manual = control.get('manual_action', {})
                human = time.time() < manual.get('expires_at', 0) and any(manual.get('value', []))
                with torch.no_grad():
                    proposed = (np.zeros(env.action_dim, dtype=np.float32) if passive else
                                actor.sample(torch.as_tensor(obs)[None], deterministic=mode == 'evaluate', **({'images': torch.as_tensor(env.images)[None]} if pixel else {}))[0][0].numpy())
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
                    classifier_probability = classifier.probability(env.classifier_images)
                    classifier_hold = classifier_hold + 1 if classifier_probability >= config['classifier_threshold'] else 0
                    detected = classifier_hold >= config['classifier_hold_steps']
                    terminal = detected or bool(env.info.get('is_failure', False))
                    reward = 10. if detected else (-10. if env.info.get('is_failure', False) else 0.)
                    env.info['classifier_success'] = detected
                if config['reward_source'] == 'human':
                    reward, terminal = 0., bool(env.info.get('is_failure', False))
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
                if not collecting:
                    queue.insert(row)
                log.write(json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in row.items() if k not in ('images', 'next_images')}) + '\n')
                last_row = row
                obs = nxt; steps += 1; episode_step += 1; human_steps += int(human)
                if terminal or truncated:
                    if collecting:
                        completed += 1; successes += int(env.info.get('is_success', False))
                        # Pause at the boundary for capture/episode review. Do not auto-reset demonstrations.
                        collection.done = True
                        with sqlite3.connect(root / 'control.sqlite3', timeout=10) as db:
                            db.execute('BEGIN IMMEDIATE')
                            current = json.loads(db.execute('SELECT body FROM documents WHERE id=?', ('session:' + session_id,)).fetchone()[0])
                            if current['state'] == 'running':
                                current['state'] = 'paused'
                                current['manual_action'] = {}
                                current['classifier_label'] = {}
                                current['episode_outcome'] = 'success' if env.info.get('is_success') else ('failure' if terminal else 'truncated')
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
            if not client.update():
                raise RuntimeError('Agentlace replay transfer failed')
            if mode == 'train' and control['state'] == 'running':
                learning.set()
            else:
                learning.clear()
            with stats_lock:
                metrics = {**stats, 'environment_steps': steps, 'policy_version': version,
                           'replay_size': len(replay), 'human_steps': human_steps,
                           'success_rate': successes / completed if completed else None,
                           'completed_episodes': completed}
            status = {'heartbeat': time.time(), 'ack_revision': last_revision, 'state': control['state'],
                      'episode': episode, 'episode_step': episode_step, 'max_episode_steps': config['max_episode_steps'], 'metrics': metrics, 'position_m': env.position.tolist(),
                      'last_action': last_row['action'].tolist() if last_row else [0.] * env.action_dim,
                      'last_policy_action': last_row['proposed_action'].tolist() if last_row else [0.] * env.action_dim,
                      'policy_enabled': not passive,
                      'last_action_source': ('human' if last_row['human'] else ('hold' if passive else 'policy')) if last_row else 'hold',
                      'last_reward': float(last_row['reward']) if last_row else None,
                      'environment_info': dict(env.info), 'observation_dim': env.observation_dim,
                      'action_dim': env.action_dim, 'schema': env.schema,
                      'observation_mode': 'pixels' if pixel else 'state', 'image_shape': [9, 64, 64] if pixel else None,
                      'capture_metadata': env.capture_metadata if pixel else None,
                      'collection': collection.state() if collection else None,
                      'classifier_probability': classifier_probability, 'effective_batch_size': batch_size,
                      'effective_replay_capacity': capacity,
                      'checkpoint_path': str(folder / 'policy.npz'), 'dataset_path': str(folder / ('demonstrations.npz' if mode == 'demo' else 'replay.npz')),
                      'transport': 'Agentlace / loopback TCP', 'error': None}
            atomic_json(folder / 'runtime.json', status)
            log.flush()
            time.sleep(max(0, 1 / config['control_hz'] - (time.monotonic() - tick)))
    finally:
        stop.set(); thread.join(timeout=5)
        try:
            client.update()
            log.close()
            if collection:
                collection.export()
            if not passive:
                np.savez_compressed(folder / 'policy.npz', __schema__=np.array(env.schema), __task__=np.array(json.dumps(task_key(config), sort_keys=True)), __cameras__=np.array(json.dumps(selected(config, 'observation'))), **learner.policy.export())
                torch.save({'policy': learner.policy.state_dict(), 'q1': learner.q1.state_dict(), 'q2': learner.q2.state_dict(),
                            'targets': [q.state_dict() for q in learner.targets], 'policy_optimizer': learner.popt.state_dict(),
                            'critic_optimizer': learner.qopt.state_dict(), 'steps': learner.steps, 'config': config}, folder / 'learner.pt')
            with replay.sample_lock:
                rows = list(replay._data_queue)
                if rows:
                    save_rows(folder / 'replay.npz', rows, env.schema, config, pixel=pixel)
            status.update(heartbeat=time.time(), ack_revision=last_revision, state='stopped', artifacts_saved=True)
            if 'metrics' in status:
                status['metrics']['learner_steps'] = learner.steps
            atomic_json(folder / 'runtime.json', status)
        finally:
            client.stop()
            # Stop REP on its owning receive thread before closing its socket.
            server.req_rep_server.is_kill = True
            server.thread.join(timeout=2)
            server.req_rep_server.socket.close(); server.req_rep_server.context.term()
            server.broadcast_server.socket.close(); server.broadcast_server.context.term()



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
