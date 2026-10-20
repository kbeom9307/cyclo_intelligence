"""Learner engine and asynchronous local/remote execution, with CPU replay storage."""
from collections import deque
from itertools import islice
import json
from pathlib import Path
import threading
import time

import numpy as np
from hilserl_profiles import prepare_jax_libraries
prepare_jax_libraries()
import torch
from agentlace import trainer
from agentlace.data.data_store import QueuedDataStore
from hilserl_learning import SAC, Replay, LocalReply, LocalBroadcast
from hilserl_pixels import PixelSAC
from hilserl_devices import memory_status
from hilserl_data import save_rows, task_key, write_npz
from hilserl_cameras import selected
from hilserl_profiles import upstream

from hilserl_remote import RemoteEndpoint, PROTOCOL


def make_spec(config, env):
    # External learner never reads actor-side paths or credentials.
    fields = ('environment', 'robot_type', 'arm', 'target_part', 'observation_cameras', 'seed',
              'learning_rate', 'discount', 'learner_device', 'pixel_batch_size', 'pixel_replay_capacity',
              'batch_size', 'replay_capacity', 'learning_backend', 'cta_ratio', 'training_starts', 'policy_publish_interval',
              'camera_crops')
    return {'protocol': PROTOCOL, 'pixel': bool(getattr(env, 'pixel_mode', False)), 'schema': env.schema,
            'observation_dim': env.observation_dim, 'action_dim': env.action_dim,
            'config': {key: config[key] for key in fields if key in config}}


class LearnerEngine:
    """Single owner thread; local Agentlace remains bound to loopback (pickle)."""
    def __init__(self, spec, folder, weights=None):
        self.spec, self.config, self.folder = spec, spec['config'], Path(folder)
        self.pixel = spec['pixel']
        self.upstream = self.pixel and upstream(self.config)
        if self.upstream:
            from hilserl_upstream import UpstreamSAC
            from hilserl_replay import DiskReplay
            self.learner = UpstreamSAC(self.config, spec['observation_dim'], spec['action_dim'])
        else:
            self.learner = (PixelSAC if self.pixel else SAC)(self.config, spec['observation_dim'], spec['action_dim'])
        if weights:
            if self.upstream:
                self.learner.restore_policy(weights)
            else:
                self.learner.policy.restore(weights)
        self.capacity = self.config['pixel_replay_capacity' if self.pixel else 'replay_capacity']
        self.batch_size = self.config['pixel_batch_size' if self.pixel else 'batch_size']
        self.replay = DiskReplay(self.folder / 'replay', spec) if self.upstream else Replay(self.capacity)
        trainer.ReqRepServer, trainer.BroadcastServer = LocalReply, LocalBroadcast
        transport = trainer.TrainerConfig()
        self.server = trainer.TrainerServer(transport)
        transport.port_number, transport.broadcast_port = self.server.req_rep_server.port, self.server.broadcast_server.port
        self.server.register_data_store('actor', self.replay)
        self.server.start(threaded=True)
        self.queue = QueuedDataStore(256)
        self.client = trainer.TrainerClient('actor', '127.0.0.1', transport, data_store=self.queue)
        self.metrics = {'learner_steps': 0, 'replay_size': 0}
        self.closed = False

    def snapshot(self, version=-1, final=False):
        step = self.learner.steps
        interval = self.config.get('policy_publish_interval', 50) if self.upstream else 5
        weights = self.learner.policy.export() if version < 0 or final or (step > version and step % interval == 0) else {}
        return {'metrics': dict(self.metrics), 'version': step if weights else version,
                'learner_memory': self.learner.memory() if self.upstream else memory_status(self.learner.device), 'artifacts_saved': self.closed,
                'artifact_directory': str(self.folder)}, weights

    def exchange(self, rows, learning, version, offline=False):
        for row in rows:
            if offline and self.upstream:
                self.replay.insert_demo(row)
            else:
                self.queue.insert(row)
        if not self.client.update():
            raise RuntimeError('Agentlace replay transfer failed')
        ready = self.replay.ready(self.config.get('training_starts', 100)) if self.upstream else len(self.replay) >= self.batch_size
        if learning and ready:
            if self.upstream:
                for _ in range(self.config.get('cta_ratio', 2) - 1):
                    self.learner.update(self.replay.sample(self.batch_size), critic_only=True)
            self.metrics.update(self.learner.update(self.replay.sample(self.batch_size)))
            if not self.upstream and self.learner.steps % 5 == 0:
                self.server.publish_network({'weights': self.learner.policy.export(), 'version': self.learner.steps})
        self.metrics.update(learner_steps=self.learner.steps, replay_size=len(self.replay))
        if self.upstream:
            self.metrics.update(demo_replay_size=self.replay.demo.size, replay_ready=ready)
        return self.snapshot(version)

    def close(self):
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            (self.folder / 'spec.json').write_text(json.dumps(self.spec, indent=2))
            write_npz(self.folder / 'policy.npz', __schema__=np.array(self.spec['schema']),
                      __task__=np.array(json.dumps(task_key(self.config), sort_keys=True)),
                      __cameras__=np.array(json.dumps(selected(self.config, 'observation'))),
                      __crops__=np.array(json.dumps(self.config.get('camera_crops', {}), sort_keys=True)), **self.learner.policy.export())
            learner = self.learner
            if self.upstream:
                learner.save(self.folder)
                self.replay.save()
            else:
                torch.save({'policy': learner.policy.state_dict(), 'q1': learner.q1.state_dict(), 'q2': learner.q2.state_dict(),
                        'targets': [q.state_dict() for q in learner.targets], 'policy_optimizer': learner.popt.state_dict(),
                        'critic_optimizer': learner.qopt.state_dict(), 'steps': learner.steps, 'config': self.config}, self.folder / 'learner.pt')
                with self.replay.sample_lock:
                    rows = list(self.replay._data_queue)
                if rows:
                    save_rows(self.folder / 'replay.npz', rows, self.spec['schema'], self.config, pixel=self.pixel)
            self.closed = True
            return self.snapshot(final=True)
        finally:
            self.client.stop()
            self.server.req_rep_server.is_kill = True
            self.server.thread.join(timeout=2)
            self.server.req_rep_server.socket.close(); self.server.req_rep_server.context.term()
            self.server.broadcast_server.socket.close(); self.server.broadcast_server.context.term()


class AsyncLearner:
    """Network/optimizer work never runs in the actor control loop. Bounded, lossless queue."""
    def __init__(self, config, spec, folder, weights, demos, heartbeat=lambda: None):
        self.remote = config.get('learner_placement', 'local') == 'remote'
        restored = spec['pixel'] and upstream(config)
        self.transport = 'HTTPS/NPZ → remote Agentlace / SAC' if self.remote else 'Agentlace / loopback TCP'
        if restored:
            self.transport += ' · upstream HIL-SERL / JAX / ResNet-10'
        self.endpoint = None
        self.spec = spec
        self.pending = deque()
        self.condition = threading.Condition()
        self.learning = False
        self.stop_requested = False
        self.busy = False
        self.error = None
        self.meta, self.weights = {}, {}
        self.generation = 0
        self.completed_generation = 0
        self.ready = threading.Event()
        # All ZMQ client operations stay on their owning thread.
        def worker():
            try:
                if self.remote:
                    self.endpoint = RemoteEndpoint(config)
                    self.meta, self.weights = self.endpoint.open(spec, weights)
                else:
                    self.endpoint = LearnerEngine(spec, folder, weights)
                    self.meta, self.weights = self.endpoint.snapshot()
                for i in range(0, len(demos), 32):
                    # The frozen ResNet policy is ~33 MiB. Do not download the
                    # unchanged network again for every demonstration chunk.
                    self.endpoint.exchange(demos[i:i + 32], False, self.meta.get('version', 0), offline=True)
                self.meta, self.weights = self.endpoint.exchange([], False, -1)
                self.ready.set()
                while True:
                    with self.condition:
                        self.condition.wait_for(lambda: self.stop_requested or self.pending or self.generation > self.completed_generation or (restored and self.learning), timeout=.5)
                        if self.stop_requested and not self.pending:
                            break
                        rows = list(islice(self.pending, 32))
                        generation, learning = self.generation, self.learning and not self.stop_requested
                        self.busy = True
                        version = self.meta.get('version', -1)
                    meta, new_weights = self.endpoint.exchange(rows, learning and (restored or generation > self.completed_generation), version)
                    with self.condition:
                        for _ in rows:
                            self.pending.popleft()
                        self.meta = meta
                        if new_weights:
                            self.weights = new_weights
                        self.busy = False
                        self.completed_generation = generation
                        self.condition.notify_all()
                    if restored and not meta.get('metrics', {}).get('replay_ready'):
                        time.sleep(.05)
            except Exception as exc:
                self.error = f'{type(exc).__name__}: {exc}'
                self.ready.set()
            finally:
                if self.pending:
                    Path(folder).mkdir(parents=True, exist_ok=True)
                    save_rows(Path(folder) / 'unsent-replay.npz', list(self.pending), spec['schema'], config, pixel=spec['pixel'])
                if self.endpoint:
                    try:
                        meta, weights_out = self.endpoint.close()
                        with self.condition:
                            self.meta = meta
                            if weights_out:
                                self.weights = weights_out
                    except Exception as exc:
                        self.error = self.error or f'Learner finalization failed: {exc}'
                with self.condition:
                    self.busy = False
                    self.condition.notify_all()
        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()
        while not self.ready.wait(.25):
            heartbeat()
        self.check()

    def check(self):
        if self.error:
            raise RuntimeError(self.error)

    def insert(self, row):
        with self.condition:
            self.check()
            if len(self.pending) >= 256:
                raise RuntimeError('Learner connection cannot keep up: transition queue is full (256); session stopped without dropping queued data')
            self.pending.append(row)

    def tick(self, running):
        with self.condition:
            self.check()
            self.learning = running
            self.generation += 1
            self.condition.notify_all()

    def pause(self, heartbeat=lambda: None):
        deadline = time.monotonic() + 300
        with self.condition:
            self.learning = False
        while True:
            with self.condition:
                if self.condition.wait_for(lambda: not self.busy or self.error, timeout=.25):
                    break
            heartbeat()
            if time.monotonic() > deadline:
                raise RuntimeError('Learner did not acknowledge pause within 300 seconds')
        with self.condition:
            self.check()

    def snapshot(self):
        with self.condition:
            self.check()
            return {**self.meta, 'pending_transitions': len(self.pending), 'placement': 'remote' if self.remote else 'local'}, self.weights

    def close(self, heartbeat=lambda: None):
        with self.condition:
            self.learning = False
            self.stop_requested = True
            self.condition.notify_all()
        deadline = time.monotonic() + 300
        while self.thread.is_alive():
            self.thread.join(timeout=.25)
            heartbeat()
            if time.monotonic() > deadline:
                raise RuntimeError('Learner did not finish saving within 300 seconds')
        self.check()
        return self.meta, self.weights
