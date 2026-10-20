"""Versioned image samples and selected demonstration episodes (no pickle)."""
import hashlib
import json
from pathlib import Path
import time
import uuid
import numpy as np
from hilserl_cameras import selected, route

FIELDS = ('observation', 'action', 'reward', 'next_observation', 'terminal', 'human', 'truncated')


def task_key(config):
    return {k: config[k] for k in ('environment', 'robot_type', 'arm', 'target_part')}


def write_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def save_rows(path, rows, schema, config, pixel=False):
    if not rows:
        raise ValueError('No demonstration transitions to save')
    fields = FIELDS + (('images', 'next_images') if pixel else ())
    write_npz(path, __schema__=np.array(schema), __task__=np.array(json.dumps(task_key(config), sort_keys=True)),
              __cameras__=np.array(json.dumps(selected(config, 'observation'))),
              **{key: np.stack([r[key] for r in rows]) for key in fields})


class Collection:
    def __init__(self, folder, config, mode, schema):
        self.folder, self.config, self.mode, self.schema = Path(folder), config, mode, schema
        self.samples = []
        self.rows = []
        self.saved = []
        self.done = False
        self.last_capture = None
        self.error = None

    @property
    def target_reached(self):
        return self.mode == 'classifier' and sum(s['label'] == 1 for s in self.samples) >= self.config.get('classifier_success_target', 200)

    def state(self):
        return {'mode': self.mode, 'success_target': self.config.get('classifier_success_target', 200), 'target_reached': self.target_reached, 'positive_samples': sum(s['label'] == 1 for s in self.samples),
                'negative_samples': sum(s['label'] == 0 for s in self.samples),
                'pending_demo_steps': len(self.rows), 'saved_demo_episodes': len(self.saved),
                'episode_finished': self.done, 'last_capture': self.last_capture, 'error': self.error}

    def capture(self, env, label, episode, *, fresh=True, automatic=False):
        # Same physical state cannot be submitted repeatedly or with contradictory labels.
        if self.target_reached:
            raise ValueError('Success sample target reached; start a new collection')
        if len(self.samples) >= 10000:
            raise ValueError('Collection limit reached (10,000 samples); stop this session')
        key = f"{episode}:{env.info['physics_steps']}"
        if any(sample['state_key'] == key for sample in self.samples):
            raise ValueError('This simulation state is already labeled; move or reset before capturing again')
        if fresh:
            env.capture()
        pixels = route(getattr(env, 'classifier_images', env.images), selected(self.config, 'classifier'))
        digest = hashlib.sha256(pixels.tobytes()).hexdigest()
        if not automatic and any(sample['image_hash'] == digest for sample in self.samples):
            raise ValueError('Identical camera sample already exists in this collection')
        identifier = str(uuid.uuid4())
        item = {'id': identifier, 'label': label, 'episode': episode, 'state_key': key,
                'image_hash': digest, 'timestamp': time.time(), 'source': 'step' if automatic else 'manual', 'capture': env.capture_metadata, 'cameras': selected(self.config, 'classifier')}
        write_npz(self.folder / 'classifier_samples' / f'{identifier}.npz', images=pixels, label=np.array(label))
        self.samples.append(item)
        self.last_capture = item
        manifest = {'task': task_key(self.config), 'cameras': selected(self.config, 'classifier'), 'samples': self.samples, 'success_target': self.config.get('classifier_success_target', 200), 'target_reached': self.target_reached}
        target = self.folder / 'classifier_samples' / 'manifest.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest))
        temporary.replace(target)
        return env.observation()

    def save_demo(self, episode):
        if not self.rows:
            raise ValueError('No pending demonstration to save')
        if sum(x['steps'] for x in self.saved) + len(self.rows) > 4096:
            raise ValueError('Demo collection limit reached (4,096 transitions); discard pending episode and stop')
        path = self.folder / 'demos' / f'episode-{episode:05d}.npz'
        # Mark the actual last transition, not an invented zero-action transition.
        self.rows[-1].update(reward=10., terminal=True, truncated=False)
        save_rows(path, self.rows, self.schema, self.config, pixel=True)
        self.saved.append({'episode': episode, 'steps': len(self.rows), 'path': str(path)})
        (self.folder / 'demos' / 'manifest.json').write_text(json.dumps({'episodes': self.saved, 'task': task_key(self.config), 'cameras': selected(self.config, 'observation')}))
        self.rows = []
        return path

    def export(self):
        if self.mode == 'demo' and self.rows:
            save_rows(self.folder / 'pending-demo.npz', self.rows, self.schema, self.config, pixel=True)
        if not self.saved:
            return None
        # A bounded collection session, independent of the online replay eviction policy.
        rows = []
        for episode in self.saved:
            with np.load(episode['path'], allow_pickle=False) as data:
                for i in range(len(data['reward'])):
                    rows.append({key: data[key][i] for key in FIELDS + ('images', 'next_images')})
        path = self.folder / 'demonstrations.npz'
        save_rows(path, rows, self.schema, self.config, pixel=True)
        return path
