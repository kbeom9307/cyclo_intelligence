from hilserl_profiles import image_shape
"""Versioned image samples and selected demonstration episodes (no pickle)."""
import hashlib
import json
from pathlib import Path
import time
import uuid
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
from hilserl_cameras import selected, route
from hilserl_profiles import upstream

FIELDS = ('observation', 'action', 'reward', 'next_observation', 'terminal', 'human', 'truncated')


def task_key(config):
    return {k: config[k] for k in ('environment', 'robot_type', 'arm', 'target_part')}


def environment_name(config):
    return (config.get('name') or '').strip() or 'Unnamed environment'


def classifier_episode_file(episode):
    episode = int(episode)
    if episode < 0:
        raise ValueError('Invalid classifier episode')
    return f'episode-{episode:05d}.npz'


class ClassifierSamples:
    """Read both per-image legacy files and episode archives without loading an episode into RAM."""
    def __init__(self, folder):
        self.folder = Path(folder)
        self.archives = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for archive in self.archives.values():
            archive.close()

    def read(self, sample):
        identifier = str(uuid.UUID(sample['id']))
        if sample.get('storage') == 'episode_npz_v1':
            filename = classifier_episode_file(sample['episode'])
            if filename not in self.archives:
                self.archives[filename] = np.load(self.folder / filename, allow_pickle=False)
            data = self.archives[filename]
            return data[f'images_{identifier}'], int(data[f'label_{identifier}'])
        try:
            with np.load(self.folder / f'{identifier}.npz', allow_pickle=False) as data:
                return data['images'], int(data['label'])
        except FileNotFoundError:
            # The writer may have finalized this episode after our manifest read.
            return self.read({**sample, 'storage': 'episode_npz_v1'})


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
              __crops__=np.array(json.dumps(config.get('camera_crops', {}), sort_keys=True)),
              **{key: np.stack([r[key] for r in rows]) for key in fields})


def write_classifier_episode(target, config, episode, samples, read_sample, heartbeat=None):
    """Publish one archive after streaming its images, keeping the previous file readable."""
    temporary = target.with_suffix('.tmp')
    last_heartbeat = time.monotonic()
    with ZipFile(temporary, 'w', compression=ZIP_DEFLATED, compresslevel=3) as archive:
        def put(key, value):
            with archive.open(key + '.npy', 'w', force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asarray(value), allow_pickle=False)
        put('__schema__', 'cyclo.classifier.episode.v1')
        put('__environment_name__', environment_name(config))
        put('__task__', json.dumps(task_key(config), sort_keys=True))
        put('__cameras__', json.dumps(selected(config, 'classifier')))
        put('__crops__', json.dumps(config.get('camera_crops', {}), sort_keys=True))
        put('episode', episode)
        put('sample_ids', [sample['id'] for sample in samples])
        for sample in samples:
            pixels, label = read_sample(sample)
            put(f"images_{sample['id']}", pixels)
            put(f"label_{sample['id']}", label)
            if heartbeat and time.monotonic() - last_heartbeat >= .5:
                heartbeat()
                last_heartbeat = time.monotonic()
    temporary.replace(target)


class Collection:
    def __init__(self, folder, config, mode, schema, *, wait_for_intervention=False):
        self.folder, self.config, self.mode, self.schema = Path(folder), config, mode, schema
        self.samples = []
        self.rows = []
        self.demo_classifier_frames = []
        self.saved = []
        self.done = False
        self.last_capture = None
        self.error = None
        self.episode_step = 0
        self.skipped_negative_samples = 0
        self.waiting_for_intervention = wait_for_intervention

    @property
    def failure_step_limit(self):
        return self.config.get('collection_episode_steps', 1000)

    @property
    def success_only(self):
        return self.mode == 'classifier' and self.episode_step >= self.failure_step_limit

    @property
    def target_reached(self):
        return self.mode == 'classifier' and sum(s['label'] == 1 for s in self.samples) >= self.config.get('classifier_success_target', 200)

    def state(self):
        return {'mode': self.mode, 'success_target': self.config.get('classifier_success_target', 200), 'target_reached': self.target_reached, 'positive_samples': sum(s['label'] == 1 for s in self.samples),
                'negative_samples': sum(s['label'] == 0 for s in self.samples),
                'failure_step_limit': self.failure_step_limit, 'success_only': self.success_only,
                'skipped_negative_samples': self.skipped_negative_samples,
                'waiting_for_intervention': self.waiting_for_intervention,
                'pending_demo_steps': len(self.rows), 'saved_demo_episodes': len(self.saved),
                'pending_demo_classifier_samples': len(self.demo_classifier_frames),
                'episode_finished': self.done, 'last_capture': self.last_capture, 'error': self.error}

    def capture(self, env, label, episode, *, fresh=True, automatic=False):
        if self.waiting_for_intervention:
            if not automatic:
                raise ValueError('Move the arm with an intervention input before capturing classifier data')
            return env.observation()
        if self.last_capture and self.last_capture['episode'] != episode:
            self.finish_episode(self.last_capture['episode'], getattr(env, 'heartbeat', None))
        if self.mode == 'classifier' and not label:
            # Automatic samples belong to the step just completed: retain 1..N.
            # A manual capture after the Nth step must not extend that window.
            outside_window = self.episode_step > self.failure_step_limit if automatic else self.success_only
            if outside_window:
                if not automatic:
                    raise ValueError('Failure collection step limit reached; capture success or reset the episode')
                self.skipped_negative_samples += 1
                return env.observation()
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
        self.write_classifier_manifest(self.samples)
        return env.observation()

    def write_classifier_manifest(self, samples):
        manifest = {'task': task_key(self.config), 'environment_name': environment_name(self.config),
                    'cameras': selected(self.config, 'classifier'), 'samples': samples, 'success_target': self.config.get('classifier_success_target', 200), 'target_reached': self.target_reached,
                    'image_size': image_shape(self.config)[1], 'learning_backend': self.config.get('learning_backend', 'legacy_torch'), 'camera_crops': self.config.get('camera_crops', {})}
        target = self.folder / 'classifier_samples' / 'manifest.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest))
        temporary.replace(target)

    def finish_episode(self, episode, heartbeat=None):
        if self.mode != 'classifier':
            return
        samples = [sample for sample in self.samples if sample['episode'] == episode]
        pending = [sample for sample in samples if sample.get('storage') != 'episode_npz_v1']
        if not pending:
            return
        folder = self.folder / 'classifier_samples'
        target = folder / classifier_episode_file(episode)
        # Stream one image at a time. Packing never holds the full collection in
        # memory and does not rewrite a growing archive during the control loop.
        with ClassifierSamples(folder) as source:
            write_classifier_episode(target, self.config, episode, samples, source.read, heartbeat)
        updated = [{**sample, 'storage': 'episode_npz_v1'} if sample['episode'] == episode else sample for sample in self.samples]
        # Publish the archive and manifest before removing any staged images.
        # A reader with the old manifest can still fall back to the new archive.
        self.write_classifier_manifest(updated)
        self.samples = updated
        if self.last_capture and self.last_capture['episode'] == episode:
            self.last_capture = next(sample for sample in updated if sample['id'] == self.last_capture['id'])
        for sample in pending:
            (folder / f"{sample['id']}.npz").unlink(missing_ok=True)

    def record_demo(self, row, env):
        # Keep the post-action view used for success detection, including the
        # terminal false positive. Policy cameras may exclude classifier views.
        cameras = selected(self.config, 'classifier')
        pixels = (row['next_images'] if cameras == selected(self.config, 'observation') else
                  route(env.classifier_images, cameras))
        self.demo_classifier_frames.append({
            'images': pixels, 'capture': dict(env.capture_metadata or {}),
            'physics_steps': env.info.get('physics_steps', row['step_id']),
            'timestamp': row['timestamp'],
        })
        self.rows.append(row)

    def clear_pending_demo(self):
        self.rows = []
        self.demo_classifier_frames = []

    def discard_demo(self, episode, *, add_classifier_failures=False, heartbeat=None):
        if self.mode != 'demo' or not self.rows:
            raise ValueError('No pending demonstration to discard')
        if add_classifier_failures:
            frames = self.demo_classifier_frames
            if len(frames) != len(self.rows):
                raise ValueError('Classifier views unavailable; start a new demo session to record them')
            if len(self.samples) + len(frames) > 10000:
                raise ValueError('Classifier collection limit reached (10,000 samples); discard only or start a new session')
            cameras = selected(self.config, 'classifier')
            added, images = [], {}
            for frame in frames:
                identifier = str(uuid.uuid4())
                images[identifier] = frame['images']
                added.append({'id': identifier, 'label': 0, 'episode': episode,
                              'state_key': f"{episode}:{frame['physics_steps']}",
                              'image_hash': hashlib.sha256(frame['images'].tobytes()).hexdigest(),
                              'timestamp': frame['timestamp'], 'source': 'discarded_demo',
                              'capture': frame['capture'], 'cameras': cameras, 'storage': 'episode_npz_v1'})
            updated = self.samples + added
            folder = self.folder / 'classifier_samples'
            folder.mkdir(parents=True, exist_ok=True)
            with ClassifierSamples(folder) as source:
                write_classifier_episode(folder / classifier_episode_file(episode), self.config, episode,
                                         [sample for sample in updated if sample['episode'] == episode],
                                         lambda sample: (images[sample['id']], 0) if sample['id'] in images else source.read(sample),
                                         heartbeat)
            # Clear the demo only after the archive and its manifest are published.
            # On a write failure the operator can retry with all pending frames.
            self.write_classifier_manifest(updated)
            self.samples = updated
            self.last_capture = added[-1]
        self.clear_pending_demo()

    def save_demo(self, episode):
        if not self.rows:
            raise ValueError('No pending demonstration to save')
        if sum(x['steps'] for x in self.saved) + len(self.rows) > 4096:
            raise ValueError('Demo collection limit reached (4,096 transitions); discard pending episode and stop')
        path = self.folder / 'demos' / f'episode-{episode:05d}.npz'
        # Mark the actual last transition, not an invented zero-action transition.
        self.rows[-1].update(reward=1. if upstream(self.config) else 10., terminal=True, truncated=False)
        save_rows(path, self.rows, self.schema, self.config, pixel=True)
        saved = self.saved + [{'episode': episode, 'steps': len(self.rows), 'path': str(path)}]
        manifest = self.folder / 'demos' / 'manifest.json'
        temporary = manifest.with_suffix('.tmp')
        temporary.write_text(json.dumps({'episodes': saved, 'task': task_key(self.config), 'environment_name': environment_name(self.config), 'cameras': selected(self.config, 'observation'),
                                           'image_size': image_shape(self.config)[1], 'learning_backend': self.config.get('learning_backend', 'legacy_torch'), 'camera_crops': self.config.get('camera_crops', {})}))
        temporary.replace(manifest)
        self.saved = saved
        self.clear_pending_demo()
        return path

    def export(self, heartbeat=None):
        if self.mode == 'classifier':
            for episode in sorted({sample['episode'] for sample in self.samples}):
                self.finish_episode(episode, heartbeat)
            return None
        if self.mode == 'demo' and self.rows:
            save_rows(self.folder / 'pending-demo.npz', self.rows, self.schema, self.config, pixel=True)
        if not self.saved:
            return None
        # A bounded collection session, independent of the online replay eviction policy.
        rows = []
        for episode in self.saved:
            with np.load(episode['path'], allow_pickle=False) as data:
                arrays = {key: data[key] for key in FIELDS + ('images', 'next_images')}
                for i in range(len(arrays['reward'])):
                    rows.append({key: value[i] for key, value in arrays.items()})
        path = self.folder / 'demonstrations.npz'
        save_rows(path, rows, self.schema, self.config, pixel=True)
        return path
