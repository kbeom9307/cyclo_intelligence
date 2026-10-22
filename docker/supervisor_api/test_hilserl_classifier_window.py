"""Classifier limits stop negative sampling without resetting the scene."""
import json
import os
import numpy as np
import pytest
import hilserl as hil
from hilserl_data import Collection
from hilserl_pixels import PIXEL_SCHEMA
import hilserl_runtime as runtime


@pytest.mark.parametrize('physical_failure', [False, True])
def test_classifier_continues_past_limits_until_target_or_physical_failure(tmp_path, monkeypatch, physical_failure):
    monkeypatch.setenv('CYCLO_HILSERL_DATA_DIR', str(tmp_path))
    monkeypatch.setattr(runtime.time, 'sleep', lambda _: None)
    config = hil.ExperimentConfig(
        environment='mujoco', learning_backend='legacy_torch', actor_device='cpu',
        collection_episode_steps=2, max_episode_steps=3, classifier_success_target=2,
    ).model_dump()
    identifier = 'classifier-window'
    current = dict(id=identifier, config=config, mode='classifier', state='running',
                   revision=0, events=[], metrics={}, episode=1)
    with hil.database() as db:
        hil.write(db, 'session:' + identifier, current)

    class Env:
        pixel_mode = True
        observation_dim = 22
        action_dim = 6
        schema = PIXEL_SCHEMA
        images = np.zeros((9, 64, 64), dtype=np.uint8)
        classifier_images = images
        capture_metadata = {}
        position = np.zeros(3)
        info = {'is_success': False, 'is_failure': False, 'physics_steps': 0}
        steps = 0

        def observation(self):
            return np.zeros(22, dtype=np.float32)

        def heartbeat(self):
            pass

        def tcp_to_action(self, action):
            return action

        def step(self, action):
            self.steps += 1
            assert self.steps <= 6, 'Success target failed to stop collection'
            self.info = dict(physics_steps=self.steps, is_success=self.steps >= 4 and not physical_failure,
                             is_failure=physical_failure and self.steps >= 4)
            self.images = self.classifier_images = np.full((9, 64, 64), self.steps, dtype=np.uint8)
            return self.observation(), float(self.info['is_success']), self.steps >= 4, self.steps >= 3

        def reset(self):
            pytest.fail('Time limit must not reset the classifier scene')

    env = Env()
    boundaries = []

    def session():
        with hil.database() as db:
            value = hil.read(db, 'session:' + identifier)
        if env.steps == 0:
            value['manual_action'] = {'value': [1, 0, 0, 0, 0, 0], 'expires_at': runtime.time.time() + 10}
        if value['state'] == 'paused':
            boundaries.append(value['episode_outcome'])
            if len(boundaries) > 1:
                value['state'] = 'stopped'
        if env.steps >= 4:
            value['classifier_label'] = {'success': True, 'expires_at': runtime.time.time() + 10}
        return value

    folder = tmp_path / identifier
    folder.mkdir()
    runtime.train(tmp_path, identifier, os.getppid(), folder, config, env, session)
    status = json.loads((folder / 'runtime.json').read_text())
    manifest = json.loads((folder / 'classifier_samples/manifest.json').read_text())
    labels = [sample['label'] for sample in manifest['samples']]
    assert labels.count(0) == 2
    assert status['episode'] == 1
    assert status['collection']['success_only']
    assert status['collection']['skipped_negative_samples'] == 2
    assert status['metrics']['learner_steps'] == 0
    if physical_failure:
        assert env.steps == 4 and boundaries == ['failure', 'failure']
        assert labels == [0, 0]
    else:
        assert not boundaries
        assert env.steps == 6 and labels == [0, 0, 1, 1]
        assert status['state'] == 'stopped' and status['collection']['target_reached']
        assert {sample['episode'] for sample in manifest['samples']} == {1}


def test_failure_window_enforces_manual_capture_and_reopens_on_reset(tmp_path):
    config = hil.ExperimentConfig(learning_backend='legacy_torch', collection_episode_steps=2).model_dump()
    collection = Collection(tmp_path, config, 'classifier', PIXEL_SCHEMA)

    class Env:
        info = {'physics_steps': 1}
        images = np.zeros((9, 64, 64), dtype=np.uint8)
        capture_metadata = {}

        def observation(self):
            return np.zeros(22)

        def capture(self):
            pass

    env = Env()
    collection.episode_step = 2
    collection.capture(env, 0, 1, automatic=True)
    assert collection.state()['negative_samples'] == 1
    with pytest.raises(ValueError, match='Failure collection step limit'):
        collection.capture(env, 0, 1)
    collection.episode_step = 3
    env.info['physics_steps'] = 2
    collection.capture(env, 0, 1, automatic=True)
    assert collection.state()['negative_samples'] == 1
    collection.capture(env, 1, 1, automatic=True)
    assert collection.state()['positive_samples'] == 1
    collection.episode_step = 0
    assert not collection.success_only
    collection.episode_step = 1
    collection.capture(env, 0, 2, automatic=True)
    assert collection.state()['negative_samples'] == 2
