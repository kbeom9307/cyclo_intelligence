"""MuJoCo classifier collection starts with the first effective manual action."""
import json
import os

import numpy as np
import pytest

import hilserl as hil
import hilserl_runtime as runtime
from hilserl_data import Collection
from hilserl_pixels import PIXEL_SCHEMA


@pytest.mark.parametrize('intervene', [False, True])
@pytest.mark.parametrize('mode', ['classifier', 'demo'])
def test_wait_for_intervention_then_keep_collecting_after_release(tmp_path, monkeypatch, intervene, mode):
    config = hil.ExperimentConfig(environment='mujoco', learning_backend='legacy_torch',
                                actor_device='cpu', collection_episode_steps=2,
                                max_episode_steps=2 if mode == 'classifier' else 100).model_dump()
    current = dict(mode=mode, state='running', revision=0, events=[])
    iteration = 0
    snapshots = []

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
        actions = []
        heartbeats = 0

        def observation(self):
            return np.zeros(22, dtype=np.float32)

        def heartbeat(self):
            self.heartbeats += 1

        def tcp_to_action(self, action):
            return action

        def step(self, action):
            self.actions.append(action.copy())
            step = len(self.actions)
            self.info = dict(physics_steps=step, is_success=False, is_failure=False)
            self.images = self.classifier_images = np.full((9, 64, 64), step, dtype=np.uint8)
            return self.observation(), 0., False, step >= config['max_episode_steps']

        def reset(self):
            pytest.fail('Waiting for input must not reset the scene')

    env = Env()

    def session():
        # Space, a takeover event, a zero action and an expired movement pulse
        # must not start recording. Nor may movement received while paused.
        state = 'stopped' if iteration >= 9 else 'paused' if iteration in (2, 6) else 'running'
        value = [0.] * 6
        expires_at = runtime.time.time() + 10
        if iteration in (1, 2) or (iteration == 4 and intervene):
            value[0] = 1.
        if iteration == 1:
            expires_at = runtime.time.time() - 1
        return {**current, 'state': state, 'revision': 1,
                'events': [{'sequence': 1, 'action': 'take_over'}],
                'manual_action': {'value': value, 'expires_at': expires_at},
                'classifier_label': {'success': iteration in (0, 8), 'expires_at': runtime.time.time() + 10}}

    def tick(_):
        nonlocal iteration
        snapshots.append(json.loads((tmp_path / 'runtime.json').read_text()))
        iteration += 1

    monkeypatch.setattr(runtime.time, 'sleep', tick)
    runtime.train(tmp_path, 'test-collection', os.getppid(), tmp_path, config, env, session)
    assert env.heartbeats >= 9
    for snapshot in snapshots[:4]:
        assert snapshot['episode_step'] == snapshot['metrics']['environment_steps'] == 0
        assert snapshot['collection']['waiting_for_intervention']
        assert snapshot['collection']['positive_samples'] == snapshot['collection']['negative_samples'] == 0
        assert snapshot['collection']['pending_demo_steps'] == snapshot['collection']['pending_demo_classifier_samples'] == 0
        assert not snapshot['collection']['success_only']
    if not intervene:
        assert not env.actions
        assert not (tmp_path / 'classifier_samples').exists()
        assert not (tmp_path / 'pending-demo.npz').exists()
        assert all(snapshot['collection']['waiting_for_intervention'] for snapshot in snapshots)
        return
    assert len(env.actions) == 4
    np.testing.assert_array_equal(env.actions[0], [1, 0, 0, 0, 0, 0])
    assert all(not action.any() for action in env.actions[1:])
    assert all(not snapshot['collection']['waiting_for_intervention'] for snapshot in snapshots[4:])
    if mode == 'demo':
        assert snapshots[-1]['collection']['pending_demo_steps'] == 4
        assert snapshots[-1]['collection']['pending_demo_classifier_samples'] == 4
        with np.load(tmp_path / 'pending-demo.npz', allow_pickle=False) as data:
            assert len(data['reward']) == 4
            np.testing.assert_array_equal(data['action'][0], [1, 0, 0, 0, 0, 0])
        assert not (tmp_path / 'classifier_samples').exists()
        return
    manifest = json.loads((tmp_path / 'classifier_samples' / 'manifest.json').read_text())
    assert [sample['label'] for sample in manifest['samples']] == [0, 0, 1]
    assert [sample['state_key'] for sample in manifest['samples']] == ['1:1', '1:2', '1:4']
    assert snapshots[-1]['episode_step'] == 4
    assert snapshots[-1]['collection']['skipped_negative_samples'] == 1
    assert (tmp_path / 'classifier_samples' / 'episode-00001.npz').is_file()


def test_manual_capture_cannot_bypass_initial_intervention_gate(tmp_path):
    class Env:
        def observation(self):
            return np.zeros(22)
        def capture(self):
            pytest.fail('Must not request an image before the first intervention')
    collection = Collection(tmp_path, {}, 'classifier', PIXEL_SCHEMA, wait_for_intervention=True)
    for label in (0, 1):
        with pytest.raises(ValueError, match='intervention input'):
            collection.capture(Env(), label, 1)
        collection.capture(Env(), label, 1, automatic=True)
    assert not collection.samples
    assert not (tmp_path / 'classifier_samples').exists()
