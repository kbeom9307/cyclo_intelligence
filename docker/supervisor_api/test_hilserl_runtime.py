"""Numerical task / optimizer checks; CPU only, no hardware dependencies."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip('torch')
pytest.importorskip('agentlace')
spec = importlib.util.spec_from_file_location('hil_runtime_test', Path(__file__).with_name('hilserl_runtime.py'))
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)

CONFIG = dict(seed=0, success_depth_m=.02, success_hold_s=.2, control_hz=10,
              success_tolerance_m=.002, translation_limit_m=.005, max_episode_steps=100,
              learning_rate=.0003, discount=.99)


def test_plate_collision_depth_hold_and_timeout():
    env = runtime.MockPegEnv(CONFIG)
    env.position = np.array([.02, .02, .001], dtype=np.float32)
    env.step([0, 0, -1])
    assert env.position[2] == 0
    env.position = np.array([0, 0, -.019], dtype=np.float32)
    assert not env.step([0, 0, -1])[2]
    assert env.step([0, 0, 0])[2]
    env.reset()
    env.steps = 99
    assert env.step([0, 0, 0])[3]


def test_sac_changes_weights_and_interventions_are_sampled():
    runtime.torch.set_num_threads(1)
    learner = runtime.SAC(CONFIG)
    before = learner.policy.export()
    store = runtime.Replay(100)
    for i in range(20):
        store.insert(dict(observation=np.zeros(4), action=np.zeros(3), reward=1.,
                          next_observation=np.ones(4) * .1, terminal=False, human=i == 0))
    rows = store.sample(10)
    assert sum(r['human'] for r in rows) >= 5
    metrics = learner.update(rows)
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(not np.array_equal(before[k], v) for k, v in learner.policy.export().items())


def test_agentlace_server_binds_loopback_only():
    reply = runtime.LocalReply()
    broadcast = runtime.LocalBroadcast(0)
    for endpoint in (reply, broadcast):
        assert endpoint.socket.getsockopt_string(runtime.zmq.LAST_ENDPOINT).startswith('tcp://127.0.0.1:')
        endpoint.socket.close(); endpoint.context.term()


@pytest.mark.parametrize('review_action', [None, 'discard_demo', 'discard_demo_as_failure'])
def test_demo_classifier_ends_episode_without_environment_success(tmp_path, monkeypatch, review_action):
    import json
    import os
    import sqlite3
    import hilserl as hil
    from hilserl_pixels import SuccessClassifier, CLASSIFIER_SCHEMA, PIXEL_SCHEMA
    from hilserl_data import task_key, ClassifierSamples
    from hilserl_cameras import route
    monkeypatch.setenv('CYCLO_HILSERL_DATA_DIR', str(tmp_path))
    runtime.torch.set_num_threads(1)
    config = hil.ExperimentConfig(environment='isaac_sim', observation_mode='pixels', reward_source='classifier', learning_backend='legacy_torch',
        observation_cameras=['head'], classifier_cameras=['right_wrist'], actor_device='cpu', learner_device='cpu',
        classifier_path=str(tmp_path / 'classifier.npz'), classifier_hold_steps=2).model_dump()
    model = SuccessClassifier()
    with runtime.torch.no_grad():
        for parameter in model.parameters(): parameter.zero_()
        model.head.bias.fill_(8.)
    np.savez_compressed(config['classifier_path'], __schema__=CLASSIFIER_SCHEMA,
        __task__=json.dumps(task_key(config)), __cameras__=json.dumps(['right_wrist']),
        **{k:v.detach().numpy() for k,v in model.state_dict().items()})
    current = dict(id='demo-test', config=config, mode='demo', state='running', revision=0, events=[], metrics={}, episode=1)
    with hil.database() as db: hil.write(db, 'session:demo-test', current)
    class Env:
        pixel_mode=True; observation_dim=22; action_dim=6; schema=PIXEL_SCHEMA
        images=np.zeros((9,64,64), dtype=np.uint8)
        classifier_images=images; capture_metadata={}; position=np.zeros(3)
        info={'is_success':False, 'is_failure':False}
        steps=0
        def observation(self): return np.zeros(22, dtype=np.float32)
        def heartbeat(self): pass
        def step(self, action):
            self.steps += 1
            assert self.steps <= 2, 'Classifier did not end the demonstration'
            self.info = dict(is_success=False, is_failure=False, physics_steps=self.steps)
            raw = np.full((9, 64, 64), self.steps, dtype=np.uint8)
            self.images = route(raw, ['head'])
            self.classifier_images = route(raw, ['right_wrist'])
            return self.observation(), 0., False, False
    env=Env()
    saw_success=[]
    def session():
        with hil.database() as db: value=hil.read(db, 'session:demo-test')
        if value['state']=='paused':
            assert value['episode_outcome']=='success'
            saw_success.append(True)
            # Let the boundary iteration publish telemetry, then stop on the next poll.
            if len(saw_success) == 2 and review_action:
                value['revision'] += 1
                value['events'].append({'sequence': value['revision'], 'action': review_action, 'episode': 1})
                with hil.database() as db: hil.write(db, 'session:demo-test', value)
            elif len(saw_success)>1:
                value['state']='stopped'
        return value
    folder=tmp_path/'demo-test'; folder.mkdir()
    runtime.train(tmp_path, 'demo-test', os.getppid(), folder, config, env, session)
    status=json.loads((folder/'runtime.json').read_text())
    assert saw_success and env.steps==2
    assert status['classifier_enabled'] and status['classifier_logit']==8.
    assert status['classifier_probability']>.99 and status['classifier_hold_steps']==2
    assert status['collection']['pending_demo_steps']==(0 if review_action else 2)
    assert status['collection']['pending_demo_classifier_samples']==(0 if review_action else 2)
    assert status['collection']['saved_demo_episodes']==0
    assert status['metrics']['learner_steps']==0
    if not review_action:
        with np.load(folder/'pending-demo.npz') as data:
            assert list(data['terminal'])==[False, True]
    else:
        assert not (folder/'pending-demo.npz').exists()
    if review_action == 'discard_demo_as_failure':
        manifest = json.loads((folder/'classifier_samples/manifest.json').read_text())
        assert manifest['cameras'] == ['right_wrist']
        assert status['collection']['negative_samples'] == 2
        with ClassifierSamples(folder/'classifier_samples') as source:
            for step, sample in enumerate(manifest['samples'], 1):
                pixels, label = source.read(sample)
                assert label == 0 and sample['source'] == 'discarded_demo'
                assert sample['state_key'] == f'1:{step}'
                assert not pixels[:6].any() and np.all(pixels[6:] == step)
    else:
        assert not (folder/'classifier_samples').exists()
