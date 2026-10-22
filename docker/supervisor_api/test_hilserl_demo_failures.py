"""Discarded demo images remain reviewable classifier data, separate from policy demos."""
import json
from uuid import uuid4

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import hilserl as hil
from hilserl_cameras import route
from hilserl_data import Collection, ClassifierSamples
from hilserl_pixels import PIXEL_SCHEMA


def record(collection, step):
    class Env:
        capture_metadata = {'sequence': step}
        info = {'physics_steps': step}
        classifier_images = np.full((9, 64, 64), step, dtype=np.uint8)
    images = route(Env.classifier_images, collection.config['observation_cameras'])
    row = dict(observation=np.zeros(22), action=np.zeros(6), reward=0., next_observation=np.ones(22),
               terminal=False, human=True, truncated=False, images=images, next_images=images.copy(),
               step_id=step, timestamp=float(step))
    collection.record_demo(row, Env())


def config():
    return hil.ExperimentConfig(name='NIST demo review', environment='mujoco', learning_backend='legacy_torch',
                                observation_cameras=['head'], classifier_cameras=['right_wrist']).model_dump()


def test_failed_discard_write_keeps_demo_and_retry_preserves_previous_episodes(tmp_path, monkeypatch):
    demo = Collection(tmp_path, config(), 'demo', PIXEL_SCHEMA)
    record(demo, 1)
    saved = demo.save_demo(1)
    saved_bytes = saved.read_bytes()
    assert demo.state()['pending_demo_classifier_samples'] == 0
    record(demo, 2)
    demo.discard_demo(2, add_classifier_failures=True)
    first_id = demo.samples[0]['id']
    record(demo, 3)
    publish = demo.write_classifier_manifest
    def fail(_):
        raise OSError('disk full')
    monkeypatch.setattr(demo, 'write_classifier_manifest', fail)
    with pytest.raises(OSError, match='disk full'):
        demo.discard_demo(3, add_classifier_failures=True)
    assert len(demo.rows) == len(demo.demo_classifier_frames) == 1
    assert len(demo.samples) == 1
    manifest = json.loads((tmp_path/'classifier_samples/manifest.json').read_text())
    assert [sample['id'] for sample in manifest['samples']] == [first_id]
    monkeypatch.setattr(demo, 'write_classifier_manifest', publish)
    demo.discard_demo(3, add_classifier_failures=True)
    assert not demo.rows and not demo.demo_classifier_frames
    assert demo.state()['negative_samples'] == 2
    assert saved.read_bytes() == saved_bytes
    with ClassifierSamples(tmp_path/'classifier_samples') as reader:
        assert [reader.read(sample)[1] for sample in demo.samples] == [0, 0]
    assert sorted(path.name for path in (tmp_path/'classifier_samples').glob('*.npz')) == ['episode-00002.npz', 'episode-00003.npz']
    with pytest.raises(ValueError, match='No pending'):
        demo.discard_demo(3, add_classifier_failures=True)


def test_unavailable_classifier_views_do_not_destroy_pending_demo(tmp_path):
    demo = Collection(tmp_path, config(), 'demo', PIXEL_SCHEMA)
    record(demo, 1)
    demo.demo_classifier_frames = []
    with pytest.raises(ValueError, match='Classifier views unavailable'):
        demo.discard_demo(1, add_classifier_failures=True)
    assert len(demo.rows) == 1
    assert not (tmp_path/'classifier_samples').exists()
    demo.discard_demo(1)
    assert not demo.rows


@pytest.mark.parametrize('state,mode,frames,expected', [
    ('paused', 'demo', 2, 200), ('running', 'demo', 2, 409),
    ('paused', 'classifier', 2, 409), ('paused', 'demo', 0, 409),
])
def test_failure_conversion_command_requires_paused_demo_and_classifier_frames(tmp_path, monkeypatch, state, mode, frames, expected):
    monkeypatch.setenv('CYCLO_HILSERL_DATA_DIR', str(tmp_path))
    identifier = str(uuid4())
    item = dict(id=identifier, config=config(), mode=mode, state=state, revision=2, events=[],
                metrics={}, episode=1, episode_outcome='success', control_source='policy')
    class Worker:
        def poll(self): return None
    monkeypatch.setattr(hil, '_workers', {identifier: Worker()})
    folder = tmp_path/identifier
    folder.mkdir()
    (folder/'runtime.json').write_text(json.dumps({'ack_revision': 2, 'collection': {
        'pending_demo_steps': 2, 'pending_demo_classifier_samples': frames}}))
    with hil.database() as db:
        hil.write(db, 'session:'+identifier, item)
    app = FastAPI(); app.include_router(hil.router)
    with TestClient(app) as client:
        response = client.post(f'/hilserl/sessions/{identifier}/commands', json={
            'expected_revision': 2, 'action': 'discard_demo_as_failure'})
        assert response.status_code == expected, response.text
        if expected == 200:
            assert response.json()['events'][-1]['action'] == 'discard_demo_as_failure'
            assert response.json()['runtime_pending']
            assert client.post(f'/hilserl/sessions/{identifier}/commands', json={
                'expected_revision': 2, 'action': 'discard_demo_as_failure'}).status_code == 409


def test_discarded_demo_combines_with_classifier_dataset_for_training_and_independent_deletion(tmp_path, monkeypatch):
    monkeypatch.setenv('CYCLO_HILSERL_DATA_DIR', str(tmp_path))
    monkeypatch.setattr(hil, '_classifier_workers', {})
    monkeypatch.setattr(hil, '_workers', {})
    settings = config()
    identifiers = [str(uuid4()), str(uuid4())]
    classifier = Collection(tmp_path/identifiers[0], settings, 'classifier', PIXEL_SCHEMA)
    class Env:
        capture_metadata = {}
        def observation(self): return np.zeros(22)
    env = Env()
    rng = np.random.default_rng(42)
    for episode in (1, 2, 3, 4):
        for step in range(10):
            env.info = {'physics_steps': step}
            env.images = rng.integers(0, 255, (9, 64, 64), dtype=np.uint8)
            classifier.capture(env, step % 2, episode, fresh=False)
    classifier.export()
    demo = Collection(tmp_path/identifiers[1], settings, 'demo', PIXEL_SCHEMA)
    record(demo, 1)
    saved = demo.save_demo(1)
    for step in range(2, 12):
        record(demo, step)
    demo.discard_demo(2, add_classifier_failures=True)
    demo.export()
    with hil.database() as db:
        for identifier, mode in zip(identifiers, ('classifier', 'demo')):
            hil.write(db, 'session:'+identifier, dict(id=identifier, mode=mode, state='stopped', config=settings,
                      revision=0, metrics={}, episode=2, episode_outcome=None))
    library = [item for item in hil.data_library() if item['session_id'] == identifiers[1]]
    assert {item['kind'] for item in library} == {'demo', 'classifier'}
    negatives = next(item for item in library if item['kind'] == 'classifier')
    assert negatives['negative'] == 10 and negatives['positive'] == 0
    assert negatives['environment_name'] == settings['name']
    assert negatives['cameras'] == ['right_wrist']
    app = FastAPI(); app.include_router(hil.router)
    with TestClient(app) as client:
        response = client.post('/hilserl/classifiers/train', json={'session_ids': identifiers, 'epochs': 1, 'device': 'cpu'})
        assert response.status_code == 202, response.text
        job = response.json()['id']
        assert hil._classifier_workers[job].wait(timeout=20) == 0
        status = next(item for item in hil.classifier_jobs() if item['id'] == job)
        assert status['state'] == 'complete'
        assert status['negative_samples'] == 30 and status['positive_samples'] == 20
        assert status['validation_split'] == 'global_label_stratified_v1'
        assert client.delete(f'/hilserl/datasets/{identifiers[1]}/classifier').status_code == 200
        assert saved.is_file() and (tmp_path/identifiers[1]/'demonstrations.npz').is_file()
