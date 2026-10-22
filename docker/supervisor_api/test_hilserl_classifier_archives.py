"""Episode archives preserve images, labels, previews and environment identity."""
import json
from uuid import uuid4

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import hilserl as hil
from hilserl_data import Collection, ClassifierSamples


CONFIG = dict(name='NIST board A', environment='mujoco', robot_type='ffw_sg2_rev1',
              arm='right', target_part='peg_round_8')


def collect(folder, episodes=(1, 2)):
    class Env:
        capture_metadata = {}
        def observation(self):
            return np.zeros(22)

    collection = Collection(folder, CONFIG, 'classifier', 'test')
    env = Env()
    for episode in episodes:
        for step in range(3):
            env.info = {'physics_steps': step}
            env.images = np.full((9, 64, 64), episode * 10 + step, dtype=np.uint8)
            collection.capture(env, step % 2, episode, fresh=False, automatic=True)
    return collection


def test_one_archive_per_episode_is_lossless_and_idempotent(tmp_path):
    collection = collect(tmp_path)
    folder = tmp_path / 'classifier_samples'
    before = json.loads((folder / 'manifest.json').read_text())
    # Earlier episode is finalized; the active episode remains durably staged.
    assert (folder / 'episode-00001.npz').is_file()
    collection.export()
    collection.export()
    assert sorted(path.name for path in folder.glob('*.npz')) == ['episode-00001.npz', 'episode-00002.npz']
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['environment_name'] == CONFIG['name']
    assert len(manifest['samples']) == 6
    # A reader that fetched the old manifest during finalization still works.
    for samples in (before['samples'], manifest['samples']):
        with ClassifierSamples(folder) as source:
            for sample in samples:
                image, label = source.read(sample)
                step = int(sample['state_key'].split(':')[1])
                np.testing.assert_array_equal(image, np.full((9, 64, 64), sample['episode'] * 10 + step, dtype=np.uint8))
                assert label == sample['label'] == step % 2
    with np.load(folder / 'episode-00001.npz', allow_pickle=False) as archive:
        assert str(archive['__environment_name__']) == CONFIG['name']
        assert len(archive['sample_ids']) == 3


def test_failed_manifest_publish_preserves_staged_images(tmp_path, monkeypatch):
    collection = collect(tmp_path, (1,))
    original = collection.write_classifier_manifest
    def fail(samples):
        raise OSError('disk write failed')
    monkeypatch.setattr(collection, 'write_classifier_manifest', fail)
    with pytest.raises(OSError, match='disk write failed'):
        collection.export()
    folder = tmp_path / 'classifier_samples'
    old = json.loads((folder / 'manifest.json').read_text())
    assert all((folder / f"{sample['id']}.npz").is_file() for sample in old['samples'])
    with ClassifierSamples(folder) as source:
        assert [source.read(sample)[1] for sample in old['samples']] == [0, 1, 0]
    monkeypatch.setattr(collection, 'write_classifier_manifest', original)
    collection.export()
    assert [path.name for path in folder.glob('*.npz')] == ['episode-00001.npz']


def test_library_recovers_historical_names_and_previews_both_formats(tmp_path, monkeypatch):
    monkeypatch.setenv('CYCLO_HILSERL_DATA_DIR', str(tmp_path))
    identifiers = [str(uuid4()), str(uuid4())]
    collections = [collect(tmp_path / identifier, (1,)) for identifier in identifiers]
    collections[0].export()
    # Legacy datasets did not store the name in their manifest.
    legacy = tmp_path / identifiers[1] / 'classifier_samples' / 'manifest.json'
    manifest = json.loads(legacy.read_text())
    del manifest['environment_name']
    legacy.write_text(json.dumps(manifest))
    config = hil.ExperimentConfig().model_dump()
    with hil.database() as db:
        for identifier in identifiers:
            hil.write(db, 'session:' + identifier, dict(id=identifier, mode='classifier', state='stopped',
                      revision=0, config={**config, 'name': 'Historical name'}))
        hil.write(db, 'config', {'revision': 1, 'config': {**config, 'name': 'Current name'}})
    job = tmp_path / 'classifiers' / str(uuid4())
    job.mkdir(parents=True)
    (job / 'status.json').write_text(json.dumps({'state': 'complete'}))
    (job / 'request.json').write_text(json.dumps({'session_ids': [identifiers[1]]}))
    library = {item['session_id']: item for item in hil.data_library()}
    assert library[identifiers[0]]['environment_name'] == CONFIG['name']
    assert library[identifiers[1]]['environment_name'] == 'Historical name'
    assert library[identifiers[0]]['files'] == [dict(episode=1, name='episode-00001.npz', positive=1, negative=2)]
    assert hil.classifier_jobs()[0]['environment_names'] == ['Historical name']
    app = FastAPI()
    app.include_router(hil.router)
    with TestClient(app) as client:
        for identifier, collection in zip(identifiers, collections):
            response = client.get(f"/hilserl/classifier-samples/{identifier}/{collection.last_capture['id']}")
            assert response.status_code == 200
            assert response.headers['content-type'] == 'image/jpeg'
            assert response.content[:2] == b'\xff\xd8'
        assert client.get(f'/hilserl/classifier-samples/{identifiers[0]}/{uuid4()}').status_code == 404
        assert client.get(f'/hilserl/classifier-samples/{identifiers[0]}/invalid').status_code == 422


def test_stopped_worker_still_saving_cannot_train_or_delete(tmp_path, monkeypatch):
    monkeypatch.setenv('CYCLO_HILSERL_DATA_DIR', str(tmp_path))
    identifier = str(uuid4())
    collection = collect(tmp_path / identifier, (1,))
    config = hil.ExperimentConfig(environment='mujoco').model_dump()
    item = dict(id=identifier, mode='classifier', state='stopped', config=config,
                revision=0, metrics={}, episode=1)
    class Saving:
        def poll(self):
            return None
    monkeypatch.setattr(hil, '_workers', {identifier: Saving()})
    monkeypatch.setattr(hil, '_classifier_workers', {})
    with hil.database() as db:
        hil.write(db, 'session:' + identifier, item)
    assert hil.decorate(item)['dataset_finalizing']
    app = FastAPI()
    app.include_router(hil.router)
    with TestClient(app) as client:
        for response in [client.post('/hilserl/classifiers/train', json={'session_ids': [identifier]}),
                         client.delete(f'/hilserl/datasets/{identifier}/classifier')]:
            assert response.status_code == 409
            assert 'finish saving' in response.json()['detail']
    assert len(collection.samples) == 3
    assert (tmp_path / identifier / 'classifier_samples' / 'manifest.json').is_file()
