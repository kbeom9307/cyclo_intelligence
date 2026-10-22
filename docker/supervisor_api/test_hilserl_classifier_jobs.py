"""Classifier placement and explicit cancellation retain completed artifacts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import hilserl as hil
from hilserl_data import Collection
from hilserl_profiles import image_shape


def labeled_collection(root, backend='hil_serl'):
    identifier = str(uuid4())
    config = hil.ExperimentConfig(learning_backend=backend).model_dump()
    collection = Collection(root / identifier, config, 'classifier', 'test')
    class Env:
        capture_metadata = {}
        def observation(self):
            return np.zeros(22)
    env = Env()
    rng = np.random.default_rng(91)
    for episode in range(4):
        for step in range(10):
            env.info = {'physics_steps': step}
            env.images = rng.integers(0, 50, image_shape(config), dtype=np.uint8) + 150 * (step % 2)
            collection.capture(env, step % 2, episode, fresh=False, automatic=True)
    collection.export()
    with hil.database() as db:
        hil.write(db, 'session:' + identifier, dict(id=identifier, mode='classifier', state='stopped',
                  config=config, revision=0, episode=1, metrics={}))
    return identifier


@pytest.mark.parametrize('backend', ['hil_serl', 'legacy_torch'])
def test_new_jobs_default_to_gpu_and_override_inherited_cpu_environment(tmp_path, monkeypatch, backend):
    monkeypatch.setenv('CYCLO_HILSERL_DATA_DIR', str(tmp_path))
    monkeypatch.setenv('JAX_PLATFORMS', 'cpu')
    monkeypatch.setattr(hil, '_classifier_workers', {})
    monkeypatch.setattr(hil, 'hardware_status', lambda: {'gpus': [{'device': 'cuda:0'}]})
    identifier = labeled_collection(tmp_path, backend)
    launched = {}
    class Worker:
        def __init__(self, command, **kwargs):
            launched.update(command=command, **kwargs)
        def poll(self):
            return None
    monkeypatch.setattr(hil.subprocess, 'Popen', Worker)
    result = hil.train_classifier(hil.TrainClassifier(session_ids=[identifier]))
    folder = tmp_path / 'classifiers' / result['id']
    assert json.loads((folder / 'request.json').read_text())['device'] == 'cuda:0'
    assert json.loads((folder / 'status.json').read_text())['device'] == 'cuda:0'
    assert launched['env']['JAX_PLATFORMS'] == 'cuda,cpu'
    assert launched['env']['XLA_PYTHON_CLIENT_PREALLOCATE'] == 'false'
    assert '--xla_gpu_autotune_level=2' in launched['env']['XLA_FLAGS']
    hil._classifier_workers.clear()
    monkeypatch.setattr(hil, 'hardware_status', lambda: {'gpus': []})
    with pytest.raises(hil.HTTPException) as error:
        hil.train_classifier(hil.TrainClassifier(session_ids=[identifier]))
    assert error.value.status_code == 503
    assert len(list((tmp_path / 'classifiers').iterdir())) == 1


@pytest.mark.parametrize('has_checkpoint', [False, True])
def test_stop_terminates_worker_preserves_data_and_is_idempotent(tmp_path, monkeypatch, has_checkpoint):
    monkeypatch.setenv('CYCLO_HILSERL_DATA_DIR', str(tmp_path))
    identifier = str(uuid4())
    folder = tmp_path / 'classifiers' / identifier
    folder.mkdir(parents=True)
    original = {'state': 'training', 'device': 'cuda:0', 'history': [{'epoch': 1}]}
    (folder / 'status.json').write_text(json.dumps(original))
    checkpoint = folder / 'classifier.npz'
    if has_checkpoint:
        # This artifact is newer than status, as in a stop during publication.
        np.savez_compressed(checkpoint, __training_metrics__=json.dumps({'epoch': 2, 'balanced_accuracy': .7}))
        before = checkpoint.read_bytes()
    source = tmp_path / 'collected-images.npz'
    source.write_bytes(b'keep source data')
    worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    monkeypatch.setattr(hil, '_classifier_workers', {identifier: worker})
    app = FastAPI()
    app.include_router(hil.router)
    try:
        with TestClient(app) as client:
            response = client.post(f'/hilserl/classifiers/{identifier}/stop')
            assert response.status_code == 200, response.text
            assert worker.poll() is not None
            saved = response.json()
            assert saved['state'] == 'stopped' and saved['history'] == original['history']
            assert source.read_bytes() == b'keep source data'
            if has_checkpoint:
                assert checkpoint.read_bytes() == before
                assert saved['checkpoint_path'] == str(checkpoint)
                assert saved['best_epoch'] == saved['checkpoint_metrics']['epoch'] == 2
            else:
                assert not saved.get('checkpoint_path')
            assert client.post(f'/hilserl/classifiers/{identifier}/stop').json()['state'] == 'stopped'
            assert client.post(f'/hilserl/classifiers/{uuid4()}/stop').status_code == 404
            assert client.post('/hilserl/classifiers/invalid/stop').status_code == 422
            assert hil.classifier_jobs()[0]['state'] == 'stopped'
    finally:
        if worker.poll() is None:
            worker.kill()
        worker.wait(timeout=3)


@pytest.mark.skipif(os.environ.get('CYCLO_TEST_CLASSIFIER_GPU') != '1', reason='Requires idle GPU')
def test_full_size_gpu_classifier_training_and_stop(tmp_path, monkeypatch):
    monkeypatch.setenv('CYCLO_HILSERL_DATA_DIR', str(tmp_path))
    monkeypatch.setattr(hil, '_classifier_workers', {})
    identifier = labeled_collection(tmp_path)
    app = FastAPI()
    app.include_router(hil.router)
    with TestClient(app) as client:
        response = client.post('/hilserl/classifiers/train', json={'session_ids': [identifier], 'epochs': 200})
        assert response.status_code == 202, response.text
        job = response.json()['id']
        process = hil._classifier_workers[job]
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                state = next(j for j in hil.classifier_jobs() if j['id'] == job)
                assert state['state'] != 'failed', (state, (tmp_path / 'classifiers' / job / 'worker.log').read_text())
                if state.get('epoch', 0) >= 1:
                    break
                time.sleep(.2)
            assert state.get('epoch', 0) >= 1, state
            assert state['device'] == 'cuda:0' and state['batch_size'] == 256
            assert state['memory']['allocated_mib'] > 0
            stopped = client.post(f'/hilserl/classifiers/{job}/stop')
            assert stopped.status_code == 200, stopped.text
            assert stopped.json()['state'] == 'stopped'
            assert process.poll() is not None
            assert Path(stopped.json()['checkpoint_path']).is_file()
            Path('/tmp/hil-classifier-gpu-stop.json').write_text(json.dumps({
                key: stopped.json().get(key) for key in ('state', 'device', 'batch_size', 'epoch', 'best_epoch', 'memory')}, indent=2))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def test_stop_preserves_checkpoint_from_an_older_cpu_job(tmp_path):
    from hilserl_classifier_jobs import stopped_status
    metrics = {'epoch': 7, 'balanced_accuracy': .8}
    (tmp_path / 'status.json').write_text(json.dumps({'state': 'training', 'device': 'cpu', 'best_metrics': metrics}))
    np.savez_compressed(tmp_path / 'classifier.npz', __schema__='existing-checkpoint', weight=np.ones(2))
    saved = stopped_status(tmp_path)
    assert saved['state'] == 'stopped'
    assert saved['best_epoch'] == 7
    assert saved['checkpoint_metrics'] == metrics
    assert saved['device'] == 'cpu'
