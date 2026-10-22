"""Real SAC update, transport round-trip, device placement and remote isolation checks."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from hilserl import ExperimentConfig
from hilserl_devices import resolve_device
from hilserl_learner import AsyncLearner, make_spec
from hilserl_remote import RemoteEndpoint, PROTOCOL, pack, unpack, rows_to_arrays
from hilserl_learner_server import create_app
from hilserl_runtime import MockPegEnv
from hilserl_pixels import PixelSAC, PixelPolicy, SuccessClassifier, PIXEL_SCHEMA

TOKEN = 'test-only-learner-token-0123456789abcdef'


def config(**values):
    return ExperimentConfig(environment='mock', actor_device='cpu', learner_device='cpu', batch_size=2, **values).model_dump()


def rows(pixel=False):
    row = dict(observation=np.zeros(22 if pixel else 4, np.float32), action=np.zeros(6 if pixel else 3, np.float32),
               reward=1., next_observation=np.ones(22 if pixel else 4, np.float32), terminal=False, human=True, truncated=False)
    if pixel:
        row.update(images=np.zeros((9,64,64), np.uint8), next_images=np.ones((9,64,64), np.uint8))
    return [row, row.copy()]


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_pixel_policy_classifier_and_optimizer_use_requested_device(device):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    torch.set_num_threads(1)
    learner = PixelSAC({'learner_device': device, 'learning_rate': .0003, 'discount': .99})
    before = learner.policy.export()
    losses = learner.update(rows(pixel=True))
    assert all(np.isfinite(v) for v in losses.values())
    for model in (learner.policy, learner.q1, learner.q2, *learner.targets):
        assert all(str(p.device) == device for p in model.parameters())
    assert any(not np.array_equal(before[k], v) for k, v in learner.policy.export().items())
    actor = PixelPolicy().to(device)
    actor.restore(learner.policy.export())
    with torch.no_grad():
        action, _ = actor.sample(torch.zeros(1,22,device=device), images=torch.zeros(1,9,64,64,device=device))
    assert str(action.device) == device and np.isfinite(action.cpu().numpy()).all()
    classifier = SuccessClassifier().to(device)
    logit, probability = classifier.predict(np.zeros((9,64,64), np.uint8))
    assert np.isfinite(logit) and 0 <= probability <= 1


def test_requested_gpu_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='unavailable'):
        resolve_device('cuda:0')


def test_remote_server_auth_schema_sequence_and_artifacts(tmp_path):
    spec = make_spec(config(), MockPegEnv(config()))
    app = create_app(tmp_path, TOKEN)
    with TestClient(app) as client:
        assert client.get('/health').status_code == 401
        client.headers['Authorization'] = 'Bearer ' + TOKEN
        assert client.get('/health').json()['protocol'] == PROTOCOL
        bad = {**spec, 'observation_dim': 200000}
        assert client.post('/runs', json=bad).status_code == 422
        result = client.post('/runs', json=spec)
        assert result.status_code == 200, result.text
        run_id = result.json()['run_id']
        assert client.post('/runs', json=spec).status_code == 409
        route = f'/runs/{run_id}'
        initial = client.post(route+'/exchange', content=pack({'sequence':0,'learning':False,'version':-1}))
        _, before = unpack(initial.content)
        for sequence in range(1,6):
            packet = pack({'sequence':sequence,'learning':True,'version':0}, rows_to_arrays(rows(), False))
            reply = client.post(route+'/exchange', content=packet)
            assert reply.status_code == 200, reply.text
        meta, weights = unpack(reply.content)
        assert meta['metrics']['learner_steps'] == 5 and meta['metrics']['replay_size'] == 10
        assert any(not np.array_equal(before[k], v) for k,v in weights.items())
        assert client.post(route+'/exchange', content=packet).status_code == 409
        stopped = client.post(route+'/stop')
        meta, weights = unpack(stopped.content)
        assert meta['artifacts_saved']
        assert (tmp_path/run_id/'learner.pt').is_file()
        with np.load(tmp_path/run_id/'replay.npz', allow_pickle=False) as data:
            assert len(data['reward']) == 10
        assert client.post(route+'/stop').content == stopped.content
        assert not client.get('/health').json()['busy']


@pytest.fixture
def live_server(tmp_path):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    environment = {**os.environ, 'CYCLO_HIL_LEARNER_TOKEN': TOKEN, 'CYCLO_HIL_LEARNER_DATA_DIR': str(tmp_path/'remote')}
    log = (tmp_path/'server.log').open('w')
    process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'hilserl_learner_server:create_app', '--factory',
                                '--host', '127.0.0.1', '--port', str(port)], cwd=Path(__file__).parent,
                                env=environment, stdout=log, stderr=log)
    tokenfile = tmp_path/'token'; tokenfile.write_text(TOKEN)
    c=config(learner_placement='remote', learner_url=f'http://127.0.0.1:{port}', learner_token_file=str(tokenfile))
    try:
        endpoint=RemoteEndpoint(c)
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            try:
                if json.loads(endpoint.request('/health'))['protocol']==PROTOCOL: break
            except RuntimeError: time.sleep(.1)
        else: pytest.fail((tmp_path/'server.log').read_text())
        yield c
    finally:
        process.terminate()
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
        log.close()


def test_remote_actor_has_no_local_learner_and_uploads_demos(live_server, tmp_path, monkeypatch):
    import hilserl_learner
    def forbidden(*args, **kwargs):
        raise AssertionError('A remote actor must not allocate a local learner')
    monkeypatch.setattr(hilserl_learner, 'LearnerEngine', forbidden)
    backend=AsyncLearner(live_server, make_spec(live_server, MockPegEnv(live_server)), tmp_path/'actor', None, rows())
    try:
        assert backend.snapshot()[0]['metrics']['replay_size']==2
        for i in range(6):
            backend.insert(rows()[0]); backend.tick(True)
            deadline=time.monotonic()+5
            while backend.snapshot()[0]['metrics']['learner_steps'] < i+1 and time.monotonic()<deadline:
                time.sleep(.01)
        meta, weights=backend.snapshot()
        assert meta['metrics']['learner_steps']==6
        assert meta['metrics']['replay_size']==8
        assert meta['version']>=5 and weights
        backend.pause()
        before=meta['metrics']['learner_steps']
        backend.tick(False); time.sleep(.1)
        assert backend.snapshot()[0]['metrics']['learner_steps']==before
    finally:
        meta, weights=backend.close()
    assert meta['artifacts_saved'] and meta['metrics']['learner_steps']==6
    assert not (tmp_path/'actor'/'learner.pt').exists()
    assert Path(meta['artifact_directory'], 'learner.pt').exists()


def test_network_failure_preserves_unacknowledged_transitions(tmp_path, monkeypatch):
    import hilserl_learner
    class Offline:
        def __init__(self, config): pass
        def open(self, spec, weights, resume=None): return {'version':0, 'metrics':{}}, {}
        def exchange(self, batch, learning, version):
            if batch: raise RuntimeError('connection lost')
            return {'version':0, 'metrics':{}}, {}
        def close(self): return {'artifacts_saved':False}, {}
    monkeypatch.setattr(hilserl_learner, 'RemoteEndpoint', Offline)
    c=config(learner_placement='remote')
    spec=make_spec(c, MockPegEnv(c))
    backend=AsyncLearner(c, spec, tmp_path, None, [])
    backend.insert(rows()[0]);backend.tick(True)
    deadline=time.monotonic()+3
    while not backend.error and time.monotonic()<deadline: time.sleep(.01)
    with pytest.raises(RuntimeError, match='connection lost'): backend.close()
    with np.load(tmp_path/'unsent-replay.npz', allow_pickle=False) as data:
        assert data['action'].shape==(1,3)
        assert data['human'][0]


def test_remote_gpu_readiness_does_not_gate_local_evaluation(tmp_path, monkeypatch):
    import hilserl
    monkeypatch.setenv('CYCLO_HILSERL_DATA_DIR', str(tmp_path))
    checkpoint=tmp_path/'policy.npz';checkpoint.write_bytes(b'fixture')
    monkeypatch.setattr(hilserl, 'learner_probe', lambda *args, **kwargs: {'connected':False,'gpus':[], 'error':'offline'})
    c=config(learner_placement='remote', checkpoint_path=str(checkpoint))
    ready=hilserl.preflight(c)
    assert not ready['can_train'] and ready['can_evaluate']


def test_remote_url_and_device_validation():
    from pydantic import ValidationError
    for fields in [dict(actor_device='auto'),dict(learner_url='http://untrusted.example:7130'),dict(learner_url='https://secret@host.example')]:
        with pytest.raises(ValidationError): ExperimentConfig(**fields)
    assert ExperimentConfig().actor_device=='cuda:0'
    assert ExperimentConfig().learner_device=='cuda:0'
