"""Control-plane contract tests without a robot, GPU, or learner."""
import importlib.util
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

spec = importlib.util.spec_from_file_location('hilserl_test_module', Path(__file__).with_name('hilserl.py'))
hil = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hil
spec.loader.exec_module(hil)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('CYCLO_HILSERL_DATA_DIR', str(tmp_path))
    with hil.database() as db:
        hil.write(db, "config", {"revision": 0, "config": hil.ExperimentConfig(environment="real_robot").model_dump()})
    app = FastAPI()
    app.include_router(hil.router)
    with TestClient(app) as client:
        yield client


def create(client):
    result = client.post('/hilserl/sessions', json={'mode': 'preview', 'expected_config_revision': 0})
    assert result.status_code == 201
    return result.json()


def action(client, session, command):
    return client.post(f"/hilserl/sessions/{session['id']}/commands", json={
        'action': command, 'expected_revision': session['revision']})


def test_hardware_workflow_preview_does_not_allow_real_execution(client):
    data = client.get('/hilserl/workspace').json()
    assert data['sessions'] == []
    assert data['readiness']['can_preview']
    assert not data['readiness']['can_train']
    for mode in ['train', 'evaluate']:
        assert client.post('/hilserl/sessions', json={'mode': mode, 'expected_config_revision': 0}).status_code == 409


def test_config_validation_revision_and_session_snapshot(client):
    session = create(client)
    config = client.get('/hilserl/workspace').json()['config']
    config['control_hz'] = 0
    assert client.put('/hilserl/config', json={'expected_revision': 0, 'config': config}).status_code == 422
    config['control_hz'] = 20
    assert client.put('/hilserl/config', json={'expected_revision': 0, 'config': config}).status_code == 200
    assert client.put('/hilserl/config', json={'expected_revision': 0, 'config': config}).status_code == 409
    assert client.get(f"/hilserl/sessions/{session['id']}").json()['config']['control_hz'] == 10
    assert client.get('/hilserl/workspace').json()['config']['control_hz'] == 20


def test_preview_lifecycle_and_metrics_are_not_simulated(client):
    session = create(client)
    assert action(client, session, 'take_over').status_code == 409
    for command in ['start', 'take_over', 'release', 'success', 'reset', 'resume', 'pause', 'stop']:
        result = action(client, session, command)
        assert result.status_code == 200, result.text
        session = result.json()
    assert session['state'] == 'stopped'
    assert session['episode'] == 2
    assert all(value is None for value in session['metrics'].values())
    assert len(session['events']) == 8
    assert action(client, session, 'start').status_code == 409
    assert action(client, session, 'reset').status_code == 409


def test_label_requires_reset_before_resume(client):
    session = action(client, create(client), 'start').json()
    session = action(client, session, 'failure').json()
    assert action(client, session, 'resume').status_code == 409
    assert action(client, session, 'success').status_code == 409


def test_duplicate_or_stale_commands_cannot_execute_twice(client):
    session = create(client)
    assert action(client, session, 'start').status_code == 200
    assert action(client, session, 'start').status_code == 409
    assert client.post('/hilserl/sessions', json={'mode': 'preview', 'expected_config_revision': 0}).status_code == 409
    assert client.get('/hilserl/sessions/missing').status_code == 404


def test_concurrent_session_creation_allows_only_one(client):
    def attempt(_):
        return client.post('/hilserl/sessions', json={'mode': 'preview', 'expected_config_revision': 0}).status_code
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, range(2)))
    assert sorted(results) == [201, 409]


def test_store_survives_application_restart(client):
    session = create(client)
    app = FastAPI()
    app.include_router(hil.router)
    with TestClient(app) as second:
        assert second.get('/hilserl/workspace').json()['sessions'][0]['id'] == session['id']


def test_mock_readiness_guards_unsupported_reward_and_artifacts(client):
    config = client.get('/hilserl/workspace').json()['config']
    config.update(environment='mock', reward_source='classifier')
    assert not hil.preflight(config)['can_train']
    config.update(reward_source='environment', checkpoint_path='/tmp/untrusted.npz')
    assert not hil.preflight(config)['can_evaluate']


def test_manual_action_requires_runtime_ownership_and_bounded_input(client):
    session = create(client)
    url = f"/hilserl/sessions/{session['id']}/action"
    assert client.post(url, json={'expected_revision': 0, 'value': [2, 0, 0]}).status_code == 422
    assert client.post(url, json={'expected_revision': 0, 'value': [1, 0, 0]}).status_code == 409


def test_agentlace_training_intervention_checkpoint_and_evaluation(client, tmp_path):
    """Actual subprocess + socket transport + optimizer, isolated from live sessions."""
    import time
    import numpy as np
    pytest.importorskip('agentlace')
    pytest.importorskip('torch')
    config = client.get('/hilserl/workspace').json()['config']
    config.update(environment='mock', batch_size=8, control_hz=50, max_episode_steps=30)
    client.put('/hilserl/config', json={'expected_revision': 0, 'config': config}).raise_for_status()
    item = client.post('/hilserl/sessions', json={'mode': 'train', 'expected_config_revision': 1}).json()

    def wait(predicate, timeout=25):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            current = client.get(f"/hilserl/sessions/{item['id']}").json()
            assert not current.get('runtime_error'), current
            if predicate(current):
                return current
            time.sleep(.1)
        pytest.fail(f'Runtime timeout: {current}')

    def send(command):
        current = wait(lambda s: not s.get('runtime_pending'))
        response = action(client, current, command)
        assert response.status_code == 200, response.text
        return wait(lambda s: not s.get('runtime_pending'))

    try:
        assert action(client, item, 'start').status_code == 200
        item = wait(lambda s: (s['metrics'].get('policy_version') or 0) >= 5)
        assert item['metrics']['replay_size'] >= 8
        assert item['metrics']['learner_steps'] > 0
        item = send('take_over')
        response = client.post(f"/hilserl/sessions/{item['id']}/action", json={'expected_revision': item['revision'], 'value': [1, 0, 0]})
        assert response.status_code == 200
        item = wait(lambda s: s.get('runtime', {}).get('last_action') == [1., 0., 0.])
        item = send('pause')
        steps = item['metrics']['environment_steps']
        time.sleep(.2)
        assert wait(lambda s: True)['metrics']['environment_steps'] == steps
        item = send('resume')
        item = send('success')
        assert item['episode_outcome'] == 'success'
        item = send('reset')
        item = send('resume')
        item = send('truncate')
        assert item['episode_outcome'] == 'truncated'
        item = send('stop')
        hil._workers[item['id']].wait(timeout=10)
        folder = tmp_path / item['id']
        assert (folder / 'learner.pt').is_file()
        with np.load(folder / 'replay.npz') as data:
            assert data['human'].any()
            assert ((data['action'][:, 0] == 1) & data['human']).any()
            assert data['terminal'].any()
            assert data['truncated'].any()
        import json
        rows = [json.loads(line) for line in (folder / 'transitions.jsonl').read_text().splitlines()]
        manual_truncations = [row for row in rows if row['label_only'] and row['truncated']]
        assert len(manual_truncations) == 1
        assert manual_truncations[0]['terminal'] is False
        assert manual_truncations[0]['reward'] == 0
        config.update(checkpoint_path=str(folder / 'policy.npz'), demonstration_path=str(folder / 'replay.npz'))
        client.put('/hilserl/config', json={'expected_revision': 1, 'config': config}).raise_for_status()
        item = client.post('/hilserl/sessions', json={'mode': 'evaluate', 'expected_config_revision': 2}).json()
        assert action(client, item, 'start').status_code == 200
        item = wait(lambda s: (s['metrics']['environment_steps'] or 0) >= 10)
        assert item['metrics']['learner_steps'] == 0
        item = send('stop')
        hil._workers[item['id']].wait(timeout=10)
        item = client.post('/hilserl/sessions', json={'mode': 'train', 'expected_config_revision': 2}).json()
        assert action(client, item, 'start').status_code == 200
        item = wait(lambda s: (s['metrics']['environment_steps'] or 0) > 0)
        process = hil._workers[item['id']]
        process.kill(); process.wait(timeout=5)
        failed = client.get(f"/hilserl/sessions/{item['id']}").json()
        assert failed['state'] == 'stopped'
        assert failed['runtime_error']
        assert not failed['runtime_pending']
        assert client.post('/hilserl/sessions', json={'mode': 'preview', 'expected_config_revision': 2}).status_code == 201
    finally:
        hil.shutdown()


def test_input_activates_intervention_without_toggle_and_expires(client, monkeypatch):
    from types import SimpleNamespace
    item = create(client)
    item.update(mode='train', state='running')
    item['config']['environment'] = 'mock'
    with hil.database() as db:
        hil.write(db, 'session:' + item['id'], item)
    monkeypatch.setitem(hil._workers, item['id'], SimpleNamespace(poll=lambda: None))
    monkeypatch.setattr(hil, 'runtime_status', lambda s: {'ack_revision': s['revision']})
    clock = [1000.]
    monkeypatch.setattr(hil.time, 'time', lambda: clock[0])
    path = '/hilserl/sessions/' + item['id']
    def send(value):
        return client.post(path + '/action', json={'expected_revision': 0, 'value': value})
    assert send([1, 0, 0]).status_code == 200
    assert client.get(path).json()['control_source'] == 'human'
    assert send([0, 0, 0]).status_code == 200
    assert client.get(path).json()['control_source'] == 'policy'
    assert send([1, 0, 0]).status_code == 200
    clock[0] += 1
    assert client.get(path).json()['control_source'] == 'policy'
    item['state'] = 'paused'
    with hil.database() as db:
        hil.write(db, 'session:' + item['id'], item)
    assert send([1, 0, 0]).status_code == 409

@pytest.mark.parametrize('paused', [False, True])
def test_truncate_requires_reset_and_cannot_repeat(client, paused):
    session = action(client, create(client), 'start').json()
    if paused:
        session = action(client, session, 'pause').json()
    response = action(client, session, 'truncate')
    assert response.status_code == 200
    session = response.json()
    assert session['state'] == 'paused'
    assert session['episode_outcome'] == 'truncated'
    assert action(client, session, 'resume').status_code == 409
    assert action(client, session, 'truncate').status_code == 409
    session = action(client, session, 'reset').json()
    assert session['episode_outcome'] is None
    assert action(client, session, 'resume').status_code == 200
