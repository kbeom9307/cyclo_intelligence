"""Isaac protocol tests; optional actual GPU/physics smoke with CYCLO_TEST_ISAAC=1."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

spec = importlib.util.spec_from_file_location('hil_isaac_api_test', Path(__file__).with_name('hilserl.py'))
hil = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hil
spec.loader.exec_module(hil)


def test_preflight_matches_robot_rate_and_bridge_schema(monkeypatch):
    monkeypatch.setattr(hil._isaac, 'health', lambda: {'ready': True, 'busy': False, 'robot_type': 'ffw_sg2_rev1', 'control_hz': 20, 'target_parts': ['peg_round_8']})
    monkeypatch.setattr(hil.importlib.util, 'find_spec', lambda _: True)
    config = hil.ExperimentConfig(environment='isaac_sim', observation_mode='state', robot_type='ffw_sg2_rev1', control_hz=20).model_dump()
    assert hil.preflight(config)['can_train']
    for key, value in [('robot_type', 'ffw_bg2_rev4'), ('control_hz', 10), ('target_part', 'peg_round_16'), ('rotation_limit_rad', .3)]:
        assert not hil.preflight({**config, key: value})['can_train']
    monkeypatch.setattr(hil._isaac, 'health', lambda: {'ready': False, 'error': 'disconnected'})
    assert not hil.preflight(config)['can_train']


def test_remote_url_cannot_be_selected_by_client(monkeypatch):
    monkeypatch.setenv('CYCLO_HILSERL_ISAAC_URL', 'http://example.com:7125')
    with pytest.raises(ValueError):
        hil._isaac.bridge_url()


@pytest.mark.skipif(os.environ.get('CYCLO_TEST_ISAAC') != '1', reason='Requires a running Isaac NIST bridge and GPU')
def test_real_isaac_actor_learner_manual_reset_and_artifacts(tmp_path, monkeypatch):
    import numpy as np
    monkeypatch.setenv('CYCLO_HILSERL_DATA_DIR', str(tmp_path))
    app = FastAPI()
    app.include_router(hil.router)
    with TestClient(app) as client:
        config = hil.ExperimentConfig(environment='isaac_sim', observation_mode='state', robot_type='ffw_sg2_rev1', control_hz=20,
                                      initialization='near_hole', batch_size=8, max_episode_steps=12, translation_limit_m=.001,
                                      rotation_limit_rad=.01).model_dump()
        client.put('/hilserl/config', json={'expected_revision': 0, 'config': config}).raise_for_status()
        assert client.get('/hilserl/workspace').json()['readiness']['can_train']
        result = client.post('/hilserl/sessions', json={'expected_config_revision': 1, 'mode': 'train'})
        assert result.status_code == 201, result.text
        item = result.json()
        url = f"/hilserl/sessions/{item['id']}"
        def wait(predicate, timeout=45):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                value = client.get(url).json()
                assert not value.get('runtime_error'), value.get('runtime_error')
                if predicate(value):
                    return value
                time.sleep(.1)
            pytest.fail(f'Isaac runtime timeout: {value}')
        def command(action):
            value = wait(lambda v: not v.get('runtime_pending'))
            response = client.post(url+'/commands', json={'expected_revision': value['revision'], 'action': action})
            assert response.status_code == 200, response.text
            return wait(lambda v: not v.get('runtime_pending'))
        try:
            command('start')
            value = wait(lambda v: (v['metrics']['policy_version'] or 0) >= 5 and v['metrics']['completed_episodes'] >= 1 and v['runtime']['environment_info']['physics_steps'] > 0)
            assert value['runtime']['observation_dim'] == 39
            assert value['runtime']['action_dim'] == 6
            assert value['runtime']['environment_info']['grasp_verified']
            assert value['runtime']['environment_info']['physics_steps'] > 0
            value = command('take_over')
            assert client.post(url+'/action', json={'expected_revision': value['revision'], 'value': [0, 0, 1]}).status_code == 422
            assert client.post(url+'/action', json={'expected_revision': value['revision'], 'value': [0]*7}).status_code == 422
            response = client.post(url+'/action', json={'expected_revision': value['revision'], 'value': [0, 0, .2, 0, 0, 0]})
            assert response.status_code == 200
            wait(lambda v: v['runtime']['last_action_source'] == 'human' and v['runtime']['last_action'][2] > .19)
            value = command('pause')
            steps = value['metrics']['environment_steps']
            time.sleep(.3)
            assert wait(lambda v: True)['metrics']['environment_steps'] == steps
            command('reset')
            command('resume')
            wait(lambda v: v['metrics']['environment_steps'] > steps)
            command('stop')
            hil._workers[item['id']].wait(timeout=15)
            folder = tmp_path / item['id']
            with np.load(folder/'replay.npz') as data:
                assert data['observation'].shape[1] == 39
                assert data['action'].shape[1] == 6
                assert data['human'].any()
                assert str(data['__schema__']) == 'nist-sg2-state39-delta6-fixedgripper-v2'
            transitions = [json.loads(line) for line in (folder/'transitions.jsonl').read_text().splitlines()]
            assert any(row['environment_info']['physics_steps'] > 0 for row in transitions)
            assert any(row['truncated'] for row in transitions)
            assert not hil._isaac.health()['busy']
            config.update(checkpoint_path=str(folder/'policy.npz'), demonstration_path=str(folder/'replay.npz'))
            client.put('/hilserl/config', json={'expected_revision': 1, 'config': config}).raise_for_status()
            evaluated = client.post('/hilserl/sessions', json={'expected_config_revision': 2, 'mode': 'evaluate'}).json()
            url = f"/hilserl/sessions/{evaluated['id']}"
            command('start')
            evaluation = wait(lambda v: (v['metrics']['environment_steps'] or 0) >= 5)
            assert evaluation['metrics']['learner_steps'] == 0
            command('stop')
            hil._workers[evaluated['id']].wait(timeout=15)
            assert not hil._isaac.health()['busy']
            print('ISAAC_SMOKE', json.dumps(value['metrics']))
        finally:
            hil.shutdown()


def test_visualization_proxy_returns_jpeg_without_starting_runtime(monkeypatch):
    app = FastAPI()
    app.include_router(hil.router)
    payload = b'\xff\xd8test-jpeg\xff\xd9'
    monkeypatch.setattr(hil._isaac, 'frame', lambda camera, quality: (payload, {'Cache-Control': 'no-store', 'X-Frame-Id': '7'}))
    with TestClient(app) as client:
        response = client.get('/hilserl/isaac/frame')
        assert response.content == payload
        assert response.headers['content-type'] == 'image/jpeg'
        assert response.headers['cache-control'] == 'no-store'
        assert response.headers['x-frame-id'] == '7'
        for camera in ('head', 'left_wrist', 'right_wrist'):
            assert client.get('/hilserl/isaac/frame', params={'camera': camera}).content == payload
        assert client.get('/hilserl/isaac/frame?camera=unknown').status_code == 422
        assert client.get('/hilserl/isaac/frame?quality=unknown').status_code == 422
        for quality in ('low', 'standard', 'high', 'ultra'):
            assert client.get('/hilserl/isaac/frame', params={'quality': quality}).content == payload
        def unavailable(camera, quality):
            raise RuntimeError('Camera unavailable')
        monkeypatch.setattr(hil._isaac, 'frame', unavailable)
        assert client.get('/hilserl/isaac/frame').status_code == 503


@pytest.mark.skipif(os.environ.get('CYCLO_TEST_GRASPS') != '1', reason='Requires idle Isaac bridge; verifies physical initialization')
@pytest.mark.parametrize('arm', ['left', 'right'])
@pytest.mark.parametrize('part', [f'peg_{shape}_{size}' for shape in ['round', 'square'] for size in [4, 8, 12, 16]])
def test_physical_grasp_initialization_and_reset(arm, part):
    config = hil.ExperimentConfig(environment='isaac_sim', observation_mode='state', robot_type='ffw_sg2_rev1',
                                  control_hz=20, arm=arm, target_part=part,
                                  initialization='grasped').model_dump()
    env = hil._isaac.IsaacNistEnv(config)
    try:
        assert env.info['grasp_verified']
        assert env.info['grasp_drift_m'] < .008
        assert env.info['physics_steps'] == 0  # initialization is not policy data
        env.reset()
        assert env.info['grasp_verified']
        assert env.info['grasp_drift_m'] < .008
    finally:
        env.close()


def test_isaac_preview_requires_live_bridge(monkeypatch, tmp_path):
    monkeypatch.setenv('CYCLO_HILSERL_DATA_DIR', str(tmp_path))
    monkeypatch.setattr(hil._isaac, 'health', lambda: {'ready': False, 'error': 'offline'})
    app = FastAPI(); app.include_router(hil.router)
    with TestClient(app) as client:
        assert not client.get('/hilserl/workspace').json()['readiness']['can_preview']
        assert client.post('/hilserl/sessions', json={'mode': 'preview', 'expected_config_revision': 0}).status_code == 409


@pytest.mark.skipif(os.environ.get('CYCLO_TEST_ISAAC') != '1', reason='Requires idle Isaac bridge')
def test_isaac_preview_initializes_manual_physics_without_learning(tmp_path, monkeypatch):
    monkeypatch.setenv('CYCLO_HILSERL_DATA_DIR', str(tmp_path))
    app = FastAPI(); app.include_router(hil.router)
    with TestClient(app) as client:
        config = hil.ExperimentConfig(environment='isaac_sim', observation_mode='state', robot_type='ffw_sg2_rev1', control_hz=20,
                                      initialization='near_hole', batch_size=8, max_episode_steps=1000).model_dump()
        client.put('/hilserl/config', json={'expected_revision': 0, 'config': config}).raise_for_status()
        response = client.post('/hilserl/sessions', json={'mode': 'preview', 'expected_config_revision': 1})
        response.raise_for_status()
        item = response.json(); url = f"/hilserl/sessions/{item['id']}"
        def wait(predicate):
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                value = client.get(url).json()
                assert not value.get('runtime_error'), value
                if predicate(value): return value
                time.sleep(.1)
            pytest.fail(f'Preview timed out: {value}')
        def command(action):
            value = client.get(url).json()
            client.post(url+'/commands', json={'expected_revision': value['revision'], 'action': action}).raise_for_status()
            return wait(lambda v: not v.get('runtime_pending'))
        try:
            value = command('start')
            assert value['control_source'] == 'policy'
            assert value['runtime']['environment_info']['grasp_verified']
            value = wait(lambda v: v['metrics']['environment_steps'] >= 10)
            assert value['metrics']['learner_steps'] == 0
            assert value['runtime']['last_action'] == [0.] * 6
            client.post(url+'/action', json={'expected_revision': value['revision'], 'value': [0,0,1,0,0,0]}).raise_for_status()
            wait(lambda v: v['runtime']['last_action'][2] > .99)
            value = wait(lambda v: v['control_source'] == 'policy' and v['runtime']['last_action'] == [0.] * 6)
            human_steps = value['metrics']['human_steps']
            assert human_steps > 0
            time.sleep(.4)
            assert client.get(url).json()['metrics']['human_steps'] == human_steps
            client.post(url+'/action', json={'expected_revision': value['revision'], 'value': [0,0,1,0,0,0]}).raise_for_status()
            client.post(url+'/action', json={'expected_revision': value['revision'], 'value': [0]*6}).raise_for_status()
            assert client.get(url).json()['control_source'] == 'policy'
            value = command('pause'); steps = value['metrics']['environment_steps']
            time.sleep(.3)
            assert client.get(url).json()['metrics']['environment_steps'] == steps
            value = command('reset')
            assert value['runtime']['environment_info']['grasp_verified']
            command('resume')
            value = wait(lambda v: v['metrics']['environment_steps'] > steps)
            assert value['metrics']['learner_steps'] == 0
            command('stop'); hil._workers[item['id']].wait(timeout=15)
            assert not hil._isaac.health()['busy']
        finally:
            hil.shutdown()


@pytest.mark.skipif(os.environ.get('CYCLO_TEST_PICK_PLACE') != '1', reason='Requires idle Isaac bridge for physical tray pickup')
def test_pick_place_prepares_vertical_peg_near_hole_before_step_zero():
    import numpy as np
    config = hil.ExperimentConfig(environment='isaac_sim', observation_mode='state', robot_type='ffw_sg2_rev1', control_hz=20,
                                  arm='right', target_part='peg_round_8', initialization='pick_place').model_dump()
    env = hil._isaac.IsaacNistEnv(config)
    try:
        for reset in (False, True):
            if reset: env.reset()
            assert env.info['grasp_verified']
            assert env.info['episode_step'] == 0
            assert env.info['physics_steps'] == 0
            offset = np.array(env.info['preinsert_offset_m'])
            error = np.array(env.info['position_m']) - np.array(env.info['target_position_m']) - offset
            assert np.linalg.norm(error) < .025
            q = env.observation()[17:21]
            assert 1 - 2 * (q[1]**2 + q[2]**2) > np.cos(np.deg2rad(10))
    finally:
        env.close()


@pytest.mark.skipif(os.environ.get('CYCLO_TEST_NEAR_HOLE') != '1', reason='Requires idle Isaac bridge for direct near-hole reset')
def test_direct_near_hole_reset_and_free_physics_hold():
    import numpy as np
    config = hil.ExperimentConfig(environment='isaac_sim', observation_mode='state', robot_type='ffw_sg2_rev1', control_hz=20,
                                  arm='right', target_part='peg_round_8').model_dump()
    assert config['initialization'] == 'near_hole'
    env = hil._isaac.IsaacNistEnv(config)
    try:
        for reset in (False, True):
            if reset:
                env.reset()
            assert env.info['initialization'] == 'near_hole'
            assert env.info['grasp_verified']
            assert env.info['episode_step'] == env.info['physics_steps'] == 0
            assert .005 < env.info['peg_clearance_above_board_m'] < .025
            assert abs(env.info['lift_position_m'] + .35) < .005
            # Camera horizon is level, optical axis points down 45 degrees.
            expected_q = np.array([.01836045, .04432602, .38224274, .92281558])
            actual_q = np.array(env.info['observer_eef_quaternion_wxyz'])
            alignment = abs(np.dot(expected_q / np.linalg.norm(expected_q), actual_q / np.linalg.norm(actual_q)))
            assert alignment > np.cos(np.deg2rad(.5))  # orientation error < 1 degree
            np.testing.assert_allclose(env.info['observer_eef_position_m'], [.4600139, .1587696, .8695936], atol=.003)

            assert env.info['gripper_mode'] == 'fixed'
            grip = list(env.info['gripper_target'])
            with pytest.raises(ValueError, match='six finite'):
                env.step(np.zeros(7))
            start = np.array(env.info['position_m'])
            start_eef = np.array(env.info['eef_position_m'])
            start_observer = np.array(env.info['observer_eef_position_m'])
            for _ in range(150):
                env.step(np.zeros(6))
                assert env.info['gripper_target'] == grip
                assert env.info['arm_joint_hold']
                assert np.linalg.norm(np.array(env.info['eef_position_m']) - start_eef) < .001
                assert np.linalg.norm(np.array(env.info['observer_eef_position_m']) - start_observer) < .001
            assert np.linalg.norm(np.array(env.info['position_m']) - start) < .001
            assert env.info['episode_step'] == 150
            held = np.array(env.info['eef_position_m'])
            for _ in range(5):
                env.step(np.array([0, 0, .2, 0, 0, 0]))
                assert env.info['gripper_target'] == grip
                assert not env.info['arm_joint_hold']
            assert np.array(env.info['eef_position_m'])[2] - held[2] > .001
            env.step(np.zeros(6))
            released = np.array(env.info['eef_position_m'])
            for _ in range(30):
                env.step(np.zeros(6))
                assert env.info['gripper_target'] == grip
                assert env.info['arm_joint_hold']
                assert np.linalg.norm(np.array(env.info['eef_position_m']) - released) < .001
    finally:
        env.close()


def test_tcp_manual_translation_and_rotation_follow_rotated_tool():
    import numpy as np
    env = hil._isaac.IsaacNistEnv.__new__(hil._isaac.IsaacNistEnv)
    env._obs = np.zeros(39, dtype=np.float32)
    # TCP rotated +90 degrees about world Z: local X becomes world Y.
    env._obs[17:21] = [2**-.5, 0, 0, 2**-.5]
    result = env.tcp_to_action(np.array([1,0,0,1,0,0]))
    np.testing.assert_allclose(result, [0,1,0,0,1,0], atol=1e-6)
    assert np.max(np.abs(env.tcp_to_action(np.ones(6)))) <= 1


def test_fixed_gripper_rejects_old_action_before_sending(monkeypatch):
    import numpy as np
    env = hil._isaac.IsaacNistEnv.__new__(hil._isaac.IsaacNistEnv)
    def unexpected(*args, **kwargs):
        pytest.fail('Invalid action must not consume a bridge sequence')
    monkeypatch.setattr(env, '_command', unexpected)
    for action in (np.zeros(7), np.full(6, np.nan), np.full(6, 2.)):
        with pytest.raises(ValueError, match='six finite'):
            env.step(action)
