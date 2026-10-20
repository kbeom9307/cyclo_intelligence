"""Restored-engine contracts; numerical upstream test is opt-in and uses CPU."""
import json
import os
import numpy as np
import pytest

from hilserl import ExperimentConfig
from hilserl_profiles import image_shape, pixel_schema, UPSTREAM_SCHEMA
from hilserl_replay import DiskReplay
from hilserl_remote import pack, unpack, arrays_to_rows, rows_to_arrays, PROTOCOL


def config(**values):
    return ExperimentConfig(**{'environment': 'isaac_sim', 'actor_device': 'cpu', 'learner_device': 'cpu',
                               'pixel_replay_capacity': 64, **values}).model_dump()


def row(reward=0., human=False):
    return {'observation': np.zeros(22, np.float32), 'next_observation': np.zeros(22, np.float32),
            'action': np.zeros(6, np.float32), 'reward': reward, 'terminal': False, 'truncated': False,
            'human': human, 'images': np.full((9, 128, 128), 90, np.uint8),
            'next_images': np.full((9, 128, 128), 100, np.uint8)}


def spec(c):
    return {'config': c, 'pixel': True, 'schema': UPSTREAM_SCHEMA, 'observation_dim': 22, 'action_dim': 6,
            'protocol': PROTOCOL}


def test_native_resolution_and_restored_defaults():
    c = ExperimentConfig().model_dump()
    assert (c['pixel_batch_size'], c['pixel_replay_capacity'], c['discount'], c['cta_ratio']) == (256, 200000, .97, 2)
    assert image_shape(c) == (9, 128, 128) and pixel_schema(c) == UPSTREAM_SCHEMA
    assert image_shape({}) == (9, 64, 64)
    with pytest.raises(ValueError):
        config(pixel_batch_size=255)
    with pytest.raises(ValueError):
        config(camera_crops={'head': [.8, 0, .2, 1]})


def test_restored_actor_readiness_when_overview_leaves_too_little_vram(monkeypatch):
    import hilserl
    monkeypatch.setattr(hilserl, 'hardware_status', lambda: {'gpus': [
        {'device': 'cuda:0', 'free_mib': 400, 'total_mib': 8151, 'used_mib': 7300}], 'error': None})
    monkeypatch.setattr(hilserl._isaac, 'health', lambda: {
        'ready': True, 'busy': False, 'robot_type': 'ffw_sg2_rev1', 'control_hz': 20,
        'target_parts': ['peg_round_8'], 'pixel_observations': True, 'pixel_sizes': [64, 128]})
    c = config(robot_type='ffw_sg2_rev1', control_hz=20, actor_device='cuda:0', learner_device='cuda:0')
    status = hilserl.preflight(c)
    actor = next(item for item in status['checks'] if item['id'] == 'actor')
    assert not actor['ready'] and 'Close Overview' in actor['detail']
    assert status['can_classifier'] and status['can_preview']
    assert status['can_demo']  # Environment-reward demo collection needs no GPU policy/classifier.


def test_remote_128_images_and_model_sized_packets():
    c = config()
    arrays = rows_to_arrays([row()] * 32, True)
    meta, restored = unpack(pack({'sequence': 1}, arrays))
    assert len(arrays_to_rows(restored, spec(c))) == 32
    assert meta['sequence'] == 1
    arrays['images'] = np.zeros((32, 9, 64, 64), np.uint8)
    with pytest.raises(ValueError, match='schema'):
        arrays_to_rows(arrays, spec(c))
    weights = {'layer': np.ones((9_000_000,), np.float32)}
    assert unpack(pack({}, weights))[1]['layer'].shape == weights['layer'].shape


def test_separate_demo_online_buffers_and_disk_checkpoint(tmp_path):
    replay = DiskReplay(tmp_path, spec(config()))
    replay.insert_demo(row(1., True))
    assert len(replay) == 0 and replay.demo.size == 1 and not replay.ready(2)
    replay.insert(row())
    replay.insert(row())
    assert replay.ready(2)
    sampled = replay.sample(8)
    assert sum(r['reward'] == 1. for r in sampled) == 4
    replay.insert(row(1., True))
    assert len(replay) == 3 and replay.demo.size == 2
    replay.save()
    assert json.loads((tmp_path / 'online/index.json').read_text())['size'] == 3
    assert isinstance(replay.online.arrays['images'], np.memmap)


@pytest.mark.skipif(os.environ.get('CYCLO_TEST_UPSTREAM') != '1', reason='opt-in full upstream numerical test')
def test_upstream_updates_frozen_backbone_temperature_and_policy(tmp_path):
    from hilserl_upstream import UpstreamSAC, UpstreamPolicy, export_tree, jax
    c = config(pixel_batch_size=2)
    learner = UpstreamSAC(c)
    before = learner.policy.export()
    initial_temperature = float(learner.agent.forward_temperature())
    batch = [row(1., True), row()]
    learner.update(batch, critic_only=True)
    metrics = learner.update(batch)
    after = learner.policy.export()
    backbone = [k for k in before if '/pretrained_encoder/' in k]
    assert backbone and all(np.array_equal(before[k], after[k]) for k in backbone)
    assert any(not np.array_equal(before[k], after[k]) for k in before if k not in backbone)
    assert float(learner.agent.forward_temperature()) != initial_temperature
    assert np.isfinite(metrics['critic_loss']) and learner.critic_steps == 2
    actor = UpstreamPolicy(c)
    actor.restore(after)
    actual = actor.action(np.zeros(22), batch[0]['images'], deterministic=True)
    assert actual.shape == (6,) and np.isfinite(actual).all() and np.max(np.abs(actual)) <= 1.
    np.testing.assert_allclose(actual, learner.policy.action(np.zeros(22), batch[0]['images'], deterministic=True))
    learner.save(tmp_path)
    from flax import serialization
    saved = serialization.from_bytes(learner.agent.state, (tmp_path / 'learner.msgpack').read_bytes())
    np.testing.assert_array_equal(saved.rng, learner.agent.state.rng)
    assert jax.tree_util.tree_structure(saved.opt_states) == jax.tree_util.tree_structure(learner.agent.state.opt_states)


@pytest.mark.skipif(os.environ.get('CYCLO_TEST_UPSTREAM_REMOTE') != '1', reason='opt-in two-process upstream transport test')
def test_remote_upstream_demos_updates_and_checkpoints(tmp_path):
    import secrets
    import socket
    import subprocess
    import sys
    import time
    from hilserl_remote import RemoteEndpoint
    from hilserl_learner import AsyncLearner, make_spec
    from types import SimpleNamespace
    from hilserl_upstream import UpstreamPolicy
    device = os.environ.get('CYCLO_TEST_UPSTREAM_DEVICE', 'cpu')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    token = tmp_path / 'token'
    token.write_text(secrets.token_urlsafe(48))
    environment = {**os.environ, 'CYCLO_HIL_LEARNER_TOKEN_FILE': str(token),
                   'CYCLO_HIL_LEARNER_DATA_DIR': str(tmp_path / 'server')}
    log = (tmp_path / 'server.log').open('w')
    server = subprocess.Popen([sys.executable, 'hilserl_learner_server.py', '--port', str(port)],
                              env=environment, stdout=log, stderr=log)
    backend = None
    try:
        c = config(actor_device=device, learner_device=device, learner_placement='remote',
                   learner_url=f'http://127.0.0.1:{port}', learner_token_file=str(token),
                   pixel_batch_size=2, training_starts=2, policy_publish_interval=1)
        endpoint = RemoteEndpoint(c)
        deadline = time.monotonic() + 45
        while True:
            try:
                endpoint.request('/health')
                break
            except RuntimeError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(.2)
        env = SimpleNamespace(pixel_mode=True, schema=UPSTREAM_SCHEMA, observation_dim=22, action_dim=6)
        backend = AsyncLearner(c, make_spec(c, env), tmp_path / 'actor', None, [row(1., True), row(1., True)])
        assert backend.snapshot()[0]['metrics']['replay_size'] == 0
        assert backend.snapshot()[0]['metrics']['demo_replay_size'] == 2
        backend.insert(row())
        backend.insert(row())
        backend.tick(True)
        deadline = time.monotonic() + 180
        while backend.snapshot()[0]['metrics']['learner_steps'] < 1:
            if time.monotonic() > deadline:
                pytest.fail('Restored external learner did not update')
            time.sleep(.1)
        backend.pause()
        metadata, weights = backend.snapshot()
        assert metadata['learner_memory']['framework'] == 'JAX'
        assert metadata['metrics']['critic_steps'] == metadata['metrics']['learner_steps'] * 2
        actor = UpstreamPolicy(c)
        actor.restore(weights)
        assert np.isfinite(actor.action(np.zeros(22), row()['images'])).all()
        metadata, weights = backend.close()
        folder = tmp_path / 'server' / metadata['run_id']
        assert (folder / 'learner.msgpack').is_file() and (folder / 'policy.npz').is_file()
        assert json.loads((folder / 'replay/demonstrations/index.json').read_text())['size'] == 2
        assert not (tmp_path / 'actor/learner.msgpack').exists()
    finally:
        if backend:
            backend.close()
        server.terminate()
        server.wait(timeout=30)
        log.close()
