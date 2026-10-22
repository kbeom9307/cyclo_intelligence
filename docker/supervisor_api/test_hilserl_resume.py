import json
from uuid import uuid4
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import hilserl as hil
from hilserl_replay import DiskReplay
from test_hilserl_upstream import config, spec, row


def test_replay_resume_preserves_ring_rng_and_source(tmp_path):
    definition = spec(config(pixel_replay_capacity=64))
    original = DiskReplay(tmp_path / 'source', definition)
    for i in range(70):
        original.insert(row(float(i), human=i % 2 == 0))
    original.save()
    restored = DiskReplay(tmp_path / 'new', definition)
    restored.restore(tmp_path / 'source')
    assert (restored.online.size, restored.online.cursor, restored.demo.size) == (64, 6, 35)
    assert restored.latest_data_id() == -1
    for a, b in zip(original.sample(8), restored.sample(8)):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    restored.insert(row(999))
    assert original.online.arrays['reward'][6] == 6
    assert restored.online.arrays['reward'][6] == 999
    (tmp_path / 'source/online/index.json').write_text(json.dumps({'format':'cyclo-memmap-replay-v1','capacity':64,'size':65,'cursor':0}))
    with pytest.raises(ValueError, match='size/cursor'):
        restored.restore(tmp_path / 'source')


def test_resume_api_preserves_training_config_counts_and_rejects_unsafe_states(tmp_path, monkeypatch):
    monkeypatch.setenv('CYCLO_HILSERL_DATA_DIR', str(tmp_path))
    monkeypatch.setattr(hil, 'preflight', lambda c: {'can_train': True})
    identifier = str(uuid4())
    c = config(checkpoint_path='/old-policy', demonstration_path='/old-demo')
    source = dict(id=identifier, mode='train', state='stopped', revision=2, config_revision=3,
                  config=c, metrics={}, episode=5, events=[])
    folder = tmp_path / identifier
    for sub in ['replay/online', 'replay/demonstrations']:
        (folder / sub).mkdir(parents=True, exist_ok=True)
        (folder / sub / 'index.json').write_text('{}')
    (folder / 'learner.msgpack').write_bytes(b'checkpoint')
    (folder / 'spec.json').write_text(json.dumps(spec(c)))
    status = dict(artifacts_saved=True, ack_revision=2, episode=5,
                  metrics=dict(environment_steps=1000, completed_episodes=4, successful_episodes=3,
                               success_rate=.75, learner_steps=80, critic_steps=160, human_steps=10))
    (folder / 'runtime.json').write_text(json.dumps(status))
    with hil.database() as db: hil.write(db, 'session:' + identifier, source)
    app = FastAPI(); app.include_router(hil.router)
    with TestClient(app) as client:
        url = f'/hilserl/sessions/{identifier}/restart-training'
        assert client.post(url, json={'expected_revision':1}).status_code == 409
        result = client.post(url, json={'expected_revision':2})
        assert result.status_code == 200, result.text
        resumed = result.json()
        assert resumed['state'] == 'created' and resumed['id'] != identifier
        assert resumed['config']['checkpoint_path'] == resumed['config']['demonstration_path'] == ''
        assert resumed['resume']['counters']['episode'] == 6
        assert resumed['resume']['counters']['learner_steps'] == 80
        assert resumed['resume']['counters']['successful_episodes'] == 3
        assert client.post(url, json={'expected_revision':2}).status_code == 409
        with hil.database() as db:
            assert hil.read(db, 'session:' + identifier) == source
            db.execute('DELETE FROM documents WHERE id=?', ('session:' + resumed['id'],))
        (folder / 'learner.msgpack').unlink()
        assert client.post(url, json={'expected_revision':2}).status_code == 422
        status['artifacts_saved'] = False
        (folder / 'runtime.json').write_text(json.dumps(status))
        assert client.post(url, json={'expected_revision':2}).status_code == 409


def test_active_time_accumulates_across_resume():
    from hilserl_resume import elapsed_active
    session = {'resume': {'active_seconds': 20}, 'events': [
        {'action':'start','at':'2026-10-21T00:00:00Z'},
        {'action':'pause','at':'2026-10-21T00:00:10Z'},
        {'action':'resume','at':'2026-10-21T00:01:00Z'},
        {'action':'stop','at':'2026-10-21T00:01:05Z'}]}
    assert elapsed_active(session, {}) == 35


def test_runtime_resume_keeps_counters_history_and_skips_demo_import(tmp_path, monkeypatch):
    import os
    import hilserl_runtime as runtime
    import hilserl_upstream as upstream
    c = config(demonstration_path='/must-not-be-imported-again', checkpoint_path='/must-not-load-policy-only')
    previous = tmp_path / 'old'; previous.mkdir()
    history = json.dumps({'episode':4, 'terminal':True, 'reward':1}) + '\n'
    (previous / 'transitions.jsonl').write_text(history)
    resume = {'session_id':'old', 'learner_directory':str(previous), 'counters':{
        'environment_steps':100, 'completed_episodes':4, 'successful_episodes':3,
        'human_steps':10, 'episode':6, 'learner_steps':80, 'critic_steps':160}}
    class Env:
        pixel_mode=True; observation_dim=22; action_dim=6; schema='test'
        images=np.zeros((9,128,128),np.uint8); position=np.zeros(3)
        info={}; capture_metadata={}; steps=0
        def observation(self): return np.zeros(22,np.float32)
        def heartbeat(self): pass
        def step(self, action):
            self.steps += 1
            return self.observation(), 0., False, False
    class Actor:
        def __init__(self, *args): pass
        def restore(self, weights): pass
        def action(self, *args, **kwargs): return np.zeros(6,np.float32)
        def export(self): return {'weights':np.zeros(1)}
        def memory(self): return {}
    class Backend:
        remote=False; transport='test'; steps=80
        def __init__(self, cfg, spec, folder, weights, demos, heartbeat, resume=None):
            assert resume['session_id']=='old' and demos==[] and weights is None
        def snapshot(self):
            return {'version':self.steps, 'metrics':{'learner_steps':self.steps}, 'artifact_directory':str(tmp_path/'new')}, {'weights':np.zeros(1)}
        def insert(self, row): assert row['step_id']==100 and row['episode']==6
        def pause(self, heartbeat): pass
        def tick(self, running): self.steps += int(running)
        def close(self, heartbeat): return self.snapshot()
    monkeypatch.setattr(runtime, 'AsyncLearner', Backend)
    monkeypatch.setattr(upstream, 'UpstreamPolicy', Actor)
    env=Env()
    def session():
        return {'mode':'train','state':'running' if env.steps==0 else 'stopped',
                'events':[], 'revision':0, 'resume':resume}
    folder=tmp_path/'new';folder.mkdir()
    runtime.train(tmp_path,'new',os.getppid(),folder,c,env,session)
    status=json.loads((folder/'runtime.json').read_text())
    assert status['artifacts_saved'] and status['episode']==6
    assert status['metrics']['environment_steps']==101
    assert status['metrics']['learner_steps']==81
    assert status['metrics']['completed_episodes']==4 and status['metrics']['success_rate']==.75
    assert (folder/'transitions.jsonl').read_text().startswith(history)
    assert (previous/'transitions.jsonl').read_text()==history
