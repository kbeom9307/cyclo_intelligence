"""Real Isaac camera/Actor collection integration; artifacts isolated in tmp_path."""
import json
from pathlib import Path
import os
import time
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import hilserl as hil


@pytest.mark.skipif(os.environ.get('CYCLO_TEST_PIXELS') != '1', reason='Requires idle Isaac pixel bridge')
def test_real_pixel_collection_demo_and_learner(tmp_path, monkeypatch):
    monkeypatch.setenv('CYCLO_HILSERL_DATA_DIR', str(tmp_path))
    app = FastAPI(); app.include_router(hil.router)
    with TestClient(app) as client:
        config = hil.ExperimentConfig(learning_backend='legacy_torch', environment='isaac_sim', robot_type='ffw_sg2_rev1', observation_mode='pixels',
            control_hz=20, pixel_batch_size=2, pixel_replay_capacity=64, max_episode_steps=100,
            observation_cameras=['head', 'left_wrist'], classifier_cameras=['right_wrist'],
            translation_limit_m=.001, rotation_limit_rad=.01).model_dump()
        client.put('/hilserl/config',json={'expected_revision':0,'config':config}).raise_for_status()
        item = None
        def wait(predicate, timeout=75):
            deadline=time.monotonic()+timeout
            while time.monotonic()<deadline:
                value=client.get('/hilserl/sessions/'+item['id']).json()
                assert not value.get('runtime_error'), value.get('runtime_error')
                if predicate(value):return value
                time.sleep(.1)
            pytest.fail(f'Timeout: {value}')
        def command(action):
            current=wait(lambda s:not s.get('runtime_pending'))
            response=client.post('/hilserl/sessions/'+item['id']+'/commands',json={'expected_revision':current['revision'],'action':action})
            response.raise_for_status()
            return wait(lambda s:not s.get('runtime_pending'))
        def create(mode, revision=1):
            response=client.post('/hilserl/sessions',json={'expected_config_revision':revision,'mode':mode})
            response.raise_for_status();return response.json()
        def stop():
            command('stop');hil._workers[item['id']].wait(timeout=20)
        try:
            item=create('classifier');command('start')
            command('pause')
            initial=wait(lambda s:True)
            assert initial['runtime']['observation_dim']==22
            assert initial['runtime']['image_shape']==[9,64,64]
            assert initial['runtime']['metrics']['learner_steps']==0
            assert initial['runtime']['policy_enabled'] is False
            captured=initial
            assert captured['runtime']['collection']['negative_samples']==captured['runtime']['episode_step']
            assert captured['runtime']['collection']['negative_samples'] > 0
            assert captured['runtime']['episode_step']==initial['runtime']['episode_step']
            assert captured['runtime']['collection']['last_capture']['source'] == 'step'
            sample=captured['runtime']['collection']['last_capture']
            assert sample['cameras'] == ['right_wrist']
            with np.load(tmp_path/item['id']/'classifier_samples'/f"{sample['id']}.npz") as data:
                assert not data['images'][:6].any() and data['images'][6:].any()
            response=client.get(f"/hilserl/classifier-samples/{item['id']}/{sample['id']}")
            assert response.status_code==200 and response.content[:2]==b'\xff\xd8'
            duplicate=command('capture_success')
            assert 'already labeled' in duplicate['runtime']['collection']['error']
            assert duplicate['runtime']['collection']['positive_samples']==0
            stop()
            # Deliberate fixture-only success annotation tests demo plumbing, not insertion achievement.
            item=create('demo');command('start')
            wait(lambda s:s['runtime']['episode_step']>=3)
            command('pause');command('success')
            saved=command('save_demo')
            assert saved['runtime']['collection']['saved_demo_episodes']==1
            assert saved['runtime']['collection']['pending_demo_steps']==0
            command('reset');stop()
            demo=tmp_path/item['id']/'demonstrations.npz'
            with np.load(demo,allow_pickle=False) as data:
                assert data['images'].shape[1:]==(9,64,64)
                assert data['images'].dtype==np.uint8
                assert data['action'].shape[1]==6
                assert data['observation'].shape[1]==22
                assert data['terminal'][-1]
                assert data['human'].all()
                assert data['images'].std()>1
                assert not data['images'][:,6:].any()
                assert data['images'][:,:6].any()
                assert json.loads(str(data['__cameras__'])) == ['head', 'left_wrist']
            config.update(demonstration_path=str(demo))
            client.put('/hilserl/config',json={'expected_revision':1,'config':config}).raise_for_status()
            item=create('train',2);command('start')
            learned=wait(lambda s:s['metrics']['policy_version']>=5)
            assert learned['runtime']['observation_mode']=='pixels'
            assert learned['runtime']['environment_info']['gripper_mode']=='fixed'
            assert learned['runtime']['classifier_probability'] is None
            assert learned['runtime']['last_action_source']=='policy'
            stop()
            with np.load(tmp_path/item['id']/'policy.npz',allow_pickle=False) as weights:
                assert any(key.startswith('encoder.') for key in weights.files)
            with np.load(tmp_path/item['id']/'replay.npz',allow_pickle=False) as data:
                assert data['images'].shape[1:]==(9,64,64)
            # Constant-positive classifier fixture verifies reward/termination wiring only.
            # It is not a trained insertion classifier and stays inside pytest's temp directory.
            import torch
            from hilserl_pixels import SuccessClassifier, CLASSIFIER_SCHEMA
            from hilserl_data import write_npz, task_key
            classifier=SuccessClassifier()
            for parameter in classifier.parameters(): parameter.data.zero_()
            classifier.head.bias.data.fill_(9.)
            classifier_path=tmp_path/'classifier-fixture.npz'
            write_npz(classifier_path,__schema__=np.array(CLASSIFIER_SCHEMA),__task__=np.array(json.dumps(task_key(config))),
                      __cameras__=np.array(json.dumps(config['classifier_cameras'])),
                      **{k:v.detach().numpy() for k,v in classifier.state_dict().items()})
            config.update(checkpoint_path=str(tmp_path/item['id']/'policy.npz'), classifier_path=str(classifier_path),
                          reward_source='classifier', classifier_hold_steps=3)
            client.put('/hilserl/config',json={'expected_revision':2,'config':config}).raise_for_status()
            item=create('evaluate',3);command('start')
            value=wait(lambda s:s['metrics']['completed_episodes']>=1)
            assert value['metrics']['learner_steps']==0
            assert value['metrics']['success_rate']==1.
            stop()
            rows=[json.loads(line) for line in (tmp_path/item['id']/'transitions.jsonl').read_text().splitlines()]
            assert any(row['reward']==10 and row['terminal'] and row['environment_info'].get('classifier_success')
                       and not row['environment_info']['is_success'] for row in rows)
        finally:
            if item and item['id'] in hil._workers and hil._workers[item['id']].poll() is None:
                value=client.get('/hilserl/sessions/'+item['id']).json()
                client.post('/hilserl/sessions/'+item['id']+'/commands',json={'expected_revision':value['revision'],'action':'stop'})
                hil._workers[item['id']].wait(timeout=30)


def test_classifier_job_api_with_stopped_labeled_collection(tmp_path, monkeypatch):
    from uuid import uuid4
    from hilserl_data import Collection
    from hilserl_pixels import IMAGE_SHAPE, PIXEL_SCHEMA
    monkeypatch.setenv('CYCLO_HILSERL_DATA_DIR',str(tmp_path))
    monkeypatch.setattr(hil,'_classifier_workers',{})
    config=hil.ExperimentConfig(environment='isaac_sim',robot_type='ffw_sg2_rev1',control_hz=20, classifier_cameras=['left_wrist'], learning_backend='legacy_torch').model_dump()
    identifier=str(uuid4());collection=Collection(tmp_path/identifier,config,'classifier',PIXEL_SCHEMA)
    class Env:
        capture_metadata={'sequence':1}
        def capture(self): pass
        def observation(self): return np.zeros(22)
    rng=np.random.default_rng(11);env=Env()
    for episode in range(4):
        for step in range(10):
            env.info={'physics_steps':step}; env.images=rng.integers(0,255,IMAGE_SHAPE,dtype=np.uint8)
            collection.capture(env,step%2,episode)
    collection.export()
    with hil.database() as db:
        hil.write(db,'session:'+identifier,{'id':identifier,'mode':'classifier','state':'stopped','config':config,
                  'revision':0,'metrics':{},'episode':1,'episode_outcome':None})
    app=FastAPI();app.include_router(hil.router)
    with TestClient(app) as client:
        response=client.post('/hilserl/classifiers/train',json={'session_ids':[identifier],'epochs':1,'device':'cpu'})
        assert response.status_code==202,response.text
        job=response.json()['id']
        process=hil._classifier_workers[job]
        assert process.wait(timeout=20)==0
        state=next(item for item in hil.classifier_jobs() if item['id']==job)
        assert state['state']=='complete' and Path(state['checkpoint_path']).is_file()
        assert state['cameras'] == ['left_wrist']
        assert state['environment_names'] == [config['name']]
        with np.load(state['checkpoint_path']) as model:
            assert json.loads(str(model['__cameras__'])) == ['left_wrist']
        assert state['validation_split'] == 'global_label_stratified_v1'
        assert client.post('/hilserl/classifiers/train',json={'session_ids':['missing']}).status_code==409


@pytest.mark.parametrize('mode,expected_limit', [('demo', 1000), ('classifier', 10000)])
def test_collection_limit_does_not_change_training_config(tmp_path, monkeypatch, mode, expected_limit):
    monkeypatch.setenv('CYCLO_HILSERL_DATA_DIR',str(tmp_path))
    monkeypatch.setattr(hil,'preflight',lambda config:{'can_' + mode:True})
    app=FastAPI();app.include_router(hil.router)
    config=hil.ExperimentConfig(max_episode_steps=10000,collection_episode_steps=1000).model_dump()
    with TestClient(app) as client:
        client.put('/hilserl/config',json={'expected_revision':0,'config':config}).raise_for_status()
        response=client.post('/hilserl/sessions',json={'expected_config_revision':1,'mode':mode})
        response.raise_for_status()
        assert response.json()['config']['max_episode_steps']==expected_limit
    with hil.database() as db:
        assert hil.settings(db)['config']['max_episode_steps']==10000


@pytest.mark.skipif(os.environ.get('CYCLO_TEST_PIXELS') != '1', reason='Requires idle Isaac pixel bridge')
def test_real_step_labels_follow_success_lease_and_pause(tmp_path, monkeypatch):
    monkeypatch.setenv('CYCLO_HILSERL_DATA_DIR', str(tmp_path))
    app = FastAPI(); app.include_router(hil.router)
    with TestClient(app) as client:
        config = hil.ExperimentConfig(learning_backend='legacy_torch', environment='isaac_sim', robot_type='ffw_sg2_rev1', control_hz=20,
            max_episode_steps=100, classifier_cameras=['right_wrist']).model_dump()
        client.put('/hilserl/config', json={'expected_revision':0,'config':config}).raise_for_status()
        item = client.post('/hilserl/sessions', json={'expected_config_revision':1,'mode':'classifier'}).json()
        url = '/hilserl/sessions/' + item['id']
        def wait(predicate):
            deadline = time.monotonic()+75
            while time.monotonic()<deadline:
                value=client.get(url).json()
                assert not value.get('runtime_error'), value.get('runtime_error')
                if not value.get('runtime_pending') and predicate(value): return value
                time.sleep(.05)
            pytest.fail('Collection timeout')
        def command(action):
            value = client.get(url).json()
            client.post(url+'/commands',json={'expected_revision':value['revision'],'action':action}).raise_for_status()
            return wait(lambda s:True)
        try:
            command('start')
            value=wait(lambda s:s['runtime']['collection']['negative_samples']>=2)
            revision=value['revision']
            assert client.post(url+'/classifier-label',json={'expected_revision':revision+1,'success':True}).status_code==409
            client.post(url+'/classifier-label',json={'expected_revision':revision,'success':True}).raise_for_status()
            value=wait(lambda s:s['runtime']['collection']['positive_samples']>=1)
            assert value['revision']==revision  # Label pulses must not interrupt manual control.
            # No further pulses: the lease expires and subsequent steps revert to failure.
            time.sleep(.6)
            value=wait(lambda s:s['runtime']['collection']['last_capture']['label']==0)
            value=command('pause')
            collection=value['runtime']['collection']; count=collection['positive_samples']+collection['negative_samples']
            assert count==value['runtime']['episode_step']
            assert collection['positive_samples']>0 and collection['negative_samples']>0
            time.sleep(.3)
            paused=client.get(url).json()['runtime']['collection']
            assert paused['positive_samples']+paused['negative_samples']==count
            assert client.post(url+'/classifier-label',json={'expected_revision':value['revision'],'success':True}).status_code==409
            command('stop'); hil._workers[item['id']].wait(timeout=20)
            manifest=json.loads((tmp_path/item['id']/'classifier_samples/manifest.json').read_text())
            assert len(manifest['samples'])==count
            assert all(s['source']=='step' for s in manifest['samples'])
            assert len(set(s['state_key'] for s in manifest['samples']))==count
        finally:
            if item['id'] in hil._workers and hil._workers[item['id']].poll() is None:
                command('stop'); hil._workers[item['id']].wait(timeout=20)


@pytest.mark.skipif(os.environ.get('CYCLO_TEST_PIXELS') != '1', reason='Requires idle Isaac pixel bridge')
def test_real_success_target_stops_and_saves_exact_count(tmp_path, monkeypatch):
    monkeypatch.setenv('CYCLO_HILSERL_DATA_DIR',str(tmp_path))
    app=FastAPI();app.include_router(hil.router)
    with TestClient(app) as client:
        config=hil.ExperimentConfig(environment='isaac_sim',robot_type='ffw_sg2_rev1',control_hz=20,classifier_success_target=2).model_dump()
        client.put('/hilserl/config',json={'expected_revision':0,'config':config}).raise_for_status()
        item=client.post('/hilserl/sessions',json={'expected_config_revision':1,'mode':'classifier'}).json()
        url='/hilserl/sessions/'+item['id']
        client.post(url+'/commands',json={'expected_revision':item['revision'],'action':'start'}).raise_for_status()
        try:
            deadline=time.monotonic()+75
            while time.monotonic()<deadline:
                value=client.get(url).json()
                assert not value.get('runtime_error'),value.get('runtime_error')
                if value['state']=='stopped':break
                if not value.get('runtime_pending') and value.get('runtime',{}).get('episode_step',0)>0:
                    response=client.post(url+'/classifier-label',json={'expected_revision':value['revision'],'success':True})
                    assert response.status_code in (200,409)
                time.sleep(.05)
            else:pytest.fail('Target completion timed out')
            hil._workers[item['id']].wait(timeout=20)
            value=client.get(url).json()
            assert value['runtime']['collection']['positive_samples']==2
            assert value['runtime']['collection']['target_reached']
            assert value['runtime']['artifacts_saved']
            assert any(e['action']=='success_target_reached' for e in value['events'])
            manifest=json.loads((tmp_path/item['id']/'classifier_samples/manifest.json').read_text())
            assert sum(s['label'] for s in manifest['samples'])==2
            assert client.post(url+'/commands',json={'expected_revision':value['revision'],'action':'resume'}).status_code==409
        finally:
            if hil._workers[item['id']].poll() is None:
                value=client.get(url).json()
                client.post(url+'/commands',json={'expected_revision':value['revision'],'action':'stop'})
                hil._workers[item['id']].wait(timeout=20)
