import json
from uuid import uuid4
from fastapi import FastAPI
from fastapi.testclient import TestClient
import hilserl as hil


def test_delete_dataset_and_checkpoint_guards(tmp_path, monkeypatch):
    monkeypatch.setenv('CYCLO_HILSERL_DATA_DIR',str(tmp_path))
    monkeypatch.setattr(hil,'_classifier_workers',{})
    monkeypatch.setattr(hil,'_workers',{})
    app=FastAPI();app.include_router(hil.router)
    identifier=str(uuid4());job=str(uuid4())
    config=hil.ExperimentConfig().model_dump()
    item={'id':identifier,'mode':'classifier','state':'paused','config':config,'revision':0,'metrics':{},'episode':1,'episode_outcome':None}
    samples=tmp_path/identifier/'classifier_samples';samples.mkdir(parents=True)
    (samples/'manifest.json').write_text('{}')
    (tmp_path/identifier/'transitions.jsonl').write_text('keep')
    folder=tmp_path/'classifiers'/job;folder.mkdir(parents=True)
    checkpoint=folder/'classifier.npz';checkpoint.write_bytes(b'checkpoint')
    with hil.database() as db:hil.write(db,'session:'+identifier,item)
    class Live:
        def poll(self): return None
    hil._workers[identifier]=Live()
    with TestClient(app) as client:
        assert client.delete('/hilserl/datasets/'+identifier+'/classifier').status_code==409
        item['state']='stopped'
        hil._workers.clear()
        with hil.database() as db:hil.write(db,'session:'+identifier,item)
        class Running:
            def poll(self):return None
        hil._classifier_workers[job]=Running()
        (folder/'request.json').write_text(json.dumps({'collections':[str(tmp_path/identifier)]}))
        assert client.delete('/hilserl/datasets/'+identifier+'/classifier').status_code==409
        assert client.delete('/hilserl/classifiers/'+job).status_code==409
        hil._classifier_workers.clear()
        assert client.delete('/hilserl/datasets/'+identifier+'/classifier').status_code==200
        assert not samples.exists() and (tmp_path/identifier/'transitions.jsonl').exists()
        assert checkpoint.exists()  # Dataset deletion must not delete trained models.
        with hil.database() as db:hil.write(db,'config',{'revision':1,'config':dict(config,classifier_path=str(checkpoint))})
        assert client.delete('/hilserl/classifiers/'+job).status_code==409
        assert checkpoint.exists()
        with hil.database() as db:hil.write(db,'config',{'revision':2,'config':config})
        assert client.delete('/hilserl/classifiers/'+job).status_code==200
        assert not folder.exists()
        assert client.delete('/hilserl/classifiers/'+job).status_code==404
        assert client.delete('/hilserl/classifiers/not-a-uuid').status_code==422


def test_demo_file_listing_download_and_path_guards(tmp_path, monkeypatch):
    monkeypatch.setenv('CYCLO_HILSERL_DATA_DIR', str(tmp_path))
    identifier=str(uuid4())
    folder=tmp_path/identifier; demos=folder/'demos'; demos.mkdir(parents=True)
    task=dict(environment='isaac_sim', robot_type='ffw_sg2_rev1', arm='right', target_part='peg_round_8')
    (demos/'manifest.json').write_text(json.dumps({'task':task,'episodes':[{'episode':1,'steps':12}]}))
    (demos/'episode-00001.npz').write_bytes(b'episode')
    (folder/'demonstrations.npz').write_bytes(b'combined')
    item=dict(id=identifier, mode='demo', state='stopped', config=hil.ExperimentConfig().model_dump(), revision=0)
    with hil.database() as db: hil.write(db, 'session:'+identifier, item)
    app=FastAPI(); app.include_router(hil.router)
    with TestClient(app) as client:
        files=hil.data_library()[0]['files']
        assert files==[{'name':'episode-00001.npz','episode':1,'steps':12}]
        base=f'/hilserl/datasets/{identifier}/demo/files/'
        assert client.get(base+'episode-00001.npz').content==b'episode'
        assert client.get(base+'demonstrations.npz').content==b'combined'
        assert client.get(base+'control.sqlite3').status_code==422
        assert client.get(base+'episode-00002.npz').status_code==404
        (demos/'episode-00003.npz').symlink_to(folder/'demonstrations.npz')
        assert client.get(base+'episode-00003.npz').status_code==422


def test_delete_selected_demo_episodes_rebuilds_export_and_preserves_other_data(tmp_path, monkeypatch):
    import numpy as np
    from hilserl_data import FIELDS
    monkeypatch.setenv('CYCLO_HILSERL_DATA_DIR', str(tmp_path))
    identifier=str(uuid4()); folder=tmp_path/identifier; demos=folder/'demos'; demos.mkdir(parents=True)
    task=dict(environment='isaac_sim',robot_type='ffw_sg2_rev1',arm='right',target_part='peg_round_8')
    entries=[]
    for episode in (1,2,3):
        path=demos/f'episode-{episode:05d}.npz'
        np.savez_compressed(path, __schema__='test', __task__=json.dumps(task),
            **{key:np.full((2,),episode) for key in FIELDS+('images','next_images')})
        entries.append(dict(episode=episode,steps=2,path=str(path)))
    manifest=demos/'manifest.json'; manifest.write_text(json.dumps(dict(task=task,episodes=entries)))
    (folder/'demonstrations.npz').write_bytes(b'old export')
    (folder/'pending-demo.npz').write_bytes(b'pending')
    config=hil.ExperimentConfig().model_dump()
    item=dict(id=identifier,mode='demo',state='stopped',config=config,revision=0)
    with hil.database() as db: hil.write(db,'session:'+identifier,item)
    app=FastAPI(); app.include_router(hil.router)
    base=f'/hilserl/datasets/{identifier}/demo/delete-episodes'
    with TestClient(app) as client:
        assert client.post(base,json={'episodes':[]}).status_code==422
        assert client.post(base,json={'episodes':[1,1]}).status_code==422
        assert client.post(base,json={'episodes':[9]}).status_code==404
        with hil.database() as db: hil.write(db,'config',dict(revision=1,config={**config,'demonstration_path':str(folder/'demonstrations.npz')}))
        assert client.post(base,json={'episodes':[2]}).status_code==409
        with hil.database() as db: hil.write(db,'config',dict(revision=2,config=config))
        class Live:
            def poll(self): return None
        monkeypatch.setitem(hil._workers, identifier, Live())
        assert client.post(base,json={'episodes':[2]}).status_code==409
        hil._workers.pop(identifier)
        assert client.post(base,json={'episodes':[2]}).status_code==200
        assert not (demos/'episode-00002.npz').exists()
        assert [e['episode'] for e in json.loads(manifest.read_text())['episodes']]==[1,3]
        with np.load(folder/'demonstrations.npz') as data: assert data['reward'].tolist()==[1,1,3,3]
        assert (folder/'pending-demo.npz').read_bytes()==b'pending'
        assert client.post(base,json={'episodes':[1,3]}).status_code==200
        assert not (folder/'demonstrations.npz').exists()
        assert json.loads(manifest.read_text())['episodes']==[]


def test_combine_demos_across_sessions_and_reject_mixed_tasks(tmp_path, monkeypatch):
    import numpy as np
    from hilserl_data import FIELDS
    monkeypatch.setenv('CYCLO_HILSERL_DATA_DIR',str(tmp_path))
    task=dict(environment='isaac_sim',robot_type='ffw_sg2_rev1',arm='right',target_part='peg_round_8')
    ids=[str(uuid4()),str(uuid4())]
    for index, identifier in enumerate(ids):
        folder=tmp_path/identifier/'demos';folder.mkdir(parents=True)
        (folder/'manifest.json').write_text(json.dumps(dict(task=task,cameras=['head'],episodes=[dict(episode=1,steps=2)])))
        np.savez_compressed(folder/'episode-00001.npz',__schema__='test',__task__=json.dumps(task),__cameras__=json.dumps(['head']),
            **{key:np.full((2,),index+1) for key in FIELDS+('images','next_images')})
        with hil.database() as db: hil.write(db,'session:'+identifier,dict(id=identifier,mode='demo',state='stopped',config=hil.ExperimentConfig().model_dump(),revision=0))
    app=FastAPI();app.include_router(hil.router)
    selections=[dict(session_id=identifier,episodes=[1]) for identifier in ids]
    with TestClient(app) as client:
        result=client.post('/hilserl/datasets/demo/combine',json=dict(selections=selections))
        assert result.status_code==200,result.text
        output=result.json();assert output['steps']==4 and output['episodes']==2
        with np.load(output['path']) as data:
            assert data['reward'].tolist()==[1,1,2,2]
            assert json.loads(str(data['__sources__']))==selections
        assert all((tmp_path/identifier/'demos/episode-00001.npz').exists() for identifier in ids)
        path=tmp_path/ids[1]/'demos/manifest.json';manifest=json.loads(path.read_text());manifest['task']['arm']='left';path.write_text(json.dumps(manifest))
        assert client.post('/hilserl/datasets/demo/combine',json=dict(selections=selections)).status_code==409
        manifest['task']['arm']='right';manifest['cameras']=['right_wrist'];path.write_text(json.dumps(manifest))
        assert client.post('/hilserl/datasets/demo/combine',json=dict(selections=selections)).status_code==409
