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
