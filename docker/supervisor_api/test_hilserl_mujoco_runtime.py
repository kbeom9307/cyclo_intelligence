"""Opt-in Lab API -> worker -> real MuJoCo bridge tests; all records are temporary."""
import json
import os
import time
from pathlib import Path
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import hilserl as hil

pytestmark = pytest.mark.skipif(os.environ.get("CYCLO_TEST_MUJOCO_RUNTIME") != "1",
                               reason="requires an idle running MuJoCo bridge")


def test_preview_teleop_stop_and_automatic_classifier_collection(tmp_path, monkeypatch):
    monkeypatch.setenv("CYCLO_HILSERL_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(hil, "_workers", {})
    assert hil._mujoco.health()["ready"] and not hil._mujoco.health()["busy"]
    config = hil.ExperimentConfig(environment="mujoco", actor_device="cuda:0", learner_placement="remote",
        classifier_success_target=2, collection_episode_steps=1000, max_episode_steps=1000,
        translation_limit_m=.001, rotation_limit_rad=.01).model_dump()
    app = FastAPI()
    app.include_router(hil.router)
    with TestClient(app) as client:
        client.put("/hilserl/config", json={"expected_revision": 0, "config": config}).raise_for_status()

        def create(mode):
            response = client.post("/hilserl/sessions", json={"expected_config_revision": 1, "mode": mode})
            response.raise_for_status()
            return response.json()

        def wait(item, predicate, timeout=40):
            deadline = time.monotonic()+timeout
            while time.monotonic() < deadline:
                current = client.get("/hilserl/sessions/"+item["id"]).json()
                assert not current.get("runtime_error"), current.get("runtime_error")
                if predicate(current):
                    return current
                time.sleep(.05)
            pytest.fail(f"MuJoCo runtime did not reach expected state: {current}")

        def command(item, action):
            current = client.get("/hilserl/sessions/"+item["id"]).json()
            client.post("/hilserl/sessions/"+item["id"]+"/commands",
                        json={"expected_revision": current["revision"], "action": action}).raise_for_status()
            return wait(item, lambda v: not v.get("runtime_pending"))

        preview = create("preview")
        try:
            preview = command(preview, "start")
            preview = wait(preview, lambda v: v.get("runtime", {}).get("episode_step", 0) >= 2)
            assert preview["runtime"]["image_shape"] == [9, 128, 128]
            assert not preview["runtime"]["policy_enabled"]
            client.post("/hilserl/sessions/"+preview["id"]+"/action", json={
                "expected_revision": preview["revision"], "value": [0, .5, 0, 0, 0, 0]}).raise_for_status()
            preview = wait(preview, lambda v: v["runtime"]["last_action_source"] == "human")
            assert len(preview["runtime"]["last_action"]) == 6
        finally:
            command(preview, "stop")
            hil._workers[preview["id"]].wait(timeout=30)
        assert not hil._mujoco.health()["busy"]

        collection = create("classifier")
        try:
            collection = command(collection, "start")
            assert collection['runtime']['collection']['waiting_for_intervention']
            client.post('/hilserl/sessions/'+collection['id']+'/classifier-label', json={
                'expected_revision': collection['revision'], 'success': True}).raise_for_status()
            time.sleep(.6)
            collection = client.get('/hilserl/sessions/'+collection['id']).json()
            assert collection['runtime']['episode_step'] == 0
            assert collection['runtime']['collection']['positive_samples'] == 0
            assert collection['runtime']['collection']['negative_samples'] == 0
            assert not (tmp_path/collection['id']/'classifier_samples').exists()
            frame = client.get('/hilserl/mujoco/frame?camera=head')
            deadline = time.monotonic() + 5
            while frame.status_code != 200 and time.monotonic() < deadline:
                time.sleep(.1)
                frame = client.get('/hilserl/mujoco/frame?camera=head')
            assert frame.status_code == 200 and frame.content[:2] == b'\xff\xd8'
            collection = command(collection, 'pause')
            for action in ('capture_success', 'capture_failure'):
                response = client.post('/hilserl/sessions/'+collection['id']+'/commands', json={
                    'expected_revision': collection['revision'], 'action': action})
                assert response.status_code == 409
                assert 'intervention input' in response.json()['detail']
            collection = command(collection, 'resume')
            client.post('/hilserl/sessions/'+collection['id']+'/action', json={
                'expected_revision': collection['revision'], 'value': [0, .5, 0, 0, 0, 0]}).raise_for_status()
            collection = wait(collection, lambda v: v.get("runtime", {}).get("collection", {}).get("negative_samples", 0) >= 2)
            assert not collection['runtime']['collection']['waiting_for_intervention']
            client.post("/hilserl/sessions/"+collection["id"]+"/classifier-label", json={
                "expected_revision": collection["revision"], "success": True}).raise_for_status()
            collection = wait(collection, lambda v: v["state"] == "stopped")
            hil._workers[collection["id"]].wait(timeout=30)
            manifest = json.loads((tmp_path/collection["id"]/"classifier_samples/manifest.json").read_text())
            assert manifest["task"]["environment"] == "mujoco"
            assert sum(sample["label"] == 1 for sample in manifest["samples"]) == 2
            assert sum(sample["label"] == 0 for sample in manifest["samples"]) >= 2
            result = dict(preview_manual_control=True, classifier_auto_stop=True, initial_intervention_gate=True,
                          positive_samples=2, total_samples=len(manifest["samples"]))
            Path("/tmp/hil-mujoco-lab-runtime.json").write_text(json.dumps(result, indent=2))
        finally:
            current = client.get("/hilserl/sessions/"+collection["id"]).json()
            if current["state"] != "stopped":
                command(current, "stop")
            hil._workers[collection["id"]].wait(timeout=30)
        assert not hil._mujoco.health()["busy"]
