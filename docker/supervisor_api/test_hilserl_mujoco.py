import json
from types import SimpleNamespace
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import hilserl as hil
import hilserl_mujoco as bridge
from hilserl_learner import make_spec
from hilserl_learner_server import validate_spec
from hilserl_profiles import pixel_schema, runtime_python


def test_mujoco_readiness_and_runtime_contract(monkeypatch):
    c = hil.ExperimentConfig(environment="mujoco", actor_device="cpu", learner_device="cpu").model_dump()
    monkeypatch.setattr(hil._mujoco, "health", lambda: dict(ready=True, busy=False,
        robot_type="ffw_sg2_rev1", control_hz=20, target_parts=["peg_round_8"],
        initializations=["near_hole"], pixel_observations=True, pixel_sizes=[64, 128]))
    result = hil.preflight(c)
    assert result["can_preview"] and result["can_classifier"] and result["can_train"]
    assert result["checks"][0]["label"] == "MuJoCo NIST bridge"
    assert hil.uses_runtime(dict(mode="preview", config=c))
    assert "hil-serl-venv" in runtime_python(c)
    assert not hil.preflight({**c, "initialization": "pick_place"})["can_preview"]
    monkeypatch.setattr(hil._mujoco, "health", lambda: dict(ready=False, error="offline"))
    assert not hil.preflight(c)["can_preview"]


def test_mujoco_remote_learner_accepts_six_axis_pixel_spec():
    c = hil.ExperimentConfig(environment="mujoco").model_dump()
    env = SimpleNamespace(pixel_mode=True, schema=pixel_schema(c), observation_dim=22, action_dim=6)
    spec = make_spec(c, env)
    assert validate_spec(spec)["config"]["environment"] == "mujoco"
    with pytest.raises(ValueError):
        validate_spec({**spec, "action_dim": 3})


def test_local_batch256_uses_measured_combined_memory_budget(monkeypatch):
    monkeypatch.delenv("XLA_PYTHON_CLIENT_MEM_FRACTION", raising=False)
    c = hil.ExperimentConfig(environment="mujoco", learner_placement="local",
                             actor_device="cuda:0", learner_device="cuda:0").model_dump()
    monkeypatch.setattr(hil._mujoco, "health", lambda: dict(ready=True, busy=False,
        robot_type="ffw_sg2_rev1", control_hz=20, target_parts=["peg_round_8"],
        initializations=["near_hole"], pixel_observations=True, pixel_sizes=[128]))
    gpu = {"device": "cuda:0", "total_mib": 8151, "free_mib": 7200, "used_mib": 507}
    monkeypatch.setattr(hil, "hardware_status", lambda: {"gpus": [gpu], "error": None})
    assert hil.preflight(c)["can_train"]  # Four-view rendering no longer blocks launch.
    gpu["free_mib"] = 6400
    assert not hil.preflight(c)["can_train"]
    gpu["free_mib"] = 7200
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", ".90")
    assert not hil.preflight(c)["can_train"]


def test_mujoco_camera_route_and_local_endpoint(monkeypatch):
    app = FastAPI()
    app.include_router(hil.router)
    payload = b"\xff\xd8mujoco-test\xff\xd9"
    monkeypatch.setattr(hil._mujoco, "frame", lambda camera, quality: (payload, {"X-Frame-Id": "1"}))
    with TestClient(app) as client:
        assert client.get("/hilserl/mujoco/frame?camera=left_wrist").content == payload
        assert client.get("/hilserl/mujoco/frame?camera=unknown").status_code == 422
    monkeypatch.setenv("CYCLO_HILSERL_MUJOCO_URL", "http://example.com:7126")
    with pytest.raises(ValueError, match="loopback"):
        bridge.bridge_url()


def test_wrong_simulator_artifacts_fail_readiness(tmp_path, monkeypatch):
    from hilserl_profiles import UPSTREAM_CLASSIFIER
    monkeypatch.setenv("CYCLO_HILSERL_DATA_DIR", str(tmp_path))
    c = hil.ExperimentConfig(environment="mujoco", actor_device="cpu", learner_device="cpu",
                             reward_source="classifier", classifier_path=str(tmp_path/"classifier.npz")).model_dump()
    wrong_task = {k: c[k] for k in ("environment", "robot_type", "arm", "target_part")}
    wrong_task["environment"] = "isaac_sim"
    np.savez(c["classifier_path"], __schema__=np.array(UPSTREAM_CLASSIFIER),
             __task__=np.array(json.dumps(wrong_task)))
    monkeypatch.setattr(hil._mujoco, "health", lambda: dict(ready=True, busy=False,
        robot_type="ffw_sg2_rev1", control_hz=20, target_parts=["peg_round_8"],
        initializations=["near_hole"], pixel_observations=True, pixel_sizes=[128]))
    readiness = hil.preflight(c)
    assert readiness["can_classifier"]
    assert not readiness["can_demo"] and not readiness["can_train"]
