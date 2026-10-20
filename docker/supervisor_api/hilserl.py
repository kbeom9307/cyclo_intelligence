"""HIL-SERL control plane: preview and a local Agentlace/SAC mock worker.

Run with one supervisor worker. Isaac Sim uses a leased local bridge; hardware remains blocked.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import importlib.util
import subprocess
import sys
import time
from typing import Literal
from uuid import uuid4
from hilserl_devices import hardware_status, device_ready
from hilserl_remote import validate_url, RemoteEndpoint, PROTOCOL
from threading import Lock

# This module is also dynamically loaded by app.py and the contract tests.
_isaac_spec = importlib.util.spec_from_file_location("hilserl_isaac", Path(__file__).with_name("hilserl_isaac.py"))
_isaac = importlib.util.module_from_spec(_isaac_spec)
_isaac_spec.loader.exec_module(_isaac)

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

_workers = {}
_classifier_workers = {}
_classifier_lock = Lock()

router = APIRouter(prefix="/hilserl", tags=["hilserl"])


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)


class ExperimentConfig(StrictModel):
    name: str = Field(default="AI Worker · NIST peg insertion", min_length=1, max_length=120)
    robot_type: Literal["ffw_bg2_rev4", "ffw_sg2_rev1", "ffw_sh5_rev1"] = "ffw_bg2_rev4"
    environment: Literal["mock", "isaac_sim", "real_robot"] = "isaac_sim"
    board: Literal["nist_atb1"] = "nist_atb1"
    task: Literal["peg_insertion"] = "peg_insertion"
    arm: Literal["left", "right"] = "right"
    initialization: Literal["near_hole", "pick_place", "grasped", "kit"] = "near_hole"
    target_part: Literal["peg_round_4", "peg_round_8", "peg_round_12", "peg_round_16", "peg_square_4", "peg_square_8", "peg_square_12", "peg_square_16"] = "peg_round_8"
    control_hz: int = Field(default=10, ge=1, le=50)
    seed: int = Field(default=0, ge=0, le=2147483647)
    learning_backend: Literal['hil_serl', 'legacy_torch'] = 'hil_serl'
    cta_ratio: int = Field(default=2, ge=1, le=16)
    training_starts: int = Field(default=100, ge=1, le=100000)
    policy_publish_interval: int = Field(default=50, ge=1, le=1000)
    camera_crops: dict[str, list[float]] = Field(default_factory=dict)

    @field_validator('camera_crops')
    @classmethod
    def valid_crops(cls, value):
        if set(value) - {'head', 'left_wrist', 'right_wrist'}:
            raise ValueError('Unknown crop camera')
        for roi in value.values():
            if len(roi) != 4 or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
                raise ValueError('Crop must be normalized [left, top, right, bottom] within [0, 1]')
        return {camera: roi for camera, roi in value.items() if roi != [0, 0, 1, 1]}
    batch_size: int = Field(default=256, ge=1, le=4096)
    replay_capacity: int = Field(default=100000, ge=4096, le=10000000)
    actor_device: str = Field(default="cuda:0", pattern=r"^(cpu|cuda:[0-9]+)$")
    learner_device: str = Field(default="cuda:0", pattern=r"^(cpu|cuda:[0-9]+)$")
    learner_placement: Literal["local", "remote"] = "local"
    learner_url: str = Field(default="http://127.0.0.1:7130", max_length=512)
    learner_token_file: str = Field(default="/workspace/hilserl/learner-token", max_length=512)

    @field_validator('learner_url')
    @classmethod
    def learner_endpoint(cls, value):
        return validate_url(value)

    learning_rate: float = Field(default=0.0003, gt=0, le=0.1)
    discount: float = Field(default=0.97, gt=0, le=1)
    max_episode_steps: int = Field(default=200, ge=1, le=10000)
    translation_limit_m: float = Field(default=0.005, gt=0, le=0.05)
    rotation_limit_rad: float = Field(default=0.05, gt=0, le=0.3)
    success_depth_m: float = Field(default=0.02, gt=0, le=0.2)
    success_tolerance_m: float = Field(default=0.002, gt=0, le=0.02)
    success_hold_s: float = Field(default=0.5, gt=0, le=10)
    reward_source: Literal["environment", "human", "classifier"] = "environment"
    camera_topic: str = Field(default="/camera_right/camera_right/color/image_rect_raw/compressed", pattern=r"^/[A-Za-z0-9_/]+$", max_length=256)
    joint_state_topic: str = Field(default="/joint_states", pattern=r"^/[A-Za-z0-9_/]+$", max_length=256)
    observation_mode: Literal["pixels", "state"] = "pixels"
    observation_cameras: list[Literal["head", "left_wrist", "right_wrist"]] = Field(default_factory=lambda: ["head", "left_wrist", "right_wrist"], min_length=1, max_length=3)
    classifier_cameras: list[Literal["head", "left_wrist", "right_wrist"]] = Field(default_factory=lambda: ["head", "left_wrist", "right_wrist"], min_length=1, max_length=3)

    @field_validator('observation_cameras', 'classifier_cameras')
    @classmethod
    def camera_selection(cls, values):
        if len(set(values)) != len(values):
            raise ValueError('Camera selection must not contain duplicates')
        return [camera for camera in ['head', 'left_wrist', 'right_wrist'] if camera in values]

    classifier_success_target: int = Field(default=200, ge=1, le=10000)
    collection_episode_steps: int = Field(default=1000, ge=1, le=1000)
    pixel_batch_size: int = Field(default=256, ge=2, le=1024)
    pixel_replay_capacity: int = Field(default=200000, ge=64, le=1000000)

    @field_validator('pixel_batch_size')
    @classmethod
    def even_pixel_batch(cls, value):
        if value % 2:
            raise ValueError('Pixel batch must be even for the 50% online / 50% demo mixture')
        return value
    classifier_path: str = Field(default="", max_length=512)
    classifier_threshold: float = Field(default=.9, gt=0, lt=1)
    classifier_hold_steps: int = Field(default=3, ge=1, le=100)
    demonstration_path: str = Field(default="", max_length=512)
    checkpoint_path: str = Field(default="", max_length=512)


class SaveConfig(StrictModel):
    expected_revision: int = Field(ge=0)
    config: ExperimentConfig


class CreateSession(StrictModel):
    expected_config_revision: int = Field(ge=0)
    mode: Literal["preview", "train", "evaluate", "classifier", "demo"]


class SessionCommand(StrictModel):
    expected_revision: int = Field(ge=0)
    action: Literal["start", "pause", "resume", "stop", "take_over", "release", "success", "failure", "truncate", "reset", "capture_success", "capture_failure", "save_demo", "discard_demo"]


class ManualAction(StrictModel):
    expected_revision: int = Field(ge=0)
    value: list[float] = Field(min_length=3, max_length=7)


class ClassifierLabel(StrictModel):
    expected_revision: int = Field(ge=0)
    success: bool


def data_root():
    return Path(os.environ.get("CYCLO_HILSERL_DATA_DIR", "/workspace/hilserl"))


def uses_runtime(session):
    return session['mode'] != 'preview' or session['config']['environment'] == 'isaac_sim'


def runtime_status(session):
    if not uses_runtime(session):
        return None
    try:
        return json.loads((data_root() / session["id"] / "runtime.json").read_text())
    except (OSError, ValueError):
        return None


def decorate(session):
    if not uses_runtime(session):
        return session
    result = dict(session)
    status = runtime_status(session)
    process = _workers.get(session["id"])
    if status:
        result["runtime"] = status
        result["metrics"] = status.get("metrics", session["metrics"])
        result["episode"] = status.get("episode", session["episode"])
    result["runtime_pending"] = session["state"] != "created" and (not status or status.get("ack_revision", -1) < session["revision"] or status.get("initializing", False))
    if session["state"] != "created" and (process is None or process.poll() is not None):
        result["state"] = "stopped"
        result["runtime_pending"] = False
        if session["state"] != "stopped":
            result["runtime_error"] = (status or {}).get("error") or "Runtime exited or supervisor restarted. Create a new session."
    if process is not None and process.poll() not in (None, 0):
        result["runtime_error"] = (status or {}).get("error") or f"Runtime exited with code {process.returncode}; artifacts may be incomplete."
    if status and status.get("error"):
        result["runtime_error"] = status["error"]
    manual = session.get('manual_action', {})
    active = result['state'] == 'running' and time.time() < manual.get('expires_at', 0) and any(manual.get('value', []))
    result['control_source'] = 'human' if active else 'policy'
    return result


def launch(session):
    from hilserl_profiles import runtime_python
    folder = data_root() / session["id"]
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "worker.log").open("a") as log:
        _workers[session["id"]] = subprocess.Popen(
            [runtime_python(session['config']), str(Path(__file__).with_name("hilserl_runtime.py")), str(data_root()), session["id"], str(os.getpid())],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def shutdown():
    # Request a graceful flush; wait outside the SQLite transaction.
    with database() as db:
        for item in sessions(db):
            if item["id"] in _workers and item["state"] != "stopped":
                item["state"] = "stopped"
                item["revision"] += 1
                write(db, "session:" + item["id"], item)
    for process in _workers.values():
        try:
            process.wait(timeout=310)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=3)
    _workers.clear()
    for process in _classifier_workers.values():
        if process.poll() is None:
            process.terminate()
            try: process.wait(timeout=3)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=3)
    _classifier_workers.clear()


def artifact_ready(path):
    if not path:
        return True
    candidate = Path(path).resolve()
    return candidate.is_relative_to(data_root().resolve()) and candidate.suffix == ".npz" and candidate.is_file() and candidate.stat().st_size < 512 * 1024 * 1024


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def database():
    root = Path(os.environ.get("CYCLO_HILSERL_DATA_DIR", "/workspace/hilserl"))
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / "control.sqlite3", timeout=10)
    try:
        db.execute("CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, body TEXT NOT NULL)")
        db.execute("BEGIN IMMEDIATE")
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def read(db, key):
    row = db.execute("SELECT body FROM documents WHERE id=?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def write(db, key, value):
    db.execute("INSERT INTO documents VALUES (?,?) ON CONFLICT(id) DO UPDATE SET body=excluded.body", (key, json.dumps(value)))


def settings(db):
    stored = read(db, "config") or {"revision": 0, "config": {}}
    if stored['config'] and 'learning_backend' not in stored['config']:
        stored['config'] = {'learning_backend': 'legacy_torch', **stored['config']}
    return {**stored, "config": {**ExperimentConfig().model_dump(), **stored["config"]}}


def sessions(db):
    return [json.loads(row[0]) for row in db.execute("SELECT body FROM documents WHERE id LIKE 'session:%' ORDER BY rowid DESC")]


_learner_probe = {}


def learner_probe(config, refresh=False):
    key = (config.get('learner_url'), config.get('learner_token_file'))
    cached = _learner_probe.get(key)
    if not refresh and cached and time.monotonic() - cached[0] < 5:
        return cached[1]
    try:
        result = json.loads(RemoteEndpoint(config).request('/health', timeout=2))
        if result.get('protocol') != PROTOCOL:
            raise ValueError('Incompatible learner protocol')
        result['connected'] = True
    except Exception as exc:
        result = {'connected': False, 'gpus': [], 'error': str(exc)}
    _learner_probe.clear()
    _learner_probe[key] = (time.monotonic(), result)
    return result


@router.post('/learner/check')
def check_learner(request: ExperimentConfig):
    return learner_probe(request.model_dump(), refresh=True)


def preflight(config):
    from hilserl_profiles import upstream, pixel_schema, UPSTREAM_CLASSIFIER, runtime_python
    mock = config["environment"] == "mock"
    deps = all(importlib.util.find_spec(name) is not None for name in ("agentlace", "torch", "numpy"))
    artifacts = all(artifact_ready(config[key]) for key in ("checkpoint_path", "demonstration_path"))
    classifier_ok = bool(config.get('classifier_path')) and artifact_ready(config.get('classifier_path', '')) and config.get('observation_mode') == 'pixels'
    reward = config["reward_source"] != "classifier" or classifier_ok
    isaac = config['environment'] == 'isaac_sim'
    bridge = _isaac.health() if isaac else {}
    compatible = (isaac and bridge.get('ready', False) and config['robot_type'] == bridge.get('robot_type')
                  and config['control_hz'] == bridge.get('control_hz')
                  and config.get('target_part', 'peg_round_8') in bridge.get('target_parts', [])
                  and config['translation_limit_m'] <= .02 and config['rotation_limit_rad'] <= .15)
    pixels = config.get('observation_mode', 'state') == 'pixels'
    if isaac and pixels and upstream(config):
        try:
            runtime_python(config)
        except ValueError:
            deps = False
        import numpy as np
        for key in ('checkpoint_path', 'demonstration_path', 'classifier_path'):
            if not config.get(key) or not artifact_ready(config[key]):
                continue
            try:
                with np.load(config[key], allow_pickle=False) as data:
                    match = str(data['__schema__']) == (UPSTREAM_CLASSIFIER if key == 'classifier_path' else pixel_schema(config))
                if key == 'classifier_path':
                    classifier_ok = classifier_ok and match
                else:
                    artifacts = artifacts and match
            except (OSError, ValueError, KeyError):
                if key == 'classifier_path':
                    classifier_ok = False
                else:
                    artifacts = False
        reward = config["reward_source"] != "classifier" or classifier_ok
    compatible = compatible and (not pixels or bridge.get('pixel_observations', False))
    if isaac and pixels and upstream(config):
        compatible = compatible and 128 in bridge.get('pixel_sizes', [])
    available = mock or compatible
    hardware = hardware_status()
    actor_device = config.get('actor_device', 'cuda:0')
    learner_device = config.get('learner_device', 'cuda:0')
    actor_ok = device_ready(actor_device, hardware)
    actor_memory_detail = ''
    if isaac and pixels and upstream(config) and actor_device.startswith('cuda:'):
        actor_gpu = next((g for g in hardware['gpus'] if g['device'] == actor_device), {})
        if actor_ok and actor_gpu.get('free_mib', 0) < 512:
            actor_ok = False
            actor_memory_detail = ' · Restored actor/classifier needs at least 512 MiB free. Close Overview or free GPU memory before starting.'
    remote = config.get('learner_placement') == 'remote'
    learner_hardware = learner_probe(config) if remote else hardware
    learner_ok = device_ready(learner_device, learner_hardware) and (not remote or (learner_hardware.get('connected') and not learner_hardware.get('busy')))
    memory_detail = ''
    if isaac and pixels and upstream(config) and not remote and learner_device.startswith('cuda:') and config['pixel_batch_size'] >= 256:
        gpu = next((g for g in hardware['gpus'] if g['device'] == learner_device), {})
        if gpu.get('free_mib', 0) < 7300:
            learner_ok = False
            memory_detail = ' · Restored batch 256 needs about 7 GiB of free VRAM for a local learner. Use an external GPU while Isaac runs on this 8 GiB device.'
    checks = [
        {"id": "environment", "label": "Isaac Sim NIST bridge" if isaac else "Mock peg environment" if mock else "Hardware adapter",
         "ready": bool(available), "detail": (("Connected: 3 RGB cameras + 22 proprioception values / 6 actions; fixed gripper" if pixels else "Connected: FFW-SG2, 39 state observations / 6 actions; fixed gripper") if compatible else
             bridge.get('error', 'Requires FFW-SG2, 20 Hz, translation ≤ 0.02 m, rotation ≤ 0.15 rad and an available target peg')) if isaac else
             "State-based XYZ toy task" if mock else "Not connected"},
        {"id": "actor", "label": "Actor and command ownership", "ready": bool(available and deps and actor_ok),
         "detail": "Leased simulator ownership / Cartesian + rotation control; fixed gripper" if isaac else "Mock policy / manual XYZ actions"},
        {"id": "learner", "label": "Agentlace / SAC / replay", "ready": bool(available and deps and artifacts and learner_ok),
         "detail": (f"{'External' if remote else 'Local'} SAC learner · {learner_device}" if learner_ok else learner_hardware.get('error') or f'{learner_device} unavailable, busy or insufficient free memory') + memory_detail + ('' if deps and artifacts else ' · Install dependencies and select artifacts compatible with this model/resolution')},
        {"id": "reward", "label": "Reward and reset", "ready": bool(available and reward),
         "detail": "Simulator part pose / velocity / orientation and scene reset" if isaac and reward else "Environment reward or human labels" if reward else "Select a trained classifier checkpoint and pixel observations"},
    ]
    checks[1]['detail'] = f"Actor · {actor_device}" + (actor_memory_detail or ('' if actor_ok else ' unavailable; enable NVIDIA GPU access'))
    if isaac and pixels:
        checks.append({'id': 'images', 'label': 'Synchronized RGB observations', 'ready': bool(bridge.get('pixel_observations')),
                       'detail': ('128×128 ROI images; ImageNet ResNet-10 + 22 proprioception values' if upstream(config) else '64×64 legacy CNN + robot proprioception') if bridge.get('pixel_observations') else 'Start the bridge with visualization enabled for pixel observations'})
    ready = all(item["ready"] for item in checks) and not bridge.get('busy', False)
    collection_ready = bool(compatible and deps and pixels and not bridge.get('busy', False))
    return {'can_classifier': collection_ready, 'can_demo': collection_ready and reward and (config['reward_source'] != 'classifier' or actor_ok), "can_preview": (bool(compatible and deps and not bridge.get("busy", False)) if isaac else True), "can_train": ready, "can_evaluate": all(item['ready'] for item in checks if item['id'] != 'learner') and artifacts and not bridge.get('busy', False) and bool(config["checkpoint_path"]), "checks": checks, "bridge": bridge, "hardware": hardware, "learner_hardware": learner_hardware}


@router.get("/workspace")
def workspace():
    with database() as db:
        config = settings(db)
        return {**config, "readiness": preflight(config["config"]), "sessions": [decorate(s) for s in sessions(db)],
                "execution_support": "mock_and_isaac_pixel_sac", "data_library": data_library(), "classifier_jobs": classifier_jobs()}


@router.put("/config")
def save_config(request: SaveConfig):
    with database() as db:
        current = settings(db)
        if request.expected_revision != current["revision"]:
            raise HTTPException(409, "Configuration changed. Reload before saving.")
        value = {"revision": current["revision"] + 1, "config": request.config.model_dump()}
        write(db, "config", value)
        return value


@router.post("/sessions", status_code=201)
def create_session(request: CreateSession):
    with database() as db:
        current = settings(db)
        if request.expected_config_revision != current["revision"]:
            raise HTTPException(409, "Configuration changed. Reload before creating a session.")
        if not preflight(current["config"])["can_" + request.mode]:
            raise HTTPException(409, "Resolve the environment readiness checks (Mock or Isaac Sim). Evaluation requires a saved policy checkpoint.")
        if any(decorate(s)["state"] != "stopped" or (_workers.get(s["id"]) is not None and _workers[s["id"]].poll() is None) for s in sessions(db)):
            raise HTTPException(409, "Stop the existing session before creating another.")
        session_config = dict(current['config'])
        if request.mode in ('demo', 'classifier'):
            session_config['max_episode_steps'] = min(session_config['max_episode_steps'], session_config['collection_episode_steps'])
        session = {"id": str(uuid4()), "revision": 0, "mode": request.mode, "state": "created",
                   "config_revision": current["revision"], "config": session_config,
                   "created_at": now(), "updated_at": now(), "control_source": "policy",
                   "episode": 1, "episode_outcome": None,
                   "metrics": {"environment_steps": None, "learner_steps": None, "success_rate": None, "policy_version": None},
                   "events": []}
        write(db, "session:" + session["id"], session)
        return session


@router.get("/sessions/{session_id}")
def get_session(session_id: str):
    with database() as db:
        session = read(db, "session:" + session_id)
        if session is None:
            raise HTTPException(404, "Session not found")
        return decorate(session)


@router.post("/sessions/{session_id}/commands")
def command(session_id: str, request: SessionCommand):
    with database() as db:
        session = read(db, "session:" + session_id)
        if session is None:
            raise HTTPException(404, "Session not found")
        if session["revision"] != request.expected_revision:
            raise HTTPException(409, "Session changed. Refresh before sending another command.")
        if uses_runtime(session):
            current_status = decorate(session)
            if current_status.get("runtime_pending") and request.action != "stop":
                raise HTTPException(409, "Wait for the runtime to acknowledge the previous command")
            if current_status["state"] == "stopped" and request.action != "stop":
                raise HTTPException(409, "Runtime stopped. Create a new session.")
        if session['mode'] == 'classifier' and request.action in ('resume', 'reset'):
            collection = (runtime_status(session) or {}).get('collection') or {}
            if collection.get('positive_samples', 0) + collection.get('negative_samples', 0) >= 10000:
                raise HTTPException(409, 'Collection limit reached; stop and create a new collection')
        action, state = request.action, session["state"]
        transitions = {"start": ({"created"}, "running"), "pause": ({"running"}, "paused"),
                       "resume": ({"paused"}, "running"), "stop": ({"created", "running", "paused"}, "stopped")}
        if action in transitions:
            allowed, target = transitions[action]
            if state not in allowed:
                raise HTTPException(409, f"Cannot {action} a {state} session")
            if action == "resume" and session["episode_outcome"] is not None:
                raise HTTPException(409, "Reset the episode before resuming")
            session["state"] = target
            if action in {"pause", "stop"}:
                session["control_source"] = "policy"
        elif action in {"take_over", "release"}:
            desired = "human" if action == "take_over" else "policy"
            if state != "running" or session["control_source"] == desired:
                raise HTTPException(409, "Control change requires a running session and a different owner")
            session["control_source"] = desired
        elif action in {"capture_success", "capture_failure", "save_demo", "discard_demo"}:
            if state not in {'running', 'paused'}:
                raise HTTPException(409, 'Start a collection session first')
            if action.startswith('capture_') and session['mode'] != 'classifier':
                raise HTTPException(409, 'Use Get classifier data to capture labeled images')
            if action in {'save_demo', 'discard_demo'}:
                if session['mode'] != 'demo' or state != 'paused':
                    raise HTTPException(409, 'Pause a demonstration session before reviewing it')
                if action == 'save_demo' and session['episode_outcome'] != 'success':
                    raise HTTPException(409, 'Label the demonstration successful before saving')
                if not (current_status.get('runtime', {}).get('collection') or {}).get('pending_demo_steps', 0):
                    raise HTTPException(409, 'No pending demonstration to review')
        elif action in {"success", "failure", "truncate"}:
            allowed = {"running", "paused"} if action == "truncate" or session["mode"] in {"demo", "classifier"} else {"running"}
            if state not in allowed or session["episode_outcome"] is not None:
                raise HTTPException(409, "Label an unfinished, running episode")
            session["episode_outcome"] = "truncated" if action == "truncate" else action
            session["state"] = "paused"
            session["control_source"] = "policy"
        elif action == "reset":
            if session['mode'] == 'demo' and (current_status.get('runtime', {}).get('collection') or {}).get('pending_demo_steps', 0):
                raise HTTPException(409, 'Save or discard the pending demonstration before resetting')
            if state != "paused":
                raise HTTPException(409, "Pause before resetting the episode")
            session["episode"] += 1
            session["episode_outcome"] = None
            session["control_source"] = "policy"
        session["revision"] += 1
        session["updated_at"] = now()
        session["events"].append({"sequence": session["revision"], "at": session["updated_at"],
                                  "action": action, "episode": session["episode"], "preview": session["mode"] == "preview"})
        # Commands are serialized; runtime acknowledges before the next command.
        session["events"] = session["events"][-200:]
        session.pop("manual_action", None)
        session.pop("classifier_label", None)
        write(db, "session:" + session_id, session)
        if uses_runtime(session) and action == "start":
            if not preflight(session["config"])["can_" + session["mode"]]:
                raise HTTPException(409, "Runtime readiness changed")
            launch(session)
        return decorate(session)


@router.post("/sessions/{session_id}/action")
def manual_action(session_id: str, request: ManualAction):
    if any(abs(value) > 1 for value in request.value):
        raise HTTPException(422, "XYZ action must be finite and within [-1, 1]")
    with database() as db:
        item = read(db, "session:" + session_id)
        if item is None:
            raise HTTPException(404, "Session not found")
        if item["revision"] != request.expected_revision:
            raise HTTPException(409, "Session changed")
        if not uses_runtime(item) or decorate(item).get("runtime_pending") or decorate(item)["state"] != "running":
            raise HTTPException(409, "Manual action requires a running session")
        expected_dim = 6 if item['config']['environment'] == 'isaac_sim' else 3
        if len(request.value) != expected_dim:
            raise HTTPException(422, f"Expected {expected_dim} action values for this environment")
        # A short pulse expires even if the browser disappears. No ROS commands.
        duration = max(.3, 1 / item["config"]["control_hz"] + .05)
        item["manual_action"] = {"value": request.value, "expires_at": time.time() + duration}
        write(db, "session:" + session_id, item)
        return {"accepted": True, "duration_s": duration}


@router.post('/sessions/{session_id}/classifier-label')
def classifier_label(session_id: str, request: ClassifierLabel):
    with database() as db:
        item = read(db, 'session:' + session_id)
        if item is None:
            raise HTTPException(404, 'Session not found')
        if item['revision'] != request.expected_revision:
            raise HTTPException(409, 'Session changed')
        if item['mode'] != 'classifier' or decorate(item)['state'] != 'running' or decorate(item).get('runtime_pending'):
            raise HTTPException(409, 'Label input requires a running classifier collection')
        item['classifier_label'] = {'success': request.success, 'expires_at': time.time() + .5}
        write(db, 'session:' + session_id, item)
        return {'accepted': True, 'duration_s': .5}


@router.get('/isaac/frame')
def isaac_frame(camera: Literal["spectator", "head", "left_wrist", "right_wrist"] = "spectator",
                quality: Literal["low", "standard", "high", "ultra"] = "standard"):
    try:
        content, headers = _isaac.frame(camera, quality)
        return Response(content, media_type='image/jpeg', headers=headers)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


class TrainClassifier(StrictModel):
    session_ids: list[str] = Field(min_length=1, max_length=32)
    epochs: int = Field(default=20, ge=1, le=200)
    seed: int = Field(default=0, ge=0)
    threshold: float = Field(default=.9, gt=0, lt=1)


def data_library():
    items = []
    for folder in data_root().iterdir():
        if not folder.is_dir() or folder.name == 'classifiers':
            continue
        for kind, relative in [('classifier', 'classifier_samples/manifest.json'), ('demo', 'demos/manifest.json')]:
            path = folder / relative
            if not path.is_file():
                continue
            try:
                manifest = json.loads(path.read_text())
                item = {'session_id': folder.name, 'kind': kind, 'task': manifest['task'], 'cameras': manifest.get('cameras', ['head', 'left_wrist', 'right_wrist']),
                        'learning_backend': manifest.get('learning_backend', 'legacy_torch'), 'camera_crops': manifest.get('camera_crops', {})}
                if kind == 'classifier':
                    samples = manifest['samples']
                    item.update(positive=sum(x['label'] == 1 for x in samples), negative=sum(x['label'] == 0 for x in samples),
                                episodes=len(set(x['episode'] for x in samples)))
                else:
                    item.update(files=[{'name': f"episode-{entry['episode']:05d}.npz", 'episode': entry['episode'], 'steps': entry['steps']} for entry in manifest['episodes']],
                                episodes=len(manifest['episodes']), steps=sum(x['steps'] for x in manifest['episodes']),
                                path=str(folder / 'demonstrations.npz') if (folder / 'demonstrations.npz').is_file() else None)
                items.append(item)
            except (OSError, ValueError, KeyError):
                continue
    return items


def classifier_jobs():
    jobs = []
    for path in sorted((data_root() / 'classifiers').glob('*/status.json'), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            value = json.loads(path.read_text())
            process = _classifier_workers.get(path.parent.name)
            if value.get('state') in ('queued', 'training') and (process is None or process.poll() is not None):
                value.update(state='failed', error='Classifier worker exited or supervisor restarted')
            jobs.append({'id': path.parent.name, **value})
        except (OSError, ValueError):
            pass
    return jobs


@router.post('/classifiers/train', status_code=202)
def train_classifier(request: TrainClassifier):
    with _classifier_lock:
        return _start_classifier_job(request)


def _start_classifier_job(request):
    if any(p.poll() is None for p in _classifier_workers.values()):
        raise HTTPException(409, 'A classifier training job is already running')
    collections = []
    task = None
    cameras = None
    learning_backend = None
    camera_crops = None
    positive = negative = 0
    with database() as db:
        for identifier in dict.fromkeys(request.session_ids):
            item = read(db, 'session:' + identifier)
            if not item or item['mode'] != 'classifier' or decorate(item)['state'] != 'stopped':
                raise HTTPException(409, 'Select stopped classifier collection sessions')
            path = data_root() / item['id'] / 'classifier_samples' / 'manifest.json'
            if not path.is_file():
                raise HTTPException(422, 'Selected collection contains no labeled images')
            manifest = json.loads(path.read_text())
            current_backend = manifest.get('learning_backend', 'legacy_torch')
            current_crops = manifest.get('camera_crops', {})
            if learning_backend is not None and (learning_backend != current_backend or camera_crops != current_crops):
                raise HTTPException(422, 'Select collections with matching resolution, model backend and camera crops; recollect legacy 64×64 data for ResNet-10')
            learning_backend, camera_crops = current_backend, current_crops
            if task is not None and task != manifest['task']:
                raise HTTPException(422, 'Collections must use the same robot, arm and target')
            task = manifest['task']
            current_cameras = manifest.get('cameras', ['head', 'left_wrist', 'right_wrist'])
            if cameras is not None and cameras != current_cameras:
                raise HTTPException(422, 'Collections must use the same classifier cameras')
            cameras = current_cameras
            positive += sum(sample['label'] == 1 for sample in manifest['samples'])
            negative += sum(sample['label'] == 0 for sample in manifest['samples'])
            collections.append(str(path.parent.parent))
    if min(positive, negative) < 10:
        raise HTTPException(422, 'At least 10 success and 10 failure images are required, across separate episodes')
    identifier = str(uuid4())
    folder = data_root() / 'classifiers' / identifier
    folder.mkdir(parents=True)
    spec = {**request.model_dump(), 'collections': collections, 'task': task, 'cameras': cameras,
            'learning_backend': learning_backend, 'camera_crops': camera_crops}
    (folder / 'request.json').write_text(json.dumps(spec))
    (folder / 'status.json').write_text(json.dumps({'state': 'queued', 'epochs': request.epochs, 'task': task, 'cameras': cameras}))
    with (folder / 'worker.log').open('a') as log:
        from hilserl_profiles import runtime_python
        _classifier_workers[identifier] = subprocess.Popen([runtime_python(spec), str(Path(__file__).with_name('hilserl_classifier.py')), str(folder)],
                                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                                          env={**os.environ, 'JAX_PLATFORMS': 'cpu'} if learning_backend == 'hil_serl' else None)
    return {'id': identifier, 'state': 'queued'}


@router.get('/classifier-samples/{session_id}/{sample_id}')
def classifier_sample(session_id: str, sample_id: str):
    # UUID validation prevents user-controlled paths escaping the sample directory.
    from uuid import UUID
    try:
        session_id, sample_id = str(UUID(session_id)), str(UUID(sample_id))
    except ValueError:
        raise HTTPException(422, 'Invalid sample identifier')
    path = data_root() / session_id / 'classifier_samples' / f'{sample_id}.npz'
    if not path.is_file():
        raise HTTPException(404, 'Sample not found')
    import numpy as np
    import cv2
    with np.load(path, allow_pickle=False) as data:
        pixels = data['images']
    cameras = ['head', 'left_wrist', 'right_wrist']
    manifest = json.loads((path.parent / 'manifest.json').read_text())
    selected = manifest.get('cameras', cameras)
    rgb = np.concatenate([pixels[i*3:i*3+3].transpose(1, 2, 0) for i, camera in enumerate(cameras) if camera in selected], axis=1)
    ok, jpg = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise HTTPException(500, 'Sample encoding failed')
    return Response(jpg.tobytes(), media_type='image/jpeg', headers={'Cache-Control': 'no-store'})


def artifact_id(value):
    from uuid import UUID
    try:
        return str(UUID(value))
    except ValueError:
        raise HTTPException(422, 'Invalid artifact identifier')


def require_unused(db, paths):
    paths = {str(path) for path in paths}
    config = settings(db)['config']
    if any(config.get(key) in paths for key in ('classifier_path', 'demonstration_path', 'checkpoint_path')):
        raise HTTPException(409, 'Artifact is selected in saved configuration; clear its path and save before deleting')
    for row in db.execute("SELECT body FROM documents WHERE id LIKE 'session:%'"):
        item = json.loads(row[0])
        if decorate(item)['state'] != 'stopped' and any(item['config'].get(key) in paths for key in ('classifier_path', 'demonstration_path', 'checkpoint_path')):
            raise HTTPException(409, 'Artifact is used by an active session; stop it before deleting')


@router.get('/datasets/{session_id}/demo/files/{filename}')
def download_demo(session_id: str, filename: str):
    from fastapi.responses import FileResponse
    import re
    session_id = artifact_id(session_id)
    with database() as db:
        item = read(db, 'session:' + session_id)
        if not item or item['mode'] != 'demo':
            raise HTTPException(404, 'Demo session not found')
        if decorate(item)['state'] != 'stopped':
            raise HTTPException(409, 'Stop collection before downloading its data')
    folder = data_root() / session_id
    if filename == 'demonstrations.npz':
        path = folder / filename
    elif re.fullmatch(r'episode-[0-9]{5,}\.npz', filename):
        path = folder / 'demos' / filename
    else:
        raise HTTPException(422, 'Invalid demo filename')
    if folder.is_symlink() or path.parent.is_symlink() or path.is_symlink():
        raise HTTPException(422, 'Invalid dataset path')
    if not path.is_file():
        raise HTTPException(404, 'Demo file not found')
    return FileResponse(path, media_type='application/octet-stream', filename=filename)


class DeleteDemoEpisodes(StrictModel):
    episodes: list[int] = Field(min_length=1, max_length=4096)

    @field_validator('episodes')
    @classmethod
    def valid_episodes(cls, values):
        if any(value < 1 for value in values) or len(set(values)) != len(values):
            raise ValueError('Select distinct positive episode numbers')
        return values


class DemoSelection(DeleteDemoEpisodes):
    session_id: str


class CombineDemos(StrictModel):
    selections: list[DemoSelection] = Field(min_length=1, max_length=100)


@router.post('/datasets/demo/combine')
def combine_demos(request: CombineDemos):
    import numpy as np
    from hilserl_data import FIELDS, write_npz
    from hilserl_cameras import CAMERAS
    with database() as db:
        paths, seen = [], set()
        expected = None
        total = 0
        for selection in request.selections:
            identifier = artifact_id(selection.session_id)
            item = read(db, 'session:' + identifier)
            worker = _workers.get(identifier)
            if not item or item['mode'] != 'demo':
                raise HTTPException(404, 'Demo session not found')
            if decorate(item)['state'] != 'stopped' or (worker and worker.poll() is None):
                raise HTTPException(409, 'Stop selected collections before exporting')
            folder = data_root() / identifier
            manifest_path = folder / 'demos' / 'manifest.json'
            if any(p.is_symlink() for p in (folder, manifest_path.parent, manifest_path)):
                raise HTTPException(422, 'Invalid dataset path')
            if not manifest_path.is_file():
                raise HTTPException(404, 'Demo manifest not found')
            manifest = json.loads(manifest_path.read_text())
            signature = (manifest['task'], manifest.get('cameras', CAMERAS))
            if expected is not None and signature != expected:
                raise HTTPException(409, 'Select demos with matching tasks and cameras')
            expected = signature
            entries = {entry['episode']: entry for entry in manifest['episodes']}
            for episode in selection.episodes:
                if (identifier, episode) in seen or episode not in entries:
                    raise HTTPException(422, 'Duplicate or missing episode')
                seen.add((identifier, episode))
                total += entries[episode]['steps']
                if total > 4096:
                    raise HTTPException(422, 'Select at most 4096 steps per training export')
                path = folder / 'demos' / f'episode-{episode:05d}.npz'
                if path.is_symlink() or not path.is_file():
                    raise HTTPException(422, 'Invalid episode file')
                paths.append(path)
        arrays = {key: [] for key in FIELDS + ('images', 'next_images')}
        metadata = None
        actual_steps = 0
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                current = {key: data[key].copy() for key in ('__schema__', '__task__')}
                current['__cameras__'] = data['__cameras__'].copy() if '__cameras__' in data else np.array(json.dumps(CAMERAS))
                current['__crops__'] = data['__crops__'].copy() if '__crops__' in data else np.array('{}')
                if metadata is not None and any(str(current[k]) != str(metadata[k]) for k in current):
                    raise HTTPException(409, 'Demo schemas, tasks or cameras do not match')
                metadata = current
                actual_steps += len(data['reward'])
                if actual_steps > 4096:
                    raise HTTPException(422, 'Select at most 4096 steps per training export')
                for key in arrays:
                    arrays[key].append(data[key].copy())
        target = data_root() / 'demo_exports' / f'{uuid4()}.npz'
        write_npz(target, **metadata, __sources__=np.array(json.dumps([s.model_dump() for s in request.selections])),
                  **{key: np.concatenate(values) for key, values in arrays.items()})
    return {'path': str(target), 'task': expected[0], 'cameras': expected[1], 'steps': actual_steps,
            'episodes': len(paths)}


@router.post('/datasets/{session_id}/demo/delete-episodes')
def delete_demo_episodes(session_id: str, request: DeleteDemoEpisodes):
    import numpy as np
    import tempfile
    from hilserl_data import FIELDS, write_npz
    session_id = artifact_id(session_id)
    with database() as db:
        item = read(db, 'session:' + session_id)
        if not item or item['mode'] != 'demo':
            raise HTTPException(404, 'Demo session not found')
        worker = _workers.get(session_id)
        if decorate(item)['state'] != 'stopped' or (worker and worker.poll() is None):
            raise HTTPException(409, 'Stop collection before deleting episodes')
        folder = data_root() / session_id
        demos = folder / 'demos'
        manifest_path = demos / 'manifest.json'
        combined = folder / 'demonstrations.npz'
        if any(path.is_symlink() for path in (folder, demos, manifest_path, combined)):
            raise HTTPException(422, 'Invalid dataset path')
        if not manifest_path.is_file():
            raise HTTPException(404, 'Demo manifest not found')
        manifest = json.loads(manifest_path.read_text())
        entries = manifest['episodes']
        selected = set(request.episodes)
        if not selected.issubset({entry['episode'] for entry in entries}):
            raise HTTPException(404, 'Selected episode no longer exists; refresh the list')
        paths = {entry['episode']: demos / f"episode-{entry['episode']:05d}.npz" for entry in entries}
        if any(path.is_symlink() or not path.is_file() for path in paths.values()):
            raise HTTPException(422, 'Invalid or missing episode file')
        require_unused(db, [combined, *(paths[n] for n in selected)])
        remaining = [entry for entry in entries if entry['episode'] not in selected]
        # Prepare a complete replacement before touching any published files.
        with tempfile.TemporaryDirectory(prefix='.demo-edit-', dir=folder) as temporary:
            stage = Path(temporary)
            if remaining:
                fields = FIELDS + ('images', 'next_images')
                arrays = {key: [] for key in fields}
                metadata = None
                for entry in remaining:
                    with np.load(paths[entry['episode']], allow_pickle=False) as data:
                        if metadata is None:
                            metadata = {key: data[key].copy() for key in data.files if key.startswith('__')}
                        for key in fields:
                            arrays[key].append(data[key].copy())
                write_npz(stage / 'combined.new', **metadata,
                          **{key: np.concatenate(values) for key, values in arrays.items()})
            (stage / 'manifest.new').write_text(json.dumps({**manifest, 'episodes': remaining}))
            backups = []
            installed = []
            try:
                for path in [manifest_path, combined, *(paths[n] for n in selected)]:
                    if path.exists():
                        backup = stage / f'backup-{len(backups)}'
                        path.replace(backup)
                        backups.append((path, backup))
                (stage / 'manifest.new').replace(manifest_path)
                installed.append(manifest_path)
                if remaining:
                    (stage / 'combined.new').replace(combined)
                    installed.append(combined)
            except OSError:
                for path in installed:
                    path.unlink(missing_ok=True)
                for path, backup in reversed(backups):
                    backup.replace(path)
                raise
    return {'deleted_episodes': request.episodes, 'remaining_episodes': len(remaining)}


@router.delete('/datasets/{session_id}/{kind}')
def delete_dataset(session_id: str, kind: Literal['classifier', 'demo']):
    import shutil
    session_id = artifact_id(session_id)
    with _classifier_lock, database() as db:
        item = read(db, 'session:' + session_id)
        if not item:
            raise HTTPException(404, 'Session not found')
        if decorate(item)['state'] != 'stopped':
            raise HTTPException(409, 'Stop collection before deleting its data')
        folder = data_root() / session_id
        for job, worker in _classifier_workers.items():
            if worker.poll() is None:
                request = json.loads((data_root() / 'classifiers' / job / 'request.json').read_text())
                if str(folder) in request['collections']:
                    raise HTTPException(409, 'Dataset is being used by classifier training')
        target = folder / ('classifier_samples' if kind == 'classifier' else 'demos')
        exports = [folder / 'demonstrations.npz', folder / 'pending-demo.npz'] if kind == 'demo' else []
        require_unused(db, exports)
        if not target.exists() and not any(p.exists() for p in exports):
            raise HTTPException(404, 'Dataset not found')
        if target.is_symlink() or folder.is_symlink():
            raise HTTPException(422, 'Invalid dataset path')
        if target.exists():
            shutil.rmtree(target)
        for path in exports:
            path.unlink(missing_ok=True)
        item.setdefault('deleted_datasets', []).append(kind)
        write(db, 'session:' + session_id, item)
    return {'deleted': True, 'session_id': session_id, 'kind': kind}


@router.delete('/classifiers/{job_id}')
def delete_classifier(job_id: str):
    import shutil
    job_id = artifact_id(job_id)
    with _classifier_lock, database() as db:
        worker = _classifier_workers.get(job_id)
        if worker is not None and worker.poll() is None:
            raise HTTPException(409, 'Classifier training is still running')
        folder = data_root() / 'classifiers' / job_id
        if not folder.is_dir():
            raise HTTPException(404, 'Classifier job not found')
        if folder.is_symlink():
            raise HTTPException(422, 'Invalid classifier path')
        require_unused(db, [folder / 'classifier.npz'])
        shutil.rmtree(folder)
        _classifier_workers.pop(job_id, None)
    return {'deleted': True, 'id': job_id}
