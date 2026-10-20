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
    batch_size: int = Field(default=256, ge=1, le=4096)
    replay_capacity: int = Field(default=100000, ge=4096, le=10000000)
    learning_rate: float = Field(default=0.0003, gt=0, le=0.1)
    discount: float = Field(default=0.99, gt=0, le=1)
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
    pixel_batch_size: int = Field(default=32, ge=1, le=128)
    pixel_replay_capacity: int = Field(default=2048, ge=64, le=4096)
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
    folder = data_root() / session["id"]
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "worker.log").open("a") as log:
        _workers[session["id"]] = subprocess.Popen(
            [sys.executable, str(Path(__file__).with_name("hilserl_runtime.py")), str(data_root()), session["id"], str(os.getpid())],
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
            process.wait(timeout=8)
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
    return {**stored, "config": {**ExperimentConfig().model_dump(), **stored["config"]}}


def sessions(db):
    return [json.loads(row[0]) for row in db.execute("SELECT body FROM documents WHERE id LIKE 'session:%' ORDER BY rowid DESC")]


def preflight(config):
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
    compatible = compatible and (not pixels or bridge.get('pixel_observations', False))
    available = mock or compatible
    checks = [
        {"id": "environment", "label": "Isaac Sim NIST bridge" if isaac else "Mock peg environment" if mock else "Hardware adapter",
         "ready": bool(available), "detail": (("Connected: 3 RGB cameras + 22 proprioception values / 6 actions; fixed gripper" if pixels else "Connected: FFW-SG2, 39 state observations / 6 actions; fixed gripper") if compatible else
             bridge.get('error', 'Requires FFW-SG2, 20 Hz, translation ≤ 0.02 m, rotation ≤ 0.15 rad and an available target peg')) if isaac else
             "State-based XYZ toy task" if mock else "Not connected"},
        {"id": "actor", "label": "Actor and command ownership", "ready": bool(available and deps),
         "detail": "Leased simulator ownership / Cartesian + rotation control; fixed gripper" if isaac else "Mock policy / manual XYZ actions"},
        {"id": "learner", "label": "Agentlace / SAC / replay", "ready": bool(available and deps and artifacts),
         "detail": "CPU runtime installed" if deps and artifacts else "Install runtime dependencies; artifacts must be existing .npz files under the HIL data directory"},
        {"id": "reward", "label": "Reward and reset", "ready": bool(available and reward),
         "detail": "Simulator part pose / velocity / orientation and scene reset" if isaac and reward else "Environment reward or human labels" if reward else "Select a trained classifier checkpoint and pixel observations"},
    ]
    if isaac and pixels:
        checks.append({'id': 'images', 'label': 'Synchronized RGB observations', 'ready': bool(bridge.get('pixel_observations')),
                       'detail': 'Head / left wrist / right wrist at 64×64; CNN + robot proprioception' if bridge.get('pixel_observations') else 'Start the bridge with visualization enabled for pixel observations'})
    ready = all(item["ready"] for item in checks) and not bridge.get('busy', False)
    collection_ready = bool(compatible and deps and pixels and not bridge.get('busy', False))
    return {'can_classifier': collection_ready, 'can_demo': collection_ready, "can_preview": (bool(compatible and deps and not bridge.get("busy", False)) if isaac else True), "can_train": ready, "can_evaluate": ready and bool(config["checkpoint_path"]), "checks": checks, "bridge": bridge}


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
                item = {'session_id': folder.name, 'kind': kind, 'task': manifest['task'], 'cameras': manifest.get('cameras', ['head', 'left_wrist', 'right_wrist'])}
                if kind == 'classifier':
                    samples = manifest['samples']
                    item.update(positive=sum(x['label'] == 1 for x in samples), negative=sum(x['label'] == 0 for x in samples),
                                episodes=len(set(x['episode'] for x in samples)))
                else:
                    item.update(episodes=len(manifest['episodes']), steps=sum(x['steps'] for x in manifest['episodes']),
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
    spec = {**request.model_dump(), 'collections': collections, 'task': task, 'cameras': cameras}
    (folder / 'request.json').write_text(json.dumps(spec))
    (folder / 'status.json').write_text(json.dumps({'state': 'queued', 'epochs': request.epochs, 'task': task, 'cameras': cameras}))
    with (folder / 'worker.log').open('a') as log:
        _classifier_workers[identifier] = subprocess.Popen([sys.executable, str(Path(__file__).with_name('hilserl_classifier.py')), str(folder)],
                                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
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
