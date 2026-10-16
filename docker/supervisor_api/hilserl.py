"""HIL-SERL control plane: preview and a local Agentlace/SAC mock worker.

Run with one supervisor worker. Isaac Sim and physical hardware fail closed.
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

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

_workers = {}

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
    demonstration_path: str = Field(default="", max_length=512)
    checkpoint_path: str = Field(default="", max_length=512)


class SaveConfig(StrictModel):
    expected_revision: int = Field(ge=0)
    config: ExperimentConfig


class CreateSession(StrictModel):
    expected_config_revision: int = Field(ge=0)
    mode: Literal["preview", "train", "evaluate"]


class SessionCommand(StrictModel):
    expected_revision: int = Field(ge=0)
    action: Literal["start", "pause", "resume", "stop", "take_over", "release", "success", "failure", "reset"]


class ManualAction(StrictModel):
    expected_revision: int = Field(ge=0)
    value: list[float] = Field(min_length=3, max_length=3)


def data_root():
    return Path(os.environ.get("CYCLO_HILSERL_DATA_DIR", "/workspace/hilserl"))


def runtime_status(session):
    if session["mode"] == "preview":
        return None
    try:
        return json.loads((data_root() / session["id"] / "runtime.json").read_text())
    except (OSError, ValueError):
        return None


def decorate(session):
    if session["mode"] == "preview":
        return session
    result = dict(session)
    status = runtime_status(session)
    process = _workers.get(session["id"])
    if status:
        result["runtime"] = status
        result["metrics"] = status.get("metrics", session["metrics"])
        result["episode"] = status.get("episode", session["episode"])
    result["runtime_pending"] = session["state"] != "created" and (not status or status.get("ack_revision", -1) < session["revision"])
    if session["state"] != "created" and (process is None or process.poll() is not None):
        result["state"] = "stopped"
        result["runtime_pending"] = False
        if session["state"] != "stopped":
            result["runtime_error"] = (status or {}).get("error") or "Runtime exited or supervisor restarted. Create a new session."
    if process is not None and process.poll() not in (None, 0):
        result["runtime_error"] = (status or {}).get("error") or f"Runtime exited with code {process.returncode}; artifacts may be incomplete."
    if status and status.get("error"):
        result["runtime_error"] = status["error"]
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
    return read(db, "config") or {"revision": 0, "config": ExperimentConfig().model_dump()}


def sessions(db):
    return [json.loads(row[0]) for row in db.execute("SELECT body FROM documents WHERE id LIKE 'session:%' ORDER BY rowid DESC")]


def preflight(config):
    mock = config["environment"] == "mock"
    deps = all(importlib.util.find_spec(name) is not None for name in ("agentlace", "torch", "numpy"))
    artifacts = all(artifact_ready(config[key]) for key in ("checkpoint_path", "demonstration_path"))
    reward = config["reward_source"] != "classifier"
    checks = [
        {"id": "environment", "label": "Mock peg environment" if mock else "Isaac Sim / AI Worker adapter", "ready": mock, "detail": "State-based XYZ toy task; no physics engine" if mock else "Not connected"},
        {"id": "actor", "label": "Actor and command ownership", "ready": mock and deps, "detail": "Mock policy / manual XYZ actions" if mock else "Runtime integration pending"},
        {"id": "learner", "label": "Agentlace / SAC / replay", "ready": mock and deps and artifacts, "detail": "CPU runtime installed" if deps and artifacts else "Install requirements-hilserl.txt; artifacts must be existing .npz files under the HIL data directory"},
        {"id": "reward", "label": "Reward and reset", "ready": mock and reward, "detail": "Environment reward or human labels" if reward else "Classifier is not implemented"},
    ]
    ready = all(item["ready"] for item in checks)
    return {"can_preview": True, "can_train": ready, "can_evaluate": ready and bool(config["checkpoint_path"]), "checks": checks}


@router.get("/workspace")
def workspace():
    with database() as db:
        config = settings(db)
        return {**config, "readiness": preflight(config["config"]), "sessions": [decorate(s) for s in sessions(db)],
                "execution_support": "mock_agentlace_sac"}


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
        if request.mode != "preview" and not preflight(current["config"])["can_" + request.mode]:
            raise HTTPException(409, "Select Mock and resolve readiness checks. Evaluation requires a saved policy checkpoint.")
        if any(decorate(s)["state"] != "stopped" or (_workers.get(s["id"]) is not None and _workers[s["id"]].poll() is None) for s in sessions(db)):
            raise HTTPException(409, "Stop the existing session before creating another.")
        session = {"id": str(uuid4()), "revision": 0, "mode": request.mode, "state": "created",
                   "config_revision": current["revision"], "config": current["config"],
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
        if session["mode"] != "preview":
            current_status = decorate(session)
            if current_status.get("runtime_pending") and request.action != "stop":
                raise HTTPException(409, "Wait for the runtime to acknowledge the previous command")
            if current_status["state"] == "stopped" and request.action != "stop":
                raise HTTPException(409, "Runtime stopped. Create a new session.")
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
        elif action in {"success", "failure"}:
            if state != "running" or session["episode_outcome"] is not None:
                raise HTTPException(409, "Label an unfinished, running episode")
            session["episode_outcome"] = action
            session["state"] = "paused"
            session["control_source"] = "policy"
        elif action == "reset":
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
        write(db, "session:" + session_id, session)
        if session["mode"] != "preview" and action == "start":
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
        if item["mode"] == "preview" or decorate(item)["state"] != "running" or item["control_source"] != "human":
            raise HTTPException(409, "Manual action requires a running Mock session with human control")
        # A short pulse expires even if the browser disappears. No ROS commands.
        duration = max(.3, 1 / item["config"]["control_hz"] + .05)
        item["manual_action"] = {"value": request.value, "expires_at": time.time() + duration}
        write(db, "session:" + session_id, item)
        return {"accepted": True, "duration_s": duration}
