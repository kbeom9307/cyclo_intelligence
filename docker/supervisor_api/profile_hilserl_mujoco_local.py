"""Opt-in, isolated Lab runtime benchmark on one local GPU.

Run with an idle MuJoCo bridge:
  python3 profile_hilserl_mujoco_local.py --output /tmp/mujoco-local-256.json

Real simulator images feed the actor, reward classifier and upstream learner.
Scripted hold transitions seed the demo bank; a constant-negative classifier
exercises inference without ending every episode. This measures execution and
memory, not policy learning quality. No user workspace configuration is changed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
import hilserl as hil


PREPARE = """
import json, sys, numpy as np
from pathlib import Path
from hilserl_profiles import prepare_jax_libraries, UPSTREAM_SCHEMA, UPSTREAM_CLASSIFIER
prepare_jax_libraries()
from hilserl_upstream import UpstreamPolicy, UpstreamClassifier
from hilserl_mujoco import MujocoNistEnv
from hilserl_data import task_key, write_npz, save_rows
p=Path(sys.argv[1]); c=json.loads((p/'config.json').read_text())
a=UpstreamPolicy({**c,'actor_device':'cpu'})
meta={'__task__':np.array(json.dumps(task_key(c),sort_keys=True)),
      '__cameras__':np.array(json.dumps(c['observation_cameras'])), '__crops__':np.array('{}')}
write_npz(p/'initial-policy.npz',__schema__=np.array(UPSTREAM_SCHEMA),**meta,**a.export())
f=UpstreamClassifier(c['classifier_cameras'],'cpu'); weights=f.export()
weights['Dense_1/kernel'][:]=0.; weights['Dense_1/bias'][:]=-8.
write_npz(p/'classifier.npz',__schema__=np.array(UPSTREAM_CLASSIFIER),**meta,**weights)
env=MujocoNistEnv(c)
rows=[]
try:
    for _ in range(32):
        obs, images=env.observation(), env.images.copy()
        nxt, reward, terminal, truncated=env.step(np.zeros(6,np.float32))
        rows.append(dict(observation=obs, action=np.zeros(6,np.float32), next_observation=nxt,
                         reward=reward, terminal=terminal, truncated=truncated, human=True,
                         images=images, next_images=env.images.copy()))
    save_rows(p/'scripted-demo.npz',rows,env.schema,c,pixel=True)
finally:
    env.close()
"""


def probe(args):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"state": "preparing", "batch_size": args.batch, "cta_ratio": 2,
              "learning_image_shape": [9, 128, 128], "control_hz": 20,
              "actor_device": "cuda:0", "learner_device": "cuda:0",
              "learner_placement": "local", "policy_publish_interval": args.publish_interval,
              "xla_flags": os.environ.get("XLA_FLAGS", ""),
              "xla_memory_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", ".80"),
              "notes": ["Temporary scripted hold demonstrations and negative classifier; not a learning-quality test.",
                        "Short runs can publish each update to exercise live policy refresh; normal interval is 50.",
                        "MuJoCo physics, four views, actor, classifier and learner share this workstation."]}
    gpu_samples, runtime_samples, frame_stats = [], [], {}
    done = threading.Event()
    views = threading.Event()
    start = time.monotonic()

    def save():
        output.write_text(json.dumps({**result, "gpu_samples": gpu_samples,
                                      "runtime_samples": runtime_samples,
                                      "frames": frame_stats}, indent=2) + "\n")

    def gpu_monitor():
        while not done.is_set():
            try:
                row = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits"], text=True).splitlines()[0]
                used, free, utilization = map(float, row.split(","))
                gpu_samples.append(dict(seconds=round(time.monotonic()-start, 3),
                                        used_mib=used, free_mib=free, utilization_pct=utilization))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            done.wait(.1)

    def camera_monitor(camera):
        quality = "ultra" if camera == "spectator" else "high"
        url = f"http://127.0.0.1:7080/api/hilserl/mujoco/frame?camera={camera}&quality={quality}"
        stats = frame_stats[camera] = dict(ok=0, unavailable=0, max_latency_ms=0.,
                                          fresh_frames=0, max_age_ms=0.)
        last_frame = None
        views.wait()
        while not done.is_set():
            before = time.monotonic()
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    image = response.read()
                    if image[:2] != b"\xff\xd8":
                        raise ValueError("Non-JPEG camera response")
                    frame_id = response.headers.get("X-Frame-Id")
                    if frame_id and frame_id != last_frame:
                        stats["fresh_frames"] += 1
                        last_frame = frame_id
                    stats["max_age_ms"] = max(stats["max_age_ms"], float(response.headers.get("X-Frame-Age-Ms", 0)))
                stats["ok"] += 1
                stats["max_latency_ms"] = max(stats["max_latency_ms"], (time.monotonic()-before)*1000)
            except (OSError, ValueError):
                stats["unavailable"] += 1
            done.wait(max(0, .25-(time.monotonic()-before)))

    threads = [threading.Thread(target=gpu_monitor, daemon=True)]
    threads += [threading.Thread(target=camera_monitor, args=(c,), daemon=True)
                for c in ("spectator", "head", "left_wrist", "right_wrist")]
    for thread in threads:
        thread.start()
    item = None
    try:
        assert hil._mujoco.health()["ready"] and not hil._mujoco.health()["busy"]
        with tempfile.TemporaryDirectory(prefix="hil-mujoco-local-") as td:
            folder = Path(td)
            os.environ["CYCLO_HILSERL_DATA_DIR"] = td
            config = hil.ExperimentConfig(
                environment="mujoco", robot_type="ffw_sg2_rev1", control_hz=20,
                learning_backend="hil_serl", learner_placement="local",
                actor_device="cuda:0", learner_device="cuda:0",
                pixel_batch_size=args.batch, batch_size=args.batch,
                policy_publish_interval=args.publish_interval, reward_source="classifier",
                checkpoint_path=str(folder/"initial-policy.npz"),
                classifier_path=str(folder/"classifier.npz"),
                demonstration_path=str(folder/"scripted-demo.npz")).model_dump()
            (folder/"config.json").write_text(json.dumps(config))
            subprocess.run(["/opt/hil-serl-venv/bin/python", "-c", PREPARE, td],
                           env={**os.environ, "JAX_PLATFORMS": "cpu"}, check=True)
            result["preparation_s"] = time.monotonic()-start
            app = FastAPI()
            app.include_router(hil.router)
            with TestClient(app) as client:
                client.put("/hilserl/config", json={"expected_revision": 0, "config": config}).raise_for_status()
                result["readiness"] = client.get("/hilserl/workspace").json()["readiness"]
                response = client.post("/hilserl/sessions",
                                       json={"expected_config_revision": 1, "mode": "train"})
                if response.status_code != 201:
                    raise RuntimeError(f"Create train: {response.text}")
                item = response.json()
                path = "/hilserl/sessions/"+item["id"]
                client.post(path+"/commands", json={"expected_revision": 0, "action": "start"}).raise_for_status()
                views.set()
                runtime_start = time.monotonic()
                result["state"] = "running"
                save()
                try:
                    while time.monotonic()-runtime_start < args.timeout:
                        item = client.get(path).json()
                        runtime = item.get("runtime", {})
                        metrics = item.get("metrics", {})
                        learner = runtime.get("compute", {}).get("learner", {})
                        sample = dict(seconds=round(time.monotonic()-start, 3),
                                      environment_steps=metrics.get("environment_steps"),
                                      learner_steps=metrics.get("learner_steps"),
                                      critic_steps=metrics.get("critic_steps"),
                                      policy_version=metrics.get("policy_version"),
                                      pending_transitions=learner.get("pending_transitions"),
                                      replay_size=metrics.get("replay_size"),
                                      actor_memory=runtime.get("compute", {}).get("actor_memory"),
                                      learner_memory=learner.get("learner_memory"),
                                      runtime_pending=item.get("runtime_pending"))
                        runtime_samples.append(sample)
                        result["latest_metrics"] = metrics
                        save()
                        if item.get("runtime_error"):
                            raise RuntimeError(item["runtime_error"])
                        if (metrics.get("learner_steps") or 0) >= args.updates:
                            result["state"] = "updates_complete"
                            break
                        time.sleep(.5)
                    else:
                        raise TimeoutError("Local training did not finish requested updates in time")
                finally:
                    item = client.get(path).json()
                    if item["state"] != "stopped":
                        client.post(path+"/commands", json={"expected_revision": item["revision"], "action": "stop"}).raise_for_status()
                    worker = hil._workers[item["id"]]
                    try:
                        worker.wait(timeout=180)
                    except subprocess.TimeoutExpired:
                        worker.terminate()
                        worker.wait(timeout=15)
                        result["forced_stop"] = True
                    final = client.get(path).json()
                    result["final"] = {k: final.get(k) for k in
                                       ("state", "runtime_error", "runtime", "metrics")}
                    log = folder/item["id"]/"worker.log"
                    output.with_suffix(".worker.log").write_text(log.read_text() if log.exists() else "")
                    artifacts = folder/item["id"]
                    result["artifacts"] = {name: (artifacts/name).stat().st_size
                                           for name in ("policy.npz", "learner.msgpack", "replay/online/index.json",
                                                        "replay/demonstrations/index.json")
                                           if (artifacts/name).is_file()}
                    result["replay_counts"] = {
                        bank: json.loads((artifacts/"replay"/bank/"index.json").read_text())["size"]
                        for bank in ("online", "demonstrations")
                        if (artifacts/"replay"/bank/"index.json").is_file()}
                    if (artifacts/"policy.npz").is_file():
                        with np.load(folder/"initial-policy.npz") as initial, np.load(artifacts/"policy.npz") as final_policy:
                            result["changed_policy_arrays"] = sum(
                                not np.array_equal(initial[k], final_policy[k]) for k in initial.files
                                if not k.startswith("__"))
                    rows = artifacts/"transitions.jsonl"
                    if rows.is_file():
                        records = [json.loads(line) for line in rows.read_text().splitlines()]
                        result["transition_count"] = len(records)
                        if len(records) > 1:
                            times = [r["timestamp"] for r in records if "timestamp" in r]
                            intervals = np.diff(times)
                            result["actor_timing"] = {
                                "effective_hz": (len(times)-1)/(times[-1]-times[0]),
                                "median_step_ms": float(np.median(intervals)*1000),
                                "p95_step_ms": float(np.percentile(intervals, 95)*1000),
                                "max_gap_s": float(np.max(intervals))}
                            first_update = next((s for s in runtime_samples
                                                 if (s.get("learner_steps") or 0) >= 1), None)
                            if first_update:
                                warm = [r["timestamp"] for r in records
                                        if r.get("step_id", -1) > (first_update["environment_steps"] or 0)]
                                if len(warm) > 1:
                                    result["actor_timing"]["after_first_update_hz"] = (
                                        (len(warm)-1)/(warm[-1]-warm[0]))
                    if result["state"] == "updates_complete":
                        assert not final.get("runtime_error"), final.get("runtime_error")
                        assert final["runtime"]["artifacts_saved"]
                        assert result["changed_policy_arrays"] > 0
                        assert len(result["artifacts"]) == 4
                        assert result["replay_counts"]["online"] == result["transition_count"]
                        assert result["replay_counts"]["demonstrations"] == 32
                        assert final["metrics"]["critic_steps"] == final["metrics"]["learner_steps"]*2
                        result["state"] = "passed"
    except Exception as exc:
        result.update(state="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        views.set()
        done.set()
        for thread in threads:
            thread.join(timeout=6)
        result.update(elapsed_s=time.monotonic()-start,
                      peak_used_mib=max((s["used_mib"] for s in gpu_samples), default=0),
                      min_free_mib=min((s["free_mib"] for s in gpu_samples), default=0))
        save()
        print(json.dumps({k: result.get(k) for k in
                          ("state", "error", "peak_used_mib", "min_free_mib",
                           "elapsed_s", "latest_metrics", "actor_timing", "artifacts")}), flush=True)
    return result["state"] == "passed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--publish-interval", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=420)
    parser.add_argument("--output", required=True)
    raise SystemExit(0 if probe(parser.parse_args()) else 1)
