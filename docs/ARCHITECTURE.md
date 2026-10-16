# Architecture - cyclo_intelligence

As-built runtime topology after `docker/container.sh start`.

Visual map: [`cyclo_brain/docs/architecture.html`](../cyclo_brain/docs/architecture.html)

When runtime structure changes, update both
[`cyclo_brain/STRUCTURE.md`](../cyclo_brain/STRUCTURE.md) and the visual map in
`cyclo_brain/docs/`.

## Container Topology

```text
Host
├── cyclo_intelligence container
│   ├── UI / nginx                 Autonomy Studio (Mission Canvas, Action Canvas)
│   ├── supervisor_api             service lifecycle, maps/missions/trees data,
│   │                              Nav2 goals, robot capability checks
│   ├── orchestrator_node          control plane (record / train / infer)
│   ├── bt_node                    behavior-tree engine, started on demand
│   ├── standalone CLI
│   └── cyclo_data
│
├── ai_worker container            SLAM / Nav2 navigation runtime
│
└── policy container per backend
    ├── main-runtime
    └── engine-process
```

## Task Execution

```text
Autonomy Studio (browser)
  │ /api/services/bt_node/start        supervisor_api starts the s6 service
  │ /api/bt/trees, /api/bt/support     saved trees + supported robots
  │ /bt/load_and_run, /bt/set_running  via rosbridge
  ▼
bt_node  ──/task/command──▶ orchestrator_node ──InferenceCommand──▶ policy container
  │
  │ joint / velocity commands
  ▼
Robot
```

`bt_node` and `orchestrator_node` live in the same ROS package but share no
Python; the boundary is enforced by
`orchestrator/tests/bt/test_package_boundary.py`. Mission sequencing
(Nav2 goal per waypoint, then that waypoint's tree) currently runs in the
browser (`orchestrator/ui/src/hooks/useMissionRunner.js`) against the
supervisor's navigation endpoints and the `/bt/*` services. Saved trees are
user data under `CYCLO_BT_TREES_DIR` (default `/workspace/bt/trees`), and the
robots the engine supports are listed once in
`shared/shared/robot_configs/schema.py`.

Policy containers are backend-isolated so each opensource model can own its
Dockerfile, Python dependencies, and upstream submodule.

## Policy Container

```text
UI / orchestrator or standalone CLI
  │
  │ command args or /<backend>/inference_command
  ▼
main-runtime
  ├── ServiceHandler
  ├── SessionState
  ├── InferenceRequester
  ├── ActionChunkProcessor
  ├── ControlLoop
  └── RobotClient command publishers
       │
       │ /cmd_vel, /leader/*/joint_trajectory
       ▼
     Robot

main-runtime
  │
  │ /<backend>/engine_command
  ▼
engine-process
  ├── PolicyLoader
  ├── optional Optimizer
  ├── Preprocessor
  ├── Predictor
  └── RobotClient observation subscribers
```

Main owns session flow and robot command publishing. Engine owns model loading,
sensor/state reads, preprocessing, and inference.

## Key Services

| Service | Owner | Purpose |
|---|---|---|
| `/<backend>/inference_command` | Main | External LOAD/START/PAUSE/RESUME/STOP/UNLOAD |
| `/<backend>/engine_command` | Engine | Internal LOAD_POLICY/GET_ACTION/UNLOAD_POLICY |

`EngineCommand` echoes `seq_id`. Main uses it to discard stale responses after
timeout.

## Code Map

| Concern | Source |
|---|---|
| Target structure | [`cyclo_brain/STRUCTURE.md`](../cyclo_brain/STRUCTURE.md) |
| Common runtime | [`cyclo_brain/policy/common/runtime/`](../cyclo_brain/policy/common/runtime/) |
| Main process | [`main_runtime`](../cyclo_brain/policy/common/runtime/main_runtime/) |
| Engine process | [`engine_process`](../cyclo_brain/policy/common/runtime/engine_process/) |
| LeRobot engine | [`lerobot_engine`](../cyclo_brain/policy/lerobot/lerobot_engine/) |
| GR00T engine | [`groot_engine`](../cyclo_brain/policy/groot/groot_engine/) |
| Robot client | [`robot_client`](../cyclo_brain/sdk/robot_client/) |
| Action processing | [`action_chunk_processing`](../cyclo_brain/sdk/action_chunk_processing/) |
| Behavior-tree engine | [`orchestrator/orchestrator/bt/`](../orchestrator/orchestrator/bt/) |
| Supervisor API | [`docker/supervisor_api/`](../docker/supervisor_api/) |
| Interfaces | [`interfaces`](../interfaces/) |
| Compose | [`docker/docker-compose.yml`](../docker/docker-compose.yml) |

## HIL-SERL Lab: pre-robot Mock runtime

The HIL-SERL Lab rail entry works without ROS. It supports configuration-only
Preview sessions and an executable CPU Mock for AI Worker / NIST peg-insertion
integration. Robot models remain configuration targets, not validated embodiments.

```text
React -> nginx /api/hilserl -> supervisor API -> SQLite session controls
                                      -> isolated hilserl_runtime.py process
                                         Actor thread / Mock XYZ environment
                                            -> Agentlace loopback TCP
                                         Replay -> SAC learner thread
                                            -> Agentlace weight broadcast -> Actor
```

Actor and learner are separate threads within one child process, communicating
through real Agentlace TrainerClient/TrainerServer sockets. They are not yet
separate deployment services. Upstream Agentlace is pinned in
`docker/supervisor_api/requirements-hilserl.txt`; local subclasses restrict its
pickle-based sockets to ephemeral loopback ports. Both Dockerfiles install the
requirements. Existing containers can install that file with `python3 -m pip
install -r /opt/supervisor_api/requirements-hilserl.txt`.

### Use without a robot

1. Open `http://localhost:7080`, choose **HIL-SERL Lab**.
2. Select **Mock · no robot required**, save configuration, and create a Mock
   training session. Default batch size 256 needs about 26 seconds of experience
   at 10 Hz before optimizer updates begin.
3. Start the runtime. Monitor environment / learner steps, consumed policy
   version, replay size, human steps, loss, success rate, and the peg XY display.
4. Take over, then click XYZ buttons for bounded action pulses (0.3 seconds, or one control period plus 50 ms at low step rates).
   With no fresh pulse the human action becomes zero. Release returns to policy.
5. Pause/resume, or label success/failure. Labels pause; reset clears the label
   before resume. Natural termination/time limits reset the Mock automatically.
6. Stop to flush artifacts. Copy the displayed policy path into checkpoint path
   to create an evaluation session. Optionally load the saved replay as a dataset.

The environment has four state observations (XYZ / hold progress) and three
normalized translation actions. It applies a geometric plate constraint and a
hold/depth success test. Rotation, camera/joint topics, embodiment and board assets
are reserved configuration. **This is not Isaac Sim physics, a calibrated NIST
board, or a reproduction of the upstream pixel-based HIL-SERL algorithm.**

The learner uses PyTorch SAC, twin critics, target averaging, and fixed entropy
coefficient 0.2. When intervention data exists, half each training batch is sampled
from it; the remainder comes from the online replay. There is no generated expert
or synthetic success metric. Human labels append explicit `label_only` absorbing
transitions with zero action and +/-10 reward. Environment rewards use geometric
distance plus success reward. Human reward mode uses zero reward until a label;
classifier mode remains blocked. Evaluation uses deterministic policy actions and
never performs optimizer steps. Reported success rate includes human-labeled
outcomes; it does not demonstrate autonomous task mastery.

### Persistence and lifecycle

`CYCLO_HILSERL_DATA_DIR` defaults to `/workspace/hilserl`. SQLite persists saved
configurations, immutable session snapshots, state and the latest 200 command
events. Revisions reject stale writes. Run **one supervisor worker**: its child
process registry owns runtime lifecycles. Only one unfinished/live session is
allowed. Commands wait for a runtime acknowledgement before another command;
Stop remains available while acknowledgement is pending. Metrics do not change
command revisions. Pause acknowledges after the in-flight optimizer step finishes.

Supervisor shutdown requests Stop and waits for artifact flushing. Abrupt parent
loss makes the worker exit; terminated or unowned sessions appear stopped with an
error instead of being silently resumed. A new session can warm-start policy and
replay. This is not exact optimizer/RNG resume. `learner.pt` contains optimizer and
critic state for inspection, but the UI deliberately loads only non-pickle `.npz`
policy/dataset artifacts. Inputs must reside under the HIL data directory; the
worker validates dataset shapes, action bounds and finite values before use.

Per-session outputs:

- `transitions.jsonl`: incremental experience, executed/proposed actions, human
  flags, policy version, episode and step metadata (label events are marked).
- `policy.npz`: final learner actor weights, loadable for evaluation/warm start.
- `replay.npz`: bounded final replay, loadable for subsequent training.
- `learner.pt`: final PyTorch learner/optimizer snapshot.
- `runtime.json`, `worker.log`: atomic telemetry and diagnostics.

Final snapshots are written on graceful Stop; a forced kill may leave only the
incremental transition log. Paths in the UI are output destinations until Stop
finishes. Runtime logs/datasets are local and require normal storage management.

| API relative to `/api/hilserl` | Contract |
|---|---|
| `GET /workspace` | Config/revision, readiness, sessions with runtime telemetry |
| `PUT /config` | `{expected_revision, config}` |
| `POST /sessions` | `{expected_config_revision, mode}`; preview/train/evaluate |
| `GET /sessions/{id}` | State, command revision, metrics, events, acknowledgement |
| `POST /sessions/{id}/commands` | `{expected_revision, action}` |
| `POST /sessions/{id}/action` | `{expected_revision, value: [x,y,z]}`; bounded human pulse |

### Remaining robot / Isaac Sim boundary

Real-robot and Isaac Sim train/evaluate requests remain blocked. Implement their
reset/step adapters, calibrated frames/units, synchronized observations, contact
and reward validation, reset safety and robot command arbitration before enabling
them. Replace the state policy with the intended visual HIL-SERL learner as needed;
this Mock is a transport/control/data integration fixture. Keep real-time robot
stopping in the Actor/hardware controller, not browser polling. Define the SH5 hand
mapping explicitly rather than treating it as a binary gripper.

Tests: `python3 -m pytest docker/supervisor_api/test_hilserl.py
docker/supervisor_api/test_hilserl_runtime.py docker/supervisor_api/test_app.py`.
UI: from `orchestrator/ui`, run `CI=true npm test -- --watchAll=false --runInBand
--runTestsByPath src/pages/HilSerlPage.test.js src/App.test.js` and `npm run build`.
The integration test uses isolated storage and actual Agentlace transport,
optimizer updates, manual actions, pause, artifact reload and evaluation.
