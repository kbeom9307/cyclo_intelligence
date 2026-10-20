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

## HIL-SERL Lab: Mock and Isaac NIST runtime

The HIL-SERL Lab rail entry works without ROS. It supports configuration-only
Preview sessions, a CPU Mock, and the actual Isaac Lab FFW-SG2 NIST task through
a leased simulator bridge. Physical robot execution remains disabled.

```text
React -> nginx /api/hilserl -> supervisor API -> SQLite session controls
                                      -> isolated hilserl_runtime.py process
                                         Actor thread / Mock XYZ or IsaacNistEnv HTTP client
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

### Mock use without a robot

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
worker validates dataset shapes, action bounds and finite values before use. Policy
and replay `.npz` files carry `__schema__`; Mock (4/3) and Isaac (39/7) artifacts
cannot be mixed. Legacy artifacts without this tag are treated as Mock.

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
| `POST /sessions/{id}/action` | `{expected_revision, value}`; 3 Mock or 7 Isaac action values |

### Isaac Worker / NIST Board integration

Simulator source: `/home/robotis/kb_ws/src/isaac_worker/cyclo_lab`, mounted at
`/workspace/cyclo_lab` in the `cyclo_lab` container. Its
`scripts/nist_board/run_hilserl_bridge.py` launches the existing
`Cyclo-NIST-Board-FFW-SG2-v0` task and serves a local JSON API. Start it separately
from the previous scene viewer (one scene process is enough):

```bash
docker exec -d -w /workspace/cyclo_lab -e PYTHONUNBUFFERED=1 cyclo_lab \
  bash -c 'exec /isaac-sim/python.sh scripts/nist_board/run_hilserl_bridge.py --headless > /tmp/hilserl-bridge.log 2>&1'
curl http://127.0.0.1:7125/health
```

The bridge must be started again after its container restarts. GUI use is possible
by running the same script without `--headless` in the configured display session.
Do not also launch `run_nist_board.py` for the same task. The default bridge enables a 640×480 spectator camera for the Lab live view.
`--no_visualization` restores physics-only operation. The head and both eye-in-hand cameras render RGB at 320×180; GPU
recorder buffering remains disabled. Both current containers use
host networking; all bridge sockets bind loopback. The supervisor-owned
`CYCLO_HILSERL_ISAAC_URL` defaults to `http://127.0.0.1:7125` and only accepts
loopback HTTP URLs. No ROS or DDS robot commands are sent.

In HIL-SERL Lab, stop the previous session, click **Use Isaac NIST settings**, choose
a target peg, save, and create/start an **Isaac Sim training session**. This selects
FFW-SG2 and the scene's 20 Hz control rate. Translation is limited to at most 0.02 m
and rotation to 0.15 rad per action. The default batch size 256 begins learning after
about 13 simulated seconds. Use smaller step limits for initial demonstrations.

The 39-value state is: chosen arm joint position(7), velocity(7), EE board-relative
position(3), world quaternion wxyz(4), peg board-relative position(3), world
quaternion(4), peg linear/angular velocity(6), assembled target board-relative
position(3), gripper joint position(1), success hold progress(1). The seven actions
are normalized XYZ and rotation-vector deltas in the articulation-root frame,
then a gripper joint delta (positive closes; 0.05 rad scale). Cartesian deltas are
converted to the existing absolute-pose IK terms. The other arm holds its reset
pose, lift/head hold defaults. Browser manual input exposes TCP-local translation, rotation and
gripper pulses. Zero input preserves the commanded gripper target, maintaining closing force.

The adapter advances actual PhysX through `env.step`; it never attaches the peg to
the gripper artificially. Success uses the target part's assembled asset pose,
XYZ tolerance, upright orientation (and square peg yaw), speed <0.05 m/s, and the
configured hold time. `success_depth_m` applies only to Mock; Isaac's insertion
height comes from its asset. Reward shapes EE-to-peg and peg-to-target distances
plus a small action penalty and terminal +/-10. A dropped peg is failure, never
success. Human reward mode suppresses automatic success reward but still terminates
on a dropped peg. Classifier rewards and visual policies are not implemented.

Automatic Isaac manager terminations are disabled so the adapter can return the
true final observation. Actor records terminal/truncated data before resetting.
Reset restores the complete kit scene and settles it for ten ticks. By default,
`initialization=grasped` places the selected peg in the selected hand with a reset-only
virtual fixture during closure, then removes that fixture and tests free-physics
retention while holding/lifting. Failed retention rejects initialization; these
ticks are excluded from replay. `initialization=kit` keeps the peg in its tray.
Each episode reset repeats the selected initialization. A paused session advances no physics and keeps its lease
with a heartbeat. HTTP handler threads only enqueue work; every scene mutation
runs on the simulation thread. The bridge grants one 30-second owner lease,
rejects out-of-order commands, and caches the last response so the same sequence
cannot step twice. Client transport failure stops the Actor instead of retrying an
ambiguous action. Stop releases the lease; after an abrupt worker loss it expires.

The existing v0 NIST assets and their clearance/contact limitations are preserved.
State-policy learning and replay transport were smoke-tested; successful autonomous
insertion and policy convergence are **not** established. Full upstream visual
HIL-SERL, camera observations, grasp demonstrations, and hardware arbitration are
separate follow-up work. Keep real-time stopping in the physical Actor/controller
when adding a hardware adapter.

Tests: `python3 -m pytest docker/supervisor_api/test_hilserl.py
docker/supervisor_api/test_hilserl_runtime.py docker/supervisor_api/test_app.py`.
UI: from `orchestrator/ui`, run `CI=true npm test -- --watchAll=false --runInBand
--runTestsByPath src/pages/HilSerlPage.test.js src/App.test.js` and `npm run build`.
The integration test uses isolated storage and actual Agentlace transport,
optimizer updates, manual actions, pause, artifact reload and evaluation.

Run the actual physics/learning smoke with a ready bridge:
`CYCLO_TEST_ISAAC=1 python3 -m pytest docker/supervisor_api/test_hilserl_isaac.py -q -s`.
It uses isolated storage, checks 39/7 observations/actions, optimizer/weight updates,
manual gripper input, time-limit reset, pause/reset/resume, checkpoint/replay reload,
and evaluation without learner updates. Without the flag, the GPU test is skipped.
The simulator repository's `scripts/nist_board/test_hilserl_transport.py` uses only
standard Python to verify ownership, sequencing and main-thread dispatch.

### Isaac physical preview

For `environment=isaac_sim`, **Start preview** now acquires the same bridge lease
as training, performs the configured grasp/kit reset, and runs real physics with
manual seven-axis controls. Input pulses automatically activate intervention; without an active pulse,
commands are zero and the gripper preserves its target. **Release input** ends
intervention; Preview continues with zero commands. Preview does not run
optimizer updates or load saved policies/datasets. Pause freezes physics; reset
repeats initialization. Bridge readiness and worker acknowledgements apply to
preview too, and startup failures are visible in the session panel. Other preview
environments retain the workflow-only behavior and never command real hardware.

### Isaac live visualization

The Lab displays a real renderer image on setup and above the session title across the full page width. Overview, head, left wrist and right wrist share one row (horizontal scrolling
on narrow windows). Four views show the overall AI Worker/NIST scene, head camera and both eye-in-hand
cameras. These feeds are separate from the
39-value policy observation. `run_hilserl_bridge.py --headless` enables rendering
without a desktop window. Optional external texture loading does not block reset (`wait_for_textures=False`).

The simulation thread renders/encodes JPEGs at up to 4 fps per camera when a viewer is active.
HTTP threads only read the latest cached image (`GET /frame.jpg?camera=spectator|head|left_wrist|right_wrist`), and the supervisor
proxies it as `GET /api/hilserl/isaac/frame?camera=...`. Responses use `Cache-Control: no-store`,
frame ID and age headers. Frames older than three seconds return 503. The UI keeps
one request in flight per view, retries failures, frees replaced blobs, suspends requests in
hidden tabs and offers **Pause video / Resume video / Fullscreen**. Video pause does
not pause training. Rendering a paused scene does not advance physics. The feed is
the current scene, never a replay of a selected historical session.

Changing renderer mode requires restarting the bridge. Stop an active session and
save its checkpoint/replay first; a restarted bridge cannot retain its owner lease.
Use the saved artifacts to warm-start a new session if needed. Visualization does
not change the learner to an image-based policy.


### Keyboard intervention and prepared episodes

During a running session, nonzero manual input automatically takes control;
zero input or pulse expiry returns control to the policy (Preview holds still).
No takeover toggle is required. W/S, A/D and R/F control TCP-local XYZ;
Q/E control roll, arrow up/down pitch, arrow left/right yaw, and Z/X close/open
the gripper. TCP translations and rotations are transformed into the existing
fixed-root action frame before stepping and recording replay. Space/Escape clear input. Key release, window blur, hidden tabs,
editing a field, leaving the session view, and relinquishing control stop keyboard
pulses. The client permits only one action request in flight; backend pulses expire.
The progress bar uses `runtime.episode_step / max_episode_steps`, not cumulative
training steps. Episode resets clear the counter; preparation ticks are excluded.

`initialization=near_hole` is the default. At reset, bounded offline IK places
robot joints directly at the pre-insertion pose, with the peg bottom 15 mm above
the board and 20 mm to the working side of opposite-side holes. The lift/head
and left observation arm use the same camera poses as pick/place. A reset-only
fixture holds the peg while the fingers close; it is removed for a free-physics
retention check before episode step 0. No tray pickup or transfer trajectory is
executed. Every episode reset repeats this placement, and failed retention blocks
startup. Policy steps have no peg fixture or attachment.

`initialization=pick_place` restores the authored tray scene, moves the selected
arm to a vertical gripper pose, picks the selected peg using contact physics,
lifts it, lowers the torso lift to -0.35 m, and transfers it near the selected
hole. The peg bottom waits 15 mm above the board surface; opposite-side holes
use a 20 mm lateral offset toward the working arm. The lower transfer waypoint
is 0.85 m high to stay within the arm limits. Head pitch is 0.69 rad; for
right-arm tasks the left wrist camera views the jig from the left at approximately
45 degrees downward. The left observation camera is lowered a further 100 mm and aimed at the
peg/hole contact region `(0.37445, 0.055176, 0.766)` m. Board Y is aligned with
image horizontal to correct the tilt, retaining the earlier quarter-turn image
orientation. The observer TCP is `(0.4280936, 0.2159192, 0.8118361)` m with
world quaternion wxyz `(0.7058411, -0.5091540, 0.0422896, -0.4906753)`.
Lift, head and observer-arm targets remain held during
the episode. The episode starts after pose and retention checks. No peg teleport or attachment is used for this mode.
`grasped` remains the older reset-fixture initialization, and `kit` starts in the tray.
Initialization requests have a bounded 180-second bridge timeout to accommodate
multi-waypoint motion; ordinary step requests retain their short timeout.

### IK motion limits

During episode steps, the selected arm checks every physics substep before
sending joint targets. An outward target inside the 0.003 rad joint-limit margin,
a joint jump above 0.12 rad, non-finite IK, or a predicted sideways/reversed
Cartesian response blocks the entire arm command. Individual joints are not
clipped independently. The blocked joint pose remains held for repeated input
in the same direction; another direction can retreat from the limit. Translation
commands retain the commanded tool orientation to prevent accumulated orientation
drift. The Lab displays the blocking joint/reason from environment info.

### Board placement for right-arm reach

The NIST board defaults to world position `(0.45, -0.10, 0.756)` m, a 100 mm
shift toward the robot’s right (-Y). `NIST_BOARD_Y` overrides the lateral offset;
`NIST_BOARD_X` overrides forward distance. Board geometry, kit parts, assembled
goals, board-relative observations and near-hole reset positions all derive
from `BOARD_POS`. Camera arm pose and joint-limit protections remain unchanged.

The overview camera uses eye `(1.02, -0.91, 1.22)` m and target
`(0.4, -0.08, 0.83)` m for a closer view of the jig and hands.

HIL-SERL Lab camera quality is selected independently per camera and persisted
in browser local storage. The frame proxy forwards `quality=low|standard|high`
and the bridge returns `X-Frame-Quality` and `X-Frame-Resolution`. Overview
profiles are 320×240 / 640×480 / 960×720; head and wrist profiles are
320×180 / 480×270 / 640×360. Native rendering uses the highest resolution.

Episode/peg telemetry and human intervention occupy two desktop columns
(stacked below 800 px). The policy table reports the latest normalized root-frame
proposal separately from the selected command, source and reward. Preview
disables policy and reports zero proposals; guards may reject a selected command.
Held keyboard keys are highlighted together and cleared on release, blur, or
control cancellation. Episode reset clears the previous action telemetry.

Isaac zero-action handling: when the six Cartesian action components are zero,
the working arm latches its current joint positions once and keeps that target
until nonzero Cartesian input arrives. Gripper-only commands preserve the arm
hold. The observation arm also holds a fixed joint target after initialization.
`environment_info.arm_joint_hold` exposes this state in the policy/action panel.
This prevents pose drift from repeated zero-delta IK targets based on measured
poses. It uses ordinary physics joint drives, so small settling remains possible.
The left camera uses world-up orientation and a 45-degree downward optical axis
from `(0.37445, 0.255176, 0.966)` toward `(0.37445, 0.055176, 0.766)` metres.

The `truncate` episode command is available while running or paused with an
unfinished episode. It clears manual input, records outcome `truncated`, pauses,
and requires reset before resume. Its boundary record has reward 0, terminal
false, truncated true (no success/failure label). Replay exports retain the
truncated flag; legacy imports lacking it default to false.
Keyboard input is normalized per-axis ±1 in TCP coordinates. At the current
5 mm / 0.05 rad limits it requests those increments per simulation step; the
gripper target increments by 0.05 rad per step. Keys refresh the command every
100 ms while held; the runtime consumes it at up to the configured control Hz.
Actual displacement is constrained by IK, guards, physics and contact.

### Fixed-gripper HIL environment (current)

Isaac HIL actions now have six dimensions: XYZ translation and rotation.
Schema: `nist-sg2-state39-delta6-fixedgripper-v2`. The 39-state observation
retains physical gripper state; the policy cannot command opening or closing.
Reset prepares the grasp, then holds the gripper position target throughout
the episode. Finger physics and contact remain active; the peg is not welded
to the hand. Z/X keyboard input and gripper pulse controls are removed.
Seven-dimensional actions are rejected. Existing seven-action checkpoints
and replay files have a different schema and cannot be loaded into this
environment; use a new training session and compatible demonstrations.

### Pixel learning, classifier data and demonstrations

Isaac sessions now default to `observation_mode=pixels`. The bridge captures
Head, Left wrist and Right wrist RGB synchronously after acquire/reset/step
and explicit capture operations. Each observation contains three 64×64 RGB
views (`uint8`, CHW 9×64×64), a monotonic capture sequence and timestamp.
Capture forces rendering without advancing physics and does not reuse viewer
JPEG caches. Viewer Low/Standard/High selection does not alter learning input.
No-visualization bridges cannot start pixel sessions.

Policy and both critics have trainable CNN encoders. Inputs are images plus
22 robot proprioception values (7 joint positions, 7 velocities, TCP position
and quaternion, fixed-gripper position). Peg pose, peg velocities, hole target
and success hold progress are excluded from policy/critic input; simulator
state remains available for telemetry and environment rewards. The implementation
is local PyTorch pixel SAC with Agentlace replay/weight transport, not an
upstream HIL-SERL/DrQ reproduction. Default pixel batch size is 32 and replay
capacity 2,048; both have separate bounded settings. Learning uses CPU to leave
the simulation GPU available. Schema: `nist-sg2-rgb3x64-proprio22-delta6-fixed-v3`.
Legacy state/7-action artifacts cannot be loaded into this pixel schema.

Workflow in Lab:

1. Save Isaac pixel settings, stop any existing session, then choose **Get
   classifier data** and Start. Manually move through success/failure states.
   Each environment step saves the selected classifier camera observations: hold
   Space for success, otherwise failure. Optional **Space key toggle mode** switches
   labels on each press; Esc, blur, tab change and pause clear the success input.
   `classifier_success_target` defaults to 200 (editable 1–10,000). Reaching this
   session-wide success count automatically stops and saves the collection.
   Episode boundaries still pause for reset; the total sample cap is 10,000.
   Explicit capture buttons remain available while paused for unlabeled states.
2. In the separate **Train classifier** tab, **Data management** groups collections
   by environment, robot, arm and target, then filters compatible camera sets.
   Select stopped datasets; the selected positive/negative totals are shown before
   **Train classifier**. At least 10 unique samples per class are required.
   Training and validation must both contain both classes from disjoint episodes;
   insufficient diversity fails with an explanation. The job shows epoch, loss,
   balanced accuracy, precision/recall, confusion counts and split counts.
   The best validation balanced-accuracy epoch is saved (validation loss breaks ties),
   rather than unconditionally saving the last epoch. Poor generalization is flagged. **Use classifier** fills
   the setup fields; Save configuration to use it. The reward source `classifier`
   runs this image CNN online, requires a configured probability threshold for
   consecutive steps, and replaces simulator success reward/termination.
   Simulator failure handling remains active. Metrics are validation results,
   not proof of general insertion performance.
3. **Get demo** starts a manual-only, no-learning/no-policy session. Image/state/
   action/next-image transitions are buffered by episode. Pause or label an
   outcome; **Save successful demo** saves the actual episode with its last
   transition labeled terminal. **Discard pending demo** rejects an attempt.
   Reset is blocked until a pending episode is saved/discarded. Natural endings
   pause for review rather than auto-reset. Stop exports saved episodes as
   `demonstrations.npz`; an unfinished episode is kept separately as
   `pending-demo.npz` and is not selected as an approved demonstration.
4. **Use demonstrations** fills the dataset path. Save settings, create a
   training session, and Start. CNN SAC trains from RGB/proprioception, sampling
   human demonstration transitions alongside new online experience.

Collection modes never update a policy and do not export random policy
checkpoints. Classifier artifacts contain schema/task metadata and weights in
`classifier.npz`; a job retains request, split and metrics files. Pixel replay
and demos store `images` and `next_images` alongside states/actions/rewards and
terminal/truncated flags. Artifacts are validated against schema, robot, arm
and target task. Dataset size limits bound memory; classifier collection is
limited to 10,000 images/session and demos to 4,096 saved transitions/session.
Collection sessions use the smaller of `max_episode_steps` and the separate
`collection_episode_steps` setting (default/max 1,000). Training episode limits
are preserved; this bounds the pending image-episode buffer.
No user dataset or classifier is automatically fabricated by validation tests;
synthetic fixtures and integration-test labels remain in temporary test folders.

### Lab dataset management and visualization

`observation_cameras` and `classifier_cameras` independently select head/left/right
wrist images. Excluded views are zeroed in the stable nine-channel tensor. Saved
artifacts record camera selections and reject incompatible loads; legacy files
without this metadata require all three cameras. Overview remains visualization-only.

Dataset and classifier-job delete controls require explicit confirmation. Running
collections, datasets used by active training, and artifacts referenced by saved
configuration or active sessions cannot be deleted. Deleting data preserves session
history and trained classifiers. Data/model files stay under `/workspace/hilserl`
(host `docker/workspace/hilserl`) and are not committed to Git.

Overview supports drag-to-crop, Reset crop and fullscreen. Ultra uses a native
1920×1440 render with JPEG quality 98 and no chroma subsampling; other robot views
remain 640×360 maximum. Preview quality/cropping never changes the learner's 64×64
camera inputs or advances physics.
