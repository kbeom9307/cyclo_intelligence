# cyclo_brain — Target Structure

`cyclo_brain` consists of **two Python processes**.
Each process is split internally into multiple classes/modules.

- Visual map: [`cyclo_brain/docs/architecture.html`](docs/architecture.html)
- Rule: when the runtime structure changes, update this file as the textual reference and `cyclo_brain/docs/architecture.html` as the visual reference.

- **Main process**: coordinates services, sessions, command publishing, and the control loop.
- **Engine process**: owns model loading, inference execution, and inference-time sensor/state subscriptions.

---

## HIL-SERL Mock / Isaac NIST runtime

HIL-SERL Lab uses the React UI, `docker/supervisor_api/hilserl.py` control plane,
and `hilserl_runtime.py` CPU Actor/SAC learner with Agentlace replay and weights.
`hilserl_isaac.py` connects to the FFW-SG2 NIST physics bridge in `cyclo_lab` using
sequenced JSON requests and exclusive ownership. Mock uses 4/3 observation/action
dimensions; Isaac pixel mode uses three RGB views and 22 proprioception values
with six Cartesian actions and a fixed gripper (legacy state mode: 39/6). Policy/replay
artifacts carry their schema and camera selection. The local pixel SAC learner
is not an upstream HIL-SERL/DrQ reproduction; hardware remains blocked. See
[`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md#hil-serl-lab-mock-and-isaac-nist-runtime).

## 1. Big Picture

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ Host / cyclo_intelligence                                                    │
│                                                                              │
│  UI / Orchestrator or Standalone CLI                                         │
│      │                                                                       │
│      │  same command shape: args or /<backend>/inference_command             │
│      │  LOAD / START / PAUSE / RESUME / STOP / UNLOAD                        │
│      ▼                                                                       │
│  External ROS2 / Zenoh                                                       │
└──────┬───────────────────────────────────────────────────────────────────────┘
       │
       │ service call
       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Policy Container: <backend>_server                                           │
│                                                                              │
│  Process 1: Main process                                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ main_runtime package                                                   │  │
│  │ one Python process, multiple classes/modules                           │  │
│  │                                                                        │  │
│  │  ┌───────────────────────┐       ┌──────────────────────────────────┐  │  │
│  │  │ ServiceHandler        │       │ SessionState                     │  │  │
│  │  │                       │       │                                  │  │  │
│  │  │ - LOAD                │──────▶│ - unloaded / loaded / running    │  │  │
│  │  │ - START / PAUSE       │       │ - paused / stopped               │  │  │
│  │  │ - RESUME / STOP       │       │ - gate inference + publish       │  │  │
│  │  │ - UNLOAD              │       └──────────────────────────────────┘  │  │
│  │  └───────────────────────┘                                             │  │
│  │                                                                        │  │
│  │  ┌───────────────────────┐       ┌──────────────────────────────────┐  │  │
│  │  │ RobotClient           │       │ InferenceRequester               │  │  │
│  │  │                       │       │                                  │  │  │
│  │  │ - publish robot cmds  │       │ - request model load             │  │  │
│  │  │ - command topic setup │       │ - request one inference step     │  │  │
│  │  │ - Main uses publish   │       │ - receive action list            │  │  │
│  │  └───────────▲───────────┘       └────────────────┬─────────────────┘  │  │
│  │              │                                   │ action_list         │  │
│  │              │ publish_action                     ▼                    │  │
│  │  ┌───────────┴───────────┐       ┌──────────────────────────────────┐  │  │
│  │  │ ControlLoop           │◀──────│ ActionChunkProcessor             │  │  │
│  │  │                       │ pop   │                                  │  │  │
│  │  │ - timer-like loop     │       │ - action list buffer             │  │  │
│  │  │ - one action per tick │       │ - optional post-processing       │  │  │
│  │  │ - cadence follows ACP │       │ - pop one action per tick        │  │  │
│  │  └───────────────────────┘       └──────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                  │                                           │
│                                  │ LOAD_POLICY / GET_ACTION / UNLOAD_POLICY  │
│                                  ▼                                           │
│  Process 2: Engine process                                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ <backend>Engine                                                        │  │
│  │                                                                        │  │
│  │ ┌─────────────────────┐  ┌───────────────────────────────────────────┐ │  │
│  │ │ PolicyLoader         │  │ Optimizer                                │ │  │
│  │ │ - load policy        │  │ - optional area                          │ │  │
│  │ │ - weights/processors │  │ - TensorRT / GPU / runtime optimization  │ │  │
│  │ └─────────────────────┘  └───────────────────────────────────────────┘ │  │
│  │ ┌─────────────────────┐  ┌───────────────────────────────────────────┐ │  │
│  │ │ Preprocessor         │  │ Predictor                                 │ │  │
│  │ │ - use RobotClient    │  │ - run inference once per request          │ │  │
│  │ │ - build model input  │  │ - return action list (T, D)               │ │  │
│  │ └─────────────────────┘  └───────────────────────────────────────────┘ │  │
│  │                                                                        │  │
│  │ Engine uses RobotClient for sensor/state topics.                       │  │
│  │ Engine never publishes robot commands.                                 │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       │ RobotClient publish
                                       │ /cmd_vel
                                       │ /leader/*/joint_trajectory
                                       ▼
                                  ┌───────────┐
                                  │   Robot   │
                                  └───────────┘
```

---

## 2. Target Directory Shape

```text
cyclo_brain/
├── sdk/
│   ├── robot_client/                  # robot topic client for command or observation
│   ├── action_chunk_processing/       # action_list post-processing
│   └── zenoh_ros2_sdk/                # ROS2-over-Zenoh transport SDK
│
└── policy/
    ├── common/
    │   ├── runtime/
    │   │   ├── engine.py              # InferenceEngine ABC
    │   │   ├── main_runtime/          # Process 1 package
    │   │   │   ├── main.py            # starts one Main Python process
    │   │   │   ├── service_handler.py # ServiceHandler class
    │   │   │   ├── session_state.py   # SessionState class
    │   │   │   ├── inference_requester.py
    │   │   │   ├── control_loop.py    # ControlLoop class, uses RobotClient
    │   │   │   └── zenoh_client.py    # internal EngineCommand service client
    │   │   ├── engine_process/        # Process 2 package
    │   │   │   ├── worker.py
    │   │   │   └── protocol.py
    │   └── s6-services/
    │       ├── main-runtime/
    │       └── engine-process/
    │
    ├── lerobot/
    │   ├── Dockerfile.{arm64,amd64}
    │   ├── lerobot/
    │   └── lerobot_engine/
    │       ├── engine.py
    │       ├── loading.py                # PolicyLoader convention
    │       ├── optimization.py           # optional TensorRT/GPU/runtime optimization
    │       ├── io_mapping.py
    │       ├── preprocessing.py          # Preprocessor convention
    │       ├── prediction.py             # Predictor convention
    │       └── constants.py
    │
    └── groot/
        ├── Dockerfile.{arm64,amd64}
        ├── Isaac-GR00T/
        └── groot_engine/
            ├── engine.py
            ├── loading.py                # PolicyLoader convention
            ├── optimization.py           # optional TensorRT GPU optimizer
            ├── io_mapping.py
            ├── preprocessing.py
            └── prediction.py
```

---

## 3. Runtime Data Flow

```text
1. LOAD

External or standalone CLI
  └─ InferenceCommand(LOAD, model_path, robot_type, task_instruction, publish_to_robot)
       └─▶ Main process
             ├─ RobotClient.configure(robot_type) for command + preview publish
             ├─ Engine process.load_policy(model_path, robot_type)
             │    └─ RobotClient.configure(robot_type) for observation
             ├─ ActionChunkProcessor.clear()
             └─ session = loaded, output mode = simulation or robot


2. START + RUN

External
  └─ InferenceCommand(START, publish_to_robot)
       └─▶ Main process
             └─ session = running, action buffer cleared if output mode changed

Main control loop
  ├─ action = ActionChunkProcessor.pop_action()
  ├─ if action exists:
  │    ├─ RobotClient.publish_action_preview(action) to /inference/trajectory_preview
  │    └─ if publish_to_robot:
  │          └─ RobotClient.publish_action(action)
  ├─ if action is empty:
  │    └─ publish nothing; never repeat the previous action
  │
  └─ if buffer is low and no request is in flight:
       ├─ Engine process.get_action(task_instruction)
       │    ├─ RobotClient.get_observation()
       │    ├─ run policy inference
       │    └─ return action_list
       └─ ActionChunkProcessor.push_actions(action_list)

Control loop cadence
  ├─ post-processing enabled:
  │    ├─ ActionChunkProcessor converts model action list to control actions
  │    ├─ matching / RTC aligner / future smoothing can run here
  │    ├─ example: 16 model actions → 100 control actions
  │    └─ ControlLoop runs at processed output cadence, normally 100Hz
  │
  └─ post-processing disabled:
       ├─ ActionChunkProcessor buffers raw action list as-is
       └─ ControlLoop runs at model/action-list cadence, not forced to 100Hz


3. PAUSE / RESUME / STOP / UNLOAD

PAUSE
  External ─▶ Main ─▶ session = paused
  ActionChunkProcessor.clear()
  in-flight action requests are discarded by generation id

RESUME
  External ─▶ Main ─▶ session = running, output mode may change
  ActionChunkProcessor.clear() if output mode changed
  control loop requests fresh actions before publishing again

STOP
  External ─▶ Main ─▶ session = stopped
  ActionChunkProcessor.clear()

UNLOAD
  External ─▶ Main
     ├─ Engine process.cleanup()
     │    └─ RobotClient.close() for observation
     ├─ RobotClient.close() for command publish
     ├─ ActionChunkProcessor.clear()
     └─ session = unloaded
```

---

## 4. Responsibility Boundary

| Area | Owner |
|---|---|
| External command service | Main process |
| Session state | Main process |
| Control loop | Main process |
| Robot sensor/state input | Engine process uses RobotClient |
| Robot command output | Main process uses RobotClient |
| Action list buffer/post-processing | ActionChunkProcessor |
| Model load/inference | Engine process |
| Optional optimization | Engine process optimizer class |
| Backend-specific policy code | `<backend>_engine/` |

---

## 5. Stable Contracts

| Contract | Shape |
|---|---|
| External service | `/<backend>/inference_command` |
| Internal engine service | `/<backend>/engine_command` via `zenoh_ros2_sdk` service |
| Main → Engine | `LOAD_POLICY`, `GET_ACTION`, `UNLOAD_POLICY` |
| Main → RobotClient | `configure`, `publish_action`, `close` |
| Engine → RobotClient | `configure`, `get_observation`, `close` |
| Engine output | `action_list` shaped `(T, D)` |
| Processor output | one action vector per control tick |
| Runtime processes | `main-runtime`, `engine-process` |
| Main internal modules | classes inside one Main process, not extra processes |

### 5.1 Internal engine service

```text
GET_ACTION request:
  seq_id
  task_instruction

GET_ACTION response:
  seq_id
  success
  message
  action_list
  chunk_size
  action_dim

Main rules:
  - one GET_ACTION in-flight by default
  - timeout is configurable per backend/model/deployment
  - timeout means "response not received in time", not "inference failed"
  - late/stale responses are discarded by seq_id
  - only the latest accepted response enters ActionChunkProcessor
```

### 5.2 Timeout policy

```text
LOAD_POLICY timeout:
  backend-specific and can be long
  includes model load, processor load, optimizer build/load

GET_ACTION timeout:
  runtime safety timeout
  configurable; default is a fallback, not a performance guarantee
  should account for model size and user hardware
```

### 5.3 Backend integration contract

```text
policy/<backend>/
├── Dockerfile.{arm64,amd64}        # per-opensource dependency isolation
├── <opensource-submodule>/         # git submodule
└── <backend>_engine/
    ├── engine.py                   # implements InferenceEngine ABC
    ├── loading.py                  # PolicyLoader convention
    ├── optimization.py             # optional TensorRT/GPU/runtime optimization
    ├── io_mapping.py               # robot/model key mapping
    ├── preprocessing.py            # RobotClient observation → model input
    └── prediction.py               # model input → action_list
```

The `InferenceEngine` ABC in `common/runtime/engine.py` is the required process-boundary contract.
`loading.py`, `optimization.py`, `preprocessing.py`, and `prediction.py` are the standard backend-internal layout, but they do not require separate abstract base classes.

`optimization.py` is optional. Backends that do not need TensorRT/GPU/runtime optimization can keep it as a no-op or omit the file.

### 5.4 Action list contract

```text
action_list:
  shape: (T, D)
  T: model action steps
  D: flattened robot action dimension
  action_keys: model output modality order

ActionChunkProcessor:
  does not reorder action dimensions
  may match / RTC-align / interpolate / blend / smooth over time

Robot publish path:
  splits final action vector by robot command schema
```

---

## 6. Design Rule

```text
Main owns session flow.
Main owns the control loop.
Main can be split into ServiceHandler, SessionState, InferenceRequester, ControlLoop classes.
RobotClient is the common robot I/O client.
Main process uses RobotClient to publish robot commands.
Engine process uses RobotClient to read sensor/state topics for inference.
Engine implements InferenceEngine ABC.
Engine may split internally into PolicyLoader, Optimizer, Preprocessor, Predictor.
Optimizer is optional.
ActionChunkProcessor owns optional action-list post-processing and buffering.
If post-processing converts 16 actions to 100 actions, the loop can run at 100Hz.
If post-processing is disabled, the loop cadence must follow the raw action list.
```
