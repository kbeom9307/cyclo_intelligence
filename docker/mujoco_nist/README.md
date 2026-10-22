# MuJoCo NIST bridge

Run from the repository root:

```sh
docker compose -f docker/docker-compose.mujoco.yml up -d --build
```

The standalone `cyclo_mujoco` container serves the existing leased Lab bridge
protocol at `127.0.0.1:7126`. MuJoCo 3.16 runs physics on the CPU at 500 Hz,
with 25 substeps per 20 Hz control action. EGL renders camera images on the GPU.
No Isaac, ROS, PyTorch or JAX dependency is loaded by this process.

Select **MuJoCo** / **Use MuJoCo NIST settings** in HIL-SERL Lab and save.
Preview, keyboard intervention, classifier collection, demonstrations, and
local/remote HIL-SERL learning use the same Lab workflow. Existing Isaac
datasets remain under their original environment identity. Collect new images
and demos for MuJoCo; Isaac checkpoints are not automatically reused.

The current task is fixed-gripper peg insertion, starting already grasped
15 mm above the board. Reset holds the peg only while closing the fingers,
then verifies retention with free dynamics. No weld or kinematic attachment is
used during an episode. The upper part of the peg is grasped so the fingers
clear the board at full insertion. Joint targets hold on zero input; bounded
Cartesian IK rejects unreachable targets. The other arm observes the task.
Tray pick-and-place starts and non-peg NIST manipulation tasks are not supported
by this adapter. Non-peg kit items are static scene geometry.

Overview is 1920x1440; its renderer closes after viewers stop requesting frames.
Head and both wrist cameras render at 640x360. Their extrinsics are fixed after
reset; the virtual cameras are aimed at the selected hole with a level horizon.
Learning receives synchronized 128x128 RGB crops (64x64 for legacy models) plus
22 proprioceptive values. Changing preview JPEG quality does not change learning.
The physical world is frozen between explicit step commands and while paused.

The NIST plate's actual holes are retained by triangulating its top surface and
extruding each triangle into a convex collision prism. Round-hole diameter
clearance defaults to 0.5 mm (`server.py --round-peg-clearance-mm 0.5`).
The geometry, friction and contact solver differ from PhysX; this environment
is not a validated sim-to-real model.
The vertical-grasp start excludes square-8 for both arms and round-16 for the
left arm because those poses are outside the validated reach at this board
location. The bridge advertises the available targets per arm. Gravity
compensation, position-servo gains and MuJoCo's no-slip contact solver are
configured for stable millimetre control; actuator torque limits are retained.

## Local learner

Select **Local** learner placement and use `cuda:0` for both actor and learner.
The local upstream engine receives Agentlace replay transitions independently
of the optimizer, so initial XLA compilation does not fill the actor's queue.
Pause and stop wait for an in-flight update before saving the checkpoint.

When MuJoCo's actor and local learner share a GPU, the JAX allocator defaults
to an 80% memory pool with on-demand allocation. Explicit
`XLA_PYTHON_CLIENT_MEM_FRACTION` overrides are respected. Batch 256, three
128x128 views, FP32, ResNet-10 and CTA 2 remain unchanged. Training keeps
convolution autotuning enabled; disabling it caused long actor stalls in the
combined benchmark. Local batch-256 readiness requires 6.5 GiB free at launch.

Run the isolated full-stack benchmark with an idle MuJoCo bridge:

```sh
docker exec -w /opt/supervisor_api cyclo_intelligence \
  python3 profile_hilserl_mujoco_local.py --updates 50 --publish-interval 50 \
  --output /tmp/mujoco-local-256.json
```

It polls all four Lab camera views, runs the GPU actor and reward classifier,
trains from actual simulator images, and verifies policy changes, replay counts
and saved optimizer state. Scripted hold demonstrations and a constant-negative
classifier are temporary test fixtures; this measures execution, memory and
throughput, not insertion success or classifier accuracy. User datasets and the
saved Lab configuration are not changed.

On the RTX 5070 Laptop 8 GiB GPU, the full batch-256 run completed 51 learner
updates (102 critic updates, including the in-flight update at stop), saved all
1,160 online transitions, and reached a peak total GPU usage of 6,792 MiB with
915 MiB free. Actor throughput averaged 13.2 Hz (12.4 Hz after the first update),
so this configuration does not sustain the requested 20 Hz. See the
[measurements and comparison runs](../../docs/hilserl/mujoco-local-2026-10-20.json).

## Assets and provenance

- FFW-SG2 MJCF and its referenced meshes: ROBOTIS
  [robotis_mujoco_menagerie](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie),
  commit `d8344c0dbe7a00208d0301111523dde65efc174a`, Apache-2.0.
  The original MJCF is retained; scene changes are applied by `model.py`.
  License: `assets/ROBOTIS_LICENSE`.
- NIST Task Board #1 STL assets are public-domain NIST task-board geometry,
  copied from the existing `isaac_worker/nist_board/downloads/taskboard1stlzip`.
  `layout_v0.json` is the existing Cyclo board layout in metres. Board position
  matches Isaac: `(0.45, -0.10, 0.756)`, table surface `0.727`.
- `transport.py` adapts the Apache-2.0 Cyclo Lab bridge at commit
  `68fd3de468eba0f8772c8ae6a9524fb34db40e16`; command ownership, sequencing and
  frame-demand behavior are preserved.

For local physics tests, install `requirements.txt` and pytest in a venv, then:

```sh
python -m pytest docker/mujoco_nist/test_physics.py -q
```
