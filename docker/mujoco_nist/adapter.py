"""CPU MuJoCo dynamics and guarded Cartesian control for the Lab bridge."""
import math
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from model import build_model, BOARD, TABLE_Z, LAYOUT, PEG_PARTS, camera_rotation, quat

SCHEMA = "nist-sg2-state39-delta6-fixedgripper-v2"
SUPPORTED_PARTS = {
    "right": [name for name in PEG_PARTS if name != "peg_square_8"],
    "left": [name for name in PEG_PARTS if name not in ("peg_square_8", "peg_round_16")],
}
DEFAULT_CONFIG = dict(environment="mujoco", robot_type="ffw_sg2_rev1", arm="right",
                      target_part="peg_round_8", initialization="near_hole", control_hz=20,
                      translation_limit_m=.001, rotation_limit_rad=.01,
                      success_tolerance_m=.002, success_hold_s=.5, max_episode_steps=10000,
                      reward_source="environment", observation_mode="pixels", learning_backend="hil_serl")


class MujocoNistAdapter:
    def __init__(self, clearance_mm=.5):
        xml, self.board_prisms = build_model(clearance_mm)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.clearance_mm = clearance_mm
        self.c = {}
        self.capture_observation = None
        self.steps = self.hold = 0
        self.physics_steps = 0
        self.reset_cache = {}
        self.configure(DEFAULT_CONFIG)

    def describe(self):
        return dict(environment="mujoco", task_id="Cyclo-NIST-Board-FFW-SG2-MuJoCo-v1",
                    robot_type="ffw_sg2_rev1", schema=SCHEMA, observation_dim=39, action_dim=6,
                    target_parts=SUPPORTED_PARTS["right"], target_parts_by_arm=SUPPORTED_PARTS,
                    control_hz=20, action_names=["dx", "dy", "dz", "rx", "ry", "rz"],
                    action_frame="articulation root; metres/radians after scaling", physics_dt=.002,
                    physics_backend="MuJoCo CPU", mujoco_version=mujoco.__version__,
                    initializations=["near_hole"], pixel_observations=True, pixel_sizes=[64, 128, 256],
                    board_collision_prisms=self.board_prisms,
                    round_peg_training_clearance_mm=self.clearance_mm)

    def configure(self, config):
        c = {**DEFAULT_CONFIG, **config}
        if c["environment"] != "mujoco" or c["robot_type"] != "ffw_sg2_rev1":
            raise ValueError("MuJoCo bridge requires FFW-SG2")
        if c["control_hz"] != 20 or c["arm"] not in ("left", "right") or c["target_part"] not in PEG_PARTS:
            raise ValueError("Unsupported arm, peg or control rate (requires 20 Hz)")
        if c["target_part"] not in SUPPORTED_PARTS[c["arm"]]:
            raise ValueError("Selected hole is outside this arm's validated vertical-grasp reach; choose an available target")
        if c["initialization"] != "near_hole":
            raise ValueError("MuJoCo currently supports the already-grasped near-hole start")
        for name, high in (("translation_limit_m", .02), ("rotation_limit_rad", .15),
                           ("success_tolerance_m", .02), ("success_hold_s", 10)):
            if not isinstance(c[name], (int, float)) or not math.isfinite(c[name]) or not 0 < c[name] <= high:
                raise ValueError(f"Invalid {name}")
        if type(c["max_episode_steps"]) is not int or not 1 <= c["max_episode_steps"] <= 10000:
            raise ValueError("Invalid episode limit")
        self.c = c
        self.side = c["arm"][0]
        self.other = "l" if self.side == "r" else "r"
        self.part_name = c["target_part"]
        self.target = BOARD + LAYOUT["parts"][self.part_name]["pose_assembled"][:3]
        self.arm_joints = [self.model.joint(f"arm_{self.side}_joint{i}").id for i in range(1, 8)]
        self.qids = self.model.jnt_qposadr[self.arm_joints]
        self.dids = self.model.jnt_dofadr[self.arm_joints]
        self.aids = [self.model.actuator(f"arm_{self.side}_joint{i}").id for i in range(1, 8)]
        self.bounds = self.model.jnt_range[self.arm_joints].copy()
        self.tcp_id = self.model.site(f"tcp_{self.side}").id
        self.peg_id = self.model.body(self.part_name).id
        joint = self.model.joint(self.part_name+"_joint").id
        self.peg_q = int(self.model.jnt_qposadr[joint])
        self.peg_v = int(self.model.jnt_dofadr[joint])
        self.grip_q = int(self.model.joint(f"gripper_{self.side}_joint1").qposadr[0])
        return {**self.describe(), **self.reset()}

    def _set_joint(self, name, value):
        joint = self.model.joint(name)
        self.data.qpos[joint.qposadr[0]] = value
        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            self.data.ctrl[aid] = value

    def _fk(self, q):
        self.ik_data.qpos[:] = self.data.qpos
        self.ik_data.qpos[self.qids] = q
        mujoco.mj_kinematics(self.model, self.ik_data)
        return self.ik_data.site_xpos[self.tcp_id].copy(), self.ik_data.site_xmat[self.tcp_id].reshape(3, 3).copy()

    def _residual(self, q, position, rotation):
        p, r = self._fk(q)
        return np.r_[p-position, .25*Rotation.from_matrix(rotation @ r.T).as_rotvec()]

    def solve_reset(self, position, rotation):
        key = (self.side, self.part_name, tuple(position))
        if key in self.reset_cache:
            return self.reset_cache[key].copy()
        initial = np.array([-1.50, -.01, 1.70, -1.35, -3.07, .22, -1.43])
        if self.side == "l":
            initial *= [1, -1, -1, 1, -1, 1, -1]
        rng = np.random.default_rng(3)
        low, high = self.bounds[:, 0]+.005, self.bounds[:, 1]-.005
        for seed in [initial] + [rng.uniform(low, high) for _ in range(24)]:
            result = least_squares(self._residual, np.clip(seed, low, high), args=(position, rotation),
                                   bounds=(low, high), max_nfev=180, ftol=1e-9, xtol=1e-9, gtol=1e-9)
            if np.linalg.norm(result.fun) < .00005:
                self.reset_cache[key] = result.x.copy()
                return result.x
        raise ValueError(f"Near-hole pose is unreachable for {self.c['arm']} / {self.part_name}")

    def reset(self):
        self.phase = "near_hole_reset"
        self.steps = self.hold = self.physics_steps = 0
        self.motion_blocked = False
        self.motion_block_reason = self.motion_block_joint = None
        mujoco.mj_resetData(self.model, self.data)
        self._set_joint("lift_joint", -.35)
        self._set_joint("head_joint1", .69)
        self._set_joint("head_joint2", 0)
        observer = [-1.04606645, .59072444, -.87294305, -.89643852, .17632946, -1.45950495, -1.52592387]
        if self.other == "r":
            observer = (np.array(observer)*[1, -1, -1, 1, -1, 1, -1]).tolist()
        for i, value in enumerate(observer, 1):
            self._set_joint(f"arm_{self.other}_joint{i}", value)
        for side in ("l", "r"):
            for i, factor in enumerate((1, -1, -1, 1), 1):
                self._set_joint(f"gripper_{side}_joint{i}", .8*factor)
        near = self.target.copy()
        near[2] = BOARD[2]+.015
        if (self.side == "r" and near[1] > 0) or (self.side == "l" and near[1] < 0):
            near[1] += -.02 if self.side == "r" else .02
        self.preinsert_offset = near-self.target
        # Pinch the upper part of the 50 mm peg. The real finger collision mesh
        # extends 23 mm below the TCP, so a 35 mm offset would hit the plate
        # before seating. 55 mm leaves the fingertips above the board at success.
        goal = near + [0, 0, .055]
        self.orientation_target = np.eye(3)
        solution = self.solve_reset(goal, self.orientation_target)
        for i, value in enumerate(solution, 1):
            self._set_joint(f"arm_{self.side}_joint{i}", value)
        self.data.qpos[self.peg_q:self.peg_q+7] = np.r_[near, 1, 0, 0, 0]
        self.data.ctrl[self.model.actuator(f"gripper_{self.side}_joint1").id] = 1.1
        # A reset-only fixture holds the peg while the fingers close. Every
        # episode step thereafter uses a free body with contact/friction only.
        for _ in range(300):
            self.data.qpos[self.peg_q:self.peg_q+7] = np.r_[near, 1, 0, 0, 0]
            self.data.qvel[self.peg_v:self.peg_v+6] = 0
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        before = self.peg_position().copy()
        for _ in range(250):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.grasp_drift = float(np.linalg.norm(self.peg_position()-before))
        self.grasp_verified = bool(self.grasp_drift < .008 and np.linalg.norm(self.peg_position()-near) < .012)
        if not self.grasp_verified:
            raise ValueError(f"MuJoCo free-physics grasp retention failed: drift {self.grasp_drift:.4f} m, peg {self.peg_position()}, tcp {self.tcp_position()}")
        self.grip_target = float(self.data.ctrl[self.model.actuator(f"gripper_{self.side}_joint1").id])
        self.grasp_offset = self.tcp_rotation().T @ (self.peg_position()-self.tcp_position())
        self.position_target = self.tcp_position()
        self.previous_translation = np.zeros(3)
        self.was_moving = False
        self.calibrate_cameras()
        self.data.time = 0
        self.phase = "ready"
        return dict(observation=self.observation(), info=self.info())

    def calibrate_cameras(self):
        focus = BOARD + LAYOUT["parts"][self.part_name]["pose_assembled"][:3]
        focus[2] = BOARD[2]+.009
        for name in ("head", "left_wrist", "right_wrist"):
            camera = self.model.camera(name)
            body = camera.bodyid[0]
            rotation = self.data.xmat[body].reshape(3, 3)
            eye = self.data.xpos[body] + rotation @ camera.pos
            camera.quat[:] = quat(rotation.T @ camera_rotation(eye, focus))
        mujoco.mj_forward(self.model, self.data)

    def tcp_position(self):
        return self.data.site_xpos[self.tcp_id].copy()

    def tcp_rotation(self):
        return self.data.site_xmat[self.tcp_id].reshape(3, 3).copy()

    def peg_position(self):
        return self.data.xpos[self.peg_id].copy()

    def _command_arm(self, action):
        self.motion_blocked = False
        self.motion_block_reason = self.motion_block_joint = None
        if not np.any(action):
            if self.was_moving:
                self.data.ctrl[self.aids] = self.data.qpos[self.qids]
                self.position_target = self.tcp_position()
                self.orientation_target = self.tcp_rotation()
            self.was_moving = False
            self.previous_translation[:] = 0
            return  # Preserve the last joint targets; no repeated IK during idle.
        current_position = self.tcp_position()
        base = self.position_target.copy()
        reversed_axes = action[:3]*self.previous_translation < 0
        base[reversed_axes] = current_position[reversed_axes]
        position = base+action[:3]*self.c["translation_limit_m"]
        # Hold uncommanded Cartesian axes instead of accumulating tracking error.
        # Bound the goal lead when contact prevents the tool from following.
        position = np.clip(position, current_position-.005, current_position+.005)
        rotation = Rotation.from_rotvec(action[3:]*self.c["rotation_limit_rad"]).as_matrix() @ self.orientation_target
        low, high = self.bounds[:, 0]+.005, self.bounds[:, 1]-.005
        current = self.data.qpos[self.qids].copy()
        candidate = np.clip(current, low, high)
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(25):
            p, r = self._fk(candidate)
            residual = np.r_[position-p, .25*Rotation.from_matrix(rotation @ r.T).as_rotvec()]
            if np.linalg.norm(residual) < 1e-6:
                break
            mujoco.mj_comPos(self.model, self.ik_data)
            mujoco.mj_jacSite(self.model, self.ik_data, jacp, jacr, self.tcp_id)
            jacobian = np.vstack((jacp[:, self.dids], .25*jacr[:, self.dids]))
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-7*np.eye(6), residual)
            candidate = np.clip(candidate + np.clip(delta, -.08, .08), low, high)
        p, r = self._fk(candidate)
        move = p-self.tcp_position()
        requested = action[:3]*self.c["translation_limit_m"]
        reverse = np.dot(move, requested) < -1e-10
        limit = np.any((candidate-low < .001) | (high-candidate < .001))
        if np.linalg.norm(self._residual(candidate, position, rotation)) > .00015 or np.max(np.abs(candidate-current)) > .15 or reverse or limit:
            self.motion_blocked = True
            self.motion_block_reason = "Joint limit or unreachable Cartesian target; holding previous target"
            if limit:
                index = int(np.argmin(np.minimum(candidate-low, high-candidate)))
                self.motion_block_joint = f"arm_{self.side}_joint{index+1}"
            return
        self.data.ctrl[self.aids] = candidate
        self.position_target = position
        self.orientation_target = rotation
        self.previous_translation = action[:3].copy()
        self.was_moving = True

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (6,) or not np.isfinite(action).all() or (np.abs(action) > 1).any():
            raise ValueError("Expected six finite normalized actions in [-1,1]")
        start = time.monotonic()
        self._command_arm(action)
        mujoco.mj_step(self.model, self.data, nstep=25)
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError("Non-finite MuJoCo state")
        self.steps += 1
        self.physics_steps += 25
        error = self.peg_position()-self.target
        upright = self.data.xmat[self.peg_id].reshape(3, 3)[2, 2] > math.cos(math.radians(10))
        if "square" in self.part_name:
            r = self.data.xmat[self.peg_id].reshape(3, 3)
            upright = upright and abs(math.remainder(math.atan2(r[1, 0], r[0, 0]), math.pi/2)) < math.radians(10)
        seated = np.linalg.norm(error[:2]) < self.c["success_tolerance_m"] and abs(error[2]) < self.c["success_tolerance_m"] and np.linalg.norm(self.data.qvel[self.peg_v:self.peg_v+3]) < .05 and upright
        self.hold = self.hold+1 if seated else 0
        success = self.hold >= math.ceil(self.c["success_hold_s"]*20)
        failure = self.peg_position()[2] < TABLE_Z-.15
        self.last_step_ms = (time.monotonic()-start)*1000
        return dict(observation=self.observation(), reward=float(success), terminated=bool(success or failure),
                    truncated=self.steps >= self.c["max_episode_steps"], info=self.info(success, failure))

    def observation(self):
        # Free-joint translation is world-frame, but angular qvel is body-frame.
        # The shared state39 contract exposes both velocities in world coordinates.
        angular_velocity = (self.data.xmat[self.peg_id].reshape(3, 3)
                            @ self.data.qvel[self.peg_v+3:self.peg_v+6])
        observation = np.r_[self.data.qpos[self.qids], self.data.qvel[self.dids],
                            self.tcp_position()-BOARD, quat(self.tcp_rotation()),
                            self.peg_position()-BOARD, self.data.xquat[self.peg_id],
                            self.data.qvel[self.peg_v:self.peg_v+3], angular_velocity, self.target-BOARD,
                            self.data.qpos[self.grip_q], self.hold/max(1, math.ceil(self.c["success_hold_s"]*20))]
        if observation.shape != (39,) or not np.isfinite(observation).all():
            raise ValueError("Invalid observation")
        return observation.tolist()

    def info(self, success=False, failure=False):
        other = self.data.site(f"tcp_{self.other}")
        return dict(position_m=(self.peg_position()-BOARD).tolist(),
                    eef_position_m=(self.tcp_position()-BOARD).tolist(),
                    target_position_m=(self.target-BOARD).tolist(),
                    board_world_position_m=BOARD.tolist(), target_world_position_m=self.target.tolist(),
                    target_part=self.part_name, initialization="near_hole", initialization_phase=self.phase,
                    grasp_verified=self.grasp_verified, grasp_drift_m=self.grasp_drift,
                    gripper_mode="fixed", gripper_target=[self.grip_target],
                    is_success=bool(success), is_failure=bool(failure), episode_step=self.steps,
                    max_episode_steps=self.c["max_episode_steps"], motion_blocked=self.motion_blocked,
                    motion_block_reason=self.motion_block_reason, motion_block_joint=self.motion_block_joint,
                    arm_joint_hold=True, preinsert_offset_m=self.preinsert_offset.tolist(),
                    peg_clearance_above_board_m=float(self.peg_position()[2]-BOARD[2]),
                    observer_eef_position_m=other.xpos.tolist(),
                    observer_eef_quaternion_wxyz=quat(other.xmat.reshape(3, 3)).tolist(),
                    lift_position_m=float(self.data.joint("lift_joint").qpos[0]), lift_offset_m=-.35,
                    physics_steps=self.physics_steps, physics_dt=.002, schema=SCHEMA,
                    physics_step_ms=getattr(self, "last_step_ms", None),
                    training_clearance=dict(diameter_clearance_mm=self.clearance_mm))

    def release(self):
        pass  # The simulator advances only on explicit step commands.
