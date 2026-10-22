import json
import mujoco
import numpy as np
import pytest
from shapely.geometry import Point

from adapter import MujocoNistAdapter
from model import BOARD, LAYOUT, PEG_PARTS, plate_polygon


@pytest.fixture(scope="module")
def env():
    return MujocoNistAdapter()


def test_plate_retains_holes_and_solid_surface():
    plate = plate_polygon(.5)
    for name in PEG_PARTS:
        assert not plate.contains(Point(LAYOUT["parts"][name]["pose_assembled"][:2]))
    assert plate.contains(Point(.12, .12))
    xy = np.array(LAYOUT["parts"]["peg_round_8"]["pose_assembled"][:2])
    assert not plate.contains(Point(xy+[.00415, 0]))
    assert plate.contains(Point(xy+[.0044, 0]))


def test_reset_grasp_and_idle_hold_are_free_physics(env):
    env.reset()
    peg, tcp = env.peg_position(), env.tcp_position()
    assert env.grasp_verified and env.steps == 0
    assert .01 < peg[2]-BOARD[2] < .02
    for _ in range(200):
        result = env.step(np.zeros(6))
    assert np.linalg.norm(env.peg_position()-peg) < .001
    assert np.linalg.norm(env.tcp_position()-tcp) < .0005
    assert result["info"]["physics_steps"] == 5000
    json.dumps(result, allow_nan=False)


def test_cartesian_commands_move_in_requested_direction_and_hold_gripper(env):
    env.reset()
    before = env.tcp_position()
    grip = env.grip_target
    for _ in range(30):
        result = env.step([0, 1, 0, 0, 0, 0])
    motion = env.tcp_position()-before
    assert motion[1] > .002 and abs(motion[0]) < .001
    assert env.grip_target == grip
    assert not result["info"]["motion_blocked"]


def test_real_contact_insertion_reaches_terminal_success(env):
    env.reset()
    for _ in range(800):
        error = env.target-env.peg_position()
        action = np.zeros(6)
        if np.linalg.norm(error[:2]) > .00015:
            action[:2] = np.clip(error[:2]/env.c["translation_limit_m"], -1, 1)
        else:
            action[2] = np.clip(error[2]/env.c["translation_limit_m"], -1, 1)
        result = env.step(action)
        if result["terminated"]:
            break
    assert result["terminated"] and result["reward"] == 1
    assert result["info"]["is_success"]
    assert abs(env.peg_position()[2]-env.target[2]) < env.c["success_tolerance_m"]


def test_unreachable_commands_are_blocked_before_exceeding_limits(env):
    env.reset()
    env.c["translation_limit_m"] = .02
    blocked = False
    for _ in range(300):
        before = env.data.ctrl[env.aids].copy()
        env.step([0, 1, 0, 0, 0, 0])
        if env.motion_blocked:
            np.testing.assert_array_equal(before, env.data.ctrl[env.aids])
            blocked = True
            break
    assert blocked
    q = env.data.ctrl[env.aids]
    assert np.all(q >= env.bounds[:, 0]) and np.all(q <= env.bounds[:, 1])
    env.c["translation_limit_m"] = .001


def test_episode_limit_and_invalid_actions(env):
    env.reset()
    env.c["max_episode_steps"] = 2
    assert not env.step(np.zeros(6))["truncated"]
    assert env.step(np.zeros(6))["truncated"]
    for action in ([0]*3, [0]*7, [float("nan")]*6, [2]*6):
        with pytest.raises(ValueError):
            env.step(action)
    env.c["max_episode_steps"] = 10000


def test_object_angular_velocity_uses_world_frame(env):
    env.reset()
    joint_id = env.model.body_jntadr[env.peg_id]
    qadr = env.model.jnt_qposadr[joint_id]
    env.data.qpos[qadr+3:qadr+7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    env.data.qvel[env.peg_v:env.peg_v+6] = [.1, .2, .3, 1, 2, 3]
    mujoco.mj_forward(env.model, env.data)
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_BODY,
                            env.peg_id, velocity, 0)
    observation = np.asarray(env.observation())
    np.testing.assert_allclose(observation[28:31], [.1, .2, .3], atol=1e-8)
    np.testing.assert_allclose(observation[31:34], velocity[:3], atol=1e-8)
    np.testing.assert_allclose(observation[31:34], [-2, 1, 3], atol=1e-8)
