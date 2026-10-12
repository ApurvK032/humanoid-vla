import mujoco
import numpy as np
import pytest

from humanoid_vla.envs.outcome import Outcome
from humanoid_vla.envs.sort import SortEnv, SortEnvConfig
from humanoid_vla.envs.task import COLORS


@pytest.fixture
def env() -> SortEnv:
    env = SortEnv(SortEnvConfig(cameras=()))
    yield env
    env.close()


def home_action(env: SortEnv, closure: float = 0.0) -> np.ndarray:
    return np.append(env.config.home_arm, closure)


def place_block(env: SortEnv, color: str, pos: tuple[float, float, float]) -> None:
    joint = env.model.joint(f"block_{color}_joint")
    adr, dof = joint.qposadr[0], joint.dofadr[0]
    env.data.qpos[adr : adr + 7] = [*pos, 1.0, 0.0, 0.0, 0.0]
    env.data.qvel[dof : dof + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)


def in_bin_position(env: SortEnv, side: str) -> tuple[float, float, float]:
    scene = env.config.scene
    x, y = scene.bin_xy[side]
    return x, y, scene.table_height + scene.bin_wall_thickness + scene.block_half_size


def run_until_done(env: SortEnv, seconds: float) -> dict:
    info: dict = {}
    for _ in range(round(seconds / env.control_dt)):
        _, _, terminated, truncated, info = env.step(home_action(env))
        if terminated or truncated:
            break
    return info


def test_reset_is_deterministic(env: SortEnv) -> None:
    _, info_a = env.reset(seed=3)
    qpos_a = env.data.qpos.copy()
    _, info_b = env.reset(seed=3)
    assert info_a["task"] == info_b["task"]
    np.testing.assert_array_equal(qpos_a, env.data.qpos)


def test_observation_has_only_robot_senses(env: SortEnv) -> None:
    obs, _ = env.reset(seed=0)
    assert set(obs) == {"state", "instruction"}  # no cameras configured, no object poses
    assert env.observation_space.contains(obs)
    np.testing.assert_allclose(obs["state"][:7], env.config.home_arm, atol=1e-6)


def test_arm_targets_are_clipped_and_rate_limited(env: SortEnv) -> None:
    env.reset(seed=0)
    far = np.append(env.arm_upper + 1.0, 5.0)
    _, _, _, _, info = env.step(far)
    applied = info["applied_action"]
    max_step = env.config.max_joint_speed * env.control_dt
    np.testing.assert_allclose(np.abs(applied[:7] - env.config.home_arm), max_step, atol=1e-9)
    assert applied[7] == 1.0
    for _ in range(100):
        _, _, _, _, info = env.step(far)
    np.testing.assert_allclose(info["applied_action"][:7], env.arm_upper)


def test_malformed_actions_are_rejected(env: SortEnv) -> None:
    env.reset(seed=0)
    with pytest.raises(ValueError, match="8 finite"):
        env.step(np.zeros(7))
    with pytest.raises(ValueError, match="8 finite"):
        env.step(np.full(8, np.nan))


def test_success_needs_the_requested_block_settled_in_the_requested_bin(env: SortEnv) -> None:
    env.reset(seed=0, options={"target": "red", "destination": "left"})
    place_block(env, "red", in_bin_position(env, "left"))
    info = run_until_done(env, seconds=2.0)
    assert info["outcome"] == Outcome.SUCCESS
    assert info["time"] >= env.config.settle_s


def test_wrong_bin_and_wrong_object_are_labelled(env: SortEnv) -> None:
    env.reset(seed=0, options={"target": "red", "destination": "left"})
    place_block(env, "red", in_bin_position(env, "right"))
    assert run_until_done(env, seconds=2.0)["outcome"] == Outcome.WRONG_BIN

    env.reset(seed=0, options={"target": "red", "destination": "left"})
    other = next(c for c in COLORS if c != "red")
    place_block(env, other, in_bin_position(env, "left"))
    assert run_until_done(env, seconds=2.0)["outcome"] == Outcome.WRONG_OBJECT


def test_a_block_off_the_table_is_dropped(env: SortEnv) -> None:
    env.reset(seed=0)
    place_block(env, "green", (0.3, -0.2, 0.3))
    assert run_until_done(env, seconds=0.2)["outcome"] == Outcome.DROPPED


def test_time_limit_truncates_with_timeout() -> None:
    env = SortEnv(SortEnvConfig(cameras=(), time_limit_s=0.4))
    env.reset(seed=0)
    for _ in range(12):
        _, _, terminated, truncated, info = env.step(home_action(env))
        if terminated or truncated:
            break
    assert truncated
    assert not terminated
    assert info["outcome"] == Outcome.TIMEOUT


@pytest.mark.render
def test_head_camera_observation() -> None:
    env = SortEnv(SortEnvConfig(image_hw=(64, 64)))
    obs, _ = env.reset(seed=0)
    assert obs["images"]["head"].shape == (64, 64, 3)
    assert env.observation_space.contains(obs)
    env.close()
