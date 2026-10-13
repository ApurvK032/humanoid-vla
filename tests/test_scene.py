import mujoco
import numpy as np
import pytest

from humanoid_vla.sim.render import FAST_QUALITY, CameraRenderer
from humanoid_vla.sim.scene import SceneConfig, look_at_xyaxes


def simulate(model: mujoco.MjModel, seconds: float) -> mujoco.MjData:
    """Step from the initial state with position targets held at the initial joint angles."""
    data = mujoco.MjData(model)
    steps = round(seconds / model.opt.timestep)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return data


def test_scene_has_policy_and_video_cameras(model: mujoco.MjModel) -> None:
    assert {model.camera(i).name for i in range(model.ncam)} == {"head", "scene"}


def test_blocks_settle_on_table(model: mujoco.MjModel, scene_cfg: SceneConfig) -> None:
    data = simulate(model, seconds=1.0)
    resting_z = scene_cfg.table_height + scene_cfg.block_half_size
    for color, (x, y) in scene_cfg.block_xy.items():
        pos = data.body(f"block_{color}").xpos
        assert pos[2] == pytest.approx(resting_z, abs=0.002), color
        assert np.hypot(pos[0] - x, pos[1] - y) < 0.005, color


def test_simulation_is_stable_and_arm_holds_targets(model: mujoco.MjModel) -> None:
    data = simulate(model, seconds=2.0)
    assert np.all(np.isfinite(data.qpos))
    assert all(warning.number == 0 for warning in data.warning)
    actuated_qpos = data.qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]]
    np.testing.assert_allclose(actuated_qpos, data.ctrl, atol=0.02)


def test_look_at_points_camera_at_target() -> None:
    pos, target = (1.0, -1.0, 1.5), (0.2, 0.0, 0.8)
    xyaxes = np.asarray(look_at_xyaxes(pos, target))
    right, up = xyaxes[:3], xyaxes[3:]
    forward = np.cross(up, right)  # camera looks along -z = up x right
    expected = np.subtract(target, pos) / np.linalg.norm(np.subtract(target, pos))
    np.testing.assert_allclose(forward, expected, atol=1e-12)
    assert abs(right[2]) < 1e-12  # no roll: image horizontal stays level


@pytest.mark.render
def test_cameras_render_non_blank_images(model: mujoco.MjModel) -> None:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with CameraRenderer(model, 64, 64, quality=FAST_QUALITY) as renderer:
        for camera in ("head", "scene"):
            image = renderer.render(data, camera)
            assert image.shape == (64, 64, 3)
            assert image.std() > 5.0, camera
