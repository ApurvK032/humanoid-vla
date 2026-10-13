import itertools

import mujoco
import numpy as np
import pytest

from humanoid_vla.envs.task import LayoutConfig
from humanoid_vla.sim.render import FAST_QUALITY, CameraRenderer, project
from humanoid_vla.sim.scene import SceneConfig, look_at_xyaxes


def simulate(model: mujoco.MjModel, seconds: float) -> mujoco.MjData:
    """Step from the initial state with position targets held at the initial joint angles."""
    data = mujoco.MjData(model)
    steps = round(seconds / model.opt.timestep)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return data


def test_scene_has_policy_and_video_cameras(model: mujoco.MjModel) -> None:
    assert {model.camera(i).name for i in range(model.ncam)} == {"head", "wrist", "scene"}


def test_wrist_camera_looks_at_the_grasp_site(model: mujoco.MjModel) -> None:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    uv = project(model, data, "wrist", data.site("grasp_site").xpos, 256, 256) / 256
    np.testing.assert_allclose(uv, [0.5, 0.5], atol=0.05)


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


def test_head_camera_sees_every_block_position_and_both_bins(
    model: mujoco.MjModel, scene_cfg: SceneConfig
) -> None:
    """The policy must see its target: every corner of the block region (with the cube rotated
    to its widest) and every bin rim corner must project inside the image with a 5% margin."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    layout = LayoutConfig()
    top = scene_cfg.table_height + 2 * scene_cfg.block_half_size
    reach = scene_cfg.block_half_size * np.sqrt(2)
    points = [
        (x + dx, y + dy, top)
        for x, y in itertools.product(layout.x_range, layout.y_range)
        for dx, dy in itertools.product((-reach, reach), repeat=2)
    ]
    outer = scene_cfg.bin_inner_half_size + scene_cfg.bin_wall_thickness
    rim = scene_cfg.table_height + scene_cfg.bin_wall_thickness + scene_cfg.bin_wall_height
    for bx, by in scene_cfg.bin_xy.values():
        points += [(bx + sx * outer, by + sy * outer, rim) for sx in (-1, 1) for sy in (-1, 1)]
    uv = np.array([project(model, data, "head", p, 256, 256) for p in points]) / 256
    assert uv.min() > 0.05
    assert uv.max() < 0.95


@pytest.mark.render
def test_projection_matches_the_rendered_image(
    model: mujoco.MjModel, scene_cfg: SceneConfig
) -> None:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with CameraRenderer(model, 256, 256, quality=FAST_QUALITY) as renderer:
        image = renderer.render(data, "head").astype(int)
    top_center = data.body("block_red").xpos + np.array([0.0, 0.0, scene_cfg.block_half_size])
    u, v = np.round(project(model, data, "head", top_center, 256, 256)).astype(int)
    r, g, b = image[v, u]
    assert r > g + 60
    assert r > b + 60
