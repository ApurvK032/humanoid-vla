import mujoco
import numpy as np
import pytest

from humanoid_vla.sim.assets import g1_scene_xml
from humanoid_vla.sim.g1 import G1Config, arm_joints, hand_joints
from humanoid_vla.sim.scene import SceneConfig, build_model


def actuated_joints(model: mujoco.MjModel) -> set[str]:
    return {model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)}


def test_active_joint_order_is_arm_then_hand() -> None:
    cfg = G1Config()
    assert cfg.active_joints == arm_joints("right") + hand_joints("right")
    assert len(arm_joints("right")) == 7
    assert len(hand_joints("right")) == 7


@pytest.mark.parametrize("side", ["left", "right"])
def test_only_active_arm_and_hand_are_actuated(side: str) -> None:
    cfg = G1Config(active_side=side)
    model = build_model(SceneConfig(robot=cfg))
    assert actuated_joints(model) == set(cfg.active_joints)
    hinges = {
        model.joint(i).name
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    }
    assert hinges == set(cfg.active_joints)


def test_only_blocks_have_free_joints(model: mujoco.MjModel, scene_cfg: SceneConfig) -> None:
    free = {
        model.joint(i).name
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
    }
    assert free == {f"block_{color}_joint" for color in scene_cfg.block_xy}


def test_frozen_robot_matches_upstream_pose(model: mujoco.MjModel, scene_cfg: SceneConfig) -> None:
    """Freezing joints must place every robot body exactly where the upstream model would."""
    upstream = mujoco.MjModel.from_xml_path(str(g1_scene_xml()))
    up_data = mujoco.MjData(upstream)
    up_data.qpos[:3] = scene_cfg.robot.base_pos
    up_data.qpos[3:7] = [1, 0, 0, 0]
    for joint, angle in scene_cfg.robot.frozen_angles.items():
        up_data.qpos[upstream.joint(joint).qposadr[0]] = angle
    mujoco.mj_forward(upstream, up_data)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for body_id in range(1, upstream.nbody):
        name = upstream.body(body_id).name
        np.testing.assert_allclose(data.body(name).xpos, up_data.body(name).xpos, atol=1e-9)
        np.testing.assert_allclose(data.body(name).xquat, up_data.body(name).xquat, atol=1e-9)


def test_feet_rest_on_floor(model: mujoco.MjModel) -> None:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    foot_bottoms = [
        data.geom_xpos[g][2] - model.geom_size[g][0]
        for g in range(model.ngeom)
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
        and "ankle_roll" in model.body(model.geom_bodyid[g]).name
    ]
    assert foot_bottoms
    assert min(foot_bottoms) == pytest.approx(0.0, abs=0.003)
