"""Fixed-base Unitree G1 with one active arm and its Dex3 hand.

Edits Menagerie's ``g1_with_hands`` model in place. The floating base is removed and every joint
outside the active arm and hand is frozen at a fixed angle. Frozen joints are deleted from the
model, so they cost nothing to simulate and cannot drift.

The joint orderings defined here are the single source of truth for robot state and action vectors.
"""

from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np

Side = Literal["left", "right"]

_LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
_ARM = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)
_HAND = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")

WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")


def leg_joints(side: Side) -> tuple[str, ...]:
    return tuple(f"{side}_{name}_joint" for name in _LEG)


def arm_joints(side: Side) -> tuple[str, ...]:
    """Arm joints, shoulder to wrist."""
    return tuple(f"{side}_{name}_joint" for name in _ARM)


def hand_joints(side: Side) -> tuple[str, ...]:
    """Dex3 hand joints: thumb (3), index (2), middle (2)."""
    return tuple(f"{side}_hand_{name}_joint" for name in _HAND)


def other_side(side: Side) -> Side:
    return "left" if side == "right" else "right"


def resting_arm_pose(side: Side) -> dict[str, float]:
    """Arm hanging at the side with the elbow bent, as in Menagerie's "stand" keyframe.

    At zero angles the G1 forearm points straight forward, which would put an idle arm over the
    table, so the inactive arm is frozen in this pose instead.
    """
    roll = 0.2 if side == "left" else -0.2
    return {
        f"{side}_shoulder_pitch_joint": 0.2,
        f"{side}_shoulder_roll_joint": roll,
        f"{side}_elbow_joint": 1.28,
    }


@dataclass(frozen=True)
class G1Config:
    active_side: Side = "right"
    # Menagerie's standing pelvis height: with straight legs the feet rest on the floor.
    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.793)

    @property
    def active_joints(self) -> tuple[str, ...]:
        """Joints that stay actuated, in state/action order: arm, then hand."""
        return arm_joints(self.active_side) + hand_joints(self.active_side)

    @property
    def frozen_angles(self) -> dict[str, float]:
        """Non-zero angles for frozen joints; all other frozen joints are fixed at zero."""
        return resting_arm_pose(other_side(self.active_side))


def make_fixed_base(spec: mujoco.MjSpec, cfg: G1Config) -> None:
    """Fix the pelvis in place and freeze every joint outside the active arm and hand."""
    pelvis = spec.body("pelvis")
    for joint in list(pelvis.joints):  # the floating-base free joint
        spec.delete(joint)
    pelvis.pos = cfg.base_pos

    keep = set(cfg.active_joints)
    to_freeze = [j for j in pelvis.find_all(mujoco.mjtObj.mjOBJ_JOINT) if j.name not in keep]
    frozen_names = {j.name for j in to_freeze}
    for actuator in list(spec.actuators):
        if actuator.target in frozen_names:
            spec.delete(actuator)
    for joint in to_freeze:
        freeze_hinge(spec, joint, cfg.frozen_angles.get(joint.name, 0.0))

    # Upstream keyframes describe the full floating-base qpos and no longer apply.
    for key in list(spec.keys):
        spec.delete(key)


def freeze_hinge(spec: mujoco.MjSpec, joint: mujoco.MjsJoint, angle: float) -> None:
    """Delete a hinge joint, leaving its body fixed at ``angle`` relative to the parent body."""
    if joint.type != mujoco.mjtJoint.mjJNT_HINGE:
        raise ValueError(f"{joint.name}: only hinge joints can be frozen")
    body = joint.parent
    if body.alt.type != mujoco.mjtOrientation.mjORIENTATION_QUAT:
        raise ValueError(f"{body.name}: body orientation must be given as a quaternion")

    rotation = np.zeros(4)
    axis = np.asarray(joint.axis, dtype=float)
    mujoco.mju_axisAngle2Quat(rotation, axis / np.linalg.norm(axis), angle - joint.ref)

    # Rotate the body about the joint anchor: the anchor stays put and the origin moves around it.
    anchor = np.asarray(joint.pos, dtype=float)
    rotated_anchor = np.zeros(3)
    mujoco.mju_rotVecQuat(rotated_anchor, anchor, rotation)
    body_quat = np.asarray(body.quat, dtype=float)
    shift = np.zeros(3)
    mujoco.mju_rotVecQuat(shift, anchor - rotated_anchor, body_quat)

    new_quat = np.zeros(4)
    mujoco.mju_mulQuat(new_quat, body_quat, rotation)
    body.pos = np.asarray(body.pos, dtype=float) + shift
    body.quat = new_quat / np.linalg.norm(new_quat)
    spec.delete(joint)
