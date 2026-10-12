"""Inverse kinematics: joint angles that place a site at a target pose.

Damped least squares on the site's position and orientation error, with joint limits enforced
after every step and an optional pull toward a preferred posture in the Jacobian's null space.
IK runs on its own ``MjData`` so it never disturbs a running simulation.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class IKConfig:
    max_iters: int = 300
    pos_tol: float = 1e-3  # m
    rot_tol: float = 1e-2  # rad
    damping: float = 0.05
    rot_weight: float = 0.3  # trades orientation against position error (m per rad)
    max_step: float = 0.2  # rad per iteration and joint
    posture_gain: float = 0.05  # null-space pull toward the preferred posture


@dataclass(frozen=True)
class IKResult:
    qpos: np.ndarray  # angles of the solved joints, in the order requested
    pos_err: float
    rot_err: float
    iters: int
    success: bool


class IKSolver:
    """Solves for a subset of hinge joints (e.g. one arm) of ``model``."""

    def __init__(
        self,
        model: mujoco.MjModel,
        site: str,
        joints: Sequence[str],
        config: IKConfig | None = None,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.site_id = model.site(site).id
        self.config = config or IKConfig()
        joint_ids = [model.joint(name).id for name in joints]
        self.qpos_adr = model.jnt_qposadr[joint_ids]
        self.dof_adr = model.jnt_dofadr[joint_ids]
        self.lower = model.jnt_range[joint_ids, 0].copy()
        self.upper = model.jnt_range[joint_ids, 1].copy()

    def site_pose(self, qpos_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World position and rotation matrix of the site for a full ``qpos``."""
        self.data.qpos[:] = qpos_full
        mujoco.mj_kinematics(self.model, self.data)
        return (
            self.data.site_xpos[self.site_id].copy(),
            self.data.site_xmat[self.site_id].reshape(3, 3).copy(),
        )

    def solve(
        self,
        qpos_full: np.ndarray,
        target_pos: np.ndarray,
        target_rot: np.ndarray | None = None,
        posture: np.ndarray | None = None,
    ) -> IKResult:
        """Solve starting from ``qpos_full``; other joints keep their values.

        ``target_rot`` is a 3x3 world rotation for the site frame; None solves position only.
        ``posture`` holds preferred angles for the solved joints.
        """
        cfg = self.config
        m, d = self.model, self.data
        d.qpos[:] = qpos_full
        q = d.qpos[self.qpos_adr].copy()
        n = len(q)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        target_quat = None
        if target_rot is not None:
            target_quat = np.zeros(4)
            mujoco.mju_mat2Quat(target_quat, np.asarray(target_rot, dtype=float).ravel())

        pos_err = rot_err = np.inf
        for it in range(cfg.max_iters + 1):
            d.qpos[self.qpos_adr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err_pos = np.asarray(target_pos, dtype=float) - d.site_xpos[self.site_id]
            err_rot = np.zeros(3)
            if target_quat is not None:
                site_quat = np.zeros(4)
                mujoco.mju_mat2Quat(site_quat, d.site_xmat[self.site_id])
                neg = np.zeros(4)
                mujoco.mju_negQuat(neg, site_quat)
                diff = np.zeros(4)
                mujoco.mju_mulQuat(diff, target_quat, neg)
                mujoco.mju_quat2Vel(err_rot, diff, 1.0)
            pos_err, rot_err = float(np.linalg.norm(err_pos)), float(np.linalg.norm(err_rot))
            if pos_err < cfg.pos_tol and rot_err < cfg.rot_tol:
                return IKResult(q.copy(), pos_err, rot_err, it, True)
            if it == cfg.max_iters:
                break

            mujoco.mj_jacSite(m, d, jacp, jacr, self.site_id)
            if target_quat is None:
                jac, err = jacp[:, self.dof_adr], err_pos
            else:
                jac = np.vstack([jacp[:, self.dof_adr], cfg.rot_weight * jacr[:, self.dof_adr]])
                err = np.concatenate([err_pos, cfg.rot_weight * err_rot])
            jjt = jac @ jac.T + cfg.damping**2 * np.eye(jac.shape[0])
            dq = jac.T @ np.linalg.solve(jjt, err)
            if posture is not None:
                # Exact null-space projector: a damped one leaks the posture pull into the task
                # and stalls convergence at millimetre-level errors.
                projector = np.eye(n) - np.linalg.pinv(jac, rcond=1e-4) @ jac
                dq += projector @ (cfg.posture_gain * (np.asarray(posture) - q))
            dq = np.clip(dq, -cfg.max_step, cfg.max_step)
            q = np.clip(q + dq, self.lower, self.upper)

        return IKResult(q.copy(), pos_err, rot_err, cfg.max_iters, False)
