"""Instruction-conditioned block sorting with the fixed-base G1 (right arm and Dex3 hand).

Timing: physics steps every 2 ms; the policy acts at ``control_hz`` and each action is held for
the physics substeps in between.

Actions are absolute targets: 7 arm joint angles in radians (``g1.arm_joints`` order) and one
hand closure in [0, 1] that ``sim.hand`` maps to the seven finger joints. Arm targets are clipped
to the joint ranges and may move at most ``max_joint_speed`` per second. Executed targets are
rounded to float32, the precision datasets store (docs/ACTION_CONTRACT.md).

Observations are what the robot could sense: camera images, the 14 measured arm and hand joint
angles (``G1Config.active_joints`` order) and the instruction. Object poses are only available
through ``privileged_state()``, for the scripted expert and the evaluator.
"""

import string
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from humanoid_vla.envs.outcome import TERMINAL, Outcome, in_bin, off_table
from humanoid_vla.envs.task import COLORS, LayoutConfig, SortTask, sample_task
from humanoid_vla.sim.hand import hand_targets
from humanoid_vla.sim.render import FULL_QUALITY, CameraRenderer, RenderQuality
from humanoid_vla.sim.scene import SceneConfig, build_model

# Arm pose at reset: hand raised palm-down at the robot's right, outside the head camera's view.
HOME_ARM = (0.875, -0.751, -1.223, -0.331, -0.24, -0.28, 0.391)
INSTRUCTION_CHARSET = string.ascii_letters + string.digits + " .,'-"


@dataclass(frozen=True)
class SortEnvConfig:
    scene: SceneConfig = field(default_factory=SceneConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    control_hz: float = 25.0
    max_joint_speed: float = 2.0  # rad/s
    time_limit_s: float = 25.0
    settle_s: float = 0.5  # a placement must stay still this long to count
    settle_speed: float = 0.02  # m/s
    released_closure: float = 0.1  # hand closure below which a block counts as let go
    cameras: tuple[str, ...] = ("head",)  # empty: no rendering
    image_hw: tuple[int, int] = (256, 256)
    render_quality: RenderQuality = FULL_QUALITY
    home_arm: tuple[float, ...] = HOME_ARM


@dataclass(frozen=True)
class PrivilegedState:
    """Simulator ground truth for the scripted expert and the evaluator only."""

    block_pos: dict[str, np.ndarray]
    block_yaw: dict[str, float]
    bin_xy: dict[str, tuple[float, float]]
    target: str
    destination: str
    qpos: np.ndarray


class SortEnv(gym.Env):
    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012 (gymnasium convention)

    def __init__(self, config: SortEnvConfig | None = None) -> None:
        self.config = cfg = config or SortEnvConfig()
        self.model = m = build_model(cfg.scene)
        self.data = mujoco.MjData(m)
        robot = cfg.scene.robot
        self.side = robot.active_side
        self.arm_joints = robot.active_joints[:7]
        self.hand_joints = robot.active_joints[7:]

        joint_ids = [m.joint(name).id for name in robot.active_joints]
        self._state_adr = m.jnt_qposadr[joint_ids]
        self._arm_adr = self._state_adr[:7]
        self._hand_adr = self._state_adr[7:]
        actuator_of = {m.joint(m.actuator_trnid[i, 0]).name: i for i in range(m.nu)}
        self._arm_act = np.array([actuator_of[j] for j in self.arm_joints])
        self._hand_act = np.array([actuator_of[j] for j in self.hand_joints])
        self.arm_lower = m.jnt_range[joint_ids[:7], 0].copy()
        self.arm_upper = m.jnt_range[joint_ids[:7], 1].copy()

        self.n_substeps = round(1.0 / (cfg.control_hz * m.opt.timestep))
        if not np.isclose(self.n_substeps * m.opt.timestep * cfg.control_hz, 1.0):
            raise ValueError("control period must be a whole number of physics steps")
        self.control_dt = self.n_substeps * m.opt.timestep

        self._block_body = {c: m.body(f"block_{c}").id for c in COLORS}
        self._block_geom = {c: m.geom(f"block_{c}").id for c in COLORS}
        block_joints = {c: m.joint(f"block_{c}_joint").id for c in COLORS}
        self._block_qpos = {c: m.jnt_qposadr[j] for c, j in block_joints.items()}
        self._block_dof = {c: m.jnt_dofadr[j] for c, j in block_joints.items()}
        pelvis = m.body("pelvis").id
        self._robot_geom = m.body_rootid[m.geom_bodyid] == pelvis

        h, w = cfg.image_hw
        spaces: dict[str, gym.Space] = {
            "state": gym.spaces.Box(-np.inf, np.inf, (len(robot.active_joints),), np.float32),
            "instruction": gym.spaces.Text(max_length=64, charset=INSTRUCTION_CHARSET),
        }
        if cfg.cameras:
            spaces["images"] = gym.spaces.Dict(
                {cam: gym.spaces.Box(0, 255, (h, w, 3), np.uint8) for cam in cfg.cameras}
            )
        self.observation_space = gym.spaces.Dict(spaces)
        self.action_space = gym.spaces.Box(
            low=np.append(self.arm_lower, 0.0).astype(np.float32),
            high=np.append(self.arm_upper, 1.0).astype(np.float32),
            dtype=np.float32,
        )

        self._renderer: CameraRenderer | None = None
        self._task: SortTask | None = None
        self._time = 0.0
        self._arm_target = np.asarray(cfg.home_arm, dtype=float)
        self._closure = 0.0
        self._outcome = Outcome.RUNNING
        self._candidate = Outcome.RUNNING
        self._settle_time = 0.0

    # Episode API ---------------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start an episode. ``options`` may fix ``task``, or ``target`` and ``destination``."""
        super().reset(seed=seed)
        options = options or {}
        task = options.get("task") or sample_task(
            self.np_random,
            self.config.layout,
            target=options.get("target"),
            destination=options.get("destination"),
        )
        m, d, scene = self.model, self.data, self.config.scene
        mujoco.mj_resetData(m, d)
        home = np.asarray(self.config.home_arm, dtype=float)
        open_hand = hand_targets(0.0, self.side)
        d.qpos[self._arm_adr] = home
        d.qpos[self._hand_adr] = open_hand
        d.ctrl[self._arm_act] = home
        d.ctrl[self._hand_act] = open_hand
        for color in COLORS:
            x, y = task.block_xy[color]
            yaw = task.block_yaw[color]
            adr = self._block_qpos[color]
            d.qpos[adr : adr + 3] = [x, y, scene.table_height + scene.block_half_size]
            d.qpos[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        mujoco.mj_forward(m, d)

        self._task = task
        self._time = 0.0
        self._arm_target = home
        self._closure = 0.0
        self._outcome = self._candidate = Outcome.RUNNING
        self._settle_time = 0.0
        return self._observe(), {"task": task, "outcome": self._outcome}

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._task is None:
            raise RuntimeError("call reset() before step()")
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape != (8,) or not np.all(np.isfinite(action)):
            raise ValueError(f"expected 8 finite values, got {action}")
        arm = np.clip(action[:7], self.arm_lower, self.arm_upper)
        max_delta = self.config.max_joint_speed * self.control_dt
        arm = self._arm_target + np.clip(arm - self._arm_target, -max_delta, max_delta)
        closure = np.clip(action[7], 0.0, 1.0)
        # Execute float32 values, the precision datasets store, so recorded actions replay
        # bit-for-bit; contact dynamics would otherwise amplify the rounding.
        arm = arm.astype(np.float32).astype(np.float64)
        closure = float(np.float32(closure))

        m, d = self.model, self.data
        d.ctrl[self._arm_act] = arm
        d.ctrl[self._hand_act] = hand_targets(closure, self.side)
        for _ in range(self.n_substeps):
            mujoco.mj_step(m, d)
        self._time += self.control_dt
        self._arm_target = arm
        self._closure = closure

        self._outcome = self._judge()
        terminated = self._outcome in TERMINAL
        truncated = not terminated and self._time >= self.config.time_limit_s - 1e-9
        if truncated:
            self._outcome = Outcome.TIMEOUT
        info = {
            "outcome": self._outcome,
            "applied_action": np.append(arm, closure),
            "time": self._time,
        }
        reward = float(self._outcome == Outcome.SUCCESS)
        return self._observe(), reward, terminated, truncated, info

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # Accessors -----------------------------------------------------------------------------

    @property
    def task(self) -> SortTask:
        if self._task is None:
            raise RuntimeError("no episode: call reset() first")
        return self._task

    @property
    def time(self) -> float:
        return self._time

    @property
    def outcome(self) -> Outcome:
        return self._outcome

    def privileged_state(self) -> PrivilegedState:
        d = self.data
        block_pos = {c: d.xpos[self._block_body[c]].copy() for c in COLORS}
        block_yaw = {c: _yaw(d.xquat[self._block_body[c]]) for c in COLORS}
        return PrivilegedState(
            block_pos=block_pos,
            block_yaw=block_yaw,
            bin_xy=dict(self.config.scene.bin_xy),
            target=self.task.target,
            destination=self.task.destination,
            qpos=d.qpos.copy(),
        )

    def render_camera(self, camera: str, height: int, width: int) -> np.ndarray:
        """Render any model camera (for videos and figures, not policy inputs)."""
        with CameraRenderer(self.model, height, width, self.config.render_quality) as renderer:
            return renderer.render(self.data, camera)

    # Internals -----------------------------------------------------------------------------

    def _observe(self) -> dict[str, Any]:
        obs: dict[str, Any] = {
            "state": self.data.qpos[self._state_adr].astype(np.float32),
            "instruction": self.task.instruction,
        }
        if self.config.cameras:
            if self._renderer is None:
                self._renderer = CameraRenderer(
                    self.model, *self.config.image_hw, quality=self.config.render_quality
                )
            obs["images"] = {
                cam: self._renderer.render(self.data, cam) for cam in self.config.cameras
            }
        return obs

    def _judge(self) -> Outcome:
        d, scene, task, cfg = self.data, self.config.scene, self.task, self.config
        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
            return Outcome.UNSTABLE
        if np.abs(d.qvel).max() > 1e3:
            return Outcome.UNSTABLE
        pos = {c: d.xpos[self._block_body[c]] for c in COLORS}
        if any(off_table(p, scene) for p in pos.values()):
            return Outcome.DROPPED

        def bin_of(color: str) -> str | None:
            bins = scene.bin_xy.items()
            return next((side for side, xy in bins if in_bin(pos[color], xy, scene)), None)

        candidate = Outcome.RUNNING
        if any(bin_of(c) is not None for c in COLORS if c != task.target):
            candidate = Outcome.WRONG_OBJECT
        elif bin_of(task.target) == task.destination:
            candidate = Outcome.SUCCESS
        elif bin_of(task.target) is not None:
            candidate = Outcome.WRONG_BIN

        dofs = (self._block_dof[c] for c in COLORS)
        still = max(np.linalg.norm(d.qvel[dof : dof + 3]) for dof in dofs) < cfg.settle_speed
        released = self._closure < cfg.released_closure and not self._touches_robot(task.target)
        needs_release = candidate in (Outcome.SUCCESS, Outcome.WRONG_BIN)
        holds = candidate != Outcome.RUNNING and still and (released or not needs_release)
        if holds and candidate == self._candidate:
            self._settle_time += self.control_dt
        elif holds:
            self._candidate, self._settle_time = candidate, self.control_dt
        else:
            self._candidate, self._settle_time = Outcome.RUNNING, 0.0
        return candidate if self._settle_time >= cfg.settle_s - 1e-9 else Outcome.RUNNING

    def _touches_robot(self, color: str) -> bool:
        d = self.data
        if d.ncon == 0:
            return False
        geom1, geom2 = d.contact.geom1[: d.ncon], d.contact.geom2[: d.ncon]
        block = self._block_geom[color]
        other = np.where(geom1 == block, geom2, np.where(geom2 == block, geom1, -1))
        other = other[other >= 0]
        return bool(np.any(self._robot_geom[other]))


def _yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
