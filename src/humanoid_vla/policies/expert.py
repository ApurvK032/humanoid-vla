"""Scripted expert: picks up the requested block and places it in the requested bin.

The expert reads simulator ground truth (block pose, bin position) through
``SortEnv.privileged_state()``, which a learned policy never sees. It plans straight-line moves
of the grasp site with a palm-down hand, and inverse kinematics turns each control step's target
into arm joint angles. It emits actions in exactly the format a learned policy does: 7 arm joint
targets and one hand closure.

Before closing and before releasing, the plan waits until the hand has actually arrived, so
tracking lag (arm targets may move at most 2 rad/s) cannot make it grasp or let go early.
"""

from dataclasses import dataclass, field

import numpy as np

from humanoid_vla.envs.sort import SortEnv
from humanoid_vla.sim.g1 import GRASP_SITE
from humanoid_vla.sim.hand import PRESHAPE, palm_down
from humanoid_vla.sim.ik import IKSolver


@dataclass(frozen=True)
class ExpertConfig:
    approach_height: float = 0.10  # pre-grasp: grasp site this far above the block center
    carry_height: float = 0.16  # grasp site height above the table while carrying
    release_height: float = 0.10  # grasp site height above the table when letting go
    move_speed: float = 0.25  # m/s for moves between waypoints
    turn_speed: float = 1.5  # rad/s of hand heading change
    min_move_s: float = 0.6
    descend_s: float = 0.8
    close_s: float = 0.6
    grip_s: float = 0.2
    lift_s: float = 0.8
    lower_s: float = 0.5
    release_s: float = 0.4
    retreat_s: float = 0.6
    # The right arm reaches palm-down without self-collision when the hand turns further left
    # the further left the block is: about 20 degrees mid-strip, 50 at its left edge
    # (D-014's reach map). Cubes look the same every 90 degrees, so the grasp heading is the
    # block-aligned heading nearest the preferred one, clipped to the reachable range. Grasp
    # trials tolerated 30 degrees of misalignment.
    preferred_heading: float = np.deg2rad(20.0)  # at y = heading_center_y
    heading_center_y: float = -0.20
    heading_per_meter: float = np.deg2rad(250.0)  # extra heading per meter to the left
    heading_range: tuple[float, float] = (0.0, np.deg2rad(60.0))
    # Hand heading over each bin while carrying, from the collision-free reach map.
    bin_heading: dict[str, float] = field(
        default_factory=lambda: {"left": np.deg2rad(40.0), "right": 0.0}
    )
    arrive_tolerance: float = 0.01  # m between the grasp site and its target
    max_wait_s: float = 1.0  # longest wait for the hand to arrive before going on anyway
    ik_tolerance: float = 0.005  # m; larger IK errors are counted as failures


@dataclass(frozen=True)
class Segment:
    name: str
    duration: float
    pos0: np.ndarray
    pos1: np.ndarray
    heading0: float
    heading1: float
    closure0: float
    closure1: float
    wait_to_arrive: bool = False  # hold at the end until the hand reaches pos1


def grasp_heading(block_yaw: float, preferred: float, lo: float, hi: float) -> float:
    """Block-aligned hand heading nearest ``preferred``, clipped to [lo, hi]."""
    k = np.round((preferred - block_yaw) / (np.pi / 2))
    return float(np.clip(block_yaw + k * np.pi / 2, lo, hi))


class ScriptedExpert:
    def __init__(self, env: SortEnv, config: ExpertConfig | None = None) -> None:
        self.env = env
        self.config = config or ExpertConfig()
        self.ik = IKSolver(env.model, GRASP_SITE, env.arm_joints)
        self._site_id = env.model.site(GRASP_SITE).id
        self._home = np.asarray(env.config.home_arm, dtype=float)
        self._arm_adr = [env.model.joint(j).qposadr[0] for j in env.arm_joints]
        self._plan: list[Segment] = []
        self._t = 0.0
        self._waited = 0.0
        self._q = self._home.copy()
        self.ik_failures = 0

    @property
    def phase(self) -> str:
        """Name of the segment the expert is executing, or "done"."""
        segment, _, _ = self._segment_at(self._t)
        return segment.name if segment is not None else "done"

    def reset(self) -> None:
        """Plan the episode from the environment's current state; call after ``env.reset``."""
        cfg, env = self.config, self.env
        state = env.privileged_state()
        table = env.config.scene.table_height
        block = state.block_pos[state.target]
        leftward = block[1] - cfg.heading_center_y
        preferred = cfg.preferred_heading + cfg.heading_per_meter * leftward
        heading = grasp_heading(state.block_yaw[state.target], preferred, *cfg.heading_range)
        bin_heading = cfg.bin_heading[state.destination]
        bin_x, bin_y = state.bin_xy[state.destination]

        start_pos, start_rot = self.ik.site_pose(state.qpos)
        start_heading = float(np.arctan2(start_rot[1, 0], start_rot[0, 0]))
        above = block + np.array([0.0, 0.0, cfg.approach_height])
        lifted = np.array([block[0], block[1], table + cfg.carry_height])
        over_bin = np.array([bin_x, bin_y, table + cfg.carry_height])
        release = np.array([bin_x, bin_y, table + cfg.release_height])

        bh, pre = bin_heading, PRESHAPE
        self._plan = [
            self._move("approach", start_pos, above, start_heading, heading, 0.0, pre),
            Segment("descend", cfg.descend_s, above, block, heading, heading, pre, pre, True),
            Segment("close", cfg.close_s, block, block, heading, heading, pre, 1.0),
            Segment("grip", cfg.grip_s, block, block, heading, heading, 1.0, 1.0),
            Segment("lift", cfg.lift_s, block, lifted, heading, heading, 1.0, 1.0),
            self._move("carry", lifted, over_bin, heading, bh, 1.0, 1.0),
            Segment("lower", cfg.lower_s, over_bin, release, bh, bh, 1.0, 1.0, True),
            Segment("release", cfg.release_s, release, release, bh, bh, 1.0, pre),
            Segment("retreat", cfg.retreat_s, release, over_bin, bh, bh, pre, pre),
            self._move("return", over_bin, start_pos, bh, start_heading, pre, 0.0),
        ]
        self._t = 0.0
        self._waited = 0.0
        self._q = state.qpos[self._arm_adr].copy()
        self.ik_failures = 0

    def act(self) -> np.ndarray:
        """Action for the next control step: 7 arm joint targets and the hand closure."""
        if not self._plan:
            raise RuntimeError("call reset() after env.reset() first")
        dt = self.env.control_dt
        segment, _, end = self._segment_at(self._t)
        next_t = self._t + dt
        if segment is not None and segment.wait_to_arrive and next_t > end:
            site = self.env.data.site_xpos[self._site_id]
            arrived = np.linalg.norm(site - segment.pos1) < self.config.arrive_tolerance
            if not arrived and self._waited < self.config.max_wait_s:
                self._waited += dt
                next_t = end  # hold the plan at this segment's end
            else:
                self._waited = 0.0
        self._t = next_t

        segment, s, _ = self._segment_at(self._t)
        if segment is None:  # plan finished: hold the final pose
            segment, s = self._plan[-1], 1.0
        smooth = 3 * s**2 - 2 * s**3
        pos = segment.pos0 + smooth * (segment.pos1 - segment.pos0)
        heading = segment.heading0 + smooth * _wrap(segment.heading1 - segment.heading0)
        closure = segment.closure0 + smooth * (segment.closure1 - segment.closure0)

        qpos = self.env.data.qpos.copy()
        qpos[self._arm_adr] = self._q
        result = self.ik.solve(qpos, pos, palm_down(heading), posture=self._home)
        if result.pos_err > self.config.ik_tolerance:
            self.ik_failures += 1
        self._q = result.qpos
        return np.append(result.qpos, closure)

    def _move(
        self,
        name: str,
        pos0: np.ndarray,
        pos1: np.ndarray,
        heading0: float,
        heading1: float,
        closure0: float,
        closure1: float,
    ) -> Segment:
        cfg = self.config
        duration = max(
            cfg.min_move_s,
            float(np.linalg.norm(pos1 - pos0)) / cfg.move_speed,
            abs(_wrap(heading1 - heading0)) / cfg.turn_speed,
        )
        return Segment(name, duration, pos0, pos1, heading0, heading1, closure0, closure1)

    def _segment_at(self, t: float) -> tuple[Segment | None, float, float]:
        """Segment active at plan time ``t``, progress through it in [0, 1], and its end time."""
        start = 0.0
        for segment in self._plan:
            end = start + segment.duration
            if t <= end + 1e-9:
                return segment, min(1.0, max(0.0, (t - start) / segment.duration)), end
            start = end
        return None, 1.0, start


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)
