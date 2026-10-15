"""Open-loop probe of instruction following: where a policy's first action chunk takes the hand.

On one layout, the policy sees the same first observation with each block named as the target,
and the same noise. Forward kinematics of the last arm targets in the chunk gives where the policy
sends the hand in its first 2 s; the scripted expert's targets at that time are above the named
block. A policy that ignores the instruction sends the hand to the same place whatever the
instruction says. Unlike closed-loop rollouts, the probe isolates the instruction from everything
that happens after the first chunk.
"""

from dataclasses import dataclass
from typing import Any, Protocol

import mujoco
import numpy as np

from humanoid_vla.envs.sort import SortEnv
from humanoid_vla.envs.task import COLORS
from humanoid_vla.policies.expert import ScriptedExpert
from humanoid_vla.sim.g1 import GRASP_SITE

CHUNK_STEPS = 50  # SmolVLA's chunk: 2 s at 25 Hz


class ChunkPolicy(Protocol):
    """What the probe needs from a policy: seeded noise and whole predicted chunks."""

    def seed(self, seed: int) -> None: ...

    def reset(self) -> None: ...

    def predict_chunk(self, obs: dict[str, Any]) -> np.ndarray: ...


@dataclass(frozen=True)
class LayoutProbe:
    seed: int
    blocks_xy: dict[str, np.ndarray]  # color -> block position on the table
    policy_xy: dict[str, np.ndarray]  # instructed color -> hand target at the chunk's end
    expert_xy: dict[str, np.ndarray]  # instructed color -> the expert's hand target then

    def nearest_block(self, hand_xy: np.ndarray) -> str:
        return min(COLORS, key=lambda c: float(np.linalg.norm(hand_xy - self.blocks_xy[c])))

    def as_dict(self) -> dict[str, Any]:
        def spread(points: dict[str, np.ndarray]) -> float:
            xy = np.array(list(points.values()))
            return float(np.max(np.linalg.norm(xy[:, None] - xy[None], axis=-1)))

        return {
            "seed": self.seed,
            "policy_spread_m": round(spread(self.policy_xy), 4),
            "expert_spread_m": round(spread(self.expert_xy), 4),
            "instructions": {
                color: {
                    "policy_nearest_block": self.nearest_block(self.policy_xy[color]),
                    "policy_to_target_m": round(
                        float(np.linalg.norm(self.policy_xy[color] - self.blocks_xy[color])), 4
                    ),
                    "expert_nearest_block": self.nearest_block(self.expert_xy[color]),
                }
                for color in COLORS
            },
        }


class HandKinematics:
    """Hand (grasp site) position for given arm joint angles, without touching the simulation."""

    def __init__(self, env: SortEnv) -> None:
        self.env = env
        self.data = mujoco.MjData(env.model)
        self.site = env.model.site(GRASP_SITE).id
        self.arm_adr = [env.model.joint(name).qposadr[0] for name in env.arm_joints]

    def xy(self, arm: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = self.env.data.qpos
        self.data.qpos[self.arm_adr] = arm
        mujoco.mj_kinematics(self.env.model, self.data)
        return self.data.site_xpos[self.site][:2].copy()


def probe_layout(
    env: SortEnv, policy: ChunkPolicy, seed: int, destination: str = "left", noise_seed: int = 0
) -> LayoutProbe:
    """Name each block in turn on one layout, with the same policy noise for each."""
    kinematics = HandKinematics(env)
    blocks_xy: dict[str, np.ndarray] = {}
    policy_xy: dict[str, np.ndarray] = {}
    expert_xy: dict[str, np.ndarray] = {}
    for target in COLORS:
        options = {"target": target, "destination": destination}
        obs, _ = env.reset(seed=seed, options=options)
        positions = env.privileged_state().block_pos
        blocks_xy = {c: np.asarray(positions[c][:2]) for c in COLORS}
        policy.seed(noise_seed)
        policy.reset()
        chunk = policy.predict_chunk(obs)
        policy_xy[target] = kinematics.xy(chunk[CHUNK_STEPS - 1][:7])

        expert = ScriptedExpert(env)
        expert.reset()
        env.render_images = False  # the expert reads the simulator's state, not images
        try:
            for _ in range(CHUNK_STEPS):
                action = expert.act()
                env.step(action)
        finally:
            env.render_images = True
        expert_xy[target] = kinematics.xy(action[:7])
    return LayoutProbe(seed, blocks_xy, policy_xy, expert_xy)
