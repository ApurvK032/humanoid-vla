"""One closed-loop episode of a policy, judged by the environment's evaluator.

Besides the outcome, a rollout records which block the policy lifted first and where every
block ended up. That separates "picked the wrong block" (the instruction was not followed) from
"picked the right block and dropped it" (a manipulation failure), even when the episode fails.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from humanoid_vla.data.scenarios import Scenario
from humanoid_vla.envs.outcome import Outcome, in_bin
from humanoid_vla.envs.sort import SortEnv
from humanoid_vla.envs.task import COLORS

LIFT_THRESHOLD = 0.03  # m above a resting block: the policy has picked this block


class Policy(Protocol):
    """What a rollout needs from a policy: chunked actions from observations."""

    @property
    def needs_observation(self) -> bool: ...

    def reset(self) -> None: ...

    def act(self, obs: dict[str, Any]) -> np.ndarray: ...


@dataclass(frozen=True)
class RolloutResult:
    scenario: Scenario
    outcome: Outcome
    duration_s: float
    first_lifted: str | None  # color of the first block raised off the table, if any
    final_bins: dict[str, str | None]  # color -> bin side it ended in, or None
    chunks: int  # observations the policy acted on
    inference_ms: float  # mean wall time per chunk prediction

    @property
    def picked_target(self) -> bool:
        return self.first_lifted == self.scenario.target

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario.scenario_id,
            "seed": self.scenario.seed,
            "target": self.scenario.target,
            "destination": self.scenario.destination,
            "outcome": str(self.outcome),
            "duration_s": round(self.duration_s, 3),
            "first_lifted": self.first_lifted,
            "picked_target": self.picked_target,
            "final_bins": self.final_bins,
            "chunks": self.chunks,
            "inference_ms": round(self.inference_ms, 1),
        }


def run_rollout(
    env: SortEnv,
    policy: Policy,
    scenario: Scenario,
    on_step: Callable[[], None] | None = None,
) -> RolloutResult:
    """Run ``policy`` from ``scenario`` until the environment ends the episode."""
    obs, _ = env.reset(seed=scenario.seed, options=scenario.options)
    policy.reset()
    scene = env.config.scene
    lifted_z = scene.table_height + scene.block_half_size + LIFT_THRESHOLD
    first_lifted: str | None = None
    chunk_seconds: list[float] = []
    try:
        while True:
            new_chunk = policy.needs_observation
            start = time.perf_counter()
            action = policy.act(obs)
            if new_chunk:
                chunk_seconds.append(time.perf_counter() - start)
            env.render_images = policy.needs_observation
            obs, _, terminated, truncated, info = env.step(action)
            if first_lifted is None:
                positions = env.privileged_state().block_pos
                first_lifted = next((c for c in COLORS if positions[c][2] > lifted_z), None)
            if on_step is not None:
                on_step()
            if terminated or truncated:
                break
    finally:
        env.render_images = True

    positions = env.privileged_state().block_pos
    final_bins = {
        color: next(
            (side for side, xy in scene.bin_xy.items() if in_bin(positions[color], xy, scene)),
            None,
        )
        for color in COLORS
    }
    return RolloutResult(
        scenario=scenario,
        outcome=info["outcome"],
        duration_s=env.time,
        first_lifted=first_lifted,
        final_bins=final_bins,
        chunks=len(chunk_seconds),
        inference_ms=1e3 * float(np.mean(chunk_seconds)) if chunk_seconds else 0.0,
    )
