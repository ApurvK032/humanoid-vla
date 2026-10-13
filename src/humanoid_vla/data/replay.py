"""Replaying recorded actions through the environment.

If observations and actions are aligned and mean what the action contract says, replaying an
episode's recorded actions from the same scenario reproduces its recorded joint states and its
outcome. Simulation is deterministic, so differences come only from the float32 rounding of
recorded actions.
"""

from dataclasses import dataclass

import numpy as np

from humanoid_vla.envs.outcome import Outcome
from humanoid_vla.envs.sort import SortEnv


@dataclass(frozen=True)
class ReplayResult:
    frames: int
    outcome: Outcome
    steps_to_end: int  # control steps until the episode terminated or was truncated
    max_state_error: float  # rad, over all recorded frames

    @property
    def length_matches(self) -> bool:
        return self.steps_to_end == self.frames


def replay_episode(
    env: SortEnv,
    seed: int,
    options: dict[str, str],
    actions: np.ndarray,
    states: np.ndarray,
) -> ReplayResult:
    """Reset to the episode's scenario, apply its actions, and compare states frame by frame.

    ``states[t]`` is the observation recorded before ``actions[t]`` was executed.
    """
    obs, _ = env.reset(seed=seed, options=options)
    errors = [float(np.abs(obs["state"] - states[0]).max())]
    outcome, steps = Outcome.RUNNING, 0
    for t, action in enumerate(actions):
        obs, _, terminated, truncated, info = env.step(action)
        steps = t + 1
        outcome = info["outcome"]
        if t + 1 < len(states):
            errors.append(float(np.abs(obs["state"] - states[t + 1]).max()))
        if terminated or truncated:
            break
    return ReplayResult(len(actions), outcome, steps, max(errors))
