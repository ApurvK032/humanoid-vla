import math
from typing import Any

import numpy as np
import pytest

from humanoid_vla.data.scenarios import Scenario, make_scenarios
from humanoid_vla.envs.outcome import Outcome
from humanoid_vla.envs.sort import SortEnv, SortEnvConfig
from humanoid_vla.evaluation.rollout import run_rollout
from humanoid_vla.policies.expert import ScriptedExpert


class ChunkedExpert:
    """The scripted expert behind the chunked-policy interface a learned policy exposes."""

    def __init__(self, env: SortEnv, n_action_steps: int) -> None:
        self.env = env
        self.expert = ScriptedExpert(env)
        self.n_action_steps = n_action_steps
        self.calls = 0
        self.images_seen: list[bool] = []

    @property
    def needs_observation(self) -> bool:
        return self.calls % self.n_action_steps == 0

    def reset(self) -> None:
        self.expert.reset()
        self.calls = 0

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        self.images_seen.append("images" in obs)
        self.calls += 1
        return self.expert.act()


@pytest.fixture(scope="module")
def scenario() -> Scenario:
    return make_scenarios("dev", 1)[0]


def test_rollout_reports_success_and_the_picked_block(scenario: Scenario) -> None:
    env = SortEnv(SortEnvConfig(cameras=()))
    policy = ChunkedExpert(env, n_action_steps=10)
    result = run_rollout(env, policy, scenario)
    assert result.outcome == Outcome.SUCCESS
    assert result.first_lifted == scenario.target
    assert result.picked_target
    assert result.final_bins[scenario.target] == scenario.destination
    steps = round(result.duration_s / env.control_dt)
    assert result.chunks == math.ceil(steps / 10)
    assert env.render_images


@pytest.mark.render
def test_cameras_render_only_when_a_chunk_starts(scenario: Scenario) -> None:
    env = SortEnv(SortEnvConfig(cameras=("head",), image_hw=(32, 32)))
    policy = ChunkedExpert(env, n_action_steps=5)
    run_rollout(env, policy, scenario)
    expected = [step % 5 == 0 for step in range(len(policy.images_seen))]
    assert policy.images_seen == expected
    env.close()
