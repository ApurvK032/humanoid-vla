import numpy as np
import pytest

from humanoid_vla.envs.outcome import Outcome
from humanoid_vla.envs.sort import SortEnv, SortEnvConfig
from humanoid_vla.policies.expert import ScriptedExpert, grasp_heading


@pytest.mark.parametrize("block_yaw_deg", [-44.0, -20.0, 0.0, 15.0, 44.0])
def test_grasp_heading_is_square_to_the_block(block_yaw_deg: float) -> None:
    yaw = np.deg2rad(block_yaw_deg)
    heading = grasp_heading(yaw, preferred=np.deg2rad(20.0), lo=-np.pi, hi=np.pi)
    quarter_turns = (heading - yaw) / (np.pi / 2)
    assert quarter_turns == pytest.approx(round(quarter_turns))
    assert abs(heading - np.deg2rad(20.0)) <= np.pi / 4 + 1e-9


def test_grasp_heading_is_clipped_to_the_reachable_range() -> None:
    assert grasp_heading(np.deg2rad(-10.0), np.deg2rad(20.0), 0.0, np.deg2rad(60.0)) == 0.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_expert_sorts_the_requested_block(seed: int) -> None:
    env = SortEnv(SortEnvConfig(cameras=()))
    expert = ScriptedExpert(env)
    env.reset(seed=seed)
    expert.reset()
    while True:
        _, _, terminated, truncated, info = env.step(expert.act())
        if terminated or truncated:
            break
    assert info["outcome"] == Outcome.SUCCESS, env.task.instruction
    assert expert.ik_failures == 0
