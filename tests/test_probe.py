from typing import Any

import numpy as np

from humanoid_vla.data.scenarios import make_scenarios
from humanoid_vla.envs.sort import SortEnv, SortEnvConfig
from humanoid_vla.envs.task import COLORS
from humanoid_vla.evaluation.probe import CHUNK_STEPS, probe_layout
from humanoid_vla.policies.expert import ScriptedExpert


class ExpertPolicy:
    """Follows the instruction: the scripted expert's targets for the named block."""

    def __init__(self, env: SortEnv) -> None:
        self.env = env

    def reset(self) -> None:
        self.expert = ScriptedExpert(self.env)
        self.expert.reset()

    def predict_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        return np.array([self.expert.act() for _ in range(CHUNK_STEPS)])


class BlindPolicy:
    """Ignores the instruction: always the same action."""

    def __init__(self, action: np.ndarray) -> None:
        self.action = action

    def reset(self) -> None:
        pass

    def predict_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        return np.tile(self.action, (CHUNK_STEPS, 1))


def test_probe_tells_instruction_following_from_ignoring_it() -> None:
    env = SortEnv(SortEnvConfig(cameras=()))
    seed = make_scenarios("dev", 1)[0].seed

    following = probe_layout(env, ExpertPolicy(env), seed)
    assert all(following.nearest_block(following.policy_xy[c]) == c for c in COLORS)
    assert following.as_dict()["policy_spread_m"] > 0.05

    home = np.append(env.config.home_arm, 0.0).astype(np.float32)  # arm at home, hand open
    blind = probe_layout(env, BlindPolicy(home), seed)
    assert blind.as_dict()["policy_spread_m"] == 0.0
    assert len({blind.nearest_block(blind.policy_xy[c]) for c in COLORS}) == 1
    # The expert's reference does not depend on the policy probed.
    assert all(np.allclose(blind.expert_xy[c], following.expert_xy[c]) for c in COLORS)
