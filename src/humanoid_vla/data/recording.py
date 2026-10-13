"""Recording expert episodes as LeRobot frames.

Each frame pairs the observation the policy would have seen with the action executed right
after it (the environment's clipped command, not the expert's raw request). The feature names,
shapes and orders below are the dataset's action contract: docs/ACTION_CONTRACT.md.
"""

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from humanoid_vla.data.scenarios import Scenario
from humanoid_vla.envs.outcome import Outcome
from humanoid_vla.envs.sort import SortEnv, SortEnvConfig
from humanoid_vla.envs.task import SortTask
from humanoid_vla.policies.expert import ExpertConfig, ScriptedExpert
from humanoid_vla.sim.g1 import G1Config

STATE_KEY = "observation.state"
ACTION_KEY = "action"
CLOSURE_NAME = "hand_closure"


def image_key(camera: str) -> str:
    return f"observation.images.{camera}"


def state_names(robot: G1Config) -> list[str]:
    return list(robot.active_joints)


def action_names(robot: G1Config) -> list[str]:
    return [*robot.active_joints[:7], CLOSURE_NAME]


def dataset_features(config: SortEnvConfig) -> dict[str, dict[str, Any]]:
    """LeRobot feature specification for an environment's observations and actions."""
    robot = config.scene.robot
    h, w = config.image_hw
    features: dict[str, dict[str, Any]] = {
        image_key(cam): {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
        for cam in config.cameras
    }
    features[STATE_KEY] = {
        "dtype": "float32",
        "shape": (len(robot.active_joints),),
        "names": state_names(robot),
    }
    features[ACTION_KEY] = {"dtype": "float32", "shape": (8,), "names": action_names(robot)}
    return features


@dataclass
class RecordedEpisode:
    scenario: Scenario
    task: SortTask  # the layout and instruction the environment produced
    frames: list[dict[str, Any]]  # LeRobot frames, including "task"
    outcome: Outcome
    duration_s: float
    ik_failures: int
    expert_config: ExpertConfig

    def checksum(self) -> str:
        """SHA-256 of the episode's states and actions, to detect silent changes."""
        digest = hashlib.sha256()
        for frame in self.frames:
            digest.update(np.ascontiguousarray(frame[STATE_KEY]).tobytes())
            digest.update(np.ascontiguousarray(frame[ACTION_KEY]).tobytes())
        return digest.hexdigest()


def record_episode(
    env: SortEnv, scenario: Scenario, expert_config: ExpertConfig
) -> RecordedEpisode:
    """Run the scripted expert on one scenario and collect its frames."""
    obs, _ = env.reset(seed=scenario.seed, options=scenario.options)
    expert = ScriptedExpert(env, expert_config)
    expert.reset()
    frames = []
    while True:
        action = expert.act()
        next_obs, _, terminated, truncated, info = env.step(action)
        frame = {image_key(cam): obs["images"][cam] for cam in env.config.cameras}
        frame[STATE_KEY] = obs["state"]
        frame[ACTION_KEY] = info["applied_action"].astype(np.float32)
        frame["task"] = obs["instruction"]
        frames.append(frame)
        obs = next_obs
        if terminated or truncated:
            break
    return RecordedEpisode(
        scenario=scenario,
        task=env.task,
        frames=frames,
        outcome=info["outcome"],
        duration_s=env.time,
        ik_failures=expert.ik_failures,
        expert_config=expert_config,
    )
