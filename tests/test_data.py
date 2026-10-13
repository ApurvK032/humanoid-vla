import collections
import itertools
from pathlib import Path

import numpy as np
import pytest

from humanoid_vla.data.recording import (
    ACTION_KEY,
    STATE_KEY,
    action_names,
    dataset_features,
    record_episode,
)
from humanoid_vla.data.replay import replay_episode
from humanoid_vla.data.scenarios import (
    INSTRUCTIONS,
    SPLIT_SIZE,
    SPLIT_START,
    Scenario,
    make_scenarios,
)
from humanoid_vla.envs.outcome import Outcome
from humanoid_vla.envs.sort import SortEnv, SortEnvConfig
from humanoid_vla.policies.expert import ExpertConfig, randomized


@pytest.fixture(scope="module")
def recorded():
    env = SortEnv(SortEnvConfig(cameras=()))
    scenario = make_scenarios("dev", 1)[0]
    episode = record_episode(env, scenario, randomized(ExpertConfig(), scenario.rng()))
    yield env, episode
    env.close()


def test_split_seed_ranges_are_disjoint() -> None:
    starts = sorted(SPLIT_START.values())
    assert all(b - a >= SPLIT_SIZE for a, b in itertools.pairwise(starts))


def test_scenarios_are_deterministic_and_range_consistent() -> None:
    assert make_scenarios("train", 30) == make_scenarios("train", 30)
    whole = make_scenarios("train", 30)
    part = make_scenarios("train", 10, first_layout=12)
    assert [s for s in whole if 100_012 <= s.seed < 100_022] == part


def test_every_block_of_six_layouts_covers_every_instruction() -> None:
    scenarios = make_scenarios("train", 60, paired_fraction=0.0)
    for block in range(10):
        chunk = scenarios[6 * block : 6 * block + 6]
        assert {(s.target, s.destination) for s in chunk} == set(INSTRUCTIONS)


def test_paired_layouts_get_two_different_instructions() -> None:
    scenarios = make_scenarios("train", 300)
    by_seed = collections.defaultdict(list)
    for s in scenarios:
        by_seed[s.seed].append((s.target, s.destination))
    paired = [v for v in by_seed.values() if len(v) == 2]
    assert 60 < len(paired) < 140  # about a third of 300 layouts
    assert all(a != b for a, b in paired)


def test_invalid_split_or_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="split"):
        make_scenarios("holdout", 1)
    with pytest.raises(ValueError, match="outside"):
        make_scenarios("dev", 2, first_layout=SPLIT_SIZE - 1)


def test_scenario_rng_is_stable() -> None:
    a = Scenario("train", 100_000, "red", "left")
    assert a.rng().random() == a.rng().random()
    assert a.rng().random() != Scenario("train", 100_000, "red", "right").rng().random()


def test_features_follow_the_action_contract() -> None:
    config = SortEnvConfig(cameras=("head", "wrist"), image_hw=(64, 64))
    features = dataset_features(config)
    assert features["observation.images.wrist"]["shape"] == (64, 64, 3)
    assert features[STATE_KEY]["shape"] == (14,)
    assert features[ACTION_KEY]["names"] == action_names(config.scene.robot)
    assert features[ACTION_KEY]["names"][-1] == "hand_closure"


def test_recorded_episode_succeeds_and_replays_exactly(recorded) -> None:
    env, episode = recorded
    assert episode.outcome == Outcome.SUCCESS
    states = np.stack([f[STATE_KEY] for f in episode.frames])
    actions = np.stack([f[ACTION_KEY] for f in episode.frames])
    assert actions.dtype == np.float32
    result = replay_episode(env, episode.scenario.seed, episode.scenario.options, actions, states)
    assert result.outcome == Outcome.SUCCESS
    assert result.length_matches
    assert result.max_state_error == 0.0


def test_lerobot_round_trip(recorded, tmp_path: Path) -> None:
    lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    env, episode = recorded
    root = tmp_path / "dataset"
    dataset = lerobot_dataset.LeRobotDataset.create(
        repo_id="local/test",
        fps=round(1 / env.control_dt),
        features=dataset_features(env.config),
        root=root,
        use_videos=False,
    )
    for frame in episode.frames:
        dataset.add_frame(dict(frame))
    dataset.save_episode()
    dataset.finalize()

    loaded = lerobot_dataset.LeRobotDataset("local/test", root=root)
    assert loaded.num_episodes == 1
    columns = loaded.hf_dataset.with_format("numpy")
    states, actions = np.stack(columns[STATE_KEY]), np.stack(columns[ACTION_KEY])
    np.testing.assert_array_equal(actions, np.stack([f[ACTION_KEY] for f in episode.frames]))
    result = replay_episode(env, episode.scenario.seed, episode.scenario.options, actions, states)
    assert result.outcome == Outcome.SUCCESS
    assert result.max_state_error == 0.0


def test_lerobot_serves_action_chunks_with_end_padding(recorded, tmp_path: Path) -> None:
    """SmolVLA trains on 50-action chunks; past an episode's end LeRobot pads and flags them."""
    lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    env, episode = recorded
    fps = round(1 / env.control_dt)
    root = tmp_path / "dataset"
    dataset = lerobot_dataset.LeRobotDataset.create(
        repo_id="local/chunks",
        fps=fps,
        features=dataset_features(env.config),
        root=root,
        use_videos=False,
    )
    for frame in episode.frames:
        dataset.add_frame(dict(frame))
    dataset.save_episode()
    dataset.finalize()

    chunk = 50
    loaded = lerobot_dataset.LeRobotDataset(
        "local/chunks", root=root, delta_timestamps={ACTION_KEY: [i / fps for i in range(chunk)]}
    )
    actions = np.stack([f[ACTION_KEY] for f in episode.frames])
    first = loaded[0]
    assert tuple(first[ACTION_KEY].shape) == (chunk, 8)
    np.testing.assert_array_equal(first[ACTION_KEY].numpy(), actions[:chunk])
    assert not first["action_is_pad"].any()

    near_end = len(actions) - 10
    item = loaded[near_end]
    assert item["action_is_pad"].numpy().tolist() == [False] * 10 + [True] * (chunk - 10)
    np.testing.assert_array_equal(item[ACTION_KEY].numpy()[:10], actions[near_end:])
