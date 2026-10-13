import itertools

import numpy as np
import pytest

from humanoid_vla.envs.task import BINS, COLORS, LayoutConfig, sample_task


def test_layout_respects_region_spacing_and_yaw() -> None:
    layout = LayoutConfig()
    rng = np.random.default_rng(0)
    for _ in range(200):
        task = sample_task(rng, layout)
        xy = np.array([task.block_xy[c] for c in COLORS])
        assert np.all((xy[:, 0] >= layout.x_range[0]) & (xy[:, 0] <= layout.x_range[1]))
        assert np.all((xy[:, 1] >= layout.y_range[0]) & (xy[:, 1] <= layout.y_range[1]))
        for a, b in itertools.combinations(xy, 2):
            assert np.linalg.norm(a - b) >= layout.min_distance
        assert all(layout.yaw_range[0] <= y <= layout.yaw_range[1] for y in task.block_yaw.values())


def test_same_seed_gives_the_same_task() -> None:
    a = sample_task(np.random.default_rng(7))
    b = sample_task(np.random.default_rng(7))
    assert a == b


def test_target_and_destination_can_be_fixed() -> None:
    task = sample_task(np.random.default_rng(0), target="green", destination="left")
    assert (task.target, task.destination) == ("green", "left")
    assert task.instruction == "Put the green block in the left bin."
    with pytest.raises(ValueError, match="unknown"):
        sample_task(np.random.default_rng(0), target="purple")


def test_random_instructions_cover_every_combination() -> None:
    rng = np.random.default_rng(0)
    seen = {(t.target, t.destination) for t in (sample_task(rng) for _ in range(200))}
    assert seen == set(itertools.product(COLORS, BINS))


def test_fixing_the_instruction_keeps_the_layout() -> None:
    free = sample_task(np.random.default_rng(11))
    for target, destination in itertools.product(COLORS, BINS):
        fixed = sample_task(np.random.default_rng(11), target=target, destination=destination)
        assert fixed.block_xy == free.block_xy
        assert fixed.block_yaw == free.block_yaw
