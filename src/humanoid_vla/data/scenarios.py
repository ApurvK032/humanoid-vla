"""Scenarios: a layout seed plus an instruction, assigned to a frozen data split.

Splits own disjoint seed ranges, so a test layout can never leak into training data. Within a
split, instructions cycle through all six (block, bin) pairs in a shuffled order per block of
six layouts, which keeps them balanced, and some layouts are used twice with different
instructions so the policy cannot ignore the language.
"""

import itertools
from dataclasses import dataclass

import numpy as np

from humanoid_vla.envs.task import BINS, COLORS

# First seed of each split; each split owns the next SPLIT_SIZE seeds.
SPLIT_START = {"train": 100_000, "dev": 200_000, "test": 300_000}
SPLIT_SIZE = 100_000
INSTRUCTIONS = tuple(itertools.product(COLORS, BINS))


@dataclass(frozen=True)
class Scenario:
    split: str
    seed: int  # layout seed: SortEnv.reset(seed=...) places the blocks from it
    target: str
    destination: str

    @property
    def scenario_id(self) -> str:
        return f"{self.split}-{self.seed}-{self.target}-{self.destination}"

    @property
    def options(self) -> dict[str, str]:
        """``SortEnv.reset`` options that fix this scenario's instruction."""
        return {"target": self.target, "destination": self.destination}

    def rng(self) -> np.random.Generator:
        """Generator for per-episode randomness (e.g. the expert's parameters)."""
        return np.random.default_rng(
            (self.seed, COLORS.index(self.target), BINS.index(self.destination))
        )


def make_scenarios(
    split: str, num_layouts: int, first_layout: int = 0, paired_fraction: float = 1 / 3
) -> list[Scenario]:
    """Scenarios for layouts ``first_layout`` to ``first_layout + num_layouts - 1`` of a split.

    The result depends only on the arguments, and a range of layouts always gets the same
    scenarios whether it is generated alone or as part of a longer range.
    """
    if split not in SPLIT_START:
        raise ValueError(f"unknown split {split!r}")
    if first_layout < 0 or first_layout + num_layouts > SPLIT_SIZE:
        raise ValueError("layout range outside the split")

    scenarios = []
    for layout in range(first_layout, first_layout + num_layouts):
        seed = SPLIT_START[split] + layout
        block, slot = divmod(layout, len(INSTRUCTIONS))
        order = np.random.default_rng((SPLIT_START[split], block)).permutation(len(INSTRUCTIONS))
        target, destination = INSTRUCTIONS[order[slot]]
        scenarios.append(Scenario(split, seed, target, destination))
        pair_rng = np.random.default_rng((seed, 1))
        if pair_rng.random() < paired_fraction:
            others = [i for i in INSTRUCTIONS if i != (target, destination)]
            alt_target, alt_destination = others[pair_rng.integers(len(others))]
            scenarios.append(Scenario(split, seed, alt_target, alt_destination))
    return scenarios
