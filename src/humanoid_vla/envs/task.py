"""Sorting tasks: which block goes into which bin, and where the blocks start."""

from dataclasses import dataclass

import numpy as np

COLORS = ("red", "green", "blue")
BINS = ("left", "right")


@dataclass(frozen=True)
class LayoutConfig:
    """Region where blocks are placed at reset: a strip in front of the bins that the right arm
    reaches palm-down without self-collision. Its left edge stops at y = -0.11: from closer to
    the robot's center, carrying a block to the left bin pressed the upper arm against the torso
    and knocked the block out of the hand (docs/reports/m1-scripted-expert.md)."""

    x_range: tuple[float, float] = (0.17, 0.26)
    y_range: tuple[float, float] = (-0.33, -0.11)
    min_distance: float = 0.09  # between block centers; leaves room for the open thumb
    yaw_range: tuple[float, float] = (-np.pi / 4, np.pi / 4)  # cubes repeat every 90 degrees
    max_tries: int = 1000


@dataclass(frozen=True)
class SortTask:
    target: str  # block color
    destination: str  # bin side
    block_xy: dict[str, tuple[float, float]]
    block_yaw: dict[str, float]

    @property
    def instruction(self) -> str:
        return f"Put the {self.target} block in the {self.destination} bin."


def sample_task(
    rng: np.random.Generator,
    layout: LayoutConfig | None = None,
    target: str | None = None,
    destination: str | None = None,
) -> SortTask:
    """Sample block positions and an instruction. Fix ``target`` or ``destination`` to balance
    a dataset across instructions."""
    layout = layout or LayoutConfig()
    # Always draw the instruction, even when it is fixed, so a seed gives the same layout under
    # every instruction (and the same layouts as the M1 evaluations, which fixed nothing).
    drawn_target, drawn_destination = str(rng.choice(COLORS)), str(rng.choice(BINS))
    target = target or drawn_target
    destination = destination or drawn_destination
    if target not in COLORS or destination not in BINS:
        raise ValueError(f"unknown target {target!r} or destination {destination!r}")

    low = (layout.x_range[0], layout.y_range[0])
    high = (layout.x_range[1], layout.y_range[1])
    for _ in range(layout.max_tries):
        xy = rng.uniform(low, high, size=(len(COLORS), 2))
        gaps = [np.linalg.norm(xy[i] - xy[j]) for i in range(len(xy)) for j in range(i)]
        if min(gaps) >= layout.min_distance:
            break
    else:
        raise RuntimeError("could not place blocks; the layout region is too small")
    block_xy = {color: (float(x), float(y)) for color, (x, y) in zip(COLORS, xy, strict=True)}
    block_yaw = {color: float(rng.uniform(*layout.yaw_range)) for color in COLORS}
    return SortTask(target, destination, block_xy, block_yaw)
