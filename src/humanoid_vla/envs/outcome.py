"""Episode outcomes, judged from simulator state by the evaluator (never seen by the policy)."""

from enum import StrEnum

import numpy as np

from humanoid_vla.sim.scene import SceneConfig


class Outcome(StrEnum):
    RUNNING = "running"
    SUCCESS = "success"  # requested block settled in the requested bin, released by the hand
    WRONG_BIN = "wrong_bin"  # requested block settled in the other bin
    WRONG_OBJECT = "wrong_object"  # another block settled in a bin
    DROPPED = "dropped"  # a block fell off the table
    TIMEOUT = "timeout"
    UNSTABLE = "unstable"  # the simulation diverged


TERMINAL = frozenset(Outcome) - {Outcome.RUNNING, Outcome.TIMEOUT}


def in_bin(block_pos: np.ndarray, bin_xy: tuple[float, float], scene: SceneConfig) -> bool:
    """Block center inside the bin's walls and below its rim."""
    inner = scene.bin_inner_half_size
    rim_z = scene.table_height + scene.bin_wall_thickness + scene.bin_wall_height
    return (
        abs(block_pos[0] - bin_xy[0]) < inner
        and abs(block_pos[1] - bin_xy[1]) < inner
        and block_pos[2] < rim_z
    )


def off_table(block_pos: np.ndarray, scene: SceneConfig) -> bool:
    return block_pos[2] < scene.table_height - 0.1
