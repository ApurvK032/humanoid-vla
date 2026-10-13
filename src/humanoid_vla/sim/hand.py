"""Dex3 hand synergy: one closure value in [0, 1] drives all seven finger joints.

The synergy implements a "claw" pinch for a palm-down hand, in two stages:

- closure 0 to 0.5 folds the index and middle fingers down (straight to fully curled);
- closure 0.5 to 1 sweeps the thumb forward against the block's back face.

So closing forms a wall of fingers in front of the block before the thumb pushes the block into
it, and opening releases the thumb first, so nothing shoves the block as it is let go. The fully
closed pose lies beyond contact, so the actuators squeeze up to their torque limits
(1.4-2.45 N·m). ``PRESHAPE`` curls the fingers most of the way while the thumb stays open: the
expert approaches and descends with it, so the fingertips come straight down in front of the
block instead of sweeping forward over neighbouring objects.

Poses come from grasp trials on 5 cm blocks; see docs/DECISIONS.md (D-013).
"""

import numpy as np

from humanoid_vla.sim.g1 import Side

# Angles in g1.hand_joints() order: thumb 0-2, then index 0-1 and middle 0-1 (right hand).
THUMB_OPEN = np.array([0.0, 0.25, 0.0])
THUMB_CLOSED = np.array([0.0, -0.5, -0.4])
FINGERS_OPEN = np.array([0.0, 0.0, 0.0, 0.0])
FINGERS_CLOSED = np.array([1.5, 0.4, 1.5, 0.4])
FINGER_STAGE = 0.5  # closure at which the fingers are fully curled and the thumb starts
PRESHAPE = 0.4  # fingers 80% curled, thumb open


def hand_targets(closure: float, side: Side = "right") -> np.ndarray:
    """Joint targets for the hand's actuators, in ``g1.hand_joints(side)`` order."""
    if side != "right":
        raise NotImplementedError("the synergy is defined for the right hand only")
    c = float(np.clip(closure, 0.0, 1.0))
    fingers = min(1.0, c / FINGER_STAGE)
    thumb = max(0.0, (c - FINGER_STAGE) / (1.0 - FINGER_STAGE))
    return np.concatenate(
        [
            THUMB_OPEN + thumb * (THUMB_CLOSED - THUMB_OPEN),
            FINGERS_OPEN + fingers * (FINGERS_CLOSED - FINGERS_OPEN),
        ]
    )


def palm_down(heading: float) -> np.ndarray:
    """World rotation of the grasp site for a palm-down hand with fingers along ``heading``.

    The site's x axis points along the fingers and its y axis toward the thumb side of the palm,
    which faces the table.
    """
    x = np.array([np.cos(heading), np.sin(heading), 0.0])
    y = np.array([0.0, 0.0, -1.0])
    return np.column_stack([x, y, np.cross(x, y)])
