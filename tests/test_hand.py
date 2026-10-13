import mujoco
import numpy as np
import pytest

from humanoid_vla.sim.g1 import hand_joints
from humanoid_vla.sim.hand import (
    FINGER_STAGE,
    FINGERS_CLOSED,
    FINGERS_OPEN,
    THUMB_CLOSED,
    THUMB_OPEN,
    hand_targets,
)


def test_open_and_closed_poses() -> None:
    np.testing.assert_allclose(hand_targets(0.0), np.concatenate([THUMB_OPEN, FINGERS_OPEN]))
    np.testing.assert_allclose(hand_targets(1.0), np.concatenate([THUMB_CLOSED, FINGERS_CLOSED]))


def test_fingers_close_before_the_thumb() -> None:
    halfway = hand_targets(FINGER_STAGE)
    np.testing.assert_allclose(halfway[:3], THUMB_OPEN)
    np.testing.assert_allclose(halfway[3:], FINGERS_CLOSED)
    for closure in np.linspace(FINGER_STAGE, 1.0, 5):
        np.testing.assert_allclose(hand_targets(closure)[3:], FINGERS_CLOSED)


def test_closure_is_clipped() -> None:
    np.testing.assert_allclose(hand_targets(-1.0), hand_targets(0.0))
    np.testing.assert_allclose(hand_targets(2.0), hand_targets(1.0))


def test_targets_stay_within_joint_ranges(model: mujoco.MjModel) -> None:
    ranges = np.array([model.joint(j).range for j in hand_joints("right")])
    for closure in np.linspace(0.0, 1.0, 21):
        targets = hand_targets(closure)
        assert np.all(targets >= ranges[:, 0] - 1e-9)
        assert np.all(targets <= ranges[:, 1] + 1e-9)


def test_left_hand_is_not_supported() -> None:
    with pytest.raises(NotImplementedError):
        hand_targets(0.5, "left")
