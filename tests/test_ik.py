import mujoco
import numpy as np
import pytest

from humanoid_vla.sim.g1 import GRASP_SITE, arm_joints
from humanoid_vla.sim.hand import palm_down
from humanoid_vla.sim.ik import IKSolver


@pytest.fixture(scope="module")
def solver(model: mujoco.MjModel) -> IKSolver:
    return IKSolver(model, GRASP_SITE, arm_joints("right"))


@pytest.mark.parametrize(
    ("target", "heading_deg"),
    [((0.22, -0.20, 0.805), 20.0), ((0.25, -0.30, 0.90), 0.0), ((0.40, -0.06, 0.90), 40.0)],
)
def test_ik_reaches_palm_down_targets(
    model: mujoco.MjModel, solver: IKSolver, target: tuple, heading_deg: float
) -> None:
    data = mujoco.MjData(model)
    rotation = palm_down(np.deg2rad(heading_deg))
    result = solver.solve(data.qpos.copy(), np.array(target), rotation)
    assert result.success
    assert np.all(result.qpos >= solver.lower)
    assert np.all(result.qpos <= solver.upper)

    qpos = data.qpos.copy()
    qpos[model.jnt_qposadr[[model.joint(j).id for j in arm_joints("right")]]] = result.qpos
    pos, rot = solver.site_pose(qpos)
    np.testing.assert_allclose(pos, target, atol=1e-3)
    np.testing.assert_allclose(rot, rotation, atol=2e-2)


def test_ik_reports_unreachable_targets(model: mujoco.MjModel, solver: IKSolver) -> None:
    data = mujoco.MjData(model)
    result = solver.solve(data.qpos.copy(), np.array([1.5, 0.0, 0.9]))
    assert not result.success
    assert result.pos_err > 0.5
