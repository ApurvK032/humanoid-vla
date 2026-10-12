from humanoid_vla.gl_env import configure_gl_defaults

configure_gl_defaults()

import mujoco
import pytest

from humanoid_vla.sim.scene import SceneConfig, build_model


@pytest.fixture(scope="session")
def scene_cfg() -> SceneConfig:
    return SceneConfig()


@pytest.fixture(scope="session")
def model(scene_cfg: SceneConfig) -> mujoco.MjModel:
    return build_model(scene_cfg)
