"""Offscreen camera rendering with explicit quality settings."""

from dataclasses import dataclass
from types import TracebackType

import mujoco
import numpy as np


@dataclass(frozen=True)
class RenderQuality:
    """Per-frame rendering passes. Each pass costs a full extra render of the scene."""

    shadows: bool = True
    reflections: bool = True
    skybox: bool = True


FULL_QUALITY = RenderQuality()
FAST_QUALITY = RenderQuality(shadows=False, reflections=False, skybox=False)


class CameraRenderer:
    """Renders RGB images from named model cameras. Use as a context manager to free GL state."""

    def __init__(
        self,
        model: mujoco.MjModel,
        height: int,
        width: int,
        quality: RenderQuality = FULL_QUALITY,
    ) -> None:
        self._renderer = mujoco.Renderer(model, height, width)
        self.quality = quality

    def render(self, data: mujoco.MjData, camera: str) -> np.ndarray:
        """Return an (H, W, 3) uint8 image of the current state from ``camera``."""
        self._renderer.update_scene(data, camera)
        flags = self._renderer.scene.flags
        flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self.quality.shadows
        flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = self.quality.reflections
        flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = self.quality.skybox
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()

    def __enter__(self) -> "CameraRenderer":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def gl_renderer_name() -> str:
    """Name of the OpenGL renderer in the current context, e.g. 'D3D12 (NVIDIA GeForce RTX 4070)'.

    Only valid while a renderer's GL context is current (right after creating one).
    """
    from OpenGL import GL

    name = GL.glGetString(GL.GL_RENDERER)
    return name.decode() if isinstance(name, bytes) else str(name)
