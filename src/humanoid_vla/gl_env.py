"""OpenGL backend defaults for headless MuJoCo rendering.

Entry points (scripts, tests/conftest.py) call ``configure_gl_defaults()`` before importing
``mujoco``, because MuJoCo reads ``MUJOCO_GL`` at import time. Variables already set in the
environment take precedence.

On WSL2, Mesa's EGL falls back to llvmpipe (CPU) unless it is pointed at the NVIDIA GPU through
its D3D12 driver. Without the adapter name it picks the first adapter, which can be an integrated
GPU. See docs/RUNBOOK.md for measurements.
"""

import os
from pathlib import Path


def is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower()
    except OSError:
        return False


def configure_gl_defaults() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    if is_wsl():
        os.environ.setdefault("GALLIUM_DRIVER", "d3d12")
        os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")
