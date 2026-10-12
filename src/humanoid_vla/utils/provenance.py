"""Provenance attached to every committed result file, so each number can be traced and re-run."""

import datetime
import json
import platform
import shutil
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

from humanoid_vla.gl_env import is_wsl
from humanoid_vla.paths import REPO_ROOT

TRACKED_PACKAGES = ("mujoco", "numpy", "torch", "lerobot", "transformers", "peft")
# WSL2 exposes the Windows driver's nvidia-smi here without adding it to PATH.
_WSL_NVIDIA_SMI = "/usr/lib/wsl/lib/nvidia-smi"


def _run(*cmd: str) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=REPO_ROOT)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def git_state() -> dict[str, Any]:
    status = _run("git", "status", "--porcelain")
    return {"commit": _run("git", "rev-parse", "HEAD"), "dirty": bool(status)}


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def gpu_info() -> dict[str, str] | None:
    smi = shutil.which("nvidia-smi") or (
        _WSL_NVIDIA_SMI if Path(_WSL_NVIDIA_SMI).exists() else None
    )
    if smi is None:
        return None
    out = _run(smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader")
    if not out:
        return None
    name, memory, driver = (field.strip() for field in out.splitlines()[0].split(","))
    return {"name": name, "memory_total": memory, "driver": driver}


def provenance() -> dict[str, Any]:
    return {
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "git": git_state(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "wsl": is_wsl(),
        "packages": package_versions(),
        "gpu": gpu_info(),
    }


def write_result(path: Path, result: dict[str, Any]) -> None:
    """Write ``result`` as JSON with a ``provenance`` block."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"provenance": provenance(), **result}
    path.write_text(json.dumps(payload, indent=2) + "\n")
