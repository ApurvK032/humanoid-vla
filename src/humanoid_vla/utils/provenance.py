"""Provenance attached to every committed result file, so each number can be traced and re-run."""

import datetime
import json
import os
import platform
import shutil
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

from humanoid_vla.gl_env import is_wsl
from humanoid_vla.paths import REPO_ROOT

TRACKED_PACKAGES = ("mujoco", "numpy", "torch", "lerobot", "transformers", "peft")
# Paths whose changes can alter a run's behavior. Edits elsewhere (docs, results) do not make a
# result "dirty".
CODE_PATHS = ("src", "scripts", "tests", "pyproject.toml", "uv.lock")
# WSL2 exposes the Windows driver's nvidia-smi here without adding it to PATH.
_WSL_NVIDIA_SMI = "/usr/lib/wsl/lib/nvidia-smi"


def _run(*cmd: str) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=REPO_ROOT)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def git_state() -> dict[str, Any]:
    """Current commit, and whether code paths have uncommitted or untracked changes."""
    status = _run("git", "status", "--porcelain", "--", *CODE_PATHS)
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


def slurm_job() -> dict[str, str] | None:
    """The Slurm job a run is part of (on a cluster such as MSI), or None outside one."""
    if "SLURM_JOB_ID" not in os.environ:
        return None
    names = ["SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURMD_NODENAME", "CUDA_VISIBLE_DEVICES"]
    return {name.lower(): os.environ[name] for name in names if name in os.environ}


def provenance() -> dict[str, Any]:
    """Capture at the start of a run, so later working-tree edits are not attributed to it."""
    started = {
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "git": git_state(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "wsl": is_wsl(),
        "packages": package_versions(),
        "gpu": gpu_info(),
    }
    if job := slurm_job():
        started["slurm"] = job
    return started


def write_result(path: Path, result: dict[str, Any], started: dict[str, Any]) -> None:
    """Write ``result`` as JSON with the ``provenance()`` captured when the run started."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"provenance": started, **result}
    path.write_text(json.dumps(payload, indent=2) + "\n")
