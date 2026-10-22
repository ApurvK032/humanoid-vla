"""Running `lerobot-train` from this project's training scripts.

Shared by ``scripts/train_smolvla.py`` and ``scripts/train_pi05.py``: reading a recorded dataset,
building the resume command, running training with the full log kept, and parsing LeRobot's
training log. Nothing here imports torch or LeRobot, so tests run without the ``train`` group.
"""

import contextlib
import fcntl
import hashlib
import json
import re
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from humanoid_vla.paths import REPO_ROOT

DATA_ROOT = REPO_ROOT / "data" / "datasets"
# LeRobot logs a line like "step:1K smpl:32K ep:170 epch:0.57 loss:0.123 grdn:4.5 ..." every
# log_freq steps. Its step counts are rounded ("1K" covers 1000 to 1499), so steps are numbered
# by the line's position instead.
LOG_LINE = re.compile(r"step:\S+ smpl:\S+ ep:\S+ epch:(\S+) (.*)")
METRIC = re.compile(r"(\w+):(\S+)")


def parse_log(lines: list[str], log_freq: int, start_step: int = 0) -> list[dict]:
    """Metrics of each logged step; ``start_step`` is the checkpoint a resumed run started from."""
    rows = []
    for line in lines:
        match = LOG_LINE.search(line)
        if not match:
            continue
        row = {"step": start_step + (len(rows) + 1) * log_freq, "epoch": float(match.group(1))}
        for name, value in METRIC.findall(match.group(2)):
            try:
                row[name] = float(value)
            except ValueError:
                continue
        rows.append(row)
    return rows


def combined_history(attempts: list[dict]) -> list[dict]:
    """One training history across the attempts of a resumed run.

    An attempt stopped by a job's time limit logged steps past its last checkpoint; the next
    attempt repeats them from that checkpoint (its ``start_step``), so its rows replace them.
    """
    history: list[dict] = []
    for attempt in attempts:
        history = [row for row in history if row["step"] <= attempt["start_step"]]
        history.extend(attempt["history"])
    return history


class RunLocked(RuntimeError):
    pass


@contextlib.contextmanager
def run_lock(output_dir: Path) -> Iterator[None]:
    """Hold a lock on a run while it trains, so a second job cannot train or delete it too.

    The lock is a file beside the run's directory; the operating system releases it when the
    process ends, however it ends.
    """
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with output_dir.with_suffix(".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RunLocked(f"another process is training {output_dir.name}") from None
        yield


@dataclass(frozen=True)
class RecordedDataset:
    root: Path
    manifest: dict
    manifest_sha256: str
    cameras: list[str]  # in the dataset's feature order

    @classmethod
    def load(cls, name: str, split: str) -> "RecordedDataset":
        root = DATA_ROOT / name / split
        manifest_path = root / "manifest.json"
        info = json.loads((root / "meta" / "info.json").read_text())
        cameras = [
            key.removeprefix("observation.images.")
            for key in info["features"]
            if key.startswith("observation.images.")
        ]
        return cls(
            root=root,
            manifest=json.loads(manifest_path.read_text()),
            manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            cameras=cameras,
        )

    def describe(self) -> dict:
        """The dataset's identity, for run manifests."""
        return {
            **self.manifest["dataset"],
            "manifest_sha256": self.manifest_sha256,
            "recorded_at_commit": self.manifest["provenance"]["git"]["commit"],
        }


def last_checkpoint_step(output_dir: Path) -> int | None:
    """Step of a run's latest checkpoint (LeRobot's ``checkpoints/last`` link), if it has one."""
    last = output_dir / "checkpoints" / "last"
    if not (last / "pretrained_model" / "train_config.json").is_file():
        return None
    return int(last.resolve().name)


def resume_arguments(output_dir: Path) -> list[str]:
    """lerobot-train arguments that continue a run from its latest checkpoint.

    LeRobot then takes every setting from the checkpoint's train_config.json, including the step
    count, the data order and the optimizer and scheduler state.
    """
    config = output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
    return [f"--config_path={config}", "--resume=true"]


@dataclass(frozen=True)
class Attempt:
    returncode: int
    wall_time_s: float
    lines: list[str]


def run_training(command: list[str], output_dir: Path, *, append: bool = False) -> Attempt:
    """Run a lerobot-train command, keeping its full log in ``<output_dir>/train.log``.

    A new run's log lives beside the output directory until lerobot-train has created it. With
    ``append`` (a resumed run) the attempt is added to the existing log. Logged training steps are
    echoed to stdout.
    """
    log_path = output_dir / "train.log" if append else output_dir.with_suffix(".log")
    lines: list[str] = []
    start = time.perf_counter()
    with log_path.open("a" if append else "w") as log:
        if append:
            log.write(f"\n===== resumed {time.strftime('%Y-%m-%d %H:%M:%S %Z')} =====\n")
        process = subprocess.Popen(
            ["uv", "run", "--no-sync", *command],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            lines.append(line)
            if LOG_LINE.search(line):
                print(line.rstrip(), flush=True)
        returncode = process.wait()
    if not append and output_dir.exists():
        log_path.replace(output_dir / "train.log")
    return Attempt(returncode, time.perf_counter() - start, lines)
