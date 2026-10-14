"""Milestone 2: record scripted-expert demonstrations as a LeRobot dataset.

Successful episodes go into the dataset; failed attempts go to a separate diagnostic archive
(rejected/). A manifest next to LeRobot's files records every episode's scenario, expert
parameters and checksum, the configs, the code's provenance, and a hash of every file.

    uv run python scripts/collect_dataset.py --name g1_sort_pilot --split train --layouts 40

``--paired-fraction`` sets the share of layouts recorded with a second instruction (D-022).

The dataset lands in data/datasets/<name>/<split>/ (gitignored); see docs/ACTION_CONTRACT.md.
"""

import argparse
import collections
import dataclasses
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import warnings
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from humanoid_vla.gl_env import configure_gl_defaults

configure_gl_defaults()
os.environ.setdefault("SVT_LOG", "1")  # SVT-AV1 encoder: errors only
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", message="Cannot enable progress bars")
# torchcodec needs system FFmpeg libraries; LeRobot falls back to PyAV and logs a long warning.
logging.getLogger("lerobot.utils.import_utils").setLevel(logging.ERROR)

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from humanoid_vla.data.recording import (
    ACTION_KEY,
    STATE_KEY,
    RecordedEpisode,
    dataset_features,
    record_episode,
)
from humanoid_vla.data.scenarios import SPLIT_START, Scenario, make_scenarios
from humanoid_vla.envs.outcome import Outcome
from humanoid_vla.envs.sort import SortEnv, SortEnvConfig
from humanoid_vla.paths import REPO_ROOT
from humanoid_vla.policies.expert import ExpertConfig, randomized
from humanoid_vla.utils.provenance import provenance

DATA_ROOT = REPO_ROOT / "data" / "datasets"
MANIFEST = "manifest.json"


# Each worker process owns one environment (and its OpenGL context) for its whole life.
_worker_env: SortEnv | None = None


def _init_worker(env_config: SortEnvConfig) -> None:
    global _worker_env
    configure_gl_defaults()
    _worker_env = SortEnv(env_config)


def _record(scenario: Scenario) -> RecordedEpisode:
    if _worker_env is None:
        raise RuntimeError("worker not initialized")
    return record_episode(_worker_env, scenario, randomized(ExpertConfig(), scenario.rng()))


def recorded_in_order(
    scenarios: list[Scenario], env_config: SortEnvConfig, workers: int
) -> collections.abc.Iterator[RecordedEpisode]:
    """Record scenarios in parallel, yielding results in scenario order.

    At most ``2 * workers`` episodes are in flight, which bounds memory: one episode's images
    take about 80 MB.
    """
    context = get_context("spawn")
    with ProcessPoolExecutor(workers, context, _init_worker, (env_config,)) as pool:
        pending: collections.deque[Future] = collections.deque()
        queue = iter(scenarios)
        for scenario in queue:
            pending.append(pool.submit(_record, scenario))
            if len(pending) >= 2 * workers:
                break
        while pending:
            result = pending.popleft().result()
            for scenario in queue:
                pending.append(pool.submit(_record, scenario))
                break
            yield result


def file_hashes(root: Path) -> dict[str, dict]:
    hashes = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != MANIFEST):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[str(path.relative_to(root))] = {"bytes": path.stat().st_size, "sha256": digest}
    return hashes


def to_jsonable(obj: object) -> object:
    if dataclasses.is_dataclass(obj):
        return to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="dataset name, e.g. g1_sort_v1")
    parser.add_argument("--split", choices=sorted(SPLIT_START), default="train")
    parser.add_argument("--layouts", type=int, required=True, help="number of layouts to use")
    parser.add_argument("--first-layout", type=int, default=0)
    parser.add_argument(
        "--paired-fraction",
        type=float,
        default=1 / 3,
        help="share of layouts also recorded with a second instruction",
    )
    parser.add_argument("--cameras", nargs="+", default=["head", "wrist"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4, help="parallel rendering processes")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    started = provenance()

    root = DATA_ROOT / args.name / args.split
    if root.exists():
        if not args.overwrite:
            print(f"error: {root} exists; pass --overwrite to replace it", file=sys.stderr)
            return 1
        shutil.rmtree(root)
    env_config = SortEnvConfig(
        cameras=tuple(args.cameras), image_hw=(args.image_size, args.image_size)
    )
    env = SortEnv(env_config)  # for the feature spec and timing; workers do the recording
    expert_base = ExpertConfig()
    dataset = LeRobotDataset.create(
        repo_id=f"local/{args.name}-{args.split}",
        fps=round(1 / env.control_dt),
        features=dataset_features(env_config),
        root=root,
        robot_type="unitree_g1_dex3_fixed_base",
        use_videos=True,
        video_backend="pyav",
    )
    rejected_dir = root / "rejected"

    scenarios = make_scenarios(args.split, args.layouts, args.first_layout, args.paired_fraction)
    episodes, rejected = [], []
    start = time.perf_counter()
    for k, record in enumerate(recorded_in_order(scenarios, env_config, args.workers)):
        scenario, expert_config = record.scenario, record.expert_config
        task = record.task
        entry = {
            "scenario_id": scenario.scenario_id,
            "seed": scenario.seed,
            "target": scenario.target,
            "destination": scenario.destination,
            "instruction": task.instruction,
            "block_xy": task.block_xy,
            "block_yaw": task.block_yaw,
            "outcome": str(record.outcome),
            "frames": len(record.frames),
            "duration_s": round(record.duration_s, 3),
            "ik_failures": record.ik_failures,
            "expert": to_jsonable(expert_config),
            "sha256": record.checksum(),
        }
        if record.outcome == Outcome.SUCCESS:
            for frame in record.frames:
                dataset.add_frame(frame)
            dataset.save_episode()
            episodes.append({"episode_index": len(episodes), **entry})
        else:
            rejected_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                rejected_dir / f"{scenario.scenario_id}.npz",
                state=np.stack([f[STATE_KEY] for f in record.frames]),
                action=np.stack([f[ACTION_KEY] for f in record.frames]),
            )
            rejected.append(entry)
        elapsed = time.perf_counter() - start
        print(
            f"[{k + 1}/{len(scenarios)}] {scenario.scenario_id}: {record.outcome} "
            f"({len(record.frames)} frames, {elapsed:.0f} s elapsed)",
            flush=True,
        )
    dataset.finalize()

    manifest = {
        "provenance": started,
        "dataset": {
            "name": args.name,
            "split": args.split,
            "repo_id": dataset.repo_id,
            "fps": dataset.fps,
            "layouts": [args.first_layout, args.first_layout + args.layouts],
            "paired_fraction": args.paired_fraction,
            "episodes": len(episodes),
            "frames": sum(e["frames"] for e in episodes),
            "rejected": len(rejected),
            "wall_time_s": round(time.perf_counter() - start, 1),
        },
        "contract": "docs/ACTION_CONTRACT.md",
        "configs": {"env": to_jsonable(env_config), "expert_base": to_jsonable(expert_base)},
        "episodes": episodes,
        "rejected": rejected,
    }
    manifest["files"] = file_hashes(root)
    (root / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
    env.close()

    info = manifest["dataset"]
    print(
        f"Recorded {info['episodes']} episodes ({info['frames']} frames), "
        f"{info['rejected']} rejected, in {info['wall_time_s']:.0f} s: {root}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
