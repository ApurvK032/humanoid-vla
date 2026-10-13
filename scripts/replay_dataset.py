"""Milestone 2 check: load a recorded dataset and replay its episodes through the environment.

Checks that LeRobot loads the dataset with the contract's features, that videos decode, that
each episode's states and actions still match the manifest checksum, and that replaying the
recorded actions from the episode's scenario reproduces its joint states and its success.

    uv run python scripts/replay_dataset.py --name g1_sort_pilot --split train [--save]
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import warnings

from humanoid_vla.gl_env import configure_gl_defaults

configure_gl_defaults()
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", message="Cannot enable progress bars")
logging.getLogger("lerobot.utils.import_utils").setLevel(logging.ERROR)

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from humanoid_vla.data.recording import ACTION_KEY, STATE_KEY, dataset_features, image_key
from humanoid_vla.data.replay import replay_episode
from humanoid_vla.envs.outcome import Outcome
from humanoid_vla.envs.sort import SortEnv, SortEnvConfig
from humanoid_vla.paths import REPO_ROOT, RESULTS_DIR
from humanoid_vla.utils.provenance import provenance, write_result

DATA_ROOT = REPO_ROOT / "data" / "datasets"
STATE_TOLERANCE = 1e-3  # rad


def checksum(states: np.ndarray, actions: np.ndarray) -> str:
    digest = hashlib.sha256()
    for state, action in zip(states, actions, strict=True):
        digest.update(np.ascontiguousarray(state, dtype=np.float32).tobytes())
        digest.update(np.ascontiguousarray(action, dtype=np.float32).tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--episodes", type=int, help="replay only the first N episodes")
    parser.add_argument("--save", action="store_true", help="write results/m2/replay_<name>.json")
    args = parser.parse_args()
    started = provenance()

    root = DATA_ROOT / args.name / args.split
    manifest = json.loads((root / "manifest.json").read_text())
    dataset = LeRobotDataset(manifest["dataset"]["repo_id"], root=root, video_backend="pyav")
    recorded = manifest["configs"]["env"]
    recording_config = SortEnvConfig(
        cameras=tuple(recorded["cameras"]), image_hw=tuple(recorded["image_hw"])
    )
    env = SortEnv(SortEnvConfig(cameras=()))  # rendering does not affect physics

    # The dataset must expose exactly the contract's features.
    problems = []
    for key, spec in dataset_features(recording_config).items():
        got = dataset.meta.features.get(key)
        if got is None:
            problems.append(f"missing feature {key}")
            continue
        if tuple(got["shape"]) != tuple(spec["shape"]) or got["dtype"] != spec["dtype"]:
            problems.append(f"{key}: expected {spec['dtype']} {spec['shape']}, found {got}")
        if key in (STATE_KEY, ACTION_KEY) and got["names"] != spec["names"]:
            problems.append(f"{key}: names differ from the contract")
    if dataset.num_episodes != len(manifest["episodes"]):
        problems.append(
            f"{dataset.num_episodes} episodes; the manifest lists {len(manifest['episodes'])}"
        )
    first = dataset[0]
    for camera in recording_config.cameras:
        image = first[image_key(camera)]
        if tuple(image.shape) != (3, *recording_config.image_hw):
            problems.append(f"{camera} decodes to {tuple(image.shape)}")

    columns = dataset.hf_dataset.with_format("numpy")
    episode_index = np.asarray(columns["episode_index"])
    states_all = np.stack(columns[STATE_KEY])
    actions_all = np.stack(columns[ACTION_KEY])

    episodes = manifest["episodes"][: args.episodes]
    rows = []
    for entry in episodes:
        mask = episode_index == entry["episode_index"]
        states, actions = states_all[mask], actions_all[mask]
        result = replay_episode(
            env,
            entry["seed"],
            {"target": entry["target"], "destination": entry["destination"]},
            actions,
            states,
        )
        row = {
            "episode_index": entry["episode_index"],
            "scenario_id": entry["scenario_id"],
            "checksum_ok": checksum(states, actions) == entry["sha256"],
            "outcome": str(result.outcome),
            "length_matches": result.length_matches,
            "max_state_error_rad": result.max_state_error,
        }
        row["ok"] = (
            row["checksum_ok"]
            and result.outcome == Outcome.SUCCESS
            and result.length_matches
            and result.max_state_error < STATE_TOLERANCE
        )
        rows.append(row)
        if not row["ok"]:
            print(f"  episode {entry['episode_index']}: {row}")
    env.close()

    passed = sum(r["ok"] for r in rows)
    worst = max(r["max_state_error_rad"] for r in rows)
    print(f"Contract problems: {problems or 'none'}")
    print(
        f"Replayed {len(rows)} episodes: {passed} reproduce their states and success "
        f"(worst state error {worst:.2e} rad)"
    )
    if args.save:
        path = RESULTS_DIR / "m2" / f"replay_{args.name}_{args.split}.json"
        summary = {
            "dataset": manifest["dataset"],
            "contract_problems": problems,
            "episodes_replayed": len(rows),
            "episodes_ok": passed,
            "worst_state_error_rad": worst,
            "state_tolerance_rad": STATE_TOLERANCE,
            "episodes": rows,
        }
        write_result(path, summary, started)
        print(f"Saved {path.relative_to(REPO_ROOT)}")
    return 0 if not problems and passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
