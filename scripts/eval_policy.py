"""Milestone 3: evaluate a fine-tuned SmolVLA checkpoint in closed loop.

Runs the policy on scenarios from a split (development by default; test layouts are reserved for
final evaluations), and reports the success rate with a 95% Wilson interval, how often the
policy lifted the instructed block first, and, on layouts used with two different instructions,
whether its behavior changed with the instruction. Videos (scene, head and wrist cameras side by
side) go to outputs/m3/.

    uv run python scripts/eval_policy.py \\
        --checkpoint outputs/train/RUN/checkpoints/020000/pretrained_model \\
        --layouts 50 --n-action-steps 10 --videos 2 --save
"""

import argparse
import collections
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

from humanoid_vla.gl_env import configure_gl_defaults

configure_gl_defaults()
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", message="Cannot enable progress bars")
logging.getLogger("lerobot.utils.import_utils").setLevel(logging.ERROR)

import imageio.v2 as imageio
import numpy as np
import torch

from humanoid_vla.data.scenarios import SPLIT_START, make_scenarios
from humanoid_vla.envs.outcome import Outcome
from humanoid_vla.envs.sort import SortEnv, SortEnvConfig
from humanoid_vla.evaluation.rollout import run_rollout
from humanoid_vla.paths import OUTPUTS_DIR, REPO_ROOT, RESULTS_DIR
from humanoid_vla.policies.runner import PolicyRunner
from humanoid_vla.sim.render import CameraRenderer
from humanoid_vla.utils.provenance import provenance, write_result
from humanoid_vla.utils.stats import wilson_interval

VIDEO_HEIGHT = 360


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    wins = sum(r["outcome"] == Outcome.SUCCESS for r in rows)
    picked = sum(r["picked_target"] for r in rows)
    low, high = wilson_interval(wins, n)
    by_instruction: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        counts = by_instruction[f"{r['target']}->{r['destination']}"]
        counts[0] += r["outcome"] == Outcome.SUCCESS
        counts[1] += 1
    by_seed = collections.defaultdict(list)
    for r in rows:
        by_seed[r["seed"]].append(r)
    pairs = [v for v in by_seed.values() if len(v) == 2]
    return {
        "episodes": n,
        "successes": wins,
        "success_rate": round(wins / n, 4),
        "success_rate_wilson95": [round(low, 4), round(high, 4)],
        "outcomes": dict(collections.Counter(r["outcome"] for r in rows)),
        "picked_instructed_block_first": round(picked / n, 4),
        "lifted_nothing": sum(r["first_lifted"] is None for r in rows),
        "by_instruction": {
            key: {"successes": s, "episodes": t} for key, (s, t) in sorted(by_instruction.items())
        },
        "paired_layouts": {
            "pairs": len(pairs),
            "both_picked_instructed_block": sum(all(r["picked_target"] for r in p) for p in pairs),
            "both_succeeded": sum(all(r["outcome"] == Outcome.SUCCESS for r in p) for p in pairs),
        },
        "mean_inference_ms": round(float(np.mean([r["inference_ms"] for r in rows])), 1),
        "mean_episode_s": round(float(np.mean([r["duration_s"] for r in rows])), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="pretrained_model directory")
    parser.add_argument("--split", choices=sorted(SPLIT_START), default="dev")
    parser.add_argument("--layouts", type=int, default=50)
    parser.add_argument("--first-layout", type=int, default=0)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--videos", type=int, default=0, help="record the first N episodes")
    parser.add_argument("--name", help="results name (default: run and step)")
    parser.add_argument("--save", action="store_true", help="write results/m3/eval_<name>.json")
    args = parser.parse_args()
    started = provenance()
    if args.split == "test":
        print("note: test layouts are reserved for final evaluations", file=sys.stderr)

    checkpoint = args.checkpoint.resolve()
    run_dir, step = checkpoint.parents[2], checkpoint.parent.name
    runner = PolicyRunner(checkpoint, n_action_steps=args.n_action_steps)
    h, w = runner.config.image_features[next(iter(runner.config.image_features))].shape[1:]
    env = SortEnv(SortEnvConfig(cameras=tuple(runner.cameras), image_hw=(h, w)))

    scenarios = make_scenarios(args.split, args.layouts, args.first_layout)
    video_dir = OUTPUTS_DIR / "m3"
    video_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    start = time.perf_counter()
    for k, scenario in enumerate(scenarios):
        frames: list[np.ndarray] = []
        renderers = None
        if k < args.videos:
            renderers = [
                CameraRenderer(env.model, VIDEO_HEIGHT, VIDEO_HEIGHT * 4 // 3),
                CameraRenderer(env.model, VIDEO_HEIGHT, VIDEO_HEIGHT),
                CameraRenderer(env.model, VIDEO_HEIGHT, VIDEO_HEIGHT),
            ]

            def record(renderers=renderers, frames=frames) -> None:
                views = zip(renderers, ("scene", "head", "wrist"), strict=True)
                frames.append(np.hstack([r.render(env.data, cam) for r, cam in views]))

        # SmolVLA samples each action chunk from noise; seeding per scenario makes a checkpoint's
        # evaluation repeatable and gives every compared policy the same noise.
        torch.manual_seed(int(scenario.rng().integers(2**31)))
        result = run_rollout(env, runner, scenario, on_step=record if renderers else None)
        if renderers:
            for r in renderers:
                r.close()
            path = video_dir / f"{run_dir.name}_{step}_{scenario.scenario_id}.mp4"
            imageio.mimwrite(path, frames, fps=round(1 / env.control_dt), macro_block_size=16)
            print(f"  video: {path}")
        rows.append(result.as_dict())
        print(
            f"[{k + 1}/{len(scenarios)}] {scenario.scenario_id}: {result.outcome} "
            f"(picked {result.first_lifted}, {result.duration_s:.1f} s)",
            flush=True,
        )
    summary = summarize(rows)
    summary["wall_time_s"] = round(time.perf_counter() - start, 1)
    env.close()

    low, high = summary["success_rate_wilson95"]
    pairs = summary["paired_layouts"]
    print(
        f"Success {summary['successes']}/{summary['episodes']} = {summary['success_rate']:.1%} "
        f"(95% CI {low:.1%}-{high:.1%}); picked the instructed block first in "
        f"{summary['picked_instructed_block_first']:.0%}; paired layouts with both instructions "
        f"followed: {pairs['both_picked_instructed_block']}/{pairs['pairs']}; "
        f"{summary['mean_inference_ms']} ms per chunk"
    )
    if args.save:
        manifest_path = run_dir / "run_manifest.json"
        run = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        result = {
            "checkpoint": {
                "run": run_dir.name,
                "step": step,
                "trained_at_commit": run.get("provenance", {}).get("git", {}).get("commit"),
                "dataset": run.get("dataset", {}).get("name"),
                "dataset_manifest_sha256": run.get("dataset", {}).get("manifest_sha256"),
                "action_expert_weights": "float32" if runner.float32_expert else "bfloat16",
            },
            "evaluation": {
                "split": args.split,
                "layouts": [args.first_layout, args.first_layout + args.layouts],
                "n_action_steps": args.n_action_steps,
                "cameras": runner.cameras,
                "noise_seed": "per scenario, from Scenario.rng()",
            },
            "summary": summary,
            "episodes": rows,
        }
        name = args.name or f"{run_dir.name}_{step}"
        path = RESULTS_DIR / "m3" / f"eval_{name}.json"
        write_result(path, result, started)
        print(f"Saved {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
