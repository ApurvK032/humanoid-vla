"""Milestone 1: run the scripted expert on seeded episodes, report outcomes, record videos.

Seed s always gives the same layout and instruction. Videos show the scene camera and the head
camera side by side and go to outputs/m1/. With --save, the summary and the per-episode table go
to results/m1/expert_eval.json.

    uv run python scripts/run_expert.py --episodes 500 --seed-start 1000 --videos 2 --save
"""

import argparse
import collections
import statistics
import sys
import time
from pathlib import Path

from humanoid_vla.gl_env import configure_gl_defaults

configure_gl_defaults()

import imageio.v2 as imageio
import numpy as np

from humanoid_vla.envs.outcome import Outcome
from humanoid_vla.envs.sort import SortEnv, SortEnvConfig
from humanoid_vla.paths import OUTPUTS_DIR, REPO_ROOT, RESULTS_DIR
from humanoid_vla.policies.expert import ScriptedExpert
from humanoid_vla.sim.render import CameraRenderer
from humanoid_vla.utils.media import write_gif
from humanoid_vla.utils.provenance import provenance, write_result
from humanoid_vla.utils.stats import wilson_interval

VIDEO_HW = (480, 640)  # scene camera; the head camera is rendered square at the same height


def run_episode(
    env: SortEnv,
    expert: ScriptedExpert,
    seed: int,
    renderers: tuple[CameraRenderer, CameraRenderer] | None,
) -> tuple[dict, list[np.ndarray]]:
    env.reset(seed=seed)
    expert.reset()
    frames: list[np.ndarray] = []
    while True:
        _, _, terminated, truncated, info = env.step(expert.act())
        if renderers is not None:
            scene, head = renderers
            frames.append(
                np.hstack([scene.render(env.data, "scene"), head.render(env.data, "head")])
            )
        if terminated or truncated:
            break
    task = env.task
    record = {
        "seed": seed,
        "instruction": task.instruction,
        "target": task.target,
        "destination": task.destination,
        "outcome": str(info["outcome"]),
        "time_s": round(env.time, 3),
        "ik_failures": expert.ik_failures,
    }
    return record, frames


def summarize(episodes: list[dict], wall_s: float) -> dict:
    n = len(episodes)
    wins = [e for e in episodes if e["outcome"] == Outcome.SUCCESS]
    low, high = wilson_interval(len(wins), n)
    by_instruction: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for e in episodes:
        counts = by_instruction[f"{e['target']}->{e['destination']}"]
        counts[0] += e["outcome"] == Outcome.SUCCESS
        counts[1] += 1
    times = [e["time_s"] for e in wins]
    return {
        "episodes": n,
        "successes": len(wins),
        "success_rate": round(len(wins) / n, 4),
        "success_rate_wilson95": [round(low, 4), round(high, 4)],
        "outcomes": dict(collections.Counter(e["outcome"] for e in episodes)),
        "by_instruction": {
            key: {"successes": wins_, "episodes": total}
            for key, (wins_, total) in sorted(by_instruction.items())
        },
        "success_time_s": {
            "median": round(statistics.median(times), 2) if times else None,
            "max": round(max(times), 2) if times else None,
        },
        "episodes_with_ik_failures": sum(e["ik_failures"] > 0 for e in episodes),
        "wall_time_s": round(wall_s, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--videos", type=int, default=0, help="record the first N episodes")
    parser.add_argument("--gif", type=Path, help="also write the first video as a small GIF")
    parser.add_argument("--save", action="store_true", help="write results/m1/expert_eval.json")
    args = parser.parse_args()
    started = provenance()

    env = SortEnv(SortEnvConfig(cameras=()))
    expert = ScriptedExpert(env)
    video_dir = OUTPUTS_DIR / "m1"
    video_dir.mkdir(parents=True, exist_ok=True)
    episodes: list[dict] = []
    start = time.perf_counter()
    for i in range(args.episodes):
        seed = args.seed_start + i
        if i < args.videos:
            h, w = VIDEO_HW
            with (
                CameraRenderer(env.model, h, w) as scene,
                CameraRenderer(env.model, h, h) as head,
            ):
                record, frames = run_episode(env, expert, seed, (scene, head))
            path = video_dir / f"expert_seed{seed}.mp4"
            imageio.mimwrite(path, frames, fps=round(1 / env.control_dt), macro_block_size=16)
            print(f"  video: {path}")
            if args.gif is not None and i == 0:
                write_gif(frames, args.gif)
                print(f"  gif: {args.gif} ({args.gif.stat().st_size / 1e6:.2f} MB)")
        else:
            record, _ = run_episode(env, expert, seed, None)
        episodes.append(record)
        if record["outcome"] != Outcome.SUCCESS:
            print(f"  seed {seed}: {record['outcome']} - {record['instruction']}")
    summary = summarize(episodes, time.perf_counter() - start)

    low, high = summary["success_rate_wilson95"]
    print(
        f"Success {summary['successes']}/{summary['episodes']} = {summary['success_rate']:.1%} "
        f"(95% CI {low:.1%}-{high:.1%}); outcomes {summary['outcomes']}; "
        f"median success time {summary['success_time_s']['median']} s"
    )
    if args.save:
        path = RESULTS_DIR / "m1" / "expert_eval.json"
        config = {"seed_start": args.seed_start, "env": "SortEnv defaults", "expert": "defaults"}
        write_result(path, {"config": config, "summary": summary, "episodes": episodes}, started)
        print(f"Saved {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
