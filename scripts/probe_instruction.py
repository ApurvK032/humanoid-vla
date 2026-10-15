"""Milestone 3: probe whether a checkpoint's first action chunk follows the instruction.

On each layout the checkpoint predicts its first 2 s of actions with each block named as the
target, from the same observation and noise (``humanoid_vla.evaluation.probe``). Reports how often
the hand is sent nearest the named block (chance is one in three) and how far apart the hand
targets for different instructions are, next to the scripted expert's.

    uv run python scripts/probe_instruction.py \\
        --checkpoint outputs/train/RUN/checkpoints/020000/pretrained_model --layouts 20 --save
"""

import argparse
import sys
import time
from pathlib import Path

from humanoid_vla.gl_env import configure_gl_defaults

configure_gl_defaults()

import numpy as np

from humanoid_vla.data.scenarios import SPLIT_START
from humanoid_vla.envs.sort import SortEnv, SortEnvConfig
from humanoid_vla.evaluation.probe import CHUNK_STEPS, probe_layout
from humanoid_vla.paths import REPO_ROOT, RESULTS_DIR
from humanoid_vla.policies.runner import PolicyRunner
from humanoid_vla.utils.provenance import provenance, write_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="pretrained_model directory")
    parser.add_argument("--split", choices=sorted(SPLIT_START), default="dev")
    parser.add_argument("--layouts", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--name", help="results name (default: run and step)")
    parser.add_argument("--save", action="store_true", help="write results/m3/probe_<name>.json")
    args = parser.parse_args()
    started = provenance()

    checkpoint = args.checkpoint.resolve()
    run_dir, step = checkpoint.parents[2], checkpoint.parent.name
    runner = PolicyRunner(checkpoint, n_action_steps=CHUNK_STEPS, device=args.device)
    h, w = runner.config.image_features[next(iter(runner.config.image_features))].shape[1:]
    env = SortEnv(SortEnvConfig(cameras=tuple(runner.cameras), image_hw=(h, w)))

    start = time.perf_counter()
    rows = []
    for layout in range(args.layouts):
        row = probe_layout(env, runner, SPLIT_START[args.split] + layout).as_dict()
        rows.append(row)
        named = [i["policy_nearest_block"] for i in row["instructions"].values()]
        print(
            f"[{layout + 1}/{args.layouts}] seed {row['seed']}: policy's nearest blocks {named}, "
            f"spread {100 * row['policy_spread_m']:.1f} cm "
            f"(expert {100 * row['expert_spread_m']:.1f} cm)",
            flush=True,
        )
    env.close()

    def followed(who: str) -> float:
        hits = [
            color == probe[f"{who}_nearest_block"]
            for row in rows
            for color, probe in row["instructions"].items()
        ]
        return round(float(np.mean(hits)), 4)

    summary = {
        "layouts": len(rows),
        "policy_nearest_block_is_instructed": followed("policy"),
        "expert_nearest_block_is_instructed": followed("expert"),
        "policy_spread_m_median": round(float(np.median([r["policy_spread_m"] for r in rows])), 4),
        "expert_spread_m_median": round(float(np.median([r["expert_spread_m"] for r in rows])), 4),
        "policy_to_target_m_median": round(
            float(
                np.median(
                    [i["policy_to_target_m"] for r in rows for i in r["instructions"].values()]
                )
            ),
            4,
        ),
        "wall_time_s": round(time.perf_counter() - start, 1),
    }
    policy, expert = (summary[f"{w}_nearest_block_is_instructed"] for w in ("policy", "expert"))
    print(
        f"Hand sent nearest the named block: policy {policy:.0%}, expert {expert:.0%} (chance "
        f"33%); median spread across instructions: policy "
        f"{100 * summary['policy_spread_m_median']:.1f} cm, expert "
        f"{100 * summary['expert_spread_m_median']:.1f} cm"
    )
    if args.save:
        result = {
            "checkpoint": {"run": run_dir.name, "step": step},
            "probe": {
                "split": args.split,
                "layouts": [0, args.layouts],
                "destination": "left",
                "noise_seed": 0,
                "device": args.device,
            },
            "summary": summary,
            "layouts": rows,
        }
        name = args.name or f"{run_dir.name}_{step}"
        path = RESULTS_DIR / "m3" / f"probe_{name}.json"
        write_result(path, result, started)
        print(f"Saved {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
