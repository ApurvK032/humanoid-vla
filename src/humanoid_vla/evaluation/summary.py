"""Summary of closed-loop evaluation episodes.

Episodes are ``RolloutResult.as_dict()`` rows, so a summary can be recomputed from saved results.
Besides the success rate with its 95% Wilson interval, the summary separates instruction following
from manipulation: which block the policy lifted first, and, on layouts evaluated with two
instructions, whether its behavior followed each instruction. Pairs that name different blocks
test the block word; pairs that name the same block test the bin word.
"""

import collections
from typing import Any

import numpy as np

from humanoid_vla.envs.outcome import Outcome
from humanoid_vla.utils.stats import wilson_interval


def _succeeded(row: dict[str, Any]) -> bool:
    return row["outcome"] == Outcome.SUCCESS


def _target_in_instructed_bin(row: dict[str, Any]) -> bool:
    return row["final_bins"][row["target"]] == row["destination"]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    wins = sum(_succeeded(r) for r in rows)
    low, high = wilson_interval(wins, n)
    by_instruction: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        counts = by_instruction[f"{r['target']}->{r['destination']}"]
        counts[0] += _succeeded(r)
        counts[1] += 1
    by_layout = collections.defaultdict(list)
    for r in rows:
        by_layout[r["seed"]].append(r)
    pairs = [p for p in by_layout.values() if len(p) == 2]
    block_pairs = [p for p in pairs if p[0]["target"] != p[1]["target"]]
    bin_pairs = [p for p in pairs if p[0]["target"] == p[1]["target"]]
    return {
        "episodes": n,
        "successes": wins,
        "success_rate": round(wins / n, 4),
        "success_rate_wilson95": [round(low, 4), round(high, 4)],
        "outcomes": dict(collections.Counter(r["outcome"] for r in rows)),
        "picked_instructed_block_first": round(sum(r["picked_target"] for r in rows) / n, 4),
        "lifted_nothing": sum(r["first_lifted"] is None for r in rows),
        "by_instruction": {
            key: {"successes": s, "episodes": t} for key, (s, t) in sorted(by_instruction.items())
        },
        "paired_layouts": {
            "pairs": len(pairs),
            "both_succeeded": sum(all(_succeeded(r) for r in p) for p in pairs),
            "different_blocks": {
                "pairs": len(block_pairs),
                "both_picked_instructed_block": sum(
                    all(r["picked_target"] for r in p) for p in block_pairs
                ),
            },
            "same_block_different_bins": {
                "pairs": len(bin_pairs),
                "both_in_instructed_bin": sum(
                    all(_target_in_instructed_bin(r) for r in p) for p in bin_pairs
                ),
            },
        },
        "mean_inference_ms": round(float(np.mean([r["inference_ms"] for r in rows])), 1),
        "mean_episode_s": round(float(np.mean([r["duration_s"] for r in rows])), 2),
    }
