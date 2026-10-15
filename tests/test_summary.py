from typing import Any

from humanoid_vla.evaluation.summary import summarize

NO_BINS = {"red": None, "green": None, "blue": None}


def row(seed: int, target: str, destination: str, outcome: str, **kwargs: Any) -> dict[str, Any]:
    first_lifted = kwargs.get("first_lifted", target)
    return {
        "seed": seed,
        "target": target,
        "destination": destination,
        "outcome": outcome,
        "first_lifted": first_lifted,
        "picked_target": first_lifted == target,
        "final_bins": {**NO_BINS, **kwargs.get("final_bins", {})},
        "inference_ms": 100.0,
        "duration_s": 10.0,
    }


def test_summary_separates_block_and_bin_following() -> None:
    rows = [
        # Different blocks on one layout: both lifted first, one placed in the wrong bin.
        row(1, "red", "left", "success", final_bins={"red": "left"}),
        row(1, "blue", "right", "wrong_bin", final_bins={"blue": "left"}),
        # Same block, different bins: the bin word was ignored.
        row(2, "red", "left", "success", final_bins={"red": "left"}),
        row(2, "red", "right", "wrong_bin", final_bins={"red": "left"}),
        # A single instruction on a layout; nothing lifted.
        row(3, "green", "left", "timeout", first_lifted=None),
    ]
    summary = summarize(rows)
    assert summary["episodes"] == 5
    assert summary["successes"] == 2
    assert summary["outcomes"] == {"success": 2, "wrong_bin": 2, "timeout": 1}
    assert summary["picked_instructed_block_first"] == 0.8
    assert summary["lifted_nothing"] == 1
    assert summary["by_instruction"]["red->left"] == {"successes": 2, "episodes": 2}
    pairs = summary["paired_layouts"]
    assert pairs["pairs"] == 2
    assert pairs["both_succeeded"] == 0
    assert pairs["different_blocks"] == {"pairs": 1, "both_picked_instructed_block": 1}
    assert pairs["same_block_different_bins"] == {"pairs": 1, "both_in_instructed_bin": 0}
