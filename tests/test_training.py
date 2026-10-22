"""Training-run helpers shared by the SmolVLA and π0.5 training scripts."""

from pathlib import Path

import pytest

from humanoid_vla.policies.pi05 import SETUPS, policy_arguments
from humanoid_vla.policies.training import (
    RunLocked,
    combined_history,
    last_checkpoint_step,
    parse_log,
    resume_arguments,
    run_lock,
)

LOG = [
    "INFO 2026-10-19 step:100 smpl:3K ep:17 epch:0.06 loss:0.512 grdn:4.1 lr:2.5e-06\n",
    "some other line\n",
    "INFO 2026-10-19 step:200 smpl:6K ep:35 epch:0.12 loss:0.250 grdn:2.0 lr:5.0e-06\n",
]


def test_parse_log_numbers_steps_from_the_resumed_checkpoint():
    rows = parse_log(LOG, log_freq=100, start_step=5000)
    assert [row["step"] for row in rows] == [5100, 5200]
    assert rows[1]["loss"] == pytest.approx(0.25)
    assert rows[0]["epoch"] == pytest.approx(0.06)


def test_combined_history_replaces_steps_repeated_after_a_resume():
    first = {"start_step": 0, "history": [{"step": s, "loss": 1.0} for s in (100, 200, 300)]}
    # The first job was stopped at step 300; its last checkpoint was at 200.
    second = {"start_step": 200, "history": [{"step": s, "loss": 0.5} for s in (300, 400)]}
    history = combined_history([first, second])
    assert [row["step"] for row in history] == [100, 200, 300, 400]
    assert [row["loss"] for row in history] == [1.0, 1.0, 0.5, 0.5]


def make_checkpoint(run: Path, step: int) -> None:
    pretrained = run / "checkpoints" / f"{step:06d}" / "pretrained_model"
    pretrained.mkdir(parents=True)
    (pretrained / "train_config.json").write_text("{}")
    last = run / "checkpoints" / "last"
    last.unlink(missing_ok=True)
    last.symlink_to(f"{step:06d}")


def test_last_checkpoint_step_follows_the_last_link(tmp_path):
    run = tmp_path / "run"
    assert last_checkpoint_step(run) is None
    make_checkpoint(run, 5000)
    make_checkpoint(run, 10000)
    assert last_checkpoint_step(run) == 10000
    config = run / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
    assert resume_arguments(run) == [f"--config_path={config}", "--resume=true"]


def test_run_lock_admits_one_process(tmp_path):
    run = tmp_path / "train" / "run"
    with run_lock(run), pytest.raises(RunLocked), run_lock(run):
        pass
    with run_lock(run):  # released when the first holder finished
        pass


def test_pi05_setups_fit_a_48gb_gpu_as_documented():
    expert = policy_arguments(SETUPS["expert-float32"])
    assert expert == [
        "--policy.dtype=float32",
        "--policy.train_expert_only=true",
        "--policy.gradient_checkpointing=false",
    ]
    full = policy_arguments(SETUPS["full-bfloat16"])
    assert "--policy.dtype=bfloat16" in full
    assert "--policy.train_expert_only=false" in full
    assert "--policy.gradient_checkpointing=true" in full
