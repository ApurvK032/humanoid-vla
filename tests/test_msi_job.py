"""The MSI job script's control flow, run with stand-ins for Slurm, the GPU and training."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from humanoid_vla.paths import REPO_ROOT

JOB = REPO_ROOT / "scripts" / "msi" / "train_pi05.sbatch"

# Stand-in for `uv run --no-sync python scripts/train_pi05.py ...`: a run named "*broken*" fails;
# any other run "trains" to --steps by writing a checkpoint and pointing checkpoints/last at it.
FAKE_UV = """#!/bin/bash
args=("$@")
for i in "${!args[@]}"; do
    case "${args[$i]}" in
        --run) run="${args[$((i + 1))]}" ;;
        --steps) steps="${args[$((i + 1))]}" ;;
    esac
done
echo "train ${run} on GPU ${CUDA_VISIBLE_DEVICES}: $*" >> calls.log
[[ ${run} == *broken* ]] && exit 1
dir="outputs/train/${run}/checkpoints"
mkdir -p "${dir}/$(printf %06d "${steps}")"
ln -sfn "$(printf %06d "${steps}")" "${dir}/last"
"""
FAKE_SBATCH = 'echo "sbatch $*" >> calls.log; echo 4242\n'
FAKE_SCANCEL = 'echo "scancel $*" >> calls.log\n'


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    if shutil.which("bash") is None:
        pytest.skip("needs bash")
    (tmp_path / "scripts" / "msi").mkdir(parents=True)
    (tmp_path / "scripts" / "train_pi05.py").touch()
    shutil.copy(JOB, tmp_path / "scripts" / "msi")
    (tmp_path / "outputs" / "slurm").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fakes = {"uv": FAKE_UV, "sbatch": FAKE_SBATCH, "scancel": FAKE_SCANCEL, "nvidia-smi": ""}
    for name, body in fakes.items():
        path = bin_dir / name
        path.write_text(body if body.startswith("#!") else "#!/bin/bash\n" + body)
        path.chmod(0o755)
    return tmp_path


def run_job(checkout: Path, *specs: str, **env: str) -> tuple[subprocess.CompletedProcess, str]:
    environment = {
        **os.environ,
        "PATH": f"{checkout / 'bin'}:{os.environ['PATH']}",
        "HOME": str(checkout),  # the script prepends $HOME/.local/bin to PATH
        "SLURM_SUBMIT_DIR": str(checkout),
        "SLURM_JOB_ID": "1000",
        "SLURM_JOB_PARTITION": "interactive-gpu",
        "SLURMD_NODENAME": "agc01",
        "CUDA_VISIBLE_DEVICES": "3,5",
        "STEPS": "300",
        **env,
    }
    result = subprocess.run(
        ["bash", "scripts/msi/train_pi05.sbatch", *specs],
        cwd=checkout,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    calls = checkout / "calls.log"
    return result, calls.read_text() if calls.exists() else ""


def test_trains_each_run_on_its_own_gpu_and_cancels_the_unneeded_successor(checkout):
    result, calls = run_job(checkout, "a:expert-float32", "b:full-bfloat16")
    assert result.returncode == 0, result.stderr
    assert "train a on GPU 3" in calls
    assert "train b on GPU 5" in calls
    # The successor is queued first, for the same runs, after this job ends.
    assert "--dependency=afterany:1000" in calls
    assert "--gres=gpu:a40:2" in calls
    assert "ATTEMPT=2" in calls
    assert "scancel 4242" in calls
    assert not list((checkout / "outputs" / "train").glob("*.failed"))


def test_a_failed_run_is_marked_and_skipped_later(checkout):
    result, calls = run_job(checkout, "good:expert-float32", "broken:full-bfloat16")
    assert result.returncode == 0, result.stderr
    assert (checkout / "outputs" / "train" / "broken.failed").exists()
    assert "scancel 4242" in calls  # nothing left to retry

    (checkout / "calls.log").unlink()
    result, calls = run_job(checkout, "good:expert-float32", "broken:full-bfloat16")
    assert result.returncode == 0, result.stderr
    assert "skip good: complete" in result.stdout
    assert "skip broken: it failed" in result.stdout
    assert calls == ""  # no training and no successor


def test_results_are_saved_unless_switched_off(checkout):
    _, calls = run_job(checkout, "a:expert-float32")
    assert "--resume --save" in calls
    _, calls = run_job(checkout, "b:expert-float32", SAVE="0")
    line = next(line for line in calls.splitlines() if line.startswith("train b"))
    assert line.endswith("--resume")


def test_last_attempt_queues_no_successor(checkout):
    result, calls = run_job(checkout, "a:expert-float32", ATTEMPT="4", MAX_ATTEMPTS="4")
    assert result.returncode == 0, result.stderr
    assert "sbatch" not in calls
    assert "train a on GPU 3" in calls


def test_refuses_more_runs_than_gpus(checkout):
    specs = ("a:expert-float32", "b:full-bfloat16")
    result, calls = run_job(checkout, *specs, CUDA_VISIBLE_DEVICES="0")
    assert result.returncode == 2
    assert "need 2 GPUs" in result.stderr
    assert calls == ""
