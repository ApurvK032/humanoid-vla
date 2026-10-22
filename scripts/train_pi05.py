"""π0.5 extension: fine-tune π0.5 on a recorded dataset with LeRobot's training pipeline.

Like ``train_smolvla.py``, this wraps `lerobot-train` so every run can be traced and repeated: the
base model is downloaded at the pinned Hub revision, the policy takes its features from the
dataset, provenance is captured when the run starts, and a run manifest with the exact commands
and the parsed training log is kept next to the checkpoints. ``--setup`` picks one of the two
fine-tuning setups that fit a 48 GB GPU (see ``humanoid_vla.policies.pi05``).

Runs are meant for a cluster GPU (MSI's A40 and L40S), whose jobs stop after 24 hours. With
``--resume`` the script continues a run from its latest checkpoint, starts it if it has none, and
exits at once if it is complete, so the same command can be repeated until the run finishes.

    uv run python scripts/train_pi05.py --dataset g1_sort_v2 --run pi05_v2_expert \\
        --setup expert-float32 --steps 30000 --resume

Checkpoints go to outputs/train/<run>/ (gitignored). With --save, a completed run's summary goes
to results/pi05/train_<run>.json.
"""

import argparse
import json
import shutil
import sys

from huggingface_hub import HfApi, snapshot_download

from humanoid_vla.paths import OUTPUTS_DIR, REPO_ROOT, RESULTS_DIR
from humanoid_vla.policies.pi05 import (
    PI05_REPO,
    PI05_REVISION,
    SETUPS,
    TOKENIZER_REPO,
    policy_arguments,
)
from humanoid_vla.policies.training import (
    RecordedDataset,
    RunLocked,
    combined_history,
    last_checkpoint_step,
    parse_log,
    resume_arguments,
    run_lock,
    run_training,
)
from humanoid_vla.utils.provenance import provenance, write_result


def target_steps(output_dir) -> int:
    """The step count a run was started with, from its latest checkpoint's config."""
    config = output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
    return json.loads(config.read_text())["steps"]


def tokenizer_revision() -> str | None:
    try:
        return HfApi().model_info(TOKENIZER_REPO).sha
    except Exception:  # gated or offline: the run itself fails clearly if it cannot load it
        return None


def main() -> int:
    args = parse_args()
    try:
        with run_lock(OUTPUTS_DIR / "train" / args.run):
            return train(args)
    except RunLocked as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, help="dataset name under data/datasets/")
    parser.add_argument("--split", default="train")
    parser.add_argument("--run", required=True, help="run name; checkpoints in outputs/train/")
    parser.add_argument("--setup", required=True, choices=sorted(SETUPS))
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-freq", type=int, default=5_000)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--cameras", help="comma-separated dataset cameras (default: all)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue the run from its latest checkpoint (start it if it has none)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save", action="store_true", help="write results/pi05/train_<run>.json")
    parser.add_argument("extra", nargs="*", help="further lerobot-train arguments, after --")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite exclude each other")
    return args


def train(args: argparse.Namespace) -> int:
    started = provenance()

    dataset = RecordedDataset.load(args.dataset, args.split)
    cameras = args.cameras.split(",") if args.cameras else dataset.cameras
    if unknown := sorted(set(cameras) - set(dataset.cameras)):
        print(f"error: the dataset has no camera {unknown}", file=sys.stderr)
        return 1
    setup = SETUPS[args.setup]
    output_dir = OUTPUTS_DIR / "train" / args.run
    manifest_path = output_dir / "run_manifest.json"
    launcher = ["lerobot-train"]
    if args.cameras:
        launcher = ["python", "-m", "humanoid_vla.policies.lerobot_train"]
        launcher.append(f"--cameras={args.cameras}")

    start_step = last_checkpoint_step(output_dir) if output_dir.exists() else None
    resuming = args.resume and start_step is not None
    if resuming:
        steps = target_steps(output_dir)
        if start_step >= steps:
            print(f"Run {args.run} is complete ({start_step} of {steps} steps).")
            return save_summary(args, manifest_path, started) if args.save else 0
        command = [*launcher, *resume_arguments(output_dir), *args.extra]
        run = json.loads(manifest_path.read_text())
    else:
        if output_dir.exists():
            if not args.overwrite and not args.resume:
                print(f"error: {output_dir} exists; pass --resume or --overwrite", file=sys.stderr)
                return 1
            # --resume on a run that never reached a checkpoint starts it again.
            shutil.rmtree(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        steps = args.steps
        start_step = 0
        base_dir = snapshot_download(PI05_REPO, revision=PI05_REVISION)
        command = [
            *launcher,
            f"--policy.path={base_dir}",
            "--policy.input_features=null",
            "--policy.output_features=null",
            "--policy.push_to_hub=false",
            "--policy.device=cuda",
            *policy_arguments(setup),
            f"--dataset.repo_id={dataset.manifest['dataset']['repo_id']}",
            f"--dataset.root={dataset.root}",
            "--dataset.video_backend=pyav",
            f"--output_dir={output_dir}",
            f"--job_name={args.run}",
            f"--batch_size={args.batch_size}",
            f"--steps={steps}",
            f"--save_freq={args.save_freq}",
            f"--log_freq={args.log_freq}",
            f"--num_workers={args.num_workers}",
            f"--seed={args.seed}",
            "--wandb.enable=false",
            *args.extra,
        ]
        run = {
            "run": args.run,
            "base_model": {"repo": PI05_REPO, "revision": PI05_REVISION},
            "tokenizer": {"repo": TOKENIZER_REPO, "revision_at_run": tokenizer_revision()},
            "dataset": dataset.describe(),
            "training": {
                "steps": steps,
                "batch_size": args.batch_size,
                "save_freq": args.save_freq,
                "log_freq": args.log_freq,
                "num_workers": args.num_workers,
                "seed": args.seed,
                "setup": args.setup,
                "fine_tuning": setup.description,
                "cameras": [cam for cam in dataset.cameras if cam in cameras],
            },
            "attempts": [],
        }
    print(" ".join(command), flush=True)

    attempt = run_training(command, output_dir, append=resuming)
    end_step = last_checkpoint_step(output_dir) if output_dir.exists() else None
    if attempt.returncode != 0:
        print("".join(attempt.lines[-30:]), file=sys.stderr)
    run["attempts"].append(
        {
            "provenance": started,
            "command": command,
            "returncode": attempt.returncode,
            "wall_time_s": round(attempt.wall_time_s, 1),
            "start_step": start_step,
            "last_checkpoint_step": end_step,
            "history": parse_log(attempt.lines, args.log_freq, start_step),
        }
    )
    if output_dir.exists():
        manifest_path.write_text(json.dumps(run, indent=1) + "\n")
    complete = end_step is not None and end_step >= steps
    print(
        f"Run {args.run}: exit {attempt.returncode}, {attempt.wall_time_s / 60:.1f} min, "
        f"last checkpoint {end_step} of {steps} steps"
    )
    if attempt.returncode == 0 and complete and args.save:
        return save_summary(args, manifest_path, started)
    return attempt.returncode


def save_summary(args: argparse.Namespace, manifest_path, started: dict) -> int:
    """Write a completed run's summary, with every attempt's command and training history."""
    run = json.loads(manifest_path.read_text())
    history = combined_history(run["attempts"])
    tail = history[-5:]
    run["final_loss_mean_last5_logs"] = sum(r["loss"] for r in tail) / len(tail) if tail else None
    run["wall_time_s"] = round(sum(a["wall_time_s"] for a in run["attempts"]), 1)
    path = RESULTS_DIR / "pi05" / f"train_{args.run}.json"
    write_result(path, run, started)
    print(f"Saved {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
