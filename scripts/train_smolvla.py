"""Milestone 3: fine-tune SmolVLA on a recorded dataset with LeRobot's training pipeline.

Wraps `lerobot-train` so every run can be traced and repeated:

- the base model is downloaded at the pinned Hub revision and passed as a local path, so the
  config, weights and processors all come from that revision;
- the policy takes its input and output features from the dataset (head and wrist cameras, the
  14-dim state, the 8-dim action) instead of the base model's SO-100 layout;
- the dataset is identified by the SHA-256 of its manifest;
- ``--float32-expert`` keeps the action expert's weights in float32 instead of LeRobot's bfloat16,
  through ``humanoid_vla.policies.lerobot_train``;
- provenance is captured when the run starts, the full log is kept, and a run manifest with the
  exact command and the parsed training log is written next to the checkpoints.

    uv run python scripts/train_smolvla.py --dataset g1_sort_v1 --run smolvla_v1 --steps 20000

Checkpoints go to outputs/train/<run>/ (gitignored). With --save, a summary goes to
results/m3/train_<run>.json.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time

from huggingface_hub import HfApi, snapshot_download

from humanoid_vla.paths import OUTPUTS_DIR, REPO_ROOT, RESULTS_DIR
from humanoid_vla.policies.smolvla import SMOLVLA_REPO, SMOLVLA_REVISION
from humanoid_vla.utils.provenance import provenance, write_result

DATA_ROOT = REPO_ROOT / "data" / "datasets"
# LeRobot logs a line like "step:1K smpl:32K ep:170 epch:0.57 loss:0.123 grdn:4.5 ..." every
# log_freq steps. Its step counts are rounded ("1K" covers 1000 to 1499), so steps are numbered
# by the line's position instead.
LOG_LINE = re.compile(r"step:\S+ smpl:\S+ ep:\S+ epch:(\S+) (.*)")
METRIC = re.compile(r"(\w+):(\S+)")


def parse_log(lines: list[str], log_freq: int) -> list[dict]:
    rows = []
    for line in lines:
        match = LOG_LINE.search(line)
        if not match:
            continue
        row = {"step": (len(rows) + 1) * log_freq, "epoch": float(match.group(1))}
        for name, value in METRIC.findall(match.group(2)):
            try:
                row[name] = float(value)
            except ValueError:
                continue
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, help="dataset name under data/datasets/")
    parser.add_argument("--split", default="train")
    parser.add_argument("--run", required=True, help="run name; checkpoints in outputs/train/")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-freq", type=int, default=5_000)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--float32-expert", action="store_true", help="train the action expert in float32"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save", action="store_true", help="write results/m3/train_<run>.json")
    parser.add_argument("extra", nargs="*", help="further lerobot-train arguments, after --")
    args = parser.parse_args()
    started = provenance()

    dataset_root = DATA_ROOT / args.dataset / args.split
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    output_dir = OUTPUTS_DIR / "train" / args.run
    if output_dir.exists():
        if not args.overwrite:
            print(f"error: {output_dir} exists; pass --overwrite to replace it", file=sys.stderr)
            return 1
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    base_dir = snapshot_download(SMOLVLA_REPO, revision=SMOLVLA_REVISION)
    vlm_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    vlm_revision = HfApi().model_info(vlm_name).sha
    launcher = ["lerobot-train"]
    if args.float32_expert:
        launcher = ["python", "-m", "humanoid_vla.policies.lerobot_train", "--float32-expert"]
    command = [
        *launcher,
        f"--policy.path={base_dir}",
        "--policy.input_features=null",
        "--policy.output_features=null",
        "--policy.push_to_hub=false",
        "--policy.device=cuda",
        f"--dataset.repo_id={manifest['dataset']['repo_id']}",
        f"--dataset.root={dataset_root}",
        "--dataset.video_backend=pyav",
        f"--output_dir={output_dir}",
        f"--job_name={args.run}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        f"--num_workers={args.num_workers}",
        f"--seed={args.seed}",
        "--wandb.enable=false",
        *args.extra,
    ]
    print(" ".join(command), flush=True)

    # lerobot-train creates output_dir itself, so the log lives beside it until the run ends.
    log_path = output_dir.with_suffix(".log")
    lines: list[str] = []
    start = time.perf_counter()
    with log_path.open("w") as log:
        process = subprocess.Popen(
            ["uv", "run", "--no-sync", *command],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            lines.append(line)
            if LOG_LINE.search(line):
                print(line.rstrip(), flush=True)
        returncode = process.wait()
    wall_s = time.perf_counter() - start
    if output_dir.exists():
        shutil.move(log_path, output_dir / "train.log")

    if returncode != 0:
        print("".join(lines[-30:]), file=sys.stderr)
    history = parse_log(lines, args.log_freq)
    run = {
        "run": args.run,
        "returncode": returncode,
        "wall_time_s": round(wall_s, 1),
        "command": command,
        "base_model": {"repo": SMOLVLA_REPO, "revision": SMOLVLA_REVISION},
        "vlm_backbone": {"repo": vlm_name, "revision_at_run": vlm_revision},
        "dataset": {
            **manifest["dataset"],
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "recorded_at_commit": manifest["provenance"]["git"]["commit"],
        },
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "save_freq": args.save_freq,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "fine_tuning": "LeRobot default for SmolVLA: VLM frozen, action expert trained",
            "action_expert_weights": "float32" if args.float32_expert else "bfloat16",
        },
        "history": history,
    }
    if output_dir.exists():
        (output_dir / "run_manifest.json").write_text(
            json.dumps({"provenance": started, **run}, indent=1) + "\n"
        )
    tail = history[-5:]
    final_loss = sum(r["loss"] for r in tail) / len(tail) if tail else None
    print(f"Run {args.run}: exit {returncode}, {wall_s / 60:.1f} min, final loss {final_loss}")
    if args.save and returncode == 0:
        summary = {k: v for k, v in run.items() if k != "history"}
        summary["final_loss_mean_last5_logs"] = final_loss
        summary["history"] = history
        path = RESULTS_DIR / "m3" / f"train_{args.run}.json"
        write_result(path, summary, started)
        print(f"Saved {path.relative_to(REPO_ROOT)}")
    return returncode


if __name__ == "__main__":
    sys.exit(main())
