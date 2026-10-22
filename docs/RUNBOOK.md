# Runbook

How to set up the project and run each check. Measured numbers live in `results/` and
`docs/reports/`.

## Requirements

- Linux x86_64, native or WSL2 (developed on Ubuntu 26.04 under WSL2)
- NVIDIA GPU with a current driver. On WSL2, install the driver on Windows only; do not install
  a Linux display driver inside WSL.
- [uv](https://docs.astral.sh/uv/) and git
- About 10 GB of disk for the environment and model weights

## Setup

```bash
uv sync --locked
uv run pre-commit install
uv run python scripts/fetch_assets.py
```

This installs the locked environment (Python 3.12, MuJoCo, PyTorch + CUDA, LeRobot), the
pre-commit hooks, and the pinned Unitree G1 model into `third_party/`. With `make` installed,
`make setup` does the same. For a light environment without the ML stack (simulation and tests
only), use `uv sync --no-group train`.

## Everyday commands

| Command | `make` shortcut | What it does |
|---|---|---|
| `uv run pytest` | `make test` | Run the test suite |
| `uv run pre-commit run --all-files` | `make lint` | Run all lint, format and file-hygiene checks |
| `uv run ruff check --fix . && uv run ruff format .` | `make format` | Apply lint fixes and formatting |
| `uv run python scripts/fetch_assets.py` | `make assets` | Re-fetch the pinned robot model |
| `uv run python scripts/check_render.py` | `make check-render` | M0: rendering and physics throughput; images in `outputs/check_render/` |
| `uv run python scripts/check_smolvla.py` | `make check-smolvla` | M0: SmolVLA memory, training-step time and inference latency |
| `uv run python scripts/run_expert.py --episodes 100 --videos 1` | `make expert-eval` | M1: scripted-expert success rate on held-out seeds; videos in `outputs/m1/` |
| `uv run python scripts/collect_dataset.py --name NAME --split train --layouts 225` | | M2: record expert demonstrations as a LeRobot dataset in `data/datasets/NAME/train/`; `--paired-fraction 1.0` records every layout under two instructions |
| `uv run python scripts/replay_dataset.py --name NAME --split train` | | M2: check a dataset loads, decodes, matches its checksums and replays exactly |
| `uv run python scripts/train_smolvla.py --dataset NAME --run RUN --steps 20000` | | M3: fine-tune SmolVLA (about 3.5 h per 20,000 steps); checkpoints in `outputs/train/RUN/`; `--float32-expert`, `--train-vlm`, `--cameras=head`; LeRobot options after `--`, e.g. `-- --policy.scheduler_decay_steps=50000` |
| `uv run python scripts/eval_policy.py --checkpoint outputs/train/RUN/checkpoints/020000/pretrained_model --layouts 50` | | M3: closed-loop evaluation on development layouts; `--videos N` and `--gif PATH` for media |
| `uv run python scripts/probe_instruction.py --checkpoint outputs/train/RUN/checkpoints/020000/pretrained_model` | | M3: open-loop probe of whether the first action chunk follows the named block (1 min on the GPU) |
| `uv run python scripts/train_pi05.py --dataset NAME --run RUN --setup expert-float32 --resume` | | π0.5: fine-tune π0.5 (needs a 48 GB GPU; see "π0.5 on MSI"); `--setup full-bfloat16` trains all weights; `--resume` continues from the latest checkpoint |

Add `--save` to a check or evaluation script to write its measurements, with provenance, to
`results/`. Run it from a clean commit: results record whether the code was modified.

Seeds 0-99 were used while developing the scripted expert, so its evaluations use seeds from
1000 up (`--seed-start`, the default).

## π0.5 on MSI

π0.5 trains on MSI's 48 GB A40 or L40S GPUs and is evaluated on the RTX 4070 (D-028). Every
command on MSI is run by hand in your own terminal; coding agents working on this repository do
not connect to MSI. Logins need your password and Duo; `agate` below is an SSH alias for
`agate.msi.umn.edu` with your MSI username.

**Hugging Face access, once per machine.** π0.5's tokenizer comes from a gated repository: accept
the license at <https://huggingface.co/google/paligemma-3b-pt-224>, create a Read token at
<https://huggingface.co/settings/tokens>, and log in on every machine that trains or runs π0.5:
`uv run hf auth login`.

**Set up the checkout, once, on an MSI login node** (no GPU needed). Training checkpoints are
13 GB or more each, so `outputs/train` points to global scratch, which MSI clears of old files:
copy back what you keep.

```bash
curl -LsSf https://astral.sh/uv/0.12.21/install.sh | sh
git clone https://github.com/ApurvK032/humanoid-vla.git ~/humanoid-vla
cd ~/humanoid-vla && git checkout BRANCH_OR_COMMIT && uv sync --locked
mkdir -p data/datasets outputs/slurm "/scratch.global/${USER}/humanoid-vla-train"
ln -s "/scratch.global/${USER}/humanoid-vla-train" outputs/train
uv run hf auth login
```

**Copy the dataset from the lab PC**, and check that both copies match: re-recording it on MSI
would render different images.

```bash
rsync -a --info=progress2 ~/claude-code/humanoid/data/datasets/g1_sort_v2/ agate:humanoid-vla/data/datasets/g1_sort_v2/
cd ~/claude-code/humanoid/data/datasets/g1_sort_v2 && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum
```

Run the second command on MSI too (from `~/humanoid-vla/data/datasets/g1_sort_v2`); the two
fingerprints must be identical.

**Test job: memory and speed** (about 30 minutes; both setups, 200 steps each):

```bash
cd ~/humanoid-vla
STEPS=200 SAVE_FREQ=200 MAX_ATTEMPTS=2 SAVE=0 sbatch --time=0:45:00 scripts/msi/train_pi05.sbatch pi05_test_expert:expert-float32 pi05_test_full:full-bfloat16
```

It also checks that a job may queue its own successor. Afterwards read
`outputs/slurm/hvla-pi05-JOB.out`, the training logs `outputs/slurm/pi05_test_*-JOB.log` (`updt_s`
is seconds per training step) and the GPU memory log `outputs/slurm/gpu-JOB.csv`, then delete
the test runs: `rm -r outputs/train/pi05_test_*`.

**Training job:** two runs side by side on two A40s, resumed across 24-hour jobs until done:

```bash
sbatch scripts/msi/train_pi05.sbatch pi05_v2_expert:expert-float32 pi05_v2_full:full-bfloat16
```

`STEPS=N sbatch ...` changes the length (default 30,000). Follow it with `squeue --me` and
`tail -n 3 outputs/slurm/pi05_v2_*-*.log`; Slurm emails when a job ends or fails. A run that
failed is marked `outputs/train/RUN.failed` and skipped by later jobs: read its log, fix the cause,
delete the marker and submit again. The job script's header documents the rest.

**Bring a checkpoint and the results back to the lab PC** and evaluate there, in bfloat16:

```bash
mkdir -p outputs/train/pi05_v2_expert/checkpoints/030000
rsync -a --info=progress2 agate:humanoid-vla/outputs/train/pi05_v2_expert/checkpoints/030000/pretrained_model outputs/train/pi05_v2_expert/checkpoints/030000/
rsync -a agate:humanoid-vla/outputs/train/pi05_v2_expert/run_manifest.json agate:humanoid-vla/outputs/train/pi05_v2_expert/train.log outputs/train/pi05_v2_expert/
rsync -a agate:humanoid-vla/results/pi05/ results/pi05/
uv run python scripts/eval_policy.py --checkpoint outputs/train/pi05_v2_expert/checkpoints/030000/pretrained_model --layouts 50 --dtype bfloat16 --save
```

## Rendering backends

Scripts call `humanoid_vla.gl_env.configure_gl_defaults()`, which sets:

| Variable | Default | Why |
|---|---|---|
| `MUJOCO_GL` | `egl` | Headless rendering without a display |
| `GALLIUM_DRIVER` | `d3d12` (WSL2 only) | Use the GPU instead of llvmpipe (CPU) |
| `MESA_D3D12_DEFAULT_ADAPTER_NAME` | `NVIDIA` (WSL2 only) | Pick the NVIDIA GPU, not an integrated GPU |

Values already in the environment win. Export `MUJOCO_GL=glfw` with `DISPLAY=:0` to use a WSLg
window instead. Check which renderer is active in the first line of `make check-render`.

## Troubleshooting

- **`nvidia-smi: command not found` on WSL2.** It is at `/usr/lib/wsl/lib/nvidia-smi`.
- **`Menagerie not found` or `expected pinned commit`.** Run `uv run python scripts/fetch_assets.py` (or `make assets`).
- **Rendering is slow and the renderer is `llvmpipe`.** The D3D12 variables are not set, or the
  process imported mujoco before calling `configure_gl_defaults()`.
- **`D3D12: Removing Device` under EGL.** Mesa picked the wrong adapter; set
  `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`.
- **`uv sync` removed PyTorch.** It was run with `--no-group train`. Run `uv sync` again.
- **A long warning about `libtorchcodec` and FFmpeg.** torchcodec needs system FFmpeg
  libraries; LeRobot falls back to PyAV, which works. The project's scripts silence it.
- **A result or dataset manifest says `"dirty": true`.** Code changed while it ran (or was
  uncommitted when it started). Commit, then re-run it.
- **`make: command not found`.** Install it (`sudo apt install make`) or use the `uv run`
  commands above; every `make` target is a one-line wrapper around one of them.
