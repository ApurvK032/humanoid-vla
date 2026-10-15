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

Add `--save` to a check or evaluation script to write its measurements, with provenance, to
`results/`. Run it from a clean commit: results record whether the code was modified.

Seeds 0-99 were used while developing the scripted expert, so its evaluations use seeds from
1000 up (`--seed-start`, the default).

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
