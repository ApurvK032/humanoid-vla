# CLAUDE.md

Reference for contributors and coding agents. Read this first, then `docs/ROADMAP.md`.

## Project

Fine-tune a vision-language-action model (SmolVLA) for language-conditioned tabletop sorting with
a simulated, fixed-base Unitree G1 (right arm + Dex3 hand) in MuJoCo. Then measure how it fails
under camera, latency and actuator perturbations, and test whether targeted recovery data helps.

- Plan and rationale: `docs/PROJECT_PLAN.md` (frozen, 2026-10-11)
- Current status and next tasks: `docs/ROADMAP.md`
- Decisions and open questions: `docs/DECISIONS.md`
- Setup, commands, troubleshooting: `docs/RUNBOOK.md`
- Upstream components and pins: `THIRD_PARTY.md`

## Layout

```text
src/humanoid_vla/
  gl_env.py           OpenGL backend defaults; call before importing mujoco
  paths.py            repo paths (third_party/, outputs/, results/)
  sim/assets.py       pinned Menagerie commit and checked asset paths
  sim/g1.py           fixed-base G1, joint groups, joint order for state and actions
  sim/scene.py        tabletop scene: table, blocks, bins, cameras
  sim/render.py       offscreen camera rendering with explicit quality settings
  policies/smolvla.py SmolVLA at a pinned Hub revision with this project's features
  utils/provenance.py git commit, versions and GPU recorded with every result file
scripts/              entry points (fetch_assets, check_*); logic belongs in src/
tests/                behavior and interface tests
docs/                 plan, roadmap, decisions, runbook, reports/, media/
results/              small committed measurement files with provenance
third_party/          fetched upstream assets (gitignored)
outputs/              generated images, videos, logs, datasets, checkpoints (gitignored)
```

Planned packages as milestones land: `envs/` (Gymnasium wrapper, resets, success detection),
`data/` (expert, recording, LeRobot export), `policies/` rollout and adapters,
`evaluation/` (perturbations, scenario replay, metrics), and `configs/` at the root.

## Commands

`make setup`, `make test`, `make lint`, `make format`, `make assets`, `make check-render`,
`make check-smolvla`. Use `uv run` for everything; never the system Python.

## Rules

- **No privileged state in policy inputs.** Object poses from the simulator are for the scripted
  expert and the evaluator only. Policies see camera images, robot joint state and the instruction.
- **No hidden assistance.** No welded grasps or invisible forces. Report grasping as it is.
- **One joint order.** State and action vectors follow `G1Config.active_joints` (arm, then hand).
  Index MuJoCo arrays by joint or actuator name, never by assumed model order: the model orders
  hand joints differently from the actuators.
- **Pinned versions.** MuJoCo and LeRobot are pinned exactly, the Menagerie commit in
  `sim/assets.py`, Hub model revisions in code. Upgrade deliberately, re-run the checks, and add
  an entry to `docs/DECISIONS.md`.
- **Results carry provenance.** Capture `utils.provenance.provenance()` when a run starts and
  write committed results with `write_result`. Run from a clean commit: a result marked dirty
  should be re-run. Scratch output goes to `outputs/`.
- **No large files in git.** Datasets, checkpoints and videos go to the Hugging Face Hub;
  pre-commit rejects files over 1 MB. README media in `docs/media/` stays small.
- **Claims follow measurements.** Numbers enter the README and docs only once measured; negative
  results are reported.
- **Keep docs current.** Tick tasks in `docs/ROADMAP.md` as they finish, and record each
  non-obvious choice in `docs/DECISIONS.md`.

## Environment gotchas

- Under WSL2, EGL silently uses llvmpipe (CPU) unless `GALLIUM_DRIVER=d3d12` and
  `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` are set; `configure_gl_defaults()` handles it.
- MuJoCo reads `MUJOCO_GL` at import time, so entry points configure GL before `import mujoco`
  (ruff's E402 is ignored for `scripts/` and `tests/conftest.py`).
- Close renderers explicitly (`CameraRenderer` is a context manager); EGL raises noisy errors
  when contexts are garbage-collected at interpreter exit.
- Under WSL2, CUDA allocations beyond the 12 GB of VRAM do not raise out-of-memory errors: the
  driver spills into system memory and training slows about 20x. Compare peak reserved memory
  with VRAM explicitly (`scripts/check_smolvla.py` does).
- LeRobot loads `smolvla_base` with the VLM and the action expert in bfloat16; the expert's
  trainable weights and AdamW state are bfloat16 too.
- `nvidia-smi` is at `/usr/lib/wsl/lib/nvidia-smi`, not on PATH.
- No sudo on the dev machine: do not depend on apt packages.
