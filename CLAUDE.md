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
  sim/g1.py           fixed-base G1, joint groups and order, grasp site, gravity compensation
  sim/hand.py         Dex3 claw synergy: one closure value drives seven finger joints
  sim/ik.py           damped least-squares inverse kinematics
  sim/scene.py        tabletop scene: table, blocks, bins, cameras, contact settings
  sim/render.py       offscreen camera rendering with explicit quality settings
  envs/task.py        seeded layouts and instructions
  envs/outcome.py     outcome labels (success, wrong bin, wrong object, dropped, ...)
  envs/sort.py        Gymnasium sorting environment: action interface, observations, judging
  policies/expert.py  scripted expert (privileged state, IK, waits for arrival, randomized())
  policies/smolvla.py SmolVLA at a pinned Hub revision; optional float32 action expert
  policies/lerobot_train.py  lerobot-train with project options (float32 expert, cameras)
  policies/runner.py  runs a fine-tuned checkpoint in closed loop, one action per control step
  data/scenarios.py   frozen split seed ranges, balanced and paired instructions
  data/recording.py   LeRobot features (the action contract) and episode recording
  data/replay.py      replaying recorded actions to verify alignment and semantics
  evaluation/rollout.py  one closed-loop episode: outcome, first block lifted, final bins
  evaluation/summary.py  success with Wilson interval, instruction following, paired layouts
  utils/provenance.py git commit, versions and GPU recorded with every result file
  utils/stats.py      Wilson intervals for success rates
  utils/media.py      small GIFs for the documentation
scripts/              entry points (check_*, run_expert, collect_dataset, train_smolvla,
                      eval_policy, ...); logic belongs in src/
tests/                behavior and interface tests
docs/                 plan, roadmap, decisions, runbook, reports/, media/
results/              small committed measurement files with provenance
third_party/          fetched upstream assets (gitignored)
outputs/              generated images, videos, logs, datasets, checkpoints (gitignored)
```

Planned as milestones land: perturbations and scenario replay in `evaluation/`. Datasets live in
`data/datasets/` (gitignored), each with a `manifest.json`; their meaning is fixed by
`docs/ACTION_CONTRACT.md`. Training checkpoints go to `outputs/train/<run>/`.

## Commands

`make setup`, `make test`, `make lint`, `make format`, `make assets`, `make check-render`,
`make check-smolvla`, `make expert-eval`. Use `uv run` for everything; never the system Python.

Seeds 0-99 were used to develop the scripted expert; evaluate it on seeds from 1000 up.

## Rules

- **No privileged state in policy inputs.** Object poses from the simulator are for the scripted
  expert and the evaluator only. Policies see camera images, robot joint state and the instruction.
- **No hidden assistance.** No welded grasps or invisible forces. Report grasping as it is.
- **Never record test scenarios.** Splits own seed ranges (`data/scenarios.py`); test layouts
  are only for final evaluations.
- **Respect the reach map.** IK ignores collisions: the right arm's palm-down workspace without
  self-collision is a band right of center (D-014). Check new targets against it.
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
  trainable weights and AdamW state are bfloat16 too, and small updates round away.
  `train_smolvla.py --float32-expert` trains it in float32; the runner detects such checkpoints.
- A dictionary passed as `--policy.input_features` is merged into the base model's SO-100
  features rather than replacing them; select cameras with `train_smolvla.py --cameras`.
- Policy inference measured while a training run shares the GPU is about 3x slower; take timings
  on an idle GPU.
- Renders of the same state can differ by one intensity level in a few pixels, so closed-loop
  evaluations are not bit-for-bit repeatable even with seeded policy noise; compare success rates.
- The environment executes float32 actions so datasets replay bit-for-bit; keep it that way.
- Editing code while a long run is going marks its result dirty; edit docs only, or re-run.
- torchcodec cannot load without system FFmpeg; datasets decode with PyAV (`video_backend="pyav"`).
- `nvidia-smi` is at `/usr/lib/wsl/lib/nvidia-smi`, not on PATH.
- No sudo on the dev machine: do not depend on apt packages.
