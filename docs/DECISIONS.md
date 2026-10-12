# Decisions

One entry per non-obvious choice: what was decided, why, and what it costs. Newest last.
Superseded entries stay, marked as such.

## D-001 Repository and package name (2026-10-11)

Repository `humanoid-vla`, Python package `humanoid_vla`. The plan's `g1_vla` package name is
replaced so the package matches the repository. Robot-specific code lives in `humanoid_vla.sim.g1`.

## D-002 Tooling (2026-10-11)

- uv with a managed Python 3.12. System Python is 3.14, which is too new for the ML stack;
  LeRobot 0.6.1 needs Python ≥ 3.12 and torch < 2.12.
- `uv.lock` is resolved for Linux x86_64 only, the one supported platform (native or WSL2).
- Ruff for lint and formatting, pytest, pre-commit, GitHub Actions on a CPU runner.
- Ruff hooks run from the project environment so pre-commit, CI and editors use one locked version.

## D-003 Exact pins for physics and ML libraries (2026-10-11)

`mujoco==3.14.0` and `lerobot==0.6.1` are pinned exactly in `pyproject.toml`, not only in the lock.
Contact dynamics change between MuJoCo releases and LeRobot's API changes between minor releases,
so upgrades must be deliberate: re-run the checks and record the change here. MuJoCo 3.15.0 was
6 days old at setup, so the previous release was chosen.

## D-004 Heavy ML stack as a default dependency group (2026-10-11)

PyTorch + CUDA, LeRobot, SmolVLA and PEFT are in a `train` dependency group that `uv sync` installs
by default on the development machine. CI opts out with `--no-group train` and only needs the
simulation stack. A group was chosen over an optional extra because a plain `uv sync` removes
extras that are not requested.

## D-005 Native MuJoCo instead of SIMPLE (2026-10-11)

SIMPLE requires Isaac Sim 4.5 on Ubuntu 22.04 in addition to MuJoCo. This project needs one robot,
one tabletop task and full control over resets, perturbations and state restoration. A small native
MuJoCo scene meets that within the local resource budget.

## D-006 Robot model: Menagerie `g1_with_hands`, fetched at a pinned commit (2026-10-11)

The model has 7-DOF arms and a 7-joint Dex3 hand per side, with position actuators that Menagerie
notes still need tuning. Its meshes are 36 MB, so they are fetched into `third_party/` with a
sparse, shallow git fetch at a pinned commit instead of being committed. Code checks the checkout
is at the pinned commit before loading it.

## D-007 Fixed base by freezing joints in the model (2026-10-11)

The free joint is removed and every joint outside the active (right) arm and hand is deleted from
the model, with its body fixed at a chosen angle. Frozen joints cost nothing to simulate and cannot
drift under contact. The inactive arm is frozen hanging at the side (Menagerie's "stand" pose),
because at zero angles the G1 forearm points forward over the table. A test checks that every
robot body lands exactly where the unmodified model puts it. Switching sides or enabling both arms
later is a configuration change.

## D-008 Headless rendering on WSL2 through Mesa D3D12 (2026-10-11)

MuJoCo's EGL backend works under WSL2 but defaults to llvmpipe (CPU). Setting
`GALLIUM_DRIVER=d3d12` and `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` routes it to the RTX 4070.
Without the adapter name Mesa can pick the Ryzen integrated GPU, and under EGL that crashed.
`humanoid_vla.gl_env.configure_gl_defaults()` sets these defaults on WSL2; entry points call it
before importing mujoco. Each render call has a fixed overhead from the translation layer, so
throughput depends mostly on the number of render passes, not resolution (results/m0/render_check.json).

## Open questions

| Question | Decide in | Notes |
|---|---|---|
| Policy cameras: head only, or head + wrist | M1 | SmolVLA was pretrained with a top and a wrist camera |
| Render profile for datasets: full (shadows) or fast | M1/M2 | Fast is ~3x quicker; shadows are a depth cue for grasping |
| Gravity compensation for the arm | M1 | If used, route it through the actuators (`actuatorgravcomp`) so force limits still apply |
| Config system for tasks and experiments | M1 | LeRobot uses draccus |
| pi0.5 extension and cloud GPU budget | after M4 | Needs > 22.5 GB for LoRA (openpi) |
| Project license | before release | Upstream code, models and assets are Apache-2.0 or BSD-3-Clause |
