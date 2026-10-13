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

## D-009 SmolVLA from the pretrained base at a pinned Hub revision (2026-10-12)

`lerobot/smolvla_base` is loaded at Hub commit `5e8d12a6` with this project's features replacing
the base's (SO-100: three cameras, 6-dim state and action). State and action are zero-padded to
32 inside the model, so every pretrained weight loads (0 missing, 0 unexpected keys). The
checkpoint was updated on the Hub 5 days before the project started, which is why the revision
is pinned rather than following `main`.

## D-010 Result provenance is captured when a run starts (2026-10-12)

Result files record the git commit, versions and GPU at the start of the run, and count only
code paths (`src`, `scripts`, `tests`, `pyproject.toml`, `uv.lock`) as dirty. The first SmolVLA
results were written at the end of the run and were marked dirty by README edits made while it
ran; they were discarded and re-run from a clean commit.

## D-011 M3 starts with LeRobot's default fine-tuning, not LoRA (2026-10-12)

The plan preferred a parameter-efficient method after a memory test. The test shows LeRobot's
default (VLM frozen; action expert and projections train, 100M of 450M parameters) needs 5.6 GiB
at batch size 32, where throughput already saturates. LoRA (r=64) saves only about 0.5 GiB there,
because activations rather than optimizer state dominate, and adds inference latency unless
merged. So M3 uses the default at batch size 32; LoRA stays an ablation, worthwhile mainly for a
small published artifact. Training the VLM's text layers (7.2 GiB at batch size 32) is the next
step if instruction following is weak. Measurements: `docs/reports/m0-feasibility.md`.

## D-012 Arm gravity compensation through the actuators (2026-10-12)

The active arm and hand bodies use MuJoCo gravity compensation routed through their actuators
(`gravcomp` plus `actuatorgravcomp`), as the real G1's arm controller compensates gravity.
Position targets hold without sag, and the compensation counts against the actuators' torque
limits. Objects held in the hand are not compensated.

## D-013 Grasp: palm-down claw pinch with a two-stage synergy (2026-10-12)

A fingers-down grasp cannot work: the Dex3 fingers reach about 22 cm from the wrist and the thumb
about 13 cm, so open fingers hit the table before the thumb meets the block. The palm-down
"claw" works: the index and middle fingers fold down in front of the block and the thumb sweeps
forward behind it. In grasp trials on 5 cm blocks it lifted the block with grasp-height errors
from -3 to +0.5 cm, ±1.5 cm along and ±1 cm across the approach, and ±30 degrees of block yaw.

One closure value in [0, 1] drives all seven finger joints in two stages: fingers first, then the
thumb. Closing all joints at once let the thumb push blocks out before the fingers formed a
wall. Pre-curling the fingers before descending stopped them sweeping into nearby bins, and
opening thumb-first stopped releases from flinging blocks over a bin's far wall. Blocks grew from
4 to 5 cm so both fingers land on the front face.

## D-014 Layout inside the arm's collision-free reach; head camera aimed at it (2026-10-12)

IK ignores collisions. Reaching the first layout's left bin needed the right upper arm to fold
across the chest, where it genuinely hits the torso. A reach map that rejects IK solutions coming
within 1 cm of self-contact puts the right arm's palm-down workspace in a band to the right of
the head's forward direction: y from about -0.44 to 0.04 m, x up to about 0.42 m.

The bins therefore sit behind the block strip (x = 0.40 m, y = -0.06 m "left" and -0.30 m
"right"), and blocks spawn in x 0.18-0.25 m, y -0.32 to -0.08 m with at least 10 cm between
centers. The G1 has no neck joint, so the simulated head camera is mounted turned 20 degrees right
and pitched 62 degrees down with a 66-degree field of view (near the D435's horizontal FOV),
which keeps both bins and all blocks in view.

## D-015 Control interface (2026-10-12)

Physics steps every 2 ms (Menagerie's default) and the policy acts at 25 Hz, so each action is
held for exactly 20 physics steps; a 50-action SmolVLA chunk spans 2 s. An action is 7 absolute
arm joint targets in radians, clipped to the joint limits and to 2 rad/s, plus one hand closure
in [0, 1]. The policy observes a 256x256 head-camera image, the 14 measured arm and hand joint
angles, and the instruction. The environment reports the executed (clipped) action with every
step, for recording.

## D-016 Scripted expert (2026-10-12)

The expert plans straight-line moves of a grasp site on the hand, palm-down, and solves damped
least-squares IK every control step, warm-started from the last solution with an exact null-space
pull toward the home posture. It waits (at most 1 s) for the hand to arrive before closing and
before releasing, because the 2 rad/s limit makes the arm lag fast plans. The grasp heading is the
block-aligned heading nearest a preferred heading that turns further left the further left the
block is, following the reach map, clipped to 0-60 degrees. Over the bins the hand turns to 40
degrees (left) or 0 degrees (right), where IK errors are under 1 mm.

## Open questions

| Question | Decide in | Notes |
|---|---|---|
| Policy cameras: head only, or head + wrist | M2/M3 | Record both in the dataset and choose by experiment; SmolVLA was pretrained with a top and a wrist camera, and the hand hides the block from the head camera while grasping |
| Render profile for datasets: full (shadows) or fast | M2 | Fast is ~3x quicker; shadows are a depth cue for grasping |
| Config system for tasks and experiments | M2 | LeRobot uses draccus |
| fp32 master weights for the action expert | M3 | LeRobot trains the expert's bfloat16 weights directly; small updates can round away |
| pi0.5 extension and cloud GPU budget | after M4 | Needs > 22.5 GB for LoRA (openpi) |
| Project license | before release | Upstream code, models and assets are Apache-2.0 or BSD-3-Clause |
