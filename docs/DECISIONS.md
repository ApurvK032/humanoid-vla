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

## D-017 Wrist camera on a bracket beside the index finger (2026-10-12)

With the palm facing down, the palm hides the block from any camera on the back of the hand, and
a camera behind the palm sees mostly the thumb. A camera 7.5 cm out on the index side, aimed at
the claw's pinch point, shows the block between the fingers and thumb during the approach and
grasp, and the bin below while carrying. A real build would need a short bracket there.

## D-018 Video decoding with PyAV (2026-10-12)

LeRobot prefers torchcodec, which needs FFmpeg shared libraries installed system-wide; they are
not available without sudo. LeRobot falls back to PyAV, which bundles its own FFmpeg. Videos are
encoded with LeRobot's default AV1 settings.

## D-019 Executed actions are float32 (2026-10-12)

Datasets store actions as float32. While the simulator executed float64 targets, replaying a
recorded episode drifted by up to 0.84 rad in a finger joint, because contact dynamics amplified
the rounding. The environment now rounds executed targets to float32, and replays reproduce
every recorded joint state exactly.

## D-020 Datasets rendered with shadows, in parallel workers (2026-10-12)

Rendering dominates recording time (about 17 s of an 18 s episode for two cameras). Under WSL2
the shadow pass costs about 17 ms per image whatever the shadow-map size or anti-aliasing, so the
only cheaper option drops shadows, and with them the cue for the hand's height above the table.
Images keep shadows; recording runs in 4 worker processes (about 9 s per episode), and the main
process writes episodes in scenario order so the dataset does not depend on the worker count.

## D-021 Varied demonstrations from randomized expert parameters (2026-10-12)

Each recorded episode draws the expert's approach, carry and release heights, speeds and pauses
from modest ranges inside the grasp tolerances (199/200 development episodes succeed). The draw
is seeded by the scenario and stored in the manifest.

## D-022 Splits, scenarios and configs (2026-10-12)

Splits own disjoint seed ranges (train from 100000, development from 200000, test from 300000).
Instructions are balanced in blocks of six layouts, and about a third of layouts are recorded with
a second instruction, so the policy cannot succeed by ignoring the language. Configs stay plain
dataclasses, serialized into each dataset's manifest; a config framework is not needed yet.

## D-023 Training through a recorded wrapper around `lerobot-train` (2026-10-13)

`scripts/train_smolvla.py` runs LeRobot's own training script rather than a custom loop, so the
optimizer, schedule and SmolVLA preprocessing are LeRobot's defaults (D-011). The base model is
downloaded at the pinned revision (D-009) and passed as a local path, and the input and output
features come from the dataset: the policy reads the head and wrist cameras and the 14 joint
angles and predicts the 8-value action. The VLM's processor and configuration are fetched from
the Hub by name when the model is built; the wrapper records the revision it saw. Each run keeps
its full log and writes a manifest with the exact command, the base and dataset identities and
the parsed loss history next to its checkpoints.

## D-024 Closed-loop evaluation protocol (2026-10-13)

Policies are judged by running them in the environment on layouts they were not trained on:
development layouts while developing, test layouts only for final numbers. SmolVLA samples each
action chunk from noise, so two unseeded evaluations of the pilot policy on the same 17 episodes
gave 1 and 2 successes; each episode now seeds the noise from its scenario, so every compared
checkpoint gets the same noise. Evaluations still do not repeat exactly: renders of the same state
differ by one intensity level in a few head-camera pixels (the GPU renderer under WSL2), and
closed-loop rollouts amplify that. Re-evaluating the 5,000-step checkpoint changed the outcome of
22 of 30 episodes (4 and then 7 successes), so checkpoints are compared by success rates and
their intervals, not episode by episode. Reported: success with
a 95% Wilson interval, which block was lifted first (instruction following, separately from
manipulation), behavior on layouts evaluated with both instructions, and inference time. The
policy predicts 50 actions, executes the first N and then observes again; cameras render only on
those observation steps.

## D-025 Every training layout under two instructions (2026-10-14)

The 20,000-step policy trained on `g1_sort_v1` placed every block in the instructed bin (41 of
41) but chose the block at chance. With a third of the layouts under two instructions, the target
can often be read off the image, so `g1_sort_v2` records 150 layouts under two instructions each
(295 episodes, the size of v1; `collect_dataset.py --paired-fraction 1.0`). At 20,000 steps it
changed nothing measurable (16 of 70 successes). It is the dataset of the final policy (D-026);
whether full pairing matters at that training length has not been separated.

## D-026 Longer training with float32 action-expert weights (2026-10-15)

Three 20,000-step runs never learned the block word. The probe's spread across instructions grew
slowly until the learning rate decayed: LeRobot fits its cosine schedule to the run, so a
20,000-step run ends at 2.5e-6, where bfloat16 weights barely change (in the pilot, 24% of the
expert's weights changed at all in the second half). The M3 recipe trains 50,000 steps with the
decay spread over the whole run (`-- --policy.scheduler_decay_steps=50000`) and keeps the action
expert in float32 (`--float32-expert`; 8.5 GB, 4% slower). The block word appeared between
10,000 and 30,000 steps, and the 50,000-step checkpoint succeeds in 51 of 70 development
episodes. Which of the two changes mattered is an open question.

## D-027 Float32 action-expert weights stay in the recipe (2026-10-15)

A bfloat16 control of the D-026 run (same data, schedule, seed and batch) learned the block word
on the same schedule in the open-loop probe (85% at 50,000 steps for both) but ended at a higher
loss (0.011 against 0.007) and succeeded less often in closed loop: 29 against 42 of 70 at
30,000 steps and 41 against 51 of 70 at 50,000. The longer schedule is what makes the block word
learnable; float32 weights make the final policy more reliable. One run per condition, so the
size of the gap is uncertain.

## D-028 π0.5 trained on MSI A40s, evaluated on the RTX 4070 (2026-10-19)

π0.5 (3.3B parameters: PaliGemma 2B plus a 300M action expert) does not fit the RTX 4070 for
training, and the lab RTX 5090 is rarely free. It trains on MSI's 48 GB A40 or L40S GPUs under
the group's allocation. Fully training it in float32 needs over 50 GB for weights, gradients and
AdamW state alone, so two setups are compared, one per GPU in a single two-GPU job:
`expert-float32` (the whole model in float32, vision-language model frozen, action expert
trained: SmolVLA's best recipe, D-026 and D-027) and `full-bfloat16` (all weights trained in
bfloat16 with gradient checkpointing, as LeRobot does when asked for bfloat16). MSI's GPU jobs
stop after 24 hours, so runs save every 5,000 steps and resume from their latest checkpoint
(`train_pi05.py --resume`); the job script queues its own successor and stops retrying a run
that fails. Evaluation runs on the RTX 4070 with the same protocol as SmolVLA (D-024), so the
renderer and the development layouts are the ones SmolVLA was measured on; π0.5's 13 GB of
float32 weights do not fit there, so it runs in bfloat16, the precision openpi serves π0.5 in.
Every MSI command is run by hand; coding agents do not connect to MSI.

## Open questions

| Question | Decide in | Notes |
|---|---|---|
| Does full pairing (v2) matter with the D-026 recipe | when GPU time allows | v1 data with the D-026 recipe answers it |
| Policy cameras: head only, or head + wrist | M4 or later | Every M3 run used both; `--cameras=head` is ready. SmolVLA was pretrained with a top and a wrist camera |
| Actions executed per chunk before observing again | M4 | Evaluations use 10 of 50 (0.4 s); compare a range alongside the observation-delay perturbation |
| π0.5: which fine-tuning setup | after the MSI runs | `expert-float32` against `full-bfloat16` (D-028); the bfloat16 caveat of D-027 applies to the second |
| Project license | before release | Upstream code, models and assets are Apache-2.0 or BSD-3-Clause |
