# humanoid-vla

Language-conditioned manipulation on a simulated Unitree G1 humanoid with a
vision-language-action (VLA) model. The project fine-tunes SmolVLA to sort
blocks into bins from instructions such as "Put the red block in the left
bin". It then measures where the policy breaks under camera, latency and
actuator errors, and tests whether targeted training data makes it more
robust.

![The scripted expert sorting a block: scene camera (left) and the robot's head camera (right)](docs/media/m1_expert.gif)
*The scripted expert placing a block in the bin it was asked for, by physical contact: scene
camera (left) and the head camera a policy will see (right).*

**Status (2026-10-13):** Milestones 0-2 are complete. A scripted expert sorts blocks in 499 of
500 held-out layouts ([M1 report](docs/reports/m1-scripted-expert.md)), and its demonstrations
are recorded as a LeRobot dataset of 300 training episodes with head and wrist
cameras, every one of which replays exactly ([M2 report](docs/reports/m2-dataset.md)). SmolVLA
fine-tuning fits on one RTX 4070 ([M0 report](docs/reports/m0-feasibility.md)). Next is
Milestone 3: fine-tuning SmolVLA on the dataset. There is no trained policy yet. Numbers,
videos and claims are added only once measured.

## Quick start

Requires Linux x86_64 or WSL2, an NVIDIA GPU, and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ApurvK032/humanoid-vla.git && cd humanoid-vla
uv sync --locked                                  # Python 3.12, MuJoCo, PyTorch, LeRobot
uv run python scripts/fetch_assets.py             # pinned Unitree G1 model
uv run pytest                                     # simulation, interface and expert tests
uv run python scripts/run_expert.py --videos 1    # scripted expert on held-out layouts
```

Each step has a `make` shortcut. Setup details and troubleshooting are in the
[runbook](docs/RUNBOOK.md).

## Pipeline

```mermaid
flowchart LR
    A["Scripted IK expert<br/>(MuJoCo, privileged state)"] --> B["LeRobot dataset<br/>(images, joints, instruction)"]
    B --> C["Fine-tune SmolVLA"]
    C --> D["Closed-loop rollouts<br/>(policy sees images, joints, instruction)"]
    D --> E["Failure suite<br/>(camera shift, delay, actuator error)"]
    E --> F["Targeted recovery data<br/>(restore failed states, expert continues)"]
    F --> C
```

## Setup

| | |
|---|---|
| Robot | Unitree G1 from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1) (`g1_with_hands`), fixed base; the right arm (7 joints) and its Dex3 three-finger hand (7 joints) are active, every other joint is frozen |
| Task | "Put the {red, green, blue} block in the {left, right} bin.": three 5 cm blocks placed at random, two bins |
| Observations | 256x256 RGB from a head camera and a wrist camera, 14 arm and hand joint angles, instruction text |
| Actions | At 25 Hz: 7 arm joint targets (limited to 2 rad/s) and 1 hand-closure value that drives a two-stage finger synergy |
| Policy | [SmolVLA](https://huggingface.co/docs/lerobot/smolvla) (450M) with [LeRobot](https://github.com/huggingface/lerobot) and PyTorch; default fine-tuning compared with LoRA |
| Simulator | MuJoCo 3.14 with a Gymnasium-style wrapper |
| Compute | One RTX 4070 (12 GB) under WSL2 |

The scripted expert and the evaluator may read simulator object poses; the
learned policy never does. Grasps are physical contact. Objects are not welded
to the hand.

## Results so far

**Scripted expert (M1).** 499 of 500 episodes succeed on held-out layouts
(99.8%, 95% interval 98.9%-100.0%); a typical episode takes 7.56 s. The
expert reads block positions from the simulator, which the learned policy never sees, and acts
through the same 25 Hz interface the policy will use. Getting there meant a palm-down "claw"
grasp that suits the Dex3 hand, a layout inside the arm's collision-free reach, and waiting for
the arm before grasping or releasing ([M1 report](docs/reports/m1-scripted-expert.md)).

**Dataset (M2).** `g1_sort_v1` holds 300 training episodes (56,364 frames,
38 minutes of demonstrations) and 55 development episodes in LeRobot format,
with balanced instructions and some layouts recorded under two instructions. Replaying every
episode's recorded actions reproduces its joint states exactly, which checks that observations
and actions are aligned and mean what the [action contract](docs/ACTION_CONTRACT.md) says.

**Compute (M0).** On one RTX 4070 (12 GB) under WSL2, from [`results/m0/`](results/m0/):

- **Physics:** 87x real time for the tabletop scene.
- **Rendering:** about 30 fps per 256x256 camera image at full quality, or over 90 fps without
  shadows and reflections, headless through Mesa's D3D12 driver.
- **SmolVLA fine-tuning (LeRobot default, 100M of 450M parameters trained):** 5.6 GiB peak at
  batch size 32 and 9.8 GiB at 64, about 88 samples/s either way. Batch 128 exceeds the GPU.
- **Inference:** about 180 ms per 50-action chunk at batch size 1.

Details, including LoRA and VLM fine-tuning, are in the
[feasibility report](docs/reports/m0-feasibility.md).

## Planned experiments

**Failure suite.** Three perturbations, each at two severities, plus the
clean condition. Every scenario is saved with its reset state, instruction,
disturbance schedule and seeds so it can be replayed.

| Perturbation | Implementation | What it probes |
|---|---|---|
| Camera mounting shift | Bounded translation and rotation of the rendered camera | Sensitivity to viewpoint mismatch |
| Observation delay | Timestamped frame queue with a declared delay | Stale visual input |
| Arm execution error | Actuator lag or gain change | Actual motion differing from commanded motion |

**Adaptation study.** Starting from the same nominal checkpoint, with matched
training-frame and optimizer-update budgets:

- **A:** more nominal demonstrations
- **B:** demonstrations under broad randomization
- **C:** targeted recovery data: failed development states are restored and
  the expert continues from them; only continuations that succeed are kept

Conditions are compared on a frozen test split of paired scenarios that is
never used for data collection or tuning. Results will be reported with
confidence intervals, including any loss of clean-task success and any
negative or mixed outcomes.

**Metrics:** success rate by severity, wrong-object rate, drops, timeouts,
completion time, inference latency, simulation throughput and peak VRAM.

## Roadmap

- [x] **M0 Feasibility:** scene, headless GPU rendering, SmolVLA training step and inference
  within 12 GB
- [x] **M1 Scripted expert:** physical-contact pick and place with seeded
  resets and labeled outcomes
- [x] **M2 Dataset:** LeRobot-format dataset with a documented action
  contract, scenario splits and a replay check
- [ ] **M3 First policy:** fine-tuned SmolVLA completing held-out layouts in
  closed loop
- [ ] **M4 Failure suite:** reproducible perturbed scenarios and failure
  categories
- [ ] **M5 Adaptation study:** conditions A, B and C on the frozen test split
- [ ] **M6 Release:** verified install, demo checkpoint, dataset and model on
  the Hugging Face Hub, report and video

Task-level status is in [docs/ROADMAP.md](docs/ROADMAP.md). Planned extensions:

- π0.5 fine-tuned on the same dataset and run through the same failure suite.
  This needs a larger GPU: openpi lists more than 22.5 GB for LoRA
  fine-tuning.
- ACT baseline, teleoperated corrections, bimanual tasks.

## Repository layout

```text
src/humanoid_vla/   simulation (sim/), environment (envs/), expert and SmolVLA (policies/), utils/
scripts/            entry points: asset fetch, checks, expert evaluation
tests/              behavior and interface tests
docs/               plan, roadmap, decisions, runbook, reports
results/            committed measurements, each with git commit, versions and GPU
```

| Document | Contents |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Contributor reference: layout, rules, environment gotchas |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Milestone status and next tasks |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Decision log and open questions |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | Setup, commands, troubleshooting |
| [docs/ACTION_CONTRACT.md](docs/ACTION_CONTRACT.md) | What every recorded observation and action value means |
| [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) | Original plan, scope and rationale |
| [docs/reports/](docs/reports/) | Milestone reports |
| [THIRD_PARTY.md](THIRD_PARTY.md) | Upstream code, models and assets, with pins and licenses |

## Scope

This project does not claim a new model architecture or method, whole-body
control or locomotion (the base is fixed), reinforcement learning,
physical-robot deployment, or sim-to-real transfer.

## Related work

The targeted-data experiment is closest to
[Kintsugi-VLA](https://arxiv.org/abs/2609.31048) (Snegirev et al., 2026). That
work uses simulator state restoration to turn failed SmolVLA rollouts on a
simulated Franka arm into recovery data. This project applies a similar idea
to a humanoid with a dexterous hand, and to sensing and actuation errors.

## Built on

[MuJoCo](https://github.com/google-deepmind/mujoco),
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (G1 model),
[LeRobot](https://github.com/huggingface/lerobot),
[SmolVLA](https://huggingface.co/lerobot/smolvla_base) and
[PyTorch](https://pytorch.org). Versions, pins and licenses are in [THIRD_PARTY.md](THIRD_PARTY.md).

## License

Not chosen yet. It will be selected after reviewing the terms of the upstream
code, models and assets.

---

Apurv Kushwaha · [apurvkushwaha.com](https://apurvkushwaha.com)
