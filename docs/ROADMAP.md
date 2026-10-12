# Roadmap

Living status of the project. The full plan, scope and rationale are in
[PROJECT_PLAN.md](PROJECT_PLAN.md); choices made along the way are in [DECISIONS.md](DECISIONS.md).

| Milestone | Status | Exit condition |
|---|---|---|
| M0 Feasibility | done (2026-10-12) | Simulation and a genuine SmolVLA parameter update work locally with memory headroom |
| M1 Scripted expert | next | Expert reliably picks and places blocks by physical contact across a declared region |
| M2 Dataset | not started | LeRobot dataset loads, and an episode replays through the same action interface |
| M3 First policy | not started | Fine-tuned SmolVLA completes held-out layouts in closed loop |
| M4 Failure suite | not started | Reproducible perturbed scenarios with categorized failures |
| M5 Adaptation study | not started | Conditions A/B/C compared on the frozen test split |
| M6 Release | not started | Verified install, demo checkpoint, dataset and model on the Hub, report, video |

## M0 Feasibility

- [x] Repository scaffold: uv + Python 3.12, ruff, pytest, pre-commit, CI, docs
- [x] Pinned fetch of the Menagerie G1 model
- [x] Fixed-base G1 with the right arm and Dex3 hand active (14 actuators), tested against the
  unmodified model
- [x] First tabletop scene: table, three blocks, two bins, head and scene cameras
- [x] Headless GPU rendering under WSL2, with rendering and physics throughput measured
- [x] SmolVLA load, forward, backward and optimizer step at this project's shapes: peak VRAM,
  step time, inference latency
- [x] Default fine-tuning versus LoRA versus also training the VLM: memory and trainable
  parameters
- [x] Overfit check on a fixed synthetic batch: loss falls and gradients reach only the intended
  parameters
- [x] [Feasibility report](reports/m0-feasibility.md) and re-estimate of the remaining milestones

## M1 Scripted expert (next)

Carried over from M0:

- [ ] Home pose for the active arm (at zero angles the forearm blocks the head camera)
- [ ] Reachability map of the right arm over the table; move blocks and bins inside it (the
  right bin is currently outside the head camera's view)
- [ ] Camera decision: head only, or head + wrist
- [ ] Hand actuator gains and grasp synergy (Menagerie marks the gains as untuned)

From the plan:

- [ ] Seeded scene resets within a declared operating region
- [ ] Action interface: bounded joint-position targets for the arm, one closure value for the hand
- [ ] IK-based expert: approach, grasp, lift, transport, release
- [ ] Outcome labels: success (block settled in the requested bin), wrong object, drop, timeout,
  instability
- [ ] Video and episode logs of repeated successful pick-and-place
