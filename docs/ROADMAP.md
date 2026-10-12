# Roadmap

Living status of the project. The full plan, scope and rationale are in
[PROJECT_PLAN.md](PROJECT_PLAN.md); choices made along the way are in [DECISIONS.md](DECISIONS.md).

| Milestone | Status | Exit condition |
|---|---|---|
| M0 Feasibility | done (2026-10-12) | Simulation and a genuine SmolVLA parameter update work locally with memory headroom |
| M1 Scripted expert | done (2026-10-12) | Expert reliably picks and places blocks by physical contact across a declared region |
| M2 Dataset | next | LeRobot dataset loads, and an episode replays through the same action interface |
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

## M1 Scripted expert

Details and measurements: [M1 report](reports/m1-scripted-expert.md).

- [x] Home pose with the hand raised at the robot's right, outside the head camera's view
- [x] Collision-free reach map of the right arm; blocks and bins moved inside it (D-014)
- [x] Head camera aimed so every block position and both bins are in view, with a test
- [x] Grasp: palm-down claw with a two-stage synergy driven by one closure value (D-013)
- [x] Seeded resets within a declared operating region; six balanced instructions
- [x] Action interface: rate-limited arm joint targets and one hand closure at 25 Hz (D-015)
- [x] IK-based expert: approach, grasp, lift, carry, release, return (D-016)
- [x] Outcome labels: success, wrong bin, wrong object, dropped, timeout, unstable
- [x] Videos and per-episode logs; held-out success rate with a Wilson interval
- [ ] Camera decision, head only or head + wrist: moved to M2/M3 (record both, choose by
  experiment)

## M2 Dataset (next)

- [ ] Wrist camera on the hand; record head and wrist images
- [ ] Rendering profile for recorded images (full or fast; decide with a timing measurement)
- [ ] Action contract document: joint order, units, limits, closure mapping, absolute targets,
  25 Hz, chunking and normalization from training data only
- [ ] Episode recorder: synchronized images, measured state, executed actions, instruction,
  outcome, timestamps, scenario id, expert version and simulator parameters
- [ ] Variation in the expert's demonstrations (timing and path) so the policy sees more than
  one stereotyped motion
- [ ] Export to LeRobot dataset format; failed expert attempts kept in a separate archive
- [ ] Scenario splits before extracting frames: train, development, frozen test
- [ ] Pilot of 30-50 successful episodes through the whole data path, then about 150-300
- [ ] Replay check: an exported episode replays through the action interface to the same outcome
- [ ] Config system for data collection (LeRobot uses draccus)
