# Roadmap

Living status of the project. The full plan, scope and rationale are in
[PROJECT_PLAN.md](PROJECT_PLAN.md); choices made along the way are in [DECISIONS.md](DECISIONS.md).

| Milestone | Status | Exit condition |
|---|---|---|
| M0 Feasibility | done (2026-10-12) | Simulation and a genuine SmolVLA parameter update work locally with memory headroom |
| M1 Scripted expert | done (2026-10-12) | Expert reliably picks and places blocks by physical contact across a declared region |
| M2 Dataset | done (2026-10-13) | LeRobot dataset loads, and an episode replays through the same action interface |
| M3 First policy | next | Fine-tuned SmolVLA completes held-out layouts in closed loop |
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

## M2 Dataset

Details and measurements: [M2 report](reports/m2-dataset.md). Contract:
[ACTION_CONTRACT.md](ACTION_CONTRACT.md).

- [x] Wrist camera beside the index finger; head and wrist images recorded (D-017)
- [x] Rendering profile: full quality with shadows, recorded in 4 parallel workers (D-020)
- [x] Action contract document
- [x] Episode recorder: images, measured state, executed actions, instruction, scenario, expert
  parameters, checksums; failed attempts archived separately
- [x] Varied demonstrations from randomized expert parameters (D-021)
- [x] LeRobot export, loaded back with 50-action chunks and end-of-episode padding
- [x] Splits by seed range: train, development, frozen test; balanced and paired instructions
  (D-022)
- [x] Pilot through the whole data path, then the v1 dataset
- [x] Replay check: every recorded episode reproduces its joint states exactly and succeeds
- [x] Config handling: dataclass configs serialized into manifests (no framework needed yet)

## M3 First policy (next)

- [ ] Training run on `g1_sort_v1` with LeRobot's default SmolVLA fine-tuning at batch size 32
  (D-011), recording base checkpoint, trainable modules, optimizer, effective batch size, seed,
  update count and dataset identity
- [ ] Data-loading throughput with PyAV decoding; cache or pre-decode if it starves the GPU
- [ ] Overfit check on the pilot dataset before the full run
- [ ] Closed-loop rollout: observe, predict a 50-action chunk, execute N actions, re-observe;
  reset the policy queue each episode; check denormalization, clipping and image preprocessing
- [ ] Development rollouts during training to catch a falling loss with broken behavior
- [ ] Cameras: head only versus head + wrist
- [ ] Exit: autonomous success on held-out development layouts, a video, and instruction
  sensitivity measured on identical layouts with different instructions
