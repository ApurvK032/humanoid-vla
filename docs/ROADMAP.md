# Roadmap

Living status of the project. The full plan, scope and rationale are in
[PROJECT_PLAN.md](PROJECT_PLAN.md); choices made along the way are in [DECISIONS.md](DECISIONS.md).

| Milestone | Status | Exit condition |
|---|---|---|
| M0 Feasibility | done (2026-10-12) | Simulation and a genuine SmolVLA parameter update work locally with memory headroom |
| M1 Scripted expert | done (2026-10-12) | Expert reliably picks and places blocks by physical contact across a declared region |
| M2 Dataset | done (2026-10-13) | LeRobot dataset loads, and an episode replays through the same action interface |
| M3 First policy | done (2026-10-15) | Fine-tuned SmolVLA completes held-out layouts in closed loop |
| M4 Failure suite | next | Reproducible perturbed scenarios with categorized failures |
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

## M3 First policy

Details and measurements: [M3 report](reports/m3-first-policy.md).

- [x] Training runs with LeRobot's SmolVLA fine-tuning at batch size 32 (D-011, D-023), each
  recording base checkpoint, trainable modules, optimizer, batch size, seed, update count and
  dataset identity: `smolvla_v1`, `smolvla_v2`, `smolvla_v2_vlm`, `smolvla_v2_long_fp32`
- [x] Data-loading throughput with PyAV decoding: 0.005 s of loading per 0.63 s step, so no
  caching or pre-decoding
- [x] Overfit check on the pilot dataset: loss 1.40 to 0.043 in 2,000 steps, but 2 of 17
  successes in closed loop on its own training layouts
- [x] Closed-loop rollout: observe, predict a 50-action chunk, execute N actions, re-observe;
  reset the policy queue each episode; denormalization, clipping and image preprocessing checked
  against the dataset; policy noise seeded per episode (D-024)
- [x] Development rollouts of every saved checkpoint, plus an open-loop instruction probe, which
  caught three runs whose falling loss hid a policy that ignored the block word
- [x] Every training layout under two instructions (`g1_sort_v2`, D-025): no effect at 20,000
  steps
- [x] VLM text layers trained (`--train-vlm`): no effect on instruction following
- [x] 2.5x longer training with float32 action-expert weights (D-026): 51 of 70 successes,
  instructed block first in 80%
- [x] Exit: autonomous success on held-out development layouts (51/70), videos, and instruction
  sensitivity on identical layouts with different instructions (both followed on 13 of 17 pairs)

Moved to later milestones: actions executed per chunk (evaluations use 10 of 50), head-only
versus head + wrist cameras, and whether the longer schedule or float32 weights made the
difference (see the open questions in [DECISIONS.md](DECISIONS.md)).
