# G1 VLA Robustness Lab

Plan date: October 11, 2026. This document is the original plan and is kept unchanged as a reference.
Current status is tracked in [ROADMAP.md](ROADMAP.md), and later decisions in [DECISIONS.md](DECISIONS.md).

## Objective

Build a reproducible, simulation-only robot-learning project that adapts SmolVLA to Unitree G1 upper-body manipulation, measures its failures under observation and execution errors, and evaluates whether targeted training data improves robustness.

The portfolio contribution is ownership of the environment, data pipeline, fine-tuning, policy execution, failure analysis, and reproducible experiments. We do not claim a new model architecture, a first-of-its-kind method, whole-body locomotion, or demonstrated sim-to-real transfer.

## Initial scope

- Robot: a simulated G1 with a supported/fixed base, one active arm, and a physically simulated hand.
- Task: instruction-conditioned block sorting on a table, with multiple colored blocks and two bins. Example: "Put the red block in the left bin."
- Observations: one external RGB camera, measured active-arm joint positions, hand state, and instruction. Add a wrist camera only if the initial task needs it.
- Actions: seven arm joint-position targets and one hand-closure parameter mapped to a documented finger-joint synergy. This simplifies dexterous control without attaching objects artificially to the hand.
- Learning: adapt the pretrained SmolVLA model using LeRobot/PyTorch; prioritize a supported parameter-efficient fine-tuning configuration after a memory test.
- Output: an autonomous simulation demo, reusable data and training tools, checkpoints/adapters, and a measured robustness report.
- First-release boundary: one robot, one task family, one VLA architecture, three training conditions, and a small fixed evaluation suite.

Physical robot access, a Quest headset, ROS, paid cloud compute, and a second model are not required for the initial release. Manual intervention collection, ACT, bimanual tasks, moving cameras, and pi0.5 are later extensions.

## Hardware and implementation strategy

Observed hardware: RTX 4070 with 12 GB VRAM, Ryzen 9 7900X, approximately 30 GiB RAM exposed to WSL2. GPU identification worked outside the tool sandbox; PyTorch training and rendering have not been verified. Ubuntu 26.04/WSL2 compatibility must be tested rather than assumed.

Preferred foundation: MuJoCo for the scene and physics, a Gymnasium-style environment wrapper, LeRobot datasets and training, and a separate policy adapter for simulation execution. Reuse upstream robot models, controllers, and training code with explicit versions and attribution.

During the first milestone, inspect SIMPLE and the published G1 simulation work as reuse candidates. SIMPLE includes Isaac Sim as well as MuJoCo dependencies, so it is not automatically a lightweight choice. A small native MuJoCo scene is the fallback if the existing stack exceeds the local resource budget. Inspect the actual model's hand joints, geometry, collision behavior, asset availability, and license before choosing it; a G1 model listing does not prove it includes the required Dex3 setup.

Run data generation and training separately initially. Profile memory before scaling batch size, cameras, or parallel simulation. Use small batches and gradient accumulation where supported. Pin a working Python/package combination; isolate it from system Python. Do not install GPU display drivers inside WSL.

## Milestone 0 — Prove feasibility

Deliverables:

- An environment specification and dependency lock or equivalent reproducible version record.
- A rendered G1 scene and a headless simulation check.
- A documented decision on the robot/hand model and simulation foundation.
- A SmolVLA load, forward pass, backward pass, and optimizer update on a representative batch with the intended image and action dimensions.
- A small resource report: peak VRAM, training-step time after warmup, inference latency, and simulation/render throughput.

Exit condition: the simulation and a genuine parameter update work locally with usable memory headroom. A tiny dataset overfit check should demonstrate that gradients reach the intended trainable parameters; it does not establish task success.

If this fails, resolve package/renderer issues and reduce workload complexity first. Consider a larger GPU only after measuring the remaining limit; no cloud spending is assumed. Timebox this investigation to a few focused sessions before expanding the stack.

## Milestone 1 — Reliable task and expert

Implement seeded scene resets, bounded action targets, IK-based expert motion, hand closing/opening, and terminal outcome detection. Keep physics running between control updates. Declare physics timestep, action rate, and policy execution horizon separately.

Success requires the requested block to remain inside the requested bin after hand release for a defined settling interval. Wrong-object placement, a dropped block, timeout, and simulation instability receive separate labels. Simulator object poses may be used by the expert and evaluator, but must not leak into the learned policy's observations.

Demonstrate repeated physical-contact pickup and placement across several reachable layouts. Save a video and episode logs. Verify that commanded actions are the same interface we will expose to the learned policy.

Exit condition: the expert can generate trustworthy demonstrations across a declared operating region. Shrink that region or fix contacts if necessary before collecting a large dataset. Do not disguise unreliable grasping with welded objects or invisible assistance.

## Milestone 2 — Dataset and action contract

Create a pilot set of roughly 30–50 successful episodes to validate the complete data path. This is a debugging target, not a promised sufficient training set. Expand coverage based on learning curves; an initial larger collection of roughly 150–300 episodes is a planning estimate.

Store synchronized images, measured states, executed command targets, instructions, outcomes, timestamps, scenario IDs, expert version, and simulator parameters. Keep rejected/failed expert attempts in a separate diagnostic archive; do not silently label them as successful training demonstrations.

Specify joint ordering, units, limits, closure mapping, absolute versus relative actions, action chunking, and normalization. Compute normalization from training data only. Fit these interfaces to the pretrained model explicitly; matching tensor widths is not sufficient.

Split by scenario/episode before extracting frames:

- Training: nominal and later adaptation demonstrations.
- Development: failure discovery, hyperparameters, severity selection, and checkpoint choice.
- Final test: frozen scenarios never used to collect corrective data or choose settings.

Balance target-object and destination instructions. Include the same scene with different valid instructions. Retain hashes and manifests, and replay exported examples to check temporal alignment and action semantics.

Exit condition: a LeRobot-compatible dataset loads correctly and an episode can be replayed through the same action interface.

## Milestone 3 — First learned policy

Fine-tune SmolVLA on nominal demonstrations. Record the base checkpoint, trainable modules, adaptation method/rank where applicable, optimizer, effective batch size, training seed, update count, and dataset identity.

Implement closed-loop rollout: read current observations, predict an action chunk, execute a declared number of actions, then observe again. Reset policy queues/history at episode boundaries. Validate action denormalization, joint clipping, and camera preprocessing.

Use a small development rollout set frequently enough to catch a declining loss with broken robot behavior. Expand data or repair the interface before committing to long training runs. If diagnostic ACT training helps localize a VLA issue, use it as an optional task-conditioned baseline, not an equivalent language model.

Exit condition: autonomous completion on held-out development layouts, a video, and measured instruction sensitivity in identical scenes. A stable nominal policy is required before interpreting robustness results.

## Milestone 4 — Reproducible failure suite

Start with three perturbations, each individually applied at two nonzero severity levels plus the shared clean condition:

| Perturbation | Implementation | What it probes |
| --- | --- | --- |
| Camera mounting shift | Change the actual rendered camera extrinsics by bounded translation/rotation | Visual sensitivity to viewpoint mismatch; do not label this explicit calibration correction when the policy has no calibration module |
| Observation delay | Deliver images through a timestamped frame queue with a declared delay | Sensitivity to stale visual observations |
| Arm execution error | Apply a documented actuator lag or gain change | Robustness when actual motion differs from desired motion |

Choose severities on development scenarios where the task remains physically feasible. Run the expert under the same perturbations as a diagnostic for whether the environment itself is unsolvable. Treat contact/friction changes as an optional fourth family after hand physics are trustworthy.

Save scenario manifests including reset state, instruction, disturbance schedule, and all random seeds. Replaying an identical scenario should be reproducible within measured simulator tolerances; GPU/backend determinism must be checked rather than promised.

For initial throughput estimates, use 10–20 development episodes per condition. For final results, target 50–100 paired test scenarios per condition, with two independent training seeds if compute permits. Log the actual counts and limitations. Report success-rate intervals and paired changes, and distinguish evaluation uncertainty from variation across training runs.

Metrics: success rate versus severity, clean-task success, wrong-object rate, drops, timeouts, completion time, inference latency, simulation throughput, and peak VRAM. Videos should include representative failures as well as successes.

## Milestone 5 — Controlled adaptation experiment

From the same nominal checkpoint, compare:

| Condition | Additional training data |
| --- | --- |
| A: more nominal data | Additional ordinary successful task demonstrations |
| B: broad randomization | Demonstrations generated with broad declared perturbations |
| C: targeted data | Demonstrations/corrections selected from failure categories observed on development rollouts |

Also retain the unadapted nominal checkpoint as a reference.

Match additional training-frame budgets and optimizer-update budgets across A/B/C, with the same base-data mixture and fine-tuning configuration. Report episode counts, generation attempts, and any operator time separately; equal frame counts do not imply equal collection cost.

For targeted recovery data, restore a saved failed or near-failed development state and attempt an expert continuation. Preserve relevant simulator/controller state and observation queues, or explicitly define the continuation as a fresh-start recovery episode. Record snapshot provenance. The expert must actually succeed before its continuation enters the successful training set; failure detection is not proof that recovery is possible.

The deployment policy receives only its declared images, robot state, and instruction. Privileged state is confined to demonstration generation and scoring. Evaluate A/B/C on the frozen test suite after configurations are selected using development data.

Main question: does targeted additional data improve disturbed task completion more than ordinary data or broad randomization under the declared budgets? Also measure whether adaptation damages clean-task success. Report negative or mixed results honestly. This experiment has prior art and is not a claim of scientific novelty.

## Milestone 6 — Public release and portfolio

Prepare a release with:

- Installation instructions verified in a fresh environment.
- A short simulation demo command with a downloadable compatible checkpoint.
- Dataset sample, schema, data-generation/export commands, and full dataset access where redistribution permits.
- Training and evaluation configurations, pinned dependencies, checkpoint identifiers, and seed manifests.
- Per-episode result tables and scripts that regenerate figures from those tables.
- A 60–90 second narrated video showing language-conditioned behavior and before/after disturbance cases.
- Architecture explanation, experiment report, limitations, model/data provenance, and upstream attribution.
- Tests for action conversion, observation timing, success conditions, state restoration, and other behavior whose failure would invalidate conclusions.

Large datasets and weights belong in an appropriate artifact/model store, not ordinary Git history. Select the project's license after checking upstream asset/code/model terms. Prepare release materials locally before public publishing.

Suggested future layout:

```text
configs/                 task, policy, perturbation, training, evaluation
src/g1_vla/envs/          scene, reset, control, observations, success
src/g1_vla/data/          expert, recording, export, manifests
src/g1_vla/policies/      model adapters and rollout
src/g1_vla/evaluation/    perturbations, scenario replay, metrics
scripts/                 demo, collect, train, evaluate, report
tests/                   behavior and interface checks
docs/                    setup, architecture, experiments, limitations
results/                 compact manifests, aggregate and episode tables
```

## Skills and evidence for the resume

| Skill | Evidence to produce |
| --- | --- |
| VLA adaptation / PEFT | Trained adapter/checkpoint, configs, trainable-parameter record |
| PyTorch / ML engineering | Working data loaders, reproducible training, profiling, debugging |
| Robot simulation | Contact-valid G1 task, resets, cameras, control interface |
| Robot data engineering | Synchronized LeRobot dataset, schema, splits, provenance |
| Closed-loop policy execution | Autonomous rollouts, action chunking, measured inference behavior |
| Robustness and failure analysis | Replayed perturbations, categorized failures, controlled comparisons |
| Technical communication | Usable release, demo, figures, and explicit limitations |

After completing the corresponding work, a resume description could say:

"Fine-tuned a vision-language-action model for simulated Unitree G1 manipulation; built a reproducible demonstration and evaluation pipeline and compared targeted-data adaptation against nominal and randomized training under camera and control disturbances."

Add measured improvements, dataset sizes, and runtime figures only after they are established. Do not describe this as RL training, physical robot deployment, or sim-to-real validation unless those extensions are actually completed.

## Schedule and scope control

Use milestone completion rather than promised dates. A provisional part-time planning envelope is 6–8 weeks at approximately 10–15 hours/week, subject to the first resource and grasping checks; this is an estimate, not a commitment or benchmark result. Re-estimate after Milestone 0.

The first useful checkpoint is Milestones 0–3: a functioning dataset-to-fine-tuned-policy loop. The portfolio release includes the failure suite and controlled adaptation study. If time is tight, reduce task variation and optional models before weakening the evaluation or omitting working training artifacts.

Optional extensions, in order: ACT comparison; Quest-assisted corrections; pi0.5 on a larger GPU; second-arm coordination; whole-body balance/locomotion. Each is a separate scope decision, and none blocks the first release.

## References and reuse candidates

- [SmolVLA documentation](https://huggingface.co/docs/lerobot/smolvla): pretrained policy and fine-tuning workflow. Custom G1 compatibility still requires implementation and validation.
- [LeRobot PEFT](https://huggingface.co/docs/lerobot/peft_training): supported adaptation mechanisms; pin the version and confirm trainable modules.
- [SIMPLE](https://github.com/physical-superintelligence-lab/SIMPLE): existing humanoid simulation/data/evaluation infrastructure; inspect resource requirements before adopting.
- [MuJoCo Menagerie G1](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1): model candidate; inspect the exact hand and actuator assets.
- [G1 research guide](https://sri299792458.github.io/g1-research-docs/): relevant control, calibration, grasp, and simulator lessons. Its MuJoCo backend is not a verified complete stacking simulator.
- [Kintsugi-VLA](https://arxiv.org/abs/2609.31048): closely related recovery-data work; acknowledge overlap.
- [OpenPI](https://github.com/Physical-Intelligence/openpi): optional pi0.5 extension. Published memory estimates exceed local VRAM for standard LoRA fine-tuning.

## Immediate implementation checklist

- [ ] Validate PyTorch/CUDA and MuJoCo rendering in isolated environments.
- [ ] Inspect and choose a compatible G1/hand model and asset licenses.
- [ ] Measure one representative SmolVLA training update and inference call.
- [ ] Select the smallest workable simulation foundation.
- [ ] Implement one scripted physical-contact pickup, release, and reset.
- [ ] Re-estimate the remaining milestones from measured results.
