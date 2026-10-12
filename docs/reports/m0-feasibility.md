# M0 feasibility report

Date: 2026-10-12. Machine: RTX 4070 (12 GB), Ryzen 9 7900X, Ubuntu 26.04 under WSL2.
Raw measurements with provenance (commit, package versions, GPU driver) are in
[`results/m0/`](../../results/m0/). Re-run them with `make check-render` and `make check-smolvla`.

## Verdict

The Milestone 0 exit condition is met:

- **Simulation:** the fixed-base G1 scene builds from the pinned Menagerie model and runs headless.
  Physics runs at 87x real time, and rendering uses the GPU.
- **Learning:** a genuine SmolVLA parameter update runs at this project's input and output shapes
  with headroom. LeRobot's default fine-tuning needs 5.6 GiB at batch size 32, of 10.8 GiB free.
  Gradients reach exactly the intended parameter groups, and the loss falls when overfitting a
  fixed batch.

Caveat: the learning checks use synthetic images, states and actions with the shapes and value
ranges of recorded data. They prove the training path and the memory budget, not task learning;
overfitting real demonstrations is part of M2 and M3.

## Simulation

Scene: 14 actuated joints (right arm and Dex3 hand), three free blocks, `nq` 35, 51 bodies,
2 ms physics timestep. From [`render_check.json`](../../results/m0/render_check.json):

| Measurement | Result |
|---|---|
| Physics | 43,000 steps/s, 87x real time |
| OpenGL renderer | `D3D12 (NVIDIA GeForce RTX 4070)` through Mesa's EGL |

| Camera | Resolution | Full quality (fps) | Fast (fps) |
|---|---|---|---|
| head | 256x256 | 28 | 97 |
| head | 640x480 | 21 | 82 |
| scene | 256x256 | 31 | 91 |
| scene | 640x480 | 21 | 81 |

- **Physics** runs well above real time, so simulation cost is negligible next to rendering and
  inference.
- **Rendering** goes through Mesa's D3D12 driver (WSL2 has no native NVIDIA OpenGL). Each render
  pass has a fixed overhead, so cost depends on the number of passes (shadows, reflections) far
  more than on resolution; a development test that batched several frames per readback showed
  no gain. The "fast" profile drops shadows, reflections and the skybox. Shadows are a depth cue
  for grasping, so the dataset profile is decided in M1/M2.
- In development tests without the D3D12 settings, the same backend fell back to llvmpipe (CPU)
  at 7 fps; with the adapter left unspecified, Mesa picked the Ryzen integrated GPU, which
  crashed under EGL.

## SmolVLA

Model: `lerobot/smolvla_base` at Hub revision `5e8d12a6`, LeRobot 0.6.1. Inputs: one 256x256
head-camera image (resized and padded to 512x512 inside the model), a 14-dim joint state (arm
and hand) and the instruction. Output: a chunk of 50 actions of 8 dims (7 arm targets and one
hand-closure value). Every pretrained weight loads (0 missing, 0 unexpected keys).

Three fine-tuning variants were measured:

- **expert:** LeRobot's default. The VLM is frozen; the action expert and the state/action
  projections train.
- **lora:** LoRA with rank 64 on LeRobot's default SmolVLA targets (the expert's query and value
  projections, plus the state/action projections). All base weights are frozen.
- **vlm:** the VLM's text layers train as well; the vision encoder stays frozen.

### Trainable parameters, gradient routing and overfitting

| Variant | Trainable parameters | Groups trained | Routing check | Overfit loss, first → last 20 steps |
|---|---|---|---|---|
| expert | 99.9M of 450M | expert, projections | pass | 1.99 → 0.73 |
| lora | 3.0M of 453M | LoRA adapters only | pass | 2.65 → 0.92 |
| vlm | 306.5M of 450M | VLM text layers, connector, expert, projections | pass | 2.03 → 0.49 |

Gradient routing passes when the parameter groups with trainable weights are exactly the
expected ones and every trainable weight receives a non-zero gradient. The overfit check trains
200 steps on one fixed batch of 8 and compares the mean loss of the first and last 20 steps.
SmolVLA's flow-matching loss samples new noise and timesteps every step, so it stays noisy and
does not reach zero.

### Training memory and speed

Peak reserved memory, time per optimizer step, and samples per second, for one forward pass,
backward pass and clipped AdamW update:

| Batch size | expert | lora | vlm |
|---|---|---|---|
| 1 | 1.8 GiB, 0.08 s, 12/s | 1.1 GiB, 0.08 s, 13/s | 3.4 GiB, 0.11 s, 9/s |
| 2 | 1.8 GiB, 0.09 s, 22/s | 1.2 GiB, 0.08 s, 24/s | 3.4 GiB, 0.11 s, 18/s |
| 4 | 1.8 GiB, 0.10 s, 40/s | 1.4 GiB, 0.09 s, 42/s | 3.4 GiB, 0.12 s, 32/s |
| 8 | 2.4 GiB, 0.12 s, 67/s | 2.0 GiB, 0.12 s, 69/s | 3.5 GiB, 0.15 s, 54/s |
| 16 | 3.4 GiB, 0.19 s, 84/s | 3.0 GiB, 0.18 s, 90/s | 4.9 GiB, 0.22 s, 72/s |
| 32 | 5.6 GiB, 0.36 s, 88/s | 5.1 GiB, 0.35 s, 92/s | 7.2 GiB, 0.41 s, 79/s |
| 64 | 9.8 GiB, 0.72 s, 88/s | 9.1 GiB, 0.71 s, 90/s | exceeds VRAM (12.3 GiB) |
| 128 | exceeds VRAM (18.3 GiB) | exceeds VRAM (17.4 GiB) | not run |

Windows keeps about 1.2 GiB of the card for the desktop: 10.8 of 12.0 GiB was free when the
checks started, so that is the practical budget for training plus anything else on the GPU.

Throughput saturates around batch size 32 for every variant. The largest batch of each sweep
(64 for vlm, 128 otherwise) needs more memory than the GPU has. Under WSL2 that does not raise an
out-of-memory error: the driver spills into system memory. In a development run the expert step
at batch 128 slowed from 0.72 s to 16 s. The check therefore compares peak memory with VRAM
explicitly and stops the sweep there.

### Inference

| Variant | Latency per 50-action chunk, median / max | Peak memory |
|---|---|---|
| expert | 181 / 194 ms | 0.90 GiB |
| lora | 212 / 222 ms | 0.91 GiB |
| vlm | 178 / 180 ms | 0.90 GiB |

One prediction is a chunk of 50 actions from 10 flow-matching steps at batch size 1. The policy
runs every few control steps, not every step, so this latency bounds evaluation speed but does
not block closed-loop control in simulation, where physics waits for the policy. LoRA is slower
because its adapters run as separate layers; merging them into the base weights before
deployment removes the difference.

### Implications for M3

- Start with LeRobot's default fine-tuning (expert only) at batch size 32: 5.6 GiB leaves room
  to render and evaluate on the same GPU, and larger batches add no throughput.
- LoRA saves little memory here, because activations rather than optimizer state dominate. Its
  advantage is a 3M-parameter artifact that is cheap to store and publish; it is an ablation, not
  a requirement.
- Training the VLM's text layers fits at batch size 32 (7.2 GiB) and is the next step up if the
  frozen VLM limits instruction following.

## Observations

- LeRobot loads the VLM and the action expert in bfloat16, so the expert's trainable weights and
  AdamW state are bfloat16. Small updates can round away; whether fp32 master weights help is an
  M3 question.
- SmolVLA was pretrained on SO-100 data with three cameras. A single head camera leaves its
  multi-view pretraining partly unused; a wrist camera is an M1 decision.
- At zero joint angles the G1 forearm points forward and the hand blocks the head camera's view
  of the table. The right bin is outside the head camera's view in the first layout. Both are
  M1 tasks.

## Re-estimate

Measured compute is not the bottleneck:

- **Training:** 20,000 steps at batch size 32 take about 2 hours with LeRobot's default
  fine-tuning.
- **Data generation:** at full rendering quality, one camera at 30 Hz costs about 20 s of
  rendering per 20 s episode; 300 episodes take under 2 hours.
- **Evaluation:** with one prediction per 25 executed actions, a 20 s episode at 30 Hz needs 24
  predictions: about 5 s of inference and rendering. The final test suite (~5,600 episodes) is
  about one night on this GPU.

The main schedule risk is M1: reliable physical grasping with the three-finger Dex3 hand, whose
actuator gains Menagerie marks as untuned. M1 should be timeboxed. If grasping stays unreliable,
shrink the operating region or simplify the objects before collecting data, as the plan says.
The 6–8 week part-time envelope stands, with M1 as the milestone to watch.
