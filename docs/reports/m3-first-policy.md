# M3 first policy report

Date: 2026-10-15. Machine: RTX 4070 (12 GB), Ryzen 9 7900X, Ubuntu 26.04 under WSL2.
Training summaries, probes and evaluation results with their provenance are in
[`results/m3/`](../../results/m3/).

![The final policy on one development layout under two instructions](../media/m3_policy_pair.gif)
*The final policy (`smolvla_v2_long_fp32`, 50,000 steps) on a development layout it never trained
on: "Put the green block in the right bin" (left) and "Put the blue block in the left bin"
(right).*

## Verdict

The Milestone 3 exit condition is met. The final policy succeeds in 51 of 70 episodes on held-out
development layouts (73%, 95% interval 61-82%), goes for the instructed block first in 80% of them,
and follows both instructions on 13 of 17 layouts evaluated with two instructions that name
different blocks. Videos of four episodes are recorded.

It took four runs. Three 20,000-step runs learned to grasp, carry and place, and to read the bin
word, but chose the block at chance. Recording every layout under two instructions and training
the VLM's text layers did not change that. A run 2.5 times longer, with the learning rate
decaying over the whole run and float32 action-expert weights, learned the block word between
10,000 and 30,000 steps.

| Run | Change | Success, final checkpoint | Instructed block first |
|---|---|---|---|
| `smolvla_v1` | LeRobot's default fine-tuning on `g1_sort_v1`, 20,000 steps | 21/70 (30%) | 30% |
| `smolvla_v2` | every training layout under two instructions (`g1_sort_v2`) | 16/70 (23%) | 26% |
| `smolvla_v2_vlm` | v2 plus the VLM's text layers trained | 12/70 (17%) | 20% |
| `smolvla_v2_long_fp32` | v2, 50,000 steps, float32 expert weights | 51/70 (73%) | 80% |

## Training setup

| | 20,000-step runs | Final run |
|---|---|---|
| Base model | `lerobot/smolvla_base` at Hub commit `5e8d12a6` (450M parameters, D-009) | same |
| Trained | action expert and state/action projections (100M); VLM frozen, LeRobot's default (D-011). `smolvla_v2_vlm` also trains the VLM's text layers (306M) | action expert and projections (100M), expert weights in float32 |
| Inputs | head and wrist cameras (256x256, padded to 512x512 inside SmolVLA), 14 joint angles, instruction | same |
| Outputs | chunks of 50 actions (2 s): 7 arm joint targets and one hand closure value | same |
| Data | `g1_sort_v1` (300 episodes) or `g1_sort_v2` (295), about 56,000 frames each | `g1_sort_v2` |
| Optimizer | AdamW, peak learning rate 1e-4, betas (0.9, 0.95), weight decay 1e-10, gradient clipping at 10 | same |
| Schedule | 666 warmup steps, cosine decay to 2.5e-6 at step 20,000 (LeRobot fits its defaults to the run) | 1,000 warmup steps, cosine decay to 2.5e-6 at step 50,000 |
| Batch and steps | 32 x 20,000 = 640,000 samples, about 11 passes over the data | 32 x 50,000 = 1.6M samples, about 29 passes |
| Seed | 1000 | 1000 |
| Throughput | 0.63 s per step (0.68 with the VLM), 7.7 GB (9.75 GB with the VLM) | 0.655 s per step, 8.5 GB; 552 min |

`scripts/train_smolvla.py` runs LeRobot's own training script and records the command, the base
and dataset identities and the loss history (D-023).

## How policies are judged

Closed loop: on 50 development layouts (70 episodes; 20 layouts carry two instructions), the policy
observes, predicts a 50-action chunk, executes the first 10 (0.4 s) and observes again, until the
environment's evaluator ends the episode or 25 s pass. Reported: success with a 95% Wilson
interval, the first block lifted, where every block ended, and the paired layouts (D-024).

Open-loop probe (`scripts/probe_instruction.py`): on 20 development layouts the policy sees the
first observation with each block named in turn, from the same noise, and forward kinematics of
the chunk's last arm targets shows where it sends the hand after 2 s. The scripted expert's hand
is then above the named block: it is nearest the named block every time, and its hand targets
for the three instructions are a median 18.1 cm apart. A policy that ignores the block word sends
the hand to the same place whatever the instruction, and scores 33% by chance.

## Pilot: a falling loss is not a working policy

Before the full runs, 2,000 steps on the 55-episode pilot dataset (22.5 min) took the training
loss from 1.40 to 0.043. In closed loop on 10 of its own training layouts (17 episodes), the
policy succeeded twice (95% interval 3-34%), lifted the instructed block first in 3 episodes,
lifted nothing in 9 and placed a wrong block in 5.

The pipeline was checked before blaming the policy: images decoded from the dataset and rendered
by the simulator differ by about 2/255 and give similar predictions, actions are unnormalized
with the dataset statistics, and the queue resets each episode. Open-loop, on recorded frames,
the predicted arm targets were 0.02-0.04 rad off (about 1.2 cm at the hand, near the claw
grasp's tolerance). Imprecision looked like the main problem at the time; the full runs showed
that it was not: with 300 episodes the policy grasps reliably once it goes for the right block,
and the remaining failures are about which block it goes for.

## First full run: the bin word without the block word

`smolvla_v1` at 20,000 steps, on the 70 development episodes:

| | Episodes |
|---|---|
| Success | 21 of 70 (30%, 95% interval 21-42%) |
| Lifted the instructed block first | 21, then succeeded in 20 |
| Lifted another block first | 25: 20 placed it in a bin (wrong object), 4 timed out, 1 recovered |
| Lifted no block | 24: 23 timed out, 1 knocked a block off the table |
| Placed block in the instructed bin | 41 of 41 |
| Layout pairs naming different blocks, both followed | 0 of 17 |

Inference took 181.5 ms per chunk on an idle GPU. Successful episodes took 6.6-21.5 s (median
8.0 s), close to the expert's 7.5 s.

The policy manipulates well and follows the bin word, but not the block word. Once it goes for
the instructed block it nearly always completes the task, and every block it places goes into
the instructed bin, which nothing in the image shows. Which block it goes for barely depends on
the instruction: it lifted the instructed block first in 21 of 46 episodes where it lifted any
(chance: one in three), and on no layout with two block instructions did it follow both. When it
cannot settle on a block it hesitates between them until the episode times out.

| Checkpoint | Training loss | Success (95% interval) | Instructed block lifted first | Lifted no block | Probe |
|---|---|---|---|---|---|
| 5,000 | 0.057 | 16/70 (15-34%) | 29% | 28 | 33%, 1.2 cm |
| 10,000 | 0.036 | 14/70 (12-31%) | 30% | 26 | 37%, 3.5 cm |
| 15,000 | 0.028 | 11/70 (9-26%) | 23% | 26 | 40%, 5.9 cm |
| 20,000 | 0.022 | 21/70 (21-42%) | 30% | 24 | 42%, 6.0 cm |

The probe column gives how often the first chunk sends the hand nearest the named block and the
median spread of hand targets across instructions. The loss fell by more than half while success
stayed within the intervals of the first checkpoint, because the choice of block never improved
beyond chance. Success is roughly the chance of choosing the instructed block (one in three)
times the chance of lifting a block at all (about two in three) times the grasp-and-place rate
once it commits (20 of 21). The probe's spread grew with training but stalled as the learning
rate decayed.

The bins never move, so "left" maps to the same motion on every layout. The blocks move, so
"red" has to be bound to a region of the image first, and only the first one to two seconds of
an episode depend on the block word at all.

## Every layout under two instructions (`g1_sort_v2`)

In `g1_sort_v1` a third of the training layouts carry two instructions; on the other two thirds
the target can be read off the image. If that let the policy skip the block word, recording every
layout under two instructions should force it to read the word (D-025). `g1_sort_v2` has the size
of v1 (295 episodes, 55,463 frames, all replaying exactly) from 150 layouts, each recorded twice;
119 of the 145 complete pairs name different blocks. Trained with the same recipe:

| Checkpoint | v1: success | v2: success | v1: probe | v2: probe |
|---|---|---|---|---|
| 5,000 | 16/70 | 17/70 | 33%, 1.2 cm | 37%, 2.8 cm |
| 10,000 | 14/70 | 15/70 | 37%, 3.5 cm | 35%, 2.1 cm |
| 15,000 | 11/70 | 15/70 | 40%, 5.9 cm | 43%, 7.0 cm |
| 20,000 | 21/70 | 16/70 | 42%, 6.0 cm | 38%, 6.0 cm |

Pairing every layout changed nothing measurable at 20,000 steps. The bin word stays followed (34
of 35 placed blocks at v2's 20,000 steps) and the block word stays ignored.

## Training the VLM's text layers (`smolvla_v2_vlm`)

D-011 named training the VLM's text layers as the next step if instruction following was weak.
With `--train-vlm` on v2 (306M trainable parameters instead of 100M; 9.75 GB, 10.8 GB of the
GPU; 0.684 s per step), the training loss fell to 0.013 against 0.022, but behavior did not
change: 6, 17, 14 and 12 successes of 70 at 5,000-20,000 steps, probe 35, 33, 42 and 37%.

## Looking for a bug

Three setups failing the same way, while the bin word was followed perfectly, called for checking
the pipeline before the learning:

- All 300 task strings in `g1_sort_v1` name the episode's target block and bin.
- All 108 block centers projected into the head camera on development layouts show the named
  color; the colors are distinct by eye in the head view.
- On their own training layouts, the 20,000-step checkpoints of all three runs are also at
  chance (30-37%), including on the exact recorded instructions (15 of 40 for pairs sharing a
  bin, 17 of 40 for pairs that differ in the bin), so the bin word is not standing in for the
  block word either.
- On the first frame of 24 v2 training episodes, the VLM run sends the hand nearest the
  instructed block 18 times on the decoded training frames, 12 times on clean renders and 14
  times on renders passed once through the dataset's AV1 settings. Training on compressed video
  and evaluating on clean renders shifts the input, which explains part of the gap, but even on
  its own training frames the policy chose correctly only three times in four.

No pipeline error turned up. The policies fit the motions but had not learned to bind the color
word to a block. SmolVLA's published benchmark runs train on roughly ten times as many samples,
and v1's probe spread kept growing until the learning rate decayed.

## The longer run (`smolvla_v2_long_fp32`)

The final recipe trains 2.5 times longer, spreads the learning-rate decay over the whole run, and
keeps the action expert in float32 so that late, small updates are not rounded away (D-026):

| Checkpoint | Training loss | Learning rate | Probe | Success (95% interval) |
|---|---|---|---|---|
| 10,000 | 0.041 | 9.1e-5 | 33%, 1.9 cm | |
| 20,000 | 0.023 | 6.6e-5 | 48%, 9.3 cm | |
| 30,000 | 0.015 | 3.6e-5 | 78%, 13.9 cm | 42/70 (48-71%) |
| 40,000 | 0.009 | 1.2e-5 | 85%, 16.7 cm | |
| 50,000 | 0.007 | 2.5e-6 | 85%, 17.1 cm | 51/70 (61-82%) |

At 10,000 steps the policy ignored the block word as the earlier runs did. By 30,000 steps it
sent the hand nearest the named block 78% of the time, and by 40,000 steps its hand targets for
different instructions were nearly as far apart as the expert's. On the 70 development episodes
at 50,000 steps:

| | Episodes |
|---|---|
| Success | 51 of 70 (73%, 95% interval 61-82%) |
| Lifted the instructed block first | 56, then succeeded in 51 (2 placed another block, 3 timed out) |
| Lifted another block first | 4: 1 placed it in a bin, 3 timed out |
| Lifted no block | 10, all timed out |
| Layout pairs naming different blocks, both followed | 13 of 17 |
| Layouts under two instructions, both succeeded | 14 of 20 |

Inference took 169 ms per chunk. Successful episodes took 6.3-15.4 s (median 7.6 s). Running out
of time is now the main failure (16 episodes, 10 of them without lifting a block), not the wrong
block (4). The 30,000-step checkpoint still lifted a wrong block first in 15 episodes; the last
20,000 steps, at a decaying learning rate, cut that to 4.

## bfloat16 action expert

LeRobot builds SmolVLA's action expert in the VLM's dtype, bfloat16, and AdamW updates those
weights directly: an update smaller than half the spacing between bfloat16 values near a weight
(0.2-0.4% of the weight) is lost. Comparing the pilot's checkpoints, 57% of the expert's 97M
bfloat16 weights changed at all during the first 1,000 steps, and 24% during the second 1,000 as
the learning rate decayed, against 99% of the float32 projection weights in both. The base
model's expert values are all exactly representable in bfloat16, so rounding of the base does not
inflate these counts. `--float32-expert` builds the expert in float32 for training and inference
(on the CPU its chunk predictions match the bfloat16 model's within 0.0004 rad). The final run
uses it together with the longer schedule; which of the two made the difference has not been
separated.

## Not measured in M3

- **Which change taught the block word**: the longer schedule, float32 weights, or both. A
  50,000-step bfloat16 run on v2 would separate them; v1 data with the final recipe would show
  whether full pairing matters at that length.
- **Actions executed per chunk**: every evaluation executed 10 of 50 (0.4 s).
- **Cameras**: every run used the head and wrist cameras; `--cameras=head` is ready.

## What came up

1. **Rounded step counts in LeRobot's log.** LeRobot logs "step:1K" for steps 1,000 to 1,499,
   so the wrapper's first parser put several log rows at the same step. Rows are now numbered by
   position (the pilot's history was re-parsed from its log; the run is unchanged).
2. **Evaluations do not repeat exactly.** SmolVLA samples each action chunk from noise; two
   evaluations of the pilot on the same 17 episodes gave 1 and 2 successes, so each episode now
   seeds the noise from its scenario (D-024). That was not enough: re-evaluating the 5,000-step
   checkpoint of `smolvla_v1` changed the outcome of 22 of 30 episodes (4, then 7 successes). Its
   chunk predictions are bit-for-bit repeatable and the simulator resets to the identical state,
   but re-rendering the same state changes a median of 7 head-camera pixels by one intensity level
   (in 86 of 90 re-renders; wrist camera 13 of 90). A policy that is undecided between blocks
   turns that into a different choice. Results are compared as rates with their intervals.
3. **Feature dictionaries merge.** Passing a dictionary as `--policy.input_features` merges it
   into the base model's SO-100 features (three cameras) rather than replacing them, so camera
   selection sets the features from the dataset before LeRobot builds the policy.
4. **Table edge.** The layout region's front row of blocks sits about 3 cm from the table's
   front edge. The expert never pushes blocks, but imprecise policies knocked some off the table
   (up to 7 of 70 episodes for early checkpoints); a later task revision could move the region
   back.
5. **Shared GPU.** Inference measured while a training run used the GPU took 626 ms per chunk
   instead of about 180 ms, so timings come from an idle GPU. VLM training leaves no room at all:
   a one-minute probe beside it pushed WSL past 12 GB, the driver spilled into system memory and
   the run stalled for good; it was restarted from scratch.
6. **CI without the training stack.** CI installs no LeRobot or PyTorch, and a module that
   imported torch only to seed the policy's noise broke test collection there. Seeding moved into
   the policy runner, and the tests now also run in an environment without the train group
   before a push.
