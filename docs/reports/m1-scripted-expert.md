# M1 scripted expert report

Date: 2026-10-12. Machine: RTX 4070 (12 GB), Ryzen 9 7900X, Ubuntu 26.04 under WSL2.
Measurements, each with its provenance, are in [`results/m1/`](../../results/m1/). Re-run the
final evaluation with:

```bash
uv run python scripts/run_expert.py --episodes 500 --seed-start 2000 --videos 2
```

![The scripted expert sorting a block: scene camera and head camera](../media/m1_expert.gif)

## Verdict

The Milestone 1 exit condition is met: the scripted expert reliably picks up and places blocks
by physical contact across a declared operating region. On 500 fresh held-out layouts it
succeeded in 499 of 500 episodes (99.8%, 95% Wilson interval
98.9%-100.0%). It acts through exactly the interface a learned policy will use, and every
episode ends with a labeled outcome.

## The task as built

- **Robot:** fixed-base G1. The right arm (7 joints) and its Dex3 hand (7 joints) are active;
  every other joint is frozen. One closure value drives the hand.
- **Objects:** three 5 cm, 50 g cubes (red, green, blue) on a table, and two bins behind them.
- **Instructions:** "Put the {red, green, blue} block in the {left, right} bin.", six in all.
- **Declared operating region:** block centers in x 0.17-0.26 m and y -0.33 to -0.11 m (robot
  frame, x forward, y left), at least 9 cm apart, at any yaw. Bins fixed at x = 0.40 m,
  y = -0.06 m (left) and -0.30 m (right), 11 cm inside.
- **Interface:** the policy acts at 25 Hz with 7 absolute arm joint targets (clipped to joint
  limits and to 2 rad/s) and one hand closure in [0, 1]. It observes a 256x256 head-camera
  image, the 14 arm and hand joint angles and the instruction. Physics runs at 2 ms.
- **Outcomes,** judged from simulator state the policy never sees:

| Label | Meaning |
|---|---|
| success | Requested block inside the requested bin, below the rim; every block slower than 2 cm/s; hand open and not touching it; held for 0.5 s |
| wrong_bin | Requested block settled in the other bin |
| wrong_object | Another block settled in a bin |
| dropped | A block fell off the table |
| timeout | 25 s elapsed |
| unstable | The simulation diverged |

## Results

Final held-out evaluation: seeds 2000-2499, at commit `091952d`
(clean tree), 134 s of wall time without rendering.

| Outcome | Episodes |
|---|---|
| success | 499 |
| timeout | 1 |

| Instruction (block -> bin) | Successes |
|---|---|
| blue -> left | 92/92 |
| blue -> right | 87/87 |
| green -> left | 71/71 |
| green -> right | 95/95 |
| red -> left | 82/83 |
| red -> right | 72/72 |

Successful episodes took 7.56 s of simulated time at the median and
8.64 s at most; the time limit is 25 s.

Failed episodes:

- seed 2121: timeout, "Put the red block in the left bin."

Seed 2121 is the first held-out run's failure mode again, at the near-left corner of the final
region (block at x = 0.18 m, y = -0.13 m): the block was knocked out of the hand while being
carried to the left bin. It is reported rather than tuned away, because these seeds are the
final test. Narrowing the region further, or a carry path that keeps the elbow off the torso,
would address it; failed attempts are simply excluded from training data.

Three evaluations, in the order they happened:

| Seeds | Layout region | Successes | Role |
|---|---|---|---|
| 0-99 | final | 100/100 | Development: used to tune the expert, so not an unbiased estimate |
| 1000-1499 | left edge at y = -0.08 m | 494/500 (98.8%, 97.4-99.5%) | First held-out run ([result](../../results/m1/expert_eval_first_holdout.json)) |
| 2000-2499 | final | 499/500 (99.8%, 98.9%-100.0%) | Final held-out run ([result](../../results/m1/expert_eval.json)) |

All six failures of the first held-out run were blocks near the robot's centerline carried to
the left bin. Contact traces showed the right upper arm and elbow pressing on the torso during
the lift and carry. The hand sagged, the held block caught the table, and it was knocked out of
the grip. No grasp angle within the claw's tolerance avoids that contact, so the operating
region's left edge moved from y = -0.08 to -0.11 m. The final evaluation then used seeds the
expert had never been run on.

## How the expert got there

| Change | Development success |
|---|---|
| First version: fingers and thumb close together, hand heading near straight ahead | 4/20 |
| Wait for the hand to arrive before closing and releasing; heading 20 degrees; turn toward the bin while carrying | 9/20 |
| Layout moved inside the arm's collision-free reach (bins behind the block strip) | 12/20 |
| Two-stage synergy (fingers, then thumb), pre-curled fingers, thumb-first release | 46/50 |
| Hand heading turns further left for blocks further left | 100/100 |

What mattered, in order of discovery:

1. **The hand's geometry dictates the grasp.** The Dex3 fingers reach about 22 cm from the
   wrist and the thumb about 13 cm, so a fingers-down grasp hits the table before the thumb meets
   the block. A palm-down claw works. It tolerated grasp-height errors from -3 to +0.5 cm, ±1.5 cm
   along and ±1 cm across the approach, and ±30 degrees of block yaw.
2. **Plans must wait for the arm.** The 2 rad/s joint-speed limit makes the arm lag fast plans.
   Closing the hand on schedule, rather than on arrival, missed the block.
3. **IK ignores collisions.** The original left bin was only reachable by folding the upper arm
   across the chest, which collides with the torso. A reach map that rejects near-self-contact
   solutions moved the bins behind the block strip and set the hand heading per position.
4. **Closing order matters.** Closing fingers and thumb together let the thumb push blocks out
   before the fingers formed a wall. Opening them together flung blocks over a bin's far wall.

## Physical honesty

- **No welds or attachment constraints.** Blocks are held only by contact friction, under the
  actuators' torque limits: 1.4 N·m for the fingers and up to 2.45 N·m for the thumb.
- **Gravity compensation** applies to the robot's own arm and hand links, as the real
  controller's does, never to the held block.
- **Contact settings** are MuJoCo's recommendation for manipulation: elliptic friction cones
  and an impedance ratio of 10, with friction 1.0.
- **Self-collisions are simulated.** When the arm hit the torso, the layout changed rather than
  the collision being switched off.

## Limitations

- One block size, mass and shape; fixed bins; a flat table; three blocks.
- Right arm only, on a fixed base.
- The expert uses privileged state and moves in straight lines with fixed timing, so its
  demonstrations are less varied than human ones. M2 can add timing and path variation.
- While grasping, the hand hides the block from the head camera. The dataset will also record a
  wrist camera.
- The head camera's mount (turned 35 degrees right) is a simulation choice: the G1 has no neck
  joint, and its right arm's workspace lies to the right of the head.
- IK missed its target by more than 5 mm on at least one step in 6 of 500 episodes (5 of them still succeeded). The plan continues with the closest solution.

## Next: Milestone 2

Record expert demonstrations as a LeRobot dataset with head and wrist cameras and a documented
action contract, split by scenario, then replay episodes through the same action interface.
