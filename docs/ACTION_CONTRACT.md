# Action contract

What every number in a recorded dataset, and every action a policy sends to `SortEnv`, means.
The code that defines it: `envs/sort.py` (interface), `sim/g1.py` (joint order), `sim/hand.py`
(closure mapping), `data/recording.py` (dataset features). A dataset's `manifest.json` records
the configs it was recorded with.

## Timing

| Quantity | Value |
|---|---|
| Physics timestep | 2 ms |
| Control and recording rate | 25 Hz: one action per 40 ms, held for 20 physics steps |
| Dataset frame | the observation at time t and the action executed from t to t + 40 ms |
| Episode end | when the outcome is decided (success after 0.5 s settled) or at 25 s |

## Observations

| Key | Type | Meaning |
|---|---|---|
| `observation.images.head` | RGB uint8, 256x256 (decoded by LeRobot as float [0, 1], CHW) | Head camera: pitched 58 degrees down, turned 35 degrees right, 70 degree field of view |
| `observation.images.wrist` | RGB uint8, 256x256 | Wrist camera beside the index finger, looking at the claw's pinch point, 80 degree field of view |
| `observation.state` | float32, 14 | Measured joint angles in radians, in the order below |
| `task` | text | The instruction, e.g. "Put the red block in the left bin." |

Images are rendered at full quality, with shadows: the hand's shadow is a depth cue for its height
above the table. Simulator object poses are never part of an observation.

## Actions

`action` is float32 with 8 values:

| Index | Name | Meaning |
|---|---|---|
| 0-6 | `right_shoulder_pitch_joint` ... `right_wrist_yaw_joint` | Absolute arm joint position targets in radians, in the state order below |
| 7 | `hand_closure` | Hand closure in [0, 1]: 0 open, 0.5 fingers fully curled with the thumb open, 1 fully closed |

The environment clips arm targets to the joint ranges, limits each target's change to 0.08 rad
per step (2 rad/s), clips the closure to [0, 1], and rounds everything to float32. Datasets store
these executed values, not the expert's raw requests, so replaying a recorded episode from its
scenario reproduces its joint states exactly.

The closure drives the seven finger joints through a two-stage synergy (`sim/hand.py`): from 0 to
0.5 the index and middle fingers curl, from 0.5 to 1 the thumb closes. Targets beyond contact
make the actuators squeeze up to their torque limits.

## Joint order and ranges

State indices 0-13; action indices 0-6 are the first seven.

| Index | Name | Range (rad) |
|---|---|---|
| 0 | `right_shoulder_pitch_joint` | [-3.089, 2.670] |
| 1 | `right_shoulder_roll_joint` | [-2.252, 1.588] |
| 2 | `right_shoulder_yaw_joint` | [-2.618, 2.618] |
| 3 | `right_elbow_joint` | [-1.047, 2.094] |
| 4 | `right_wrist_roll_joint` | [-1.972, 1.972] |
| 5 | `right_wrist_pitch_joint` | [-1.614, 1.614] |
| 6 | `right_wrist_yaw_joint` | [-1.614, 1.614] |
| 7 | `right_hand_thumb_0_joint` | [-1.047, 1.047] |
| 8 | `right_hand_thumb_1_joint` | [-1.047, 0.724] |
| 9 | `right_hand_thumb_2_joint` | [-1.745, 0.000] |
| 10 | `right_hand_index_0_joint` | [0.000, 1.571] |
| 11 | `right_hand_index_1_joint` | [0.000, 1.745] |
| 12 | `right_hand_middle_0_joint` | [0.000, 1.571] |
| 13 | `right_hand_middle_1_joint` | [0.000, 1.745] |

## For learned policies

- **Chunking.** SmolVLA predicts 50 actions at a time (2 s at 25 Hz). How many it executes
  before predicting again is set when evaluating (M3).
- **Normalization.** State and action statistics come from the training split's dataset only
  (LeRobot's `meta/stats.json`); development and test data are separate datasets.
- **Fit to the pretrained model.** SmolVLA's base checkpoint was trained on 6-dimensional SO-100
  arms. It zero-pads state and action to 32 values, so this contract's 14 and 8 dimensions load
  without changing any weights, but nothing about their meaning carries over: the state and
  action projections must learn it from these demonstrations.

## Scenarios and splits

A scenario is a layout seed plus an instruction (`data/scenarios.py`). Splits own disjoint seed
ranges: train from 100000, development from 200000, test from 300000, 100000 seeds each. Test
layouts are frozen: never recorded for training and never used to choose settings.
