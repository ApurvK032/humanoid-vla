"""`lerobot-train` with this project's model options.

    python -m humanoid_vla.policies.lerobot_train [--float32-expert] <lerobot-train arguments>

``--float32-expert`` builds SmolVLA's action expert with float32 weights (see
``float32_action_expert``). Every other argument goes to LeRobot unchanged.
"""

import contextlib
import sys

from lerobot.scripts.lerobot_train import main

from humanoid_vla.policies.smolvla import float32_action_expert

FLOAT32_EXPERT = "--float32-expert"

if __name__ == "__main__":
    float32 = FLOAT32_EXPERT in sys.argv
    if float32:
        sys.argv.remove(FLOAT32_EXPERT)
    with float32_action_expert() if float32 else contextlib.nullcontext():
        main()
