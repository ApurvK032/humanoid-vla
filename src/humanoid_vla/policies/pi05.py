"""π0.5 (Physical Intelligence's openpi model, as ported to LeRobot) at a pinned Hub revision.

The pretrained base pairs a PaliGemma vision-language model (Gemma 2B with a SigLIP vision encoder)
with a 300M Gemma action expert, about 3.3B parameters in all, stored in float32. State and action
vectors are zero-padded to 32 inside the model, so this project's 14-dim state and 8-dim action fit
without changing the pretrained projections. Its preprocessor tokenizes instructions with the
PaliGemma tokenizer from ``google/paligemma-3b-pt-224``, a gated Hub repository: accept its license
on the Hub and log in (``hf auth login``) before training or running π0.5.

Two fine-tuning setups fit a 48 GB GPU (A40 or L40S):

- ``expert-float32``: the whole model in float32, the vision-language model frozen and only the
  action expert and projections trained. This is SmolVLA's best recipe (D-026, D-027): float32
  weights keep small updates.
- ``full-bfloat16``: everything trained, with weights and optimizer state in bfloat16 (the vision
  encoder stays float32) and gradient checkpointing. Fully training in float32 would need over
  50 GB for weights, gradients and AdamW state alone.
"""

from dataclasses import dataclass

PI05_REPO = "lerobot/pi05_base"
PI05_REVISION = "22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0"  # Hub commit of 2026-10-06
TOKENIZER_REPO = "google/paligemma-3b-pt-224"  # gated; read by LeRobot's π0.5 preprocessor


@dataclass(frozen=True)
class Setup:
    description: str
    dtype: str  # LeRobot's PI05Config.dtype
    train_expert_only: bool
    gradient_checkpointing: bool


SETUPS = {
    "expert-float32": Setup(
        "float32 weights; vision-language model frozen, action expert and projections trained",
        dtype="float32",
        train_expert_only=True,
        gradient_checkpointing=False,
    ),
    "full-bfloat16": Setup(
        "bfloat16 weights (vision encoder float32); all weights trained, gradient checkpointing",
        dtype="bfloat16",
        train_expert_only=False,
        gradient_checkpointing=True,
    ),
}


def policy_arguments(setup: Setup) -> list[str]:
    """lerobot-train ``--policy.*`` arguments for a fine-tuning setup."""
    return [
        f"--policy.dtype={setup.dtype}",
        f"--policy.train_expert_only={str(setup.train_expert_only).lower()}",
        f"--policy.gradient_checkpointing={str(setup.gradient_checkpointing).lower()}",
    ]
