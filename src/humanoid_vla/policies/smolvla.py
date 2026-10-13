"""SmolVLA, loaded from the pretrained base at a pinned Hub revision with this project's features.

The pretrained base was trained on SO-100 data (6-dim state and action, three cameras). State and
action vectors are zero-padded to 32 inside the model, so the pretrained projections load
unchanged for any dimension up to 32; camera inputs are matched by feature name.
"""

from collections.abc import Sequence

import torch
from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

SMOLVLA_REPO = "lerobot/smolvla_base"
SMOLVLA_REVISION = "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"  # Hub commit of 2026-10-06


def camera_key(camera: str) -> str:
    return f"{OBS_IMAGES}.{camera}"


def make_features(
    cameras: Sequence[str],
    state_dim: int,
    action_dim: int,
    image_hw: tuple[int, int],
) -> tuple[dict[str, PolicyFeature], dict[str, PolicyFeature]]:
    """Input and output features: RGB cameras plus robot state in, joint targets out."""
    inputs = {
        camera_key(camera): PolicyFeature(type=FeatureType.VISUAL, shape=(3, *image_hw))
        for camera in cameras
    }
    inputs[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))
    outputs = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}
    return inputs, outputs


def load_smolvla(
    input_features: dict[str, PolicyFeature],
    output_features: dict[str, PolicyFeature],
    *,
    train_vlm: bool = False,
    device: str = "cuda",
) -> SmolVLAPolicy:
    """Load pretrained SmolVLA for the given features.

    By default (LeRobot's fine-tuning setup) the vision-language model is frozen and only the action
    expert and the state/action projections train. ``train_vlm=True`` also trains the VLM's text
    layers; the vision encoder stays frozen either way.
    """
    config = PreTrainedConfig.from_pretrained(SMOLVLA_REPO, revision=SMOLVLA_REVISION)
    if not isinstance(config, SmolVLAConfig):
        raise TypeError(f"{SMOLVLA_REPO} is not a SmolVLA checkpoint: {type(config).__name__}")
    config.input_features = dict(input_features)
    config.output_features = dict(output_features)
    config.train_expert_only = not train_vlm
    config.device = device
    # Recorded in checkpoints and PEFT adapters as the base model to load.
    config.pretrained_path = SMOLVLA_REPO
    config.pretrained_revision = SMOLVLA_REVISION
    policy = SmolVLAPolicy.from_pretrained(SMOLVLA_REPO, config=config, revision=SMOLVLA_REVISION)
    return policy.to(torch.device(device))
