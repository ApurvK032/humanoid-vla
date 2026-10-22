"""Closed-loop execution of a fine-tuned SmolVLA or π0.5 checkpoint.

The runner converts environment observations into the batch format LeRobot's dataset produced
during training (CHW float images in [0, 1], float32 state, the instruction as the task), runs
the checkpoint's own pre- and post-processors (tokenizing, normalization, unnormalization), and
returns one action per control step. The policy predicts a chunk of actions and executes
``n_action_steps`` of them before it needs a new observation; ``needs_observation`` tells the
caller when, so cameras only have to render on those steps. A SmolVLA checkpoint trained with a
float32 action expert is built with one, so it runs as it was trained. A π0.5 checkpoint runs in the
precision it was trained in unless ``dtype`` says otherwise: its float32 weights (13 GB) do not fit
the RTX 4070, so it is evaluated there in bfloat16, the precision openpi serves π0.5 in.
"""

import contextlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_STATE

from humanoid_vla.data.recording import image_key
from humanoid_vla.policies.smolvla import float32_action_expert, stored_expert_dtype


class PolicyRunner:
    def __init__(
        self,
        checkpoint: Path,
        n_action_steps: int | None = None,
        device: str = "cuda",
        dtype: str | None = None,
    ) -> None:
        config = PreTrainedConfig.from_pretrained(checkpoint)
        config.device = device
        if n_action_steps is not None:
            if not 1 <= n_action_steps <= config.chunk_size:
                raise ValueError(f"n_action_steps must be in [1, {config.chunk_size}]")
            config.n_action_steps = n_action_steps
        self.config = config
        self.policy_type = config.type
        self.float32_expert = False
        if config.type == "smolvla":
            if dtype is not None:
                raise ValueError("dtype applies to π0.5 checkpoints only")
            self.float32_expert = stored_expert_dtype(checkpoint) == "F32"
            with float32_action_expert() if self.float32_expert else contextlib.nullcontext():
                self.policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config)
            self.weights = "action expert " + ("float32" if self.float32_expert else "bfloat16")
        elif config.type == "pi05":
            trained_dtype = config.dtype
            if dtype is not None:
                config.dtype = dtype
            self.policy = PI05Policy.from_pretrained(checkpoint, config=config)
            self.weights = f"trained in {trained_dtype}, run in {config.dtype}"
        else:
            raise ValueError(f"unsupported policy type {config.type!r} in {checkpoint}")
        self.policy.eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.cameras = [key.removeprefix("observation.images.") for key in config.image_features]
        self._calls = 0

    @property
    def needs_observation(self) -> bool:
        """Whether the next ``act`` call starts a new chunk and so reads the images."""
        return self._calls % self.config.n_action_steps == 0

    def seed(self, seed: int) -> None:
        """Seed the noise the policy samples its action chunks from."""
        torch.manual_seed(seed)

    def reset(self) -> None:
        """Clear the action queue; call at the start of every episode."""
        self.policy.reset()
        self._calls = 0

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        """Action for this control step: 7 arm joint targets and the hand closure (float32)."""
        if self.needs_observation:
            self._check_images(obs)
        with torch.inference_mode():
            action = self.policy.select_action(self.preprocess(self._batch(obs)))
            action = self.postprocess(action)
        self._calls += 1
        return action.squeeze(0).float().cpu().numpy()

    def predict_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        """The whole chunk the policy predicts from ``obs``: ``chunk_size`` actions (float32)."""
        self._check_images(obs)
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(self.preprocess(self._batch(obs)))
            chunk = self.postprocess(chunk)
        return chunk.squeeze(0).float().cpu().numpy()

    def _check_images(self, obs: dict[str, Any]) -> None:
        missing = [cam for cam in self.cameras if cam not in obs.get("images", {})]
        if missing:
            raise ValueError(f"a new chunk needs images from {missing}")

    def _batch(self, obs: dict[str, Any]) -> dict[str, Any]:
        batch: dict[str, Any] = {
            OBS_STATE: torch.from_numpy(np.asarray(obs["state"], dtype=np.float32)).unsqueeze(0),
            "task": [obs["instruction"]],
        }
        for cam in self.cameras:
            image = obs.get("images", {}).get(cam)
            if image is None:  # mid-chunk: the policy does not read images
                image = np.zeros((*self._image_hw(cam), 3), dtype=np.uint8)
            tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
            batch[image_key(cam)] = tensor.unsqueeze(0)
        return batch

    def _image_hw(self, camera: str) -> tuple[int, int]:
        _, height, width = self.config.image_features[image_key(camera)].shape
        return height, width
