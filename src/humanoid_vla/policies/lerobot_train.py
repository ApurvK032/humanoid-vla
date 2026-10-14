"""`lerobot-train` with this project's model options.

    python -m humanoid_vla.policies.lerobot_train [--float32-expert] [--cameras=head,wrist] \\
        <lerobot-train arguments>

``--float32-expert`` builds SmolVLA's action expert with float32 weights (see
``float32_action_expert``). ``--cameras`` gives the policy only the named cameras of the dataset;
LeRobot would otherwise give it every camera the dataset has. Every other argument goes to LeRobot
unchanged.
"""

import contextlib
import sys
from collections.abc import Iterator, Sequence

import lerobot.scripts.lerobot_train as lerobot_train
from lerobot.configs import FeatureType
from lerobot.utils.feature_utils import dataset_to_policy_features

from humanoid_vla.data.recording import image_key
from humanoid_vla.policies.smolvla import float32_action_expert

FLOAT32_EXPERT = "--float32-expert"
CAMERAS = "--cameras="


@contextlib.contextmanager
def only_cameras(cameras: Sequence[str]) -> Iterator[None]:
    """Make lerobot-train build its policy with only these dataset cameras as image inputs."""
    original = lerobot_train.make_policy

    def make_policy(cfg, ds_meta=None, **kwargs):
        features = dataset_to_policy_features(ds_meta.features)
        missing = [cam for cam in cameras if image_key(cam) not in features]
        if missing:
            raise ValueError(f"the dataset has no camera {missing}")
        keep = {image_key(cam) for cam in cameras}
        cfg.input_features = {
            key: feature
            for key, feature in features.items()
            if feature.type is not FeatureType.ACTION
            and (feature.type is not FeatureType.VISUAL or key in keep)
        }
        return original(cfg=cfg, ds_meta=ds_meta, **kwargs)

    lerobot_train.make_policy = make_policy
    try:
        yield
    finally:
        lerobot_train.make_policy = original


def main() -> None:
    options = contextlib.ExitStack()
    if FLOAT32_EXPERT in sys.argv:
        sys.argv.remove(FLOAT32_EXPERT)
        options.enter_context(float32_action_expert())
    for arg in [a for a in sys.argv if a.startswith(CAMERAS)]:
        sys.argv.remove(arg)
        options.enter_context(only_cameras(arg.removeprefix(CAMERAS).split(",")))
    with options:
        lerobot_train.main()  # parses the remaining arguments


if __name__ == "__main__":
    main()
