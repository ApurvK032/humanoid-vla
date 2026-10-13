# Third-party code, models and assets

Nothing listed here is vendored into this repository. Assets are fetched at pinned versions and
Python packages are locked in `uv.lock`.

| Component | Use | Version pin | License |
|---|---|---|---|
| [MuJoCo](https://github.com/google-deepmind/mujoco) | Physics and rendering | `mujoco==3.14.0` (pyproject.toml) | Apache-2.0 |
| [MuJoCo Menagerie: Unitree G1](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1) | Robot model (`g1_with_hands`), fetched into `third_party/` by `make assets` | commit `71f066ad0be9cd271f7ed58c030243ef157af9f4` (src/humanoid_vla/sim/assets.py) | BSD-3-Clause |
| [LeRobot](https://github.com/huggingface/lerobot) | Datasets, policy training, PEFT | `lerobot==0.6.1` (pyproject.toml) | Apache-2.0 |
| [SmolVLA base](https://huggingface.co/lerobot/smolvla_base) | Pretrained VLA weights | Hub revision pinned in code | Apache-2.0 |
| [SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) | Vision-language backbone of SmolVLA | via SmolVLA | Apache-2.0 |
| [PyTorch](https://github.com/pytorch/pytorch) | Training and inference | `uv.lock` | BSD-3-Clause |

The Menagerie G1 model is derived from Unitree's robot description. Its README and LICENSE are in
`third_party/mujoco_menagerie/unitree_g1/` after fetching.
