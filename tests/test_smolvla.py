from pathlib import Path

import pytest

# These need the train dependency group, which CI does not install.
torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
smolvla = pytest.importorskip("humanoid_vla.policies.smolvla")


def test_stored_expert_dtype_reads_the_action_expert(tmp_path: Path) -> None:
    prefix = "model.vlm_with_expert"
    for dtype, expected in [(torch.bfloat16, "BF16"), (torch.float32, "F32")]:
        weights = {
            f"{prefix}.vlm.model.text_model.layers.0.self_attn.q_proj.weight": torch.zeros(
                2, 2, dtype=torch.bfloat16
            ),
            f"{prefix}.lm_expert.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2, dtype=dtype),
        }
        safetensors_torch.save_file(weights, tmp_path / "model.safetensors")
        assert smolvla.stored_expert_dtype(tmp_path) == expected

    safetensors_torch.save_file(
        {"model.state_proj.weight": torch.zeros(2, 2)}, tmp_path / "model.safetensors"
    )
    with pytest.raises(ValueError, match="no SmolVLA action expert"):
        smolvla.stored_expert_dtype(tmp_path)
