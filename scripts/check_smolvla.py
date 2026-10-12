"""Milestone 0 check: SmolVLA memory, training speed and inference latency at this project's shapes.

Loads the pretrained SmolVLA base with this project's features (head camera, joint state of the
active arm and hand, arm targets plus one hand-closure value), then on synthetic batches checks:

1. gradient routing: which parameter groups train and receive gradients;
2. overfitting one fixed batch: the training loss must fall;
3. training steps (forward, backward, clipped AdamW update) at increasing batch sizes until the
   GPU runs out of memory: peak memory and step time;
4. inference: latency of one action-chunk prediction at batch size 1.

    uv run python scripts/check_smolvla.py --variant expert [--save]

Variants:
    expert  LeRobot's default fine-tuning: VLM frozen; action expert and projections train
    lora    LoRA (r=64) on LeRobot's default SmolVLA targets; all base weights frozen
    vlm     the VLM's text layers train too (the vision encoder stays frozen)
"""

import argparse
import gc
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable

import torch
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE

from humanoid_vla.paths import RESULTS_DIR
from humanoid_vla.policies.smolvla import (
    SMOLVLA_REPO,
    SMOLVLA_REVISION,
    load_smolvla,
    make_features,
)
from humanoid_vla.sim.g1 import G1Config, arm_joints
from humanoid_vla.utils.provenance import write_result

INSTRUCTION = "Put the red block in the left bin."
IMAGE_HW = (256, 256)
ROBOT = G1Config()
STATE_DIM = len(ROBOT.active_joints)  # arm and hand joint positions
ACTION_DIM = len(arm_joints(ROBOT.active_side)) + 1  # arm targets and one hand-closure value
BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)
TIMED_STEPS = 5
OVERFIT_BATCH_SIZE = 8
GIB = 2**30

# Parameter groups by name, checked in order (LoRA weights sit inside the expert's modules).
GROUPS = (
    ("lora", ("lora_",)),
    ("vision", ("vision_model",)),
    ("connector", ("connector",)),
    ("vlm_text", ("text_model",)),
    ("lm_head", ("vlm.lm_head",)),
    ("expert", ("lm_expert",)),
    ("projections", ("state_proj", "action_in_proj", "action_out_proj", "action_time_mlp")),
)
EXPECTED_TRAINABLE = {
    "expert": {"expert", "projections"},
    "lora": {"lora"},
    "vlm": {"vlm_text", "connector", "expert", "projections"},
}


def group_of(name: str) -> str:
    for group, patterns in GROUPS:
        if any(pattern in name for pattern in patterns):
            return group
    return "other"


def make_batch(config, batch_size: int, seed: int, with_action: bool = True) -> dict:
    """Synthetic raw batch with the shapes and value ranges of recorded data."""
    gen = torch.Generator().manual_seed(seed)
    batch: dict = {
        key: torch.rand(batch_size, *feature.shape, generator=gen)
        for key, feature in config.image_features.items()
    }
    batch[OBS_STATE] = torch.randn(batch_size, STATE_DIM, generator=gen)
    if with_action:
        batch[ACTION] = torch.randn(batch_size, config.chunk_size, ACTION_DIM, generator=gen)
        batch["action_is_pad"] = torch.zeros(batch_size, config.chunk_size, dtype=torch.bool)
    batch["task"] = [INSTRUCTION] * batch_size
    return batch


def identity_stats() -> dict:
    """Normalization statistics that leave state and action unchanged."""
    return {
        OBS_STATE: {"mean": torch.zeros(STATE_DIM), "std": torch.ones(STATE_DIM)},
        ACTION: {"mean": torch.zeros(ACTION_DIM), "std": torch.ones(ACTION_DIM)},
    }


def gradient_routing(model: torch.nn.Module) -> dict[str, dict[str, int]]:
    """Parameter counts per group: total, trainable, and receiving a non-zero gradient."""
    counts: dict[str, Counter] = {}
    for name, param in model.named_parameters():
        group = counts.setdefault(group_of(name), Counter())
        group["params"] += param.numel()
        if param.requires_grad:
            group["trainable"] += param.numel()
            if param.grad is not None and bool(param.grad.abs().sum() > 0):
                group["with_grad"] += param.numel()
    keys = ("params", "trainable", "with_grad")
    return {group: {k: c[k] for k in keys} for group, c in sorted(counts.items())}


def sync_time(fn: Callable[[], object], repeats: int) -> list[float]:
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return times


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=sorted(EXPECTED_TRAINABLE), default="expert")
    parser.add_argument("--cameras", nargs="+", default=["head"], help="camera feature names")
    parser.add_argument("--overfit-steps", type=int, default=200)
    parser.add_argument("--save", action="store_true", help="write results/m0/smolvla_*.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("error: CUDA is not available", file=sys.stderr)
        return 1
    free_at_start, total_memory = torch.cuda.mem_get_info()

    inputs, outputs = make_features(args.cameras, STATE_DIM, ACTION_DIM, IMAGE_HW)
    policy = load_smolvla(inputs, outputs, train_vlm=args.variant == "vlm")
    config = policy.config
    preprocess, _ = make_smolvla_pre_post_processors(config, dataset_stats=identity_stats())
    model: torch.nn.Module = policy
    lr = config.optimizer_lr
    if args.variant == "lora":
        model = policy.wrap_with_peft(peft_cli_overrides={"method_type": "LORA", "r": 64})
        lr *= 10  # LeRobot's guidance for LoRA learning rates
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=lr,
        betas=config.optimizer_betas,
        eps=config.optimizer_eps,
        weight_decay=config.optimizer_weight_decay,
    )

    def train_step(batch: dict) -> float:
        loss, _ = model.forward(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, config.optimizer_grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return loss.item()

    # 1. Gradient routing. One update first: LoRA's B matrices start at zero, so its A matrices
    # receive exactly zero gradient until B has moved.
    batch = preprocess(make_batch(config, 2, seed=0))
    train_step(batch)
    loss, _ = model.forward(batch)
    loss.backward()
    routing = gradient_routing(model)
    optimizer.zero_grad(set_to_none=True)
    trainable_groups = {group for group, c in routing.items() if c["trainable"]}
    missing_grads = {group: c["trainable"] - c["with_grad"] for group, c in routing.items()}
    routing_ok = trainable_groups == EXPECTED_TRAINABLE[args.variant] and not any(
        missing_grads.values()
    )

    # 2. Overfit one fixed batch.
    batch = preprocess(make_batch(config, OVERFIT_BATCH_SIZE, seed=1))
    torch.manual_seed(0)
    losses = [train_step(batch) for _ in range(args.overfit_steps)]
    window = max(1, args.overfit_steps // 10)
    first, last = statistics.mean(losses[:window]), statistics.mean(losses[-window:])
    overfit_ok = last < 0.5 * first

    # 3. Training memory and speed by batch size. Under WSL2 the driver spills allocations beyond
    # VRAM into system memory (about 20x slower) instead of failing, so exceeding physical VRAM is
    # detected explicitly and ends the sweep like an out-of-memory error.
    sweep = []
    for batch_size in BATCH_SIZES:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        row: dict = {"batch_size": batch_size}
        try:
            batch = preprocess(make_batch(config, batch_size, seed=2))
            train_step(batch)  # warm-up
            if torch.cuda.max_memory_reserved() > total_memory:
                row["exceeds_vram"] = True
            else:
                times = sync_time(lambda b=batch: train_step(b), TIMED_STEPS)
        except torch.OutOfMemoryError:
            row["out_of_memory"] = True
            optimizer.zero_grad(set_to_none=True)
        row["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / GIB, 2)
        row["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / GIB, 2)
        if "exceeds_vram" in row or "out_of_memory" in row:
            sweep.append(row)
            print(f"  batch {batch_size:3d}: {row}", flush=True)
            break
        step_s = statistics.median(times)
        row["step_s"] = round(step_s, 4)
        row["samples_per_s"] = round(batch_size / step_s, 1)
        sweep.append(row)
        print(f"  batch {batch_size:3d}: {row}", flush=True)
    batch = None

    # 4. Inference latency at batch size 1, without optimizer state.
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    policy.eval()
    policy.reset()
    observation = preprocess(make_batch(config, 1, seed=3, with_action=False))
    chunk = policy.predict_action_chunk(observation)
    times = sync_time(lambda: policy.predict_action_chunk(observation), 3)  # warm-up
    times = sync_time(lambda: policy.predict_action_chunk(observation), 20)
    inference = {
        "chunk_shape": list(chunk.shape),
        "flow_steps": config.num_steps,
        "latency_ms_median": round(1e3 * statistics.median(times), 1),
        "latency_ms_max": round(1e3 * max(times), 1),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / GIB, 2),
    }

    result = {
        "variant": args.variant,
        "model": {"repo": SMOLVLA_REPO, "revision": SMOLVLA_REVISION},
        "shapes": {
            "cameras": args.cameras,
            "image_hw": list(IMAGE_HW),
            "model_image_hw": list(config.resize_imgs_with_padding),
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "chunk_size": config.chunk_size,
        },
        "gpu_memory_gib": {
            "total": round(total_memory / GIB, 2),
            "free_at_start": round(free_at_start / GIB, 2),
        },
        "optimizer": {"type": "AdamW", "lr": lr, "grad_clip_norm": config.optimizer_grad_clip_norm},
        "gradient_routing": routing,
        "gradient_routing_ok": routing_ok,
        "overfit": {
            "batch_size": OVERFIT_BATCH_SIZE,
            "steps": args.overfit_steps,
            "mean_loss_first": round(first, 4),
            "mean_loss_last": round(last, 4),
            "loss_every_10_steps": [round(x, 4) for x in losses[::10]],
            "ok": overfit_ok,
        },
        "training_sweep": sweep,
        "inference": inference,
    }

    trainable = sum(c["trainable"] for c in routing.values())
    total = sum(c["params"] for c in routing.values())
    print(f"Variant {args.variant}: {trainable / 1e6:.1f}M of {total / 1e6:.1f}M params trainable")
    print(f"Gradient routing ok: {routing_ok} (trainable groups: {sorted(trainable_groups)})")
    print(f"Overfit: loss {first:.3f} -> {last:.3f} ({'ok' if overfit_ok else 'NOT ok'})")
    print(f"Inference: {inference['latency_ms_median']} ms per chunk of {config.chunk_size}")
    if args.save:
        suffix = "" if args.cameras == ["head"] else f"_{len(args.cameras)}cam"
        path = RESULTS_DIR / "m0" / f"smolvla_{args.variant}{suffix}.json"
        write_result(path, result)
        print(f"Saved {path}")
    return 0 if routing_ok and overfit_ok else 1


if __name__ == "__main__":
    sys.exit(main())
