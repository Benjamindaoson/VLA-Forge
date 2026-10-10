"""Precision scope for separately registered optimized closed-loop trials."""

import torch

from .protocol import load_protocol
from .schema import content_hash


def verify_quality_protocol(root):
    from pathlib import Path

    root = Path(root)
    quality = load_protocol(root / "configs/optimization_quality_protocol_v1.json")
    if quality["status"] != "FROZEN":
        raise ValueError("quality protocol is not frozen")
    for field, filename in [
        ("data_protocol_identity", "data_protocol_v1.json"),
        ("compute_protocol_identity", "compute_protocol.json"),
        ("inference_protocol_identity", "inference_protocol_v1.json"),
    ]:
        if quality[field] != load_protocol(root / "configs" / filename)["identity"]:
            raise ValueError("quality protocol dependency changed")
    for path, expected in quality["source_hashes"].items():
        if content_hash((root / path).read_text(encoding="utf-8")) != expected:
            raise ValueError(f"quality source changed: {path}")
    return quality


def validate_runtime_pair(reference, candidate):
    reference_packages = {
        name.lower().replace("_", "-"): version for name, version in reference["packages"].items()
    }
    candidate_packages = {
        name.lower().replace("_", "-"): version for name, version in candidate["packages"].items()
    }
    for name in [
        "torch",
        "torchvision",
        "transformers",
        "lerobot",
        "hf-libero",
        "robosuite",
        "mujoco",
        "numpy",
    ]:
        if not reference_packages.get(name) or reference_packages.get(
            name
        ) != candidate_packages.get(name):
            raise ValueError(f"paired runtime dependency differs or missing: {name}")


def validate_baseline_pair(config, parent_run_id, checkpoint_sha256, seed):
    if (
        config.get("experiment_type") != "formal_eval"
        or config.get("parent_run_id") != parent_run_id
        or config.get("checkpoint_sha256") != checkpoint_sha256
        or config.get("seed") != seed
        or config.get("model") != "smolvla"
    ):
        raise ValueError("baseline is not the same SmolVLA checkpoint and seed")


def autocast_chunks(policy, device, precision):
    if precision not in {"fp32", "bf16"}:
        raise ValueError("unsupported inference precision")
    method = "_get_action_chunk" if hasattr(policy, "_get_action_chunk") else "predict_action_chunk"
    original = getattr(policy, method)

    def predict(*args, **kwargs):
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=precision == "bf16"):
            return original(*args, **kwargs)

    setattr(policy, method, predict)
    return policy
