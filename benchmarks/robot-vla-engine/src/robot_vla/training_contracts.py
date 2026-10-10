"""Pure training identities: bounded smoke and step-addressable samples."""

from pathlib import Path

import numpy as np

from .schema import content_hash


def verify_implementation(root, frozen):
    manifest = frozen.get("source_text_hashes", {})
    if not manifest:
        raise ValueError("missing frozen implementation")
    for relative, expected in manifest.items():
        if content_hash((Path(root) / relative).read_text(encoding="utf-8")) != expected:
            raise ValueError(f"frozen implementation mismatch: {relative}")


def batch_indices(length, seed, step, microbatch, batch_size):
    if min(length, batch_size) <= 0 or min(seed, step, microbatch) < 0:
        raise ValueError("invalid sampler arguments")
    return (
        np.random.default_rng(np.random.SeedSequence([seed, step, microbatch]))
        .integers(0, length, batch_size)
        .tolist()
    )


def validate_budget(recipe, frozen):
    if min(recipe["steps"], recipe["batch_size"], recipe["accumulation"]) < 1:
        raise ValueError("budget values must be positive")
    if recipe.get("profile"):
        if recipe["smoke"]:
            raise ValueError("smoke and profiling are mutually exclusive")
        if recipe["steps"] > 10:
            raise ValueError("engineering profiling limited to 10 updates")
        return
    if recipe["smoke"]:
        if recipe["steps"] > 10:
            raise ValueError("engineering smoke limited to 10 updates")
        return
    if not frozen or frozen.get("status") != "FROZEN":
        raise ValueError("formal training requires a frozen compute protocol")
    if frozen.get("identity") != content_hash({k: v for k, v in frozen.items() if k != "identity"}):
        raise ValueError("compute protocol identity mismatch")
    settings = frozen.get("policies", {}).get(recipe.get("policy"), {})
    if (
        recipe["steps"] != frozen["optimizer_updates"]
        or recipe["batch_size"] * recipe["accumulation"] != frozen["effective_batch"]
        or recipe["seed"] not in frozen["seeds"]
        or recipe.get("protocol_identity") != frozen.get("data_protocol_identity")
        or recipe.get("gpu_memory_gib") != frozen.get("gpu_memory_gib")
        or recipe.get("device") != "cuda"
        or recipe.get("condition") not in settings.get("conditions", [])
        or any(
            recipe.get(key) != settings.get(key)
            for key in ["precision", "batch_size", "accumulation"]
        )
    ):
        raise ValueError("recipe violates matched compute budget")
    return frozen["identity"]
