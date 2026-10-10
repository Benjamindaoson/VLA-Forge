"""Absolute allocator ceilings; whole-device consumption is measured separately."""

import math


def allocator_fraction(limit_gib, physical_bytes):
    if not math.isfinite(limit_gib) or limit_gib <= 0 or physical_bytes <= 0:
        raise ValueError("invalid memory budget")
    budget = limit_gib * 2**30
    if budget > physical_bytes:
        raise ValueError("memory budget exceeds physical device")
    return budget / physical_bytes


def configure_gpu_budget(limit_gib, device=0):
    import torch

    properties = torch.cuda.get_device_properties(device)
    fraction = allocator_fraction(limit_gib, properties.total_memory)
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    return dict(
        device_name=properties.name,
        physical_bytes=properties.total_memory,
        allocator_limit_bytes=int(limit_gib * 2**30),
        allocator_fraction=fraction,
        scope="PyTorch caching allocator only; CUDA context and renderer require separate device monitoring",
    )
