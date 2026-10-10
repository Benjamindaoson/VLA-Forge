"""Explicit oracle task IDs for non-language closed-set baseline policies."""

import torch


def task_condition(batch, protocol):
    mapping = {name: i for i, name in enumerate(sorted(protocol["task_mapping"]))}
    names = batch["task"]
    if any(name not in mapping for name in names):
        raise ValueError("unknown task condition")
    state = batch["observation.state"]
    if state.shape[-1] != 8 or len(names) != state.shape[0]:
        raise ValueError("task conditioning requires batched 8D source state")
    ids = torch.tensor([mapping[name] for name in names], device=state.device)
    goals = torch.nn.functional.one_hot(ids, num_classes=len(mapping)).to(state.dtype)
    while goals.ndim < state.ndim:
        goals = goals.unsqueeze(1)
    goals = goals.expand(*state.shape[:-1], len(mapping))
    return {**batch, "observation.state": torch.cat([state, goals], dim=-1)}
