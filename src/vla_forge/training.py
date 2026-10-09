"""Flow-matching compatible, per-example correction weighting.

This loss helper is not an integrated SmolVLA trainer. An actual training job
must supply model-predicted velocity, target velocity and validated examples
from the pinned LeRobot/SmolVLA training implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def weighted_flow_matching_loss(
    predicted_velocity: Tensor,
    target_velocity: Tensor,
    weights: Tensor,
    *,
    valid_action_mask: Tensor | None = None,
) -> Tensor:
    """Normalize per-sample FM squared error and use detached nonnegative weights.

    predicted_velocity and target_velocity: [batch, time, action_dim].
    weights: [batch] with nonnegative finite values.
    valid_action_mask: optional [batch, time] boolean indicating actual labels.
    """
    import torch

    if predicted_velocity.shape != target_velocity.shape or predicted_velocity.ndim != 3:
        raise ValueError("Velocity tensors must match with shape [B, T, D].")
    batch, steps, _ = predicted_velocity.shape
    if weights.shape != (batch,) or not torch.isfinite(weights).all():
        raise ValueError("Weights must be finite and have shape [B].")
    if bool((weights < 0).any()) or not bool((weights > 0).any()):
        raise ValueError("Weights must be nonnegative and not all zero.")
    if valid_action_mask is None:
        mask = torch.ones((batch, steps), device=predicted_velocity.device)
    else:
        if valid_action_mask.shape != (batch, steps):
            raise ValueError("Invalid action mask shape.")
        mask = valid_action_mask.to(device=predicted_velocity.device, dtype=torch.float32)
    if bool((mask.sum(dim=1) == 0).any()):
        raise ValueError("Every example must contain at least one valid target step.")
    if not torch.isfinite(predicted_velocity).all() or not torch.isfinite(target_velocity).all():
        raise ValueError("Non-finite model or target velocities.")
    errors = (predicted_velocity - target_velocity).square().mean(dim=-1)
    per_sample = (errors * mask).sum(dim=1) / mask.sum(dim=1)
    w = weights.detach().to(device=predicted_velocity.device, dtype=per_sample.dtype)
    return (per_sample * w).sum() / w.sum()
