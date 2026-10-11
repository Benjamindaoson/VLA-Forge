"""Pure, auditable transforms used by M04 physical intervention rollouts."""

from __future__ import annotations

import numpy as np


def scale_environment_action(action: np.ndarray, scale: float) -> np.ndarray:
    """Scale the post-processed policy command, then enforce LIBERO's [-1, 1] bounds."""
    value = float(scale)
    if not np.isfinite(value) or not 0.5 <= value <= 2.0:
        raise ValueError("action scale must be finite and within [0.5, 2.0]")
    command = np.asarray(action, dtype=np.float32)
    if command.ndim != 1 or not np.isfinite(command).all():
        raise ValueError("action must be a finite one-dimensional vector")
    return np.clip(command * value, -1.0, 1.0).astype(np.float32, copy=False)


def brightness_camera_keys(camera: str) -> tuple[str, ...]:
    """Return model-batch image keys targeted by an observation intervention."""
    cameras = {
        "both": ("observation.images.image", "observation.images.image2"),
        "agentview": ("observation.images.image",),
        "wrist": ("observation.images.image2",),
    }
    try:
        return cameras[camera]
    except KeyError as exc:
        raise ValueError(f"unsupported camera intervention: {camera}") from exc


def intervention_is_active(step: int, start: int, duration: int) -> bool:
    """Return whether an observation-time intervention is active at a policy call."""
    if start < 0 or duration < 0:
        raise ValueError("intervention start and duration must be non-negative")
    return start <= step < start + duration


def benchmark_suite_name(suite: str) -> str:
    """Map M02 suite labels to the pinned LIBERO benchmark registry names."""
    names = {
        "spatial": "libero_spatial",
        "object": "libero_object",
        "goal": "libero_goal",
        "long": "libero_10",
    }
    try:
        return names[suite]
    except KeyError as exc:
        raise ValueError(f"unsupported LIBERO suite: {suite}") from exc


def m02_source_episode_name(suite: str, task_id: int, episode_id: int) -> str:
    """Return the frozen M02 registry directory key for a source episode."""
    benchmark_suite_name(suite)
    if not 0 <= task_id < 10 or not 0 <= episode_id < 10:
        raise ValueError("LIBERO task and episode ids must be in [0, 9]")
    return f"{suite}_task{task_id:02d}_episode{episode_id:02d}_attempt1"
