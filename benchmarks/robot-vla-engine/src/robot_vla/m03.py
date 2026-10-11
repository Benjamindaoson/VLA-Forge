"""Evidence-preserving helpers for matched M03 LIBERO interventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StateError:
    qpos_linf: float
    qvel_linf: float
    time_abs: float
    exact: bool
    within_tolerance: bool


def compare_sim_state(
    reference: dict[str, Any], candidate: dict[str, Any], *, atol: float = 1e-8
) -> StateError:
    """Compare physical simulator state; pixel similarity is deliberately excluded."""
    if atol < 0:
        raise ValueError("atol must be non-negative")
    errors = {}
    for key in ("qpos", "qvel"):
        left = np.asarray(reference[key], dtype=np.float64)
        right = np.asarray(candidate[key], dtype=np.float64)
        if left.shape != right.shape:
            raise ValueError(f"simulator {key} shape differs: {left.shape} != {right.shape}")
        errors[key] = float(np.max(np.abs(left - right), initial=0.0))
    time_abs = abs(float(reference["time"]) - float(candidate["time"]))
    exact = errors["qpos"] == errors["qvel"] == time_abs == 0.0
    return StateError(
        errors["qpos"],
        errors["qvel"],
        time_abs,
        exact,
        errors["qpos"] <= atol and errors["qvel"] <= atol and time_abs <= atol,
    )


def transform_observation(
    observation: dict[str, Any], mode: str, *, camera: str = "image2"
) -> dict[str, Any]:
    """Copy observations and apply a single deterministic visual treatment.

    ``sham`` is an exact no-op. ``brightness`` applies a fixed 1.15 gain to both
    cameras. ``occlusion`` masks the central 12.5% square of one named camera.
    This helper never edits the caller's observation in place.
    """
    if mode not in {"baseline", "sham", "brightness", "occlusion"}:
        raise ValueError(f"unknown intervention mode: {mode}")
    pixels = observation.get("pixels")
    if not isinstance(pixels, dict):
        raise ValueError("observation has no pixels mapping")
    result = {k: v for k, v in observation.items()}
    result["pixels"] = dict(pixels)
    for key, value in pixels.items():
        result["pixels"][key] = (
            np.array(value, copy=True) if isinstance(value, np.ndarray) else value
        )
    if mode in {"baseline", "sham"}:
        return result
    if mode == "brightness":
        for key in ("image", "image2"):
            frame = np.asarray(result["pixels"][key])
            result["pixels"][key] = np.clip(frame.astype(np.float32) * 1.15, 0, 255).astype(
                np.uint8
            )
    else:
        frame = np.asarray(result["pixels"][camera])
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError("expected HxWx3 RGB image")
        h, w = frame.shape[:2]
        y0, y1 = int(h * 7 / 16), int(h * 9 / 16)
        x0, x1 = int(w * 7 / 16), int(w * 9 / 16)
        frame[y0:y1, x0:x1] = 0
        result["pixels"][camera] = frame
    return result


def action_delta(baseline: np.ndarray, treatment: np.ndarray) -> dict[str, float]:
    left, right = np.asarray(baseline, dtype=np.float64), np.asarray(treatment, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("paired action arrays must have equal shape")
    delta = np.abs(left - right)
    return {
        "mean_abs": float(delta.mean()) if delta.size else 0.0,
        "linf": float(delta.max(initial=0.0)),
    }
