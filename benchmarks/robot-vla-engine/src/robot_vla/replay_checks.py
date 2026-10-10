"""Strict comparison for exploratory replay qualification."""

import numbers
from contextlib import contextmanager

import numpy as np


def compare_arrays(reference, candidate, *, atol):
    if isinstance(atol, (bool, np.bool_)) or not isinstance(atol, numbers.Real):
        raise ValueError("tolerance must be a finite nonnegative number")
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("tolerance must be a finite nonnegative number")
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.size == 0 or a.shape != b.shape:
        raise ValueError("paired arrays must be nonempty with identical shapes")
    if a.dtype.kind not in "buif" or b.dtype.kind not in "buif":
        raise ValueError("paired arrays must be real numeric values")
    a, b = a.astype(np.float64), b.astype(np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired arrays must be finite")
    error = float(np.max(np.abs(a - b)))
    return dict(shape=list(a.shape), max_abs_error=error, within_tolerance=error <= atol)


@contextmanager
def isolated_chunk_profile(policy, synchronize=None):
    from .instrumentation import ChunkProfiler

    method = "_get_action_chunk" if hasattr(policy, "_get_action_chunk") else "predict_action_chunk"
    original = getattr(policy, method)
    try:
        yield ChunkProfiler(policy, synchronize)
    finally:
        setattr(policy, method, original)
