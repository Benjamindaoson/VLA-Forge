"""Predeclared paired M06 robot-policy statistics, independent of score tuning."""
from __future__ import annotations

import math

import numpy as np


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict:
    if total < 1 or not 0 <= successes <= total:
        raise ValueError('Invalid binomial count')
    p = successes / total
    z2 = z * z
    den = 1 + z2 / total
    center = (p + z2 / (2 * total)) / den
    radius = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / den
    return {'success': successes, 'total': total, 'rate': p,
            'lower_95': max(0.0, center - radius), 'upper_95': min(1.0, center + radius)}


def exact_one_sided_p(improved: int, worsened: int) -> float:
    """P(X>=improved), X~Binomial(improved+worsened, 0.5)."""
    if improved < 0 or worsened < 0:
        raise ValueError('Negative discordance')
    n = improved + worsened
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(improved, n + 1)) / (2**n)


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=lambda key: pvalues[key])
    adjusted = {}
    running = 0.0
    n = len(ordered)
    for rank, key in enumerate(ordered):
        raw = pvalues[key]
        if not 0 <= raw <= 1:
            raise ValueError('Invalid p-value')
        running = max(running, min(1.0, (n - rank) * raw))
        adjusted[key] = running
    return adjusted


def paired_target(base: list[bool], candidate: list[bool], *, seed: int = 606,
                  replicates: int = 10_000) -> dict:
    if len(base) != 10 or len(candidate) != 10:
        raise ValueError('M06 target requires ten matched frozen states')
    a = np.asarray(base, dtype=np.int8)
    b = np.asarray(candidate, dtype=np.int8)
    delta = b - a
    improved = int(np.count_nonzero(delta == 1))
    worsened = int(np.count_nonzero(delta == -1))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, 10, size=(replicates, 10))
    boots = delta[draws].mean(axis=1)
    return {'frozen_success': int(a.sum()), 'candidate_success': int(b.sum()),
            'paired_difference': float(delta.mean()), 'improved': improved,
            'worsened': worsened, 'concordant': 10 - improved - worsened,
            'one_sided_exact_p': exact_one_sided_p(improved, worsened),
            'paired_bootstrap_95': [float(np.quantile(boots, q)) for q in (0.025, 0.975)],
            'independent_scenarios': 10}


def task_cluster_difference(task_differences: list[float], *, seed: int = 606,
                            replicates: int = 10_000) -> dict:
    if len(task_differences) != 39:
        raise ValueError('M06 legacy comparison requires 39 task clusters')
    values = np.asarray(task_differences, dtype=np.float64)
    if not np.isfinite(values).all() or np.max(np.abs(values)) > 1:
        raise ValueError('Invalid task difference')
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, 39, size=(replicates, 39))
    boots = values[draws].mean(axis=1)
    return {'paired_difference': float(values.mean()), 'clusters': 39,
            'two_sided_95': [float(np.quantile(boots, q)) for q in (0.025, 0.975)],
            'one_sided_lower_95': float(np.quantile(boots, 0.05)),
            'bootstrap_replicates': replicates, 'seed': seed}


def release_decision(*, target: dict, legacy: dict, holm_p: float,
                     candidate_chunk_p95_ms: float, frozen_chunk_p95_ms: float,
                     illegal_actions: int, missing_or_corrupt: int,
                     margin: float = -0.03) -> dict:
    """Apply the immutable protocol's conjunction; uncertainty never passes."""
    if illegal_actions > 0 or missing_or_corrupt > 0:
        return {'decision': 'REJECT', 'reason': 'safety_or_integrity_failure'}
    if frozen_chunk_p95_ms <= 0 or candidate_chunk_p95_ms <= 0:
        return {'decision': 'INSUFFICIENT_EVIDENCE', 'reason': 'latency_unmeasured'}
    ratio = candidate_chunk_p95_ms / frozen_chunk_p95_ms
    if ratio > 1.2:
        return {'decision': 'REJECT', 'reason': 'chunk_latency_regression', 'p95_ratio': ratio}
    if target['paired_difference'] < 0 or legacy['two_sided_95'][1] < margin:
        return {'decision': 'REJECT', 'reason': 'observed_target_or_supported_legacy_harm', 'p95_ratio': ratio}
    if target['paired_difference'] <= 0 or holm_p > 0.05 or legacy['one_sided_lower_95'] <= margin:
        return {'decision': 'INSUFFICIENT_EVIDENCE', 'reason': 'benefit_or_noninferiority_unproven',
                'p95_ratio': ratio}
    return {'decision': 'PROMOTE', 'reason': 'preregistered_simulation_gate_passed',
            'deployment_scope': 'authorized simulation only', 'p95_ratio': ratio}
