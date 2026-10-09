"""Paired strategy evaluation and conservative locked-test release gating."""

from __future__ import annotations

import math
import random
from collections import defaultdict

from pydantic import Field

from vla_forge.models import EvaluationSummary, PairOutcome, StrictModel


def _percentile(sorted_values: list[float], fraction: float) -> float:
    k = (len(sorted_values) - 1) * fraction
    lower = math.floor(k)
    upper = math.ceil(k)
    return sorted_values[lower] + (k - lower) * (
        sorted_values[upper] - sorted_values[lower]
    )


def summarize_pairs(
    outcomes: list[PairOutcome],
    *,
    n_bootstrap: int = 2000,
    seed: int = 1729,
    alpha: float = 0.05,
) -> EvaluationSummary:
    """Two-stage task-cluster bootstrap of paired success differences.

    Rows must represent *verified matching initial states*, not merely matching
    seeds. The LIBERO batch/reset scheduler can otherwise destroy pairing.
    """
    if not outcomes:
        raise ValueError("Evaluation needs paired rollouts.")
    if n_bootstrap < 200:
        raise ValueError("Use at least 200 bootstrap resamples.")
    if not 0 < alpha < 1:
        raise ValueError("Invalid confidence level.")
    splits = {r.split for r in outcomes}
    if len(splits) != 1:
        raise ValueError("Development and locked-test observations cannot be combined.")
    keys = [(r.task_id, r.scenario_id, r.seed) for r in outcomes]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate paired scenario/seed identity.")
    tasks: dict[str, list[PairOutcome]] = defaultdict(list)
    for r in outcomes:
        tasks[r.task_id].append(r)
    task_ids = sorted(tasks)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(n_bootstrap):
        sampled: list[PairOutcome] = []
        for _ in task_ids:
            task = rng.choice(task_ids)
            family = tasks[task]
            sampled.extend(rng.choice(family) for _ in family)
        estimates.append(
            sum(int(r.candidate_success) - int(r.baseline_success) for r in sampled)
            / len(sampled)
        )
    estimates.sort()
    n = len(outcomes)
    baseline = sum(r.baseline_success for r in outcomes) / n
    candidate = sum(r.candidate_success for r in outcomes) / n
    return EvaluationSummary(
        n=n,
        baseline_success_rate=baseline,
        candidate_success_rate=candidate,
        improvement=candidate - baseline,
        ci_lower=max(-1.0, _percentile(estimates, alpha / 2)),
        ci_upper=min(1.0, _percentile(estimates, 1 - alpha / 2)),
        baseline_unsafe=sum(r.baseline_unsafe for r in outcomes),
        candidate_unsafe=sum(r.candidate_unsafe for r in outcomes),
        task_count=len(tasks),
        split=next(iter(splits)),
    )


class ReleaseThresholds(StrictModel):
    min_pairs: int = Field(default=100, ge=1)
    min_tasks: int = Field(default=4, ge=1)
    minimum_target_improvement: float = Field(default=0.0, ge=0, le=1)
    retention_noninferiority_margin: float = Field(default=0.03, ge=0, le=1)
    forbid_increased_unsafe_events: bool = True
    max_p95_latency_ms: float = Field(default=200.0, gt=0)


class ReleaseDecision(StrictModel):
    approved: bool
    reasons: list[str]
    target_improvement: float
    retention_improvement: float
    p95_latency_ms: float


def evaluate_release(
    *,
    target: EvaluationSummary,
    retention: EvaluationSummary,
    p95_latency_ms: float,
    thresholds: ReleaseThresholds | None = None,
) -> ReleaseDecision:
    """Fail closed; an inconclusive noninferiority interval does not pass."""
    t = thresholds or ReleaseThresholds()
    failures: list[str] = []
    for label, summary in (("target", target), ("retention", retention)):
        if summary.split != "locked_test":
            failures.append(label + " evaluation is not locked_test.")
        if summary.n < t.min_pairs or summary.task_count < t.min_tasks:
            failures.append(label + " lacks required sample/task coverage.")
    if target.ci_lower <= t.minimum_target_improvement:
        failures.append("Target improvement lower confidence bound fails threshold.")
    if retention.ci_lower <= -t.retention_noninferiority_margin:
        failures.append("Skill retention noninferiority is not established.")
    if t.forbid_increased_unsafe_events and (
        target.candidate_unsafe > target.baseline_unsafe
        or retention.candidate_unsafe > retention.baseline_unsafe
    ):
        failures.append("Candidate increases observed unsafe events.")
    if not math.isfinite(p95_latency_ms) or p95_latency_ms < 0:
        failures.append("Invalid latency measurement.")
    elif p95_latency_ms > t.max_p95_latency_ms:
        failures.append("P95 latency exceeds deployment budget.")
    return ReleaseDecision(
        approved=not failures,
        reasons=failures or ["All configured evidence gates passed."],
        target_improvement=target.improvement,
        retention_improvement=retention.improvement,
        p95_latency_ms=p95_latency_ms,
    )
