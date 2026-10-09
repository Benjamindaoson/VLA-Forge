import pytest

from vla_forge.evaluation import ReleaseThresholds, evaluate_release, summarize_pairs
from vla_forge.models import PairOutcome


def pairs(*, target, n=25, tasks=4, split="locked_test"):
    return [
        PairOutcome(
            scenario_id=f"scene-{task}-{i}",
            task_id=f"task-{task}", seed=i,
            source_episode_id=f"source-{task}-{i}",
            baseline_success=not target,
            candidate_success=True,
            split=split,
        )
        for task in range(tasks)
        for i in range(n)
    ]


def test_paired_improvement_and_confidence():
    summary = summarize_pairs(pairs(target=True))
    assert summary.n == 100
    assert summary.improvement == pytest.approx(1)
    assert summary.ci_lower == pytest.approx(1)
    assert summary.task_count == 4


def test_regression_blocks_release():
    target = summarize_pairs(pairs(target=True))
    retention = pairs(target=False)
    for row in retention[:8]:
        row.candidate_success = False
    decision = evaluate_release(
        target=target, retention=summarize_pairs(retention), p95_latency_ms=30,
        thresholds=ReleaseThresholds(min_pairs=100, min_tasks=4),
    )
    assert decision.approved is False
    assert any("noninferiority" in item for item in decision.reasons)


def test_clear_improvement_and_retention_can_pass():
    result = evaluate_release(
        target=summarize_pairs(pairs(target=True)),
        retention=summarize_pairs(pairs(target=False)),
        p95_latency_ms=30,
    )
    assert result.approved is True


def test_release_rejects_development_data():
    result = evaluate_release(
        target=summarize_pairs(pairs(target=True, split="development")),
        retention=summarize_pairs(pairs(target=False, split="development")),
        p95_latency_ms=30,
    )
    assert result.approved is False


def test_release_rejects_unsafe_events_and_latency():
    target = pairs(target=True)
    target[0].candidate_unsafe = True
    result = evaluate_release(
        target=summarize_pairs(target),
        retention=summarize_pairs(pairs(target=False)),
        p95_latency_ms=1000,
    )
    assert result.approved is False
    assert any("unsafe" in x for x in result.reasons)
    assert any("latency" in x for x in result.reasons)


def test_duplicate_pair_identity_is_invalid():
    rows = pairs(target=True, n=1, tasks=1)
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_pairs(rows + rows)


def test_mixed_splits_are_invalid():
    x = pairs(target=True, n=1, tasks=1)
    other = PairOutcome(
        scenario_id="x", task_id="task-other", seed=1, source_episode_id="episode-x",
        baseline_success=False, candidate_success=True, split="development",
    )
    with pytest.raises(ValueError, match="Development"):
        summarize_pairs(x + [other])


def test_insufficient_sample_size_blocks_release():
    outcome = evaluate_release(
        target=summarize_pairs(pairs(target=True, n=2, tasks=2)),
        retention=summarize_pairs(pairs(target=False, n=2, tasks=2)),
        p95_latency_ms=2,
    )
    assert outcome.approved is False
