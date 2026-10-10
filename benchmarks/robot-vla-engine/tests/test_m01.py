import pytest

from robot_vla.m01 import (
    M01_SUITES,
    episode_success_metrics,
    fit_libero_state_to_checkpoint,
    validate_m01_coverage,
    validate_smoke_coverage,
)


def test_m01_suites_bind_long_to_official_libero_10_api():
    assert M01_SUITES == {
        "spatial": ("libero_spatial", 0),
        "object": ("libero_object", 0),
        "goal": ("libero_goal", 0),
        "long": ("libero_10", 0),
    }


def test_episode_metrics_keep_failures_in_denominator():
    rows = [
        {"status": "complete", "success": True},
        {"status": "complete", "success": False},
    ]
    assert episode_success_metrics(rows) == {
        "completed_episodes": 2,
        "successes": 1,
        "failures": 1,
        "success_rate": 0.5,
    }


def test_m01_coverage_rejects_missing_or_duplicate_formal_episodes():
    rows = [
        {"phase": "formal", "suite": suite, "task_id": 0, "episode_id": str(i), "status": "complete", "success": i == 0}
        for suite in M01_SUITES
        for i in range(10)
    ]
    assert validate_m01_coverage(rows) == {suite: 10 for suite in M01_SUITES}
    with pytest.raises(ValueError, match="coverage"):
        validate_m01_coverage(rows[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        validate_m01_coverage(rows + [rows[0]])


def test_m01_coverage_rejects_runtime_errors_as_completed_episodes():
    rows = [
        {"phase": "formal", "suite": suite, "task_id": 0, "episode_id": str(i), "status": "complete", "success": i == 0}
        for suite in M01_SUITES
        for i in range(10)
    ]
    rows[0] = {**rows[0], "status": "error", "success": None}
    with pytest.raises(ValueError, match="coverage"):
        validate_m01_coverage(rows)


def test_smoke_gate_requires_three_completed_spatial_rollouts():
    rows = [
        {"phase": "smoke", "suite": "spatial", "task_id": 0, "episode_id": str(i), "status": "complete", "success": i == 0}
        for i in range(3)
    ]
    assert validate_smoke_coverage(rows) == 3
    with pytest.raises(ValueError, match="incomplete"):
        validate_smoke_coverage([{**rows[0], "status": "error", "success": None}, *rows[1:]])


def test_official_checkpoint_state_uses_observed_normalizer_width():
    import torch

    state = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    adapted, semantics = fit_libero_state_to_checkpoint(
        state, expected_dim=6, normalizer_dim=8
    )
    assert adapted.shape == (1, 8)
    assert adapted.tolist() == [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]
    assert semantics == (
        "full LIBERO pose and gripper state; official normalizer width 8; "
        "policy feature schema width 6"
    )
    with pytest.raises(ValueError, match="unsupported"):
        fit_libero_state_to_checkpoint(state, expected_dim=6, normalizer_dim=7)


def test_reproduction_clears_foreign_cudnn_search_path():
    from pathlib import Path

    script = Path(__file__).parents[1] / "reports/vla_forge_2_transition_20261010/milestones/M01_real_libero/reproduce.sh"
    contents = script.read_text(encoding="utf-8")
    assert "unset LD_LIBRARY_PATH" in contents
