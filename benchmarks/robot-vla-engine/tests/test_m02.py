import pytest

from robot_vla.m02 import (
    M02_SUITES,
    artifact_is_valid,
    classify_failure,
    count_verified_video_frames,
    episode_seed,
    expected_episode_identities,
    select_manual_review,
    validate_m02_coverage,
    wilson_interval,
)


def _complete_rows():
    return [
        {
            "suite": suite,
            "task_id": task_id,
            "episode_id": episode_id,
            "init_state_id": episode_id,
            "seed": 100_000 + suite_index * 1_000 + task_id * 10 + episode_id,
            "status": "complete",
            "success": (task_id + episode_id) % 2 == 0,
        }
        for suite_index, suite in enumerate(M02_SUITES)
        for task_id in range(10)
        for episode_id in range(10)
    ]


def test_expected_matrix_is_exactly_40_tasks_and_400_episodes():
    identities = expected_episode_identities()
    assert len(identities) == 400
    assert len({(suite, task) for suite, task, _ in identities}) == 40
    assert {suite for suite, _, _ in identities} == set(M02_SUITES)
    seeds = {
        episode_seed(82_000, suite, task_id, episode_id)
        for suite, task_id, episode_id in identities
    }
    assert len(seeds) == 400
    assert min(seeds) == 82_000
    assert max(seeds) == 85_099


def test_m02_coverage_requires_all_unique_completed_episode_rows():
    rows = _complete_rows()
    assert validate_m02_coverage(rows) == {suite: 100 for suite in M02_SUITES}
    with pytest.raises(ValueError, match="coverage"):
        validate_m02_coverage(rows[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        validate_m02_coverage(rows + [rows[0]])


def test_m02_coverage_rejects_runtime_errors_and_missing_success_signals():
    rows = _complete_rows()
    rows[0] = {**rows[0], "status": "error", "success": None}
    with pytest.raises(ValueError, match="incomplete"):
        validate_m02_coverage(rows)
    rows = _complete_rows()
    rows[0] = {**rows[0], "success": None}
    with pytest.raises(ValueError, match="boolean"):
        validate_m02_coverage(rows)


def test_wilson_interval_has_sample_size_sensitive_bounds():
    low_n = wilson_interval(5, 10)
    high_n = wilson_interval(50, 100)
    assert low_n[0] == pytest.approx(0.2366, abs=0.001)
    assert low_n[1] == pytest.approx(0.7634, abs=0.001)
    assert high_n[1] - high_n[0] < low_n[1] - low_n[0]
    assert wilson_interval(0, 0) is None


def test_failure_classification_does_not_infer_cause_from_false_success():
    row = {"success": False, "termination_reason": "environment_terminated", "total_action_steps": 80, "horizon": 220}
    assert classify_failure(row)["category"] == "Unknown / Insufficient Evidence"
    timeout = {**row, "termination_reason": "timeout", "total_action_steps": 220}
    assert classify_failure(timeout) == {
        "category": "Timeout",
        "basis": "environment success signal remained false through the frozen horizon",
        "confidence": "high",
    }


def test_manual_failure_review_requires_evidence_and_valid_category():
    row = {"success": False, "termination_reason": "timeout", "total_action_steps": 220, "horizon": 220}
    with pytest.raises(ValueError, match="evidence"):
        classify_failure(row, {"category": "Grasp Failure", "evidence": ""})
    reviewed = classify_failure(
        row,
        {"category": "Grasp Failure", "evidence": "object remains on table after gripper closes", "confidence": "medium"},
    )
    assert reviewed["category"] == "Grasp Failure"
    assert reviewed["basis"] == "human_review"
    assert classify_failure(row, {"category": "Object Drop", "evidence": "visible drop", "failure_step": 17})["failure_step"] == 17
    with pytest.raises(ValueError, match="failure_step"):
        classify_failure(row, {"category": "Object Drop", "evidence": "visible drop", "failure_step": -1})


def test_artifact_resume_requires_matching_hash(tmp_path):
    from robot_vla.schema import file_hash

    artifact = tmp_path / "episode.mp4"
    artifact.write_bytes(b"real test artifact")
    digest = file_hash(artifact)
    assert artifact_is_valid(artifact, digest)
    assert not artifact_is_valid(artifact, "0" * 64)
    assert not artifact_is_valid(tmp_path / "missing.mp4", digest)


def test_manual_review_sample_is_balanced_across_suites_and_spread_within_each():
    rows = [
        {"suite": suite, "task_id": 0, "episode_id": index, "status": "complete", "success": False}
        for suite in M02_SUITES
        for index in range(25)
    ]
    sample = select_manual_review(rows)
    assert len(sample) == 20
    assert {suite: sum(row["suite"] == suite for row in sample) for suite in M02_SUITES} == {suite: 5 for suite in M02_SUITES}
    for suite in M02_SUITES:
        assert [row["episode_id"] for row in sample if row["suite"] == suite] == [0, 6, 12, 18, 24]
    assert len(select_manual_review(rows[:4])) == 4


def test_video_frame_total_reads_verified_nested_video_receipts():
    rows = [
        {"_video_info": {"video_decoded": True, "video": {"frames": 301}}},
        {"_video_info": {"video_decoded": True, "video": {"frames": 152}}},
        {"_video_info": {"video_decoded": False, "video": {}}},
        {},
    ]
    assert count_verified_video_frames(rows) == 453
