import pytest
from pydantic import ValidationError

from vla_forge.corrections import approved_positive_corrections, training_manifest
from vla_forge.models import (
    CorrectionRecord, FailureCase, InterventionFactor, InterventionSpec, RepairCase,
)
from vla_forge.splits import LeakageError, assert_no_leakage, assign_split


def failure():
    return FailureCase(
        case_id="case-1", task_id="pick", source_episode_id="ep-1",
        policy_version="base", environment_version="libero-v1",
        split_group_id="scene-family-a", snapshot_ref="snapshot-1",
    )


def correction(**changes):
    payload = dict(
        correction_id="corr-1", case_id="case-1",
        source_episode_id="ep-1", source_checkpoint="base",
        correction_episode_id="ep-corr-1", split_group_id="scene-family-a",
        supervisor="human", replay_verified=True, outcome_verified=True,
        outcome_success=True, label="positive", parent_episode_ids=["ep-1"],
        artifact_sha256="a" * 64, expert_cost_seconds=45,
    )
    payload.update(changes)
    return CorrectionRecord(**payload)


def test_positive_needs_verified_success():
    with pytest.raises(ValidationError, match="Positive corrections"):
        correction(replay_verified=False)
    with pytest.raises(ValidationError, match="Positive corrections"):
        correction(outcome_success=False)


def test_case_rejects_foreign_split_group():
    with pytest.raises(ValidationError, match="source split group"):
        RepairCase(failure=failure(), corrections=[correction(split_group_id="different")])


def test_parent_episode_is_required():
    with pytest.raises(ValidationError, match="Parent lineage"):
        correction(parent_episode_ids=["unrelated"])


def test_only_verified_positive_is_trainable():
    f = failure()
    negative = correction(
        correction_id="neg", label="negative", outcome_success=False,
        replay_verified=False, correction_episode_id="ep-neg",
    )
    records = training_manifest([correction(), negative], {"case-1": f})
    assert len(records) == 1
    assert records[0]["positive_supervision"]
    assert records[0]["split"] == assign_split(f.split_group_id)


def test_wrong_source_checkpoint_rejected():
    with pytest.raises(ValueError, match="checkpoint"):
        approved_positive_corrections(
            [correction(source_checkpoint="other")], {"case-1": failure()}
        )


def test_deterministic_family_split():
    assert assign_split("family", salt="a") == assign_split("family", salt="a")


def test_parent_leakage_guard():
    rows = [
        {
            "split_group_id": "a", "split": "train",
            "source_episode_id": "shared", "parent_episode_ids": ["root"],
        },
        {
            "split_group_id": "b", "split": "test",
            "source_episode_id": "shared", "parent_episode_ids": ["root"],
        },
    ]
    with pytest.raises(LeakageError):
        assert_no_leakage(rows)


def test_task_changing_expert_cannot_be_assumed_invariant():
    with pytest.raises(ValidationError, match="identical expert actions"):
        InterventionSpec(
            intervention_id="i", factor=InterventionFactor.OBJECT_POSE,
            value="5cm", kind="task_changing", task_goal_preserved=False,
            expert_action_should_match=True, environment_version="libero-v1",
            snapshot_ref="snapshot-1",
        )


def test_case_rejects_different_simulator_version():
    intervention = InterventionSpec(
        intervention_id="i", factor=InterventionFactor.CAMERA_INTRINSICS,
        value="zoom", kind="task_preserving", task_goal_preserved=True,
        environment_version="libero-v2", snapshot_ref="snapshot-1",
    )
    with pytest.raises(ValidationError, match="environment version mismatch"):
        RepairCase(failure=failure(), interventions=[intervention])


def test_complete_case_valid():
    result = RepairCase(failure=failure(), corrections=[correction()])
    assert result.corrections[0].correction_id == "corr-1"
