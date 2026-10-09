"""Strict, auditable data contracts shared by research and product components."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class FailureMechanism(str, Enum):
    UNKNOWN = "unknown"
    OBSERVATION = "observation"
    TIMING = "timing"
    CALIBRATION = "calibration"
    PERCEPTION = "perception"
    GEOMETRY = "geometry"
    CONTACT = "contact"
    SKILL = "skill"


class RepairKind(str, Enum):
    NO_OP = "no_op"
    RUNTIME_FIX = "runtime_fix"
    CORRECTIVE_DATA = "corrective_data"
    POLICY_UPDATE = "policy_update"
    HUMAN_ESCALATION = "human_escalation"


class InterventionFactor(str, Enum):
    ACTION_HORIZON = "action_horizon"
    ACTION_SCALE = "action_scale"
    CAMERA_EXTRINSICS = "camera_extrinsics"
    CAMERA_INTRINSICS = "camera_intrinsics"
    LIGHTING = "lighting"
    OBJECT_POSE = "object_pose"
    ROBOT_INITIAL_POSE = "robot_initial_pose"
    FRICTION = "friction"
    MASS = "mass"


class FailureCase(StrictModel):
    case_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    source_episode_id: str = Field(min_length=1)
    split_group_id: str = Field(min_length=1)
    mechanism: FailureMechanism = FailureMechanism.UNKNOWN
    first_failure_step: int | None = Field(default=None, ge=0)
    snapshot_ref: str | None = None
    environment_version: str = Field(min_length=1)
    evidence_uris: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class InterventionSpec(StrictModel):
    intervention_id: str = Field(min_length=1)
    factor: InterventionFactor
    value: str = Field(min_length=1)
    kind: Literal["task_preserving", "task_changing"]
    task_goal_preserved: bool
    expert_action_should_match: bool | None = None
    environment_version: str = Field(min_length=1)
    snapshot_ref: str = Field(min_length=1)
    notes: str = ""

    @model_validator(mode="after")
    def check_semantics(self) -> InterventionSpec:
        if self.kind == "task_preserving" and not self.task_goal_preserved:
            raise ValueError("Task-preserving interventions must preserve the task goal.")
        if self.kind == "task_changing" and self.expert_action_should_match is True:
            raise ValueError("Changed tasks cannot assume identical expert actions.")
        return self


class CorrectionRecord(StrictModel):
    correction_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    source_episode_id: str = Field(min_length=1)
    source_checkpoint: str = Field(min_length=1)
    correction_episode_id: str = Field(min_length=1)
    split_group_id: str = Field(min_length=1)
    supervisor: str = Field(min_length=1)
    replay_verified: bool
    outcome_verified: bool
    outcome_success: bool
    label: Literal["positive", "negative"]
    parent_episode_ids: list[str] = Field(min_length=1)
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    expert_cost_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_positive(self) -> CorrectionRecord:
        if self.source_episode_id not in self.parent_episode_ids:
            raise ValueError("Parent lineage must include source episode.")
        if self.label == "positive" and not (
            self.replay_verified and self.outcome_verified and self.outcome_success
        ):
            raise ValueError("Positive corrections require verified replay and success.")
        return self


class RepairCase(StrictModel):
    failure: FailureCase
    interventions: list[InterventionSpec] = Field(default_factory=list)
    corrections: list[CorrectionRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def cross_check(self) -> RepairCase:
        if len({item.intervention_id for item in self.interventions}) != len(self.interventions):
            raise ValueError("Duplicate intervention IDs.")
        if len({item.correction_id for item in self.corrections}) != len(self.corrections):
            raise ValueError("Duplicate correction IDs.")
        for item in self.corrections:
            if item.case_id != self.failure.case_id:
                raise ValueError("Correction belongs to a different case.")
            if item.split_group_id != self.failure.split_group_id:
                raise ValueError("Correction must inherit its source split group.")
        for item in self.interventions:
            if item.environment_version != self.failure.environment_version:
                raise ValueError("Intervention environment version mismatch.")
        return self


class PairOutcome(StrictModel):
    scenario_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    seed: int
    baseline_success: bool
    candidate_success: bool
    baseline_unsafe: bool = False
    candidate_unsafe: bool = False
    split: Literal["development", "locked_test"]
    source_episode_id: str = Field(min_length=1)


class EvaluationSummary(StrictModel):
    n: int = Field(ge=1)
    baseline_success_rate: float = Field(ge=0, le=1)
    candidate_success_rate: float = Field(ge=0, le=1)
    improvement: float = Field(ge=-1, le=1)
    ci_lower: float = Field(ge=-1, le=1)
    ci_upper: float = Field(ge=-1, le=1)
    baseline_unsafe: int = Field(ge=0)
    candidate_unsafe: int = Field(ge=0)
    task_count: int = Field(ge=1)
    split: Literal["development", "locked_test"]
