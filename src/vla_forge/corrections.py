"""Convert only fully verified corrections into positive supervision manifests."""

from __future__ import annotations

from collections.abc import Iterable

from vla_forge.models import CorrectionRecord, FailureCase
from vla_forge.splits import assert_no_leakage, assign_split


def approved_positive_corrections(
    records: Iterable[CorrectionRecord],
    failures: dict[str, FailureCase],
) -> list[CorrectionRecord]:
    output: list[CorrectionRecord] = []
    seen: set[str] = set()
    for correction in records:
        if correction.correction_id in seen:
            raise ValueError("Duplicate correction ID.")
        seen.add(correction.correction_id)
        failure = failures.get(correction.case_id)
        if failure is None:
            raise ValueError("Correction has no known source failure.")
        if correction.source_episode_id != failure.source_episode_id:
            raise ValueError("Correction lineage does not match source failure.")
        if correction.source_checkpoint != failure.policy_version:
            raise ValueError("Correction checkpoint does not match source failure.")
        if correction.split_group_id != failure.split_group_id:
            raise ValueError("Correction does not inherit source split group.")
        if correction.label == "positive":
            output.append(correction)
    return output


def training_manifest(
    corrections: list[CorrectionRecord],
    failures: dict[str, FailureCase],
    *,
    salt: str = "vla-forge-v1",
) -> list[dict[str, object]]:
    """Build deterministic, traceable records. Never mutates the input episodes."""
    approved = approved_positive_corrections(corrections, failures)
    manifest = [
        {
            "correction_id": c.correction_id,
            "source_episode_id": c.source_episode_id,
            "correction_episode_id": c.correction_episode_id,
            "parent_episode_ids": c.parent_episode_ids,
            "split_group_id": c.split_group_id,
            "split": assign_split(c.split_group_id, salt=salt),
            "artifact_sha256": c.artifact_sha256,
            "expert_cost_seconds": c.expert_cost_seconds,
            "supervisor": c.supervisor,
            "positive_supervision": True,
        }
        for c in approved
    ]
    assert_no_leakage(manifest)
    return manifest
