"""Simulator-neutral, paired diagnostic interventions.

Adapters must implement the protocol against a *real* simulator. The runner does
not assume that equality of snapshots implies bit-exact contact dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from vla_forge.models import InterventionSpec


@dataclass(frozen=True)
class RolloutResult:
    succeeded: bool
    steps: int
    unsafe: bool = False
    evidence_uri: str | None = None

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError("Rollout steps cannot be negative.")


class SimulatorAdapter(Protocol):
    def restore(self, snapshot_ref: str) -> None: ...
    def state_fingerprint(self) -> str: ...
    def apply_intervention(self, intervention: InterventionSpec) -> None: ...
    def rollout(self, *, checkpoint: str, seed: int) -> RolloutResult: ...


@dataclass(frozen=True)
class InterventionComparison:
    snapshot_ref: str
    state_fingerprint: str
    intervention_id: str
    checkpoint: str
    seed: int
    baseline: RolloutResult
    intervened: RolloutResult

    @property
    def success_delta(self) -> int:
        return int(self.intervened.succeeded) - int(self.baseline.succeeded)


def run_paired_intervention(
    adapter: SimulatorAdapter,
    *,
    snapshot_ref: str,
    checkpoint: str,
    seed: int,
    intervention: InterventionSpec,
) -> InterventionComparison:
    """Compare with repeated restoration; do not infer a unique causal root.

    A source-specific adapter must validate task semantics and correct action
    targets for a task-changing intervention before exposing it here.
    """
    if snapshot_ref != intervention.snapshot_ref:
        raise ValueError("Intervention snapshot does not match paired rollout.")
    if not checkpoint.strip():
        raise ValueError("Checkpoint must be nonempty.")
    adapter.restore(snapshot_ref)
    source_fingerprint = adapter.state_fingerprint()
    if not source_fingerprint:
        raise ValueError("Simulator did not provide a state fingerprint.")
    baseline = adapter.rollout(checkpoint=checkpoint, seed=seed)
    adapter.restore(snapshot_ref)
    if adapter.state_fingerprint() != source_fingerprint:
        raise RuntimeError("Replay mismatch: restored physical state fingerprint differs.")
    adapter.apply_intervention(intervention)
    intervened = adapter.rollout(checkpoint=checkpoint, seed=seed)
    return InterventionComparison(
        snapshot_ref=snapshot_ref,
        state_fingerprint=source_fingerprint,
        intervention_id=intervention.intervention_id,
        checkpoint=checkpoint,
        seed=seed,
        baseline=baseline,
        intervened=intervened,
    )
