"""Budgeted repair planning with explicit uncertainty and diagnostic information value.

This is a *decision baseline*, not a trained causal model. Utility estimates are
inputs supplied by validated experiments or human experts; no outcomes are invented.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, field_validator, model_validator

from vla_forge.models import RepairKind, StrictModel


class FailureBelief(StrictModel):
    probabilities: dict[str, float] = Field(min_length=1)

    @field_validator("probabilities")
    @classmethod
    def valid_probability_distribution(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not math.isfinite(p) or p < 0 or p > 1 for p in value.values()):
            raise ValueError("Probabilities must be finite values in [0, 1].")
        if not math.isclose(sum(value.values()), 1.0, abs_tol=1e-7):
            raise ValueError("Prior/posterior probabilities must sum to 1.")
        return value


class DiagnosticProbe(StrictModel):
    probe_id: str = Field(min_length=1)
    cost: float = Field(ge=0)
    safety_risk: float = Field(ge=0, le=1)
    likelihoods: dict[str, dict[str, float]] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_likelihoods(self) -> DiagnosticProbe:
        outcomes = {tuple(sorted(row)) for row in self.likelihoods.values()}
        if len(outcomes) != 1 or not next(iter(outcomes), ()):
            raise ValueError("Every hypothesis needs the same nonempty outcome set.")
        for row in self.likelihoods.values():
            if any(not math.isfinite(p) or p < 0 or p > 1 for p in row.values()):
                raise ValueError("Invalid likelihood.")
            if not math.isclose(sum(row.values()), 1.0, abs_tol=1e-7):
                raise ValueError("Conditional outcomes must sum to 1.")
        return self


class RepairOption(StrictModel):
    option_id: str = Field(min_length=1)
    kind: RepairKind
    cost: float = Field(ge=0)
    safety_risk: float = Field(ge=0, le=1)
    regression_risk: float = Field(ge=0, le=1)
    estimated_gain_by_hypothesis: dict[str, float] = Field(min_length=1)

    @field_validator("estimated_gain_by_hypothesis")
    @classmethod
    def gains_in_range(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not math.isfinite(p) or p < -1 or p > 1 for p in value.values()):
            raise ValueError("Estimated success-rate gains must be in [-1, 1].")
        return value


class PlannerSettings(StrictModel):
    budget: float = Field(ge=0)
    cost_weight: float = Field(default=0.01, ge=0)
    safety_weight: float = Field(default=1.0, ge=0)
    regression_weight: float = Field(default=0.5, ge=0)
    max_safety_risk: float = Field(default=0.05, ge=0, le=1)
    min_probe_voi: float = Field(default=0.001, ge=0)


class Recommendation(StrictModel):
    mode: Literal["diagnose", "repair", "escalate"]
    choice_id: str | None
    expected_utility: float | None
    net_information_value: float | None = None
    reason: str


class DiagnosticResult(StrictModel):
    probe_id: str
    outcome: str
    prior: FailureBelief
    posterior: FailureBelief
    outcome_probability: float


def posterior_from_probe(
    prior: FailureBelief, probe: DiagnosticProbe, outcome: str
) -> DiagnosticResult:
    if set(prior.probabilities) != set(probe.likelihoods):
        raise ValueError("Probe hypotheses do not match the belief.")
    choices = next(iter(probe.likelihoods.values()))
    if outcome not in choices:
        raise ValueError("Unknown diagnostic outcome.")
    unnormalized = {
        name: p * probe.likelihoods[name][outcome]
        for name, p in prior.probabilities.items()
    }
    normalizer = sum(unnormalized.values())
    if normalizer <= 0:
        raise ValueError("Zero-probability outcome cannot update belief.")
    posterior = FailureBelief(
        probabilities={name: p / normalizer for name, p in unnormalized.items()}
    )
    return DiagnosticResult(
        probe_id=probe.probe_id,
        outcome=outcome,
        prior=prior,
        posterior=posterior,
        outcome_probability=normalizer,
    )


class BudgetedRepairPlanner:
    def __init__(
        self,
        options: list[RepairOption],
        probes: list[DiagnosticProbe],
        settings: PlannerSettings,
    ):
        if len({x.option_id for x in options}) != len(options):
            raise ValueError("Duplicate repair option IDs.")
        if len({x.probe_id for x in probes}) != len(probes):
            raise ValueError("Duplicate probe IDs.")
        self.options = options
        self.probes = probes
        self.settings = settings

    def _validate(self, belief: FailureBelief) -> None:
        hypotheses = set(belief.probabilities)
        for option in self.options:
            if set(option.estimated_gain_by_hypothesis) != hypotheses:
                raise ValueError("Repair action hypotheses do not match belief.")
        for probe in self.probes:
            if set(probe.likelihoods) != hypotheses:
                raise ValueError("Probe hypotheses do not match belief.")

    def _utility(self, belief: FailureBelief, option: RepairOption) -> float:
        s = self.settings
        gain = sum(
            p * option.estimated_gain_by_hypothesis[name]
            for name, p in belief.probabilities.items()
        )
        return (
            gain - s.cost_weight * option.cost
            - s.safety_weight * option.safety_risk
            - s.regression_weight * option.regression_risk
        )

    def _best(self, belief: FailureBelief, budget: float) -> tuple[RepairOption, float] | None:
        eligible = [
            option for option in self.options
            if option.cost <= budget and option.safety_risk <= self.settings.max_safety_risk
        ]
        if not eligible:
            return None
        return max(
            ((option, self._utility(belief, option)) for option in eligible),
            key=lambda pair: (pair[1], pair[0].option_id),
        )

    def net_value_of_information(
        self, belief: FailureBelief, probe: DiagnosticProbe, budget: float
    ) -> float:
        """Expected net gain from probing, after reserving cost for subsequent repair."""
        self._validate(belief)
        if probe.cost > budget or probe.safety_risk > self.settings.max_safety_risk:
            return float("-inf")
        base = self._best(belief, budget)
        if base is None:
            return float("-inf")
        expected_after = 0.0
        for outcome in next(iter(probe.likelihoods.values())):
            probability = sum(
                p * probe.likelihoods[hypothesis][outcome]
                for hypothesis, p in belief.probabilities.items()
            )
            if probability <= 0:
                continue
            after = posterior_from_probe(belief, probe, outcome).posterior
            best_after = self._best(after, budget - probe.cost)
            if best_after is None:
                return float("-inf")
            expected_after += probability * best_after[1]
        return expected_after - base[1] - self.settings.cost_weight * probe.cost

    def recommend(self, belief: FailureBelief, budget: float | None = None) -> Recommendation:
        self._validate(belief)
        available_budget = self.settings.budget if budget is None else budget
        if available_budget < 0:
            raise ValueError("Budget must be nonnegative.")
        best = self._best(belief, available_budget)
        if best is None:
            return Recommendation(
                mode="escalate", choice_id=None, expected_utility=None,
                reason="No repair option satisfies cost and safety constraints.",
            )
        ranked = [
            (self.net_value_of_information(belief, probe, available_budget), probe)
            for probe in self.probes
        ]
        if ranked:
            value, probe = max(ranked, key=lambda pair: (pair[0], pair[1].probe_id))
            if value > self.settings.min_probe_voi:
                return Recommendation(
                    mode="diagnose", choice_id=probe.probe_id,
                    expected_utility=best[1] + value, net_information_value=value,
                    reason="Diagnostic test has positive expected net information value.",
                )
        return Recommendation(
            mode="repair", choice_id=best[0].option_id,
            expected_utility=best[1],
            reason="Best eligible repair under the current belief and budget.",
        )
