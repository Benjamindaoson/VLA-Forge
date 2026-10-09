import math

import pytest
from pydantic import ValidationError

from vla_forge.belief import (
    BudgetedRepairPlanner, DiagnosticProbe, FailureBelief, PlannerSettings,
    RepairOption, posterior_from_probe,
)
from vla_forge.models import RepairKind


def example():
    prior = FailureBelief(probabilities={"timing": 0.5, "geometry": 0.5})
    options = [
        RepairOption(
            option_id="runtime", kind=RepairKind.RUNTIME_FIX, cost=1,
            safety_risk=0, regression_risk=0,
            estimated_gain_by_hypothesis={"timing": 0.6, "geometry": 0.0},
        ),
        RepairOption(
            option_id="sft", kind=RepairKind.POLICY_UPDATE, cost=2,
            safety_risk=0, regression_risk=0,
            estimated_gain_by_hypothesis={"timing": 0.0, "geometry": 0.7},
        ),
    ]
    probe = DiagnosticProbe(
        probe_id="horizon_test", cost=0.2, safety_risk=0,
        likelihoods={
            "timing": {"yes": 0.9, "no": 0.1},
            "geometry": {"yes": 0.1, "no": 0.9},
        },
    )
    return prior, options, probe, PlannerSettings(budget=5, cost_weight=0.01)


def test_bayesian_update():
    prior, _, probe, _ = example()
    result = posterior_from_probe(prior, probe, "yes")
    assert result.posterior.probabilities["timing"] == pytest.approx(0.9)
    assert result.outcome_probability == pytest.approx(0.5)


def test_positive_information_value():
    belief, options, probe, settings = example()
    planner = BudgetedRepairPlanner(options, [probe], settings)
    result = planner.recommend(belief)
    assert result.mode == "diagnose"
    assert result.choice_id == "horizon_test"
    assert result.net_information_value > 0


def test_update_changes_best_repair():
    belief, options, probe, settings = example()
    updated = posterior_from_probe(belief, probe, "yes").posterior
    result = BudgetedRepairPlanner(options, [], settings).recommend(updated)
    assert result.choice_id == "runtime"


def test_probe_requires_both_probe_and_repair_budget():
    belief, options, probe, settings = example()
    result = BudgetedRepairPlanner(options, [probe], settings).recommend(belief, budget=1)
    assert result.mode == "repair"


def test_unsafe_repair_requires_escalation():
    belief, options, probe, settings = example()
    options = [o.model_copy(update={"safety_risk": 0.9}) for o in options]
    result = BudgetedRepairPlanner(options, [probe], settings).recommend(belief)
    assert result.mode == "escalate"


def test_invalid_prior_rejected():
    with pytest.raises(ValidationError):
        FailureBelief(probabilities={"a": 0.5, "b": 0.8})


def test_invalid_probe_likelihood_rejected():
    with pytest.raises(ValidationError):
        DiagnosticProbe(
            probe_id="bad", cost=1, safety_risk=0,
            likelihoods={"x": {"yes": 0.8, "no": 0.8}},
        )


def test_missing_hypothesis_is_error():
    belief, options, probe, settings = example()
    options[0].estimated_gain_by_hypothesis = {"other": 1.0}
    planner = BudgetedRepairPlanner(options, [probe], settings)
    with pytest.raises(ValueError, match="hypotheses"):
        planner.recommend(belief)


def test_impossible_probe_result_is_error():
    belief = FailureBelief(probabilities={"a": 1, "b": 0})
    probe = DiagnosticProbe(
        probe_id="p", cost=1, safety_risk=0,
        likelihoods={
            "a": {"yes": 0, "no": 1},
            "b": {"yes": 1, "no": 0},
        },
    )
    with pytest.raises(ValueError, match="Zero-probability"):
        posterior_from_probe(belief, probe, "yes")


def test_voi_is_finite():
    prior, options, probe, settings = example()
    planner = BudgetedRepairPlanner(options, [probe], settings)
    assert math.isfinite(planner.net_value_of_information(prior, probe, 5))


def test_negative_expected_repair_value_leads_to_escalation():
    prior = FailureBelief(probabilities={"timing": 1.0})
    negative = RepairOption(
        option_id="harmful_update", kind=RepairKind.POLICY_UPDATE,
        cost=1, safety_risk=0, regression_risk=0,
        estimated_gain_by_hypothesis={"timing": -0.2},
    )
    decision = BudgetedRepairPlanner(
        [negative], [], PlannerSettings(budget=2, cost_weight=0.01)
    ).recommend(prior)
    assert decision.mode == "escalate"
    assert decision.choice_id is None


def test_diagnostic_safety_risk_is_included_in_value():
    prior, options, probe, settings = example()
    settings.max_safety_risk = 0.5
    low_risk = BudgetedRepairPlanner(options, [probe], settings)
    risky_probe = probe.model_copy(update={"safety_risk": 0.2})
    high_risk = BudgetedRepairPlanner(options, [risky_probe], settings)
    normal_value = low_risk.net_value_of_information(prior, probe, 5)
    risk_value = high_risk.net_value_of_information(prior, risky_probe, 5)
    assert risk_value == pytest.approx(normal_value - settings.safety_weight * 0.2)
