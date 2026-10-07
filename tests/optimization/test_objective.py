from __future__ import annotations

import arviz as az
import pandas as pd
import pytest

from cge.optimization.objective import (
    ObjectiveResult,
    _decision_to_intervention,
    evaluate_objective,
    objective_value,
)
from cge.optimization.problem import DecisionVariable
from cge.simulation.counterfactual import (
    CounterfactualSummary,
    TreatmentIntervention,
)


def make_decision() -> dict[DecisionVariable, float]:
    """Return a valid optimization decision for testing."""

    return {
        DecisionVariable.PAID_UA_SPEND: 10_000.0,
        DecisionVariable.INFLUENCER_SPEND: 4_000.0,
        DecisionVariable.SOCIAL_MEDIA_POSTS: 5.0,
        DecisionVariable.PRODUCT_TEST_RELEASE: 2.0,
        DecisionVariable.PRODUCT_VERSION_UPDATE: 1.0,
    }


def make_summary(
    incremental_mean: float = 250.0,
    incremental_lower: float = 100.0,
    incremental_upper: float = 400.0,
    probability_positive: float = 0.98,
) -> CounterfactualSummary:
    """Create a deterministic counterfactual summary for testing."""

    return CounterfactualSummary(
        baseline_mean=10_000.0,
        counterfactual_mean=10_250.0,
        incremental_mean=incremental_mean,
        incremental_lower=incremental_lower,
        incremental_upper=incremental_upper,
        probability_positive=probability_positive,
    )


def test_decision_converts_to_treatment_intervention() -> None:
    decision = make_decision()

    intervention = _decision_to_intervention(decision)

    assert isinstance(intervention, TreatmentIntervention)
    assert intervention.paid_ua_spend == 10_000.0
    assert intervention.influencer_spend == 4_000.0
    assert intervention.social_media_posts == 5.0
    assert intervention.product_test_release == 2.0
    assert intervention.product_version_update == 1.0


def test_objective_result_from_summary() -> None:
    summary = make_summary()

    result = ObjectiveResult.from_summary(summary)

    assert result.mean_incremental_installs == 250.0
    assert result.lower == 100.0
    assert result.upper == 400.0
    assert result.probability_positive == 0.98


def test_objective_rejects_invalid_decision() -> None:
    data = pd.DataFrame()
    trace = az.InferenceData()

    decision = make_decision()
    decision[DecisionVariable.PAID_UA_SPEND] = 20_000.0

    with pytest.raises(ValueError, match="exceeds its maximum"):
        evaluate_objective(
            data=data,
            trace=trace,
            decision=decision,
        )


def test_objective_value_returns_mean_incremental_installs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = make_summary()

    def fake_compare_strategy(
        data: pd.DataFrame,
        trace: az.InferenceData,
        intervention: TreatmentIntervention,
    ) -> CounterfactualSummary:
        return expected

    monkeypatch.setattr(
        "cge.optimization.objective.compare_strategy",
        fake_compare_strategy,
    )

    result = objective_value(
        data=pd.DataFrame({"value": [1]}),
        trace=az.InferenceData(),
        decision=make_decision(),
    )

    assert result == 250.0


def test_evaluate_objective_returns_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = make_summary(
        incremental_mean=150.0,
        incremental_lower=50.0,
        incremental_upper=250.0,
        probability_positive=0.90,
    )

    def fake_compare_strategy(
        data: pd.DataFrame,
        trace: az.InferenceData,
        intervention: TreatmentIntervention,
    ) -> CounterfactualSummary:
        return expected

    monkeypatch.setattr(
        "cge.optimization.objective.compare_strategy",
        fake_compare_strategy,
    )

    result = evaluate_objective(
        data=pd.DataFrame({"value": [1]}),
        trace=az.InferenceData(),
        decision=make_decision(),
    )

    assert result == ObjectiveResult(
        mean_incremental_installs=150.0,
        lower=50.0,
        upper=250.0,
        probability_positive=0.90,
    )
