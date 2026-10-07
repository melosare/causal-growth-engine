from __future__ import annotations

import pytest

from cge.optimization.bayesian import (
    BayesianOptimizationResult,
    OptimizationObservation,
)
from cge.optimization.diagnostics import (
    decision_difference,
    summarize_optimization,
)
from cge.optimization.objective import ObjectiveResult
from cge.optimization.problem import DecisionVariable


def make_observation(
    value: float,
    paid_ua_spend: float = 5_000.0,
) -> OptimizationObservation:
    decision = {
        DecisionVariable.PAID_UA_SPEND: paid_ua_spend,
        DecisionVariable.INFLUENCER_SPEND: 0.0,
        DecisionVariable.SOCIAL_MEDIA_POSTS: 0.0,
        DecisionVariable.PRODUCT_TEST_RELEASE: 0.0,
        DecisionVariable.PRODUCT_VERSION_UPDATE: 0.0,
    }

    objective = ObjectiveResult(
        mean_incremental_installs=value,
        lower=value - 10.0,
        upper=value + 10.0,
        probability_positive=0.75,
    )

    return OptimizationObservation(
        decision=decision,
        objective=objective,
    )


def make_result() -> BayesianOptimizationResult:
    observations = (
        make_observation(10.0, 5_000.0),
        make_observation(25.0, 10_000.0),
        make_observation(18.0, 7_500.0),
    )

    return BayesianOptimizationResult(
        best_decision=observations[1].decision,
        best_objective=observations[1].objective,
        observations=observations,
    )


def test_summarize_optimization() -> None:
    result = make_result()

    diagnostics = summarize_optimization(result)

    assert diagnostics.n_observations == 3
    assert diagnostics.initial_best_objective == 10.0
    assert diagnostics.best_objective == 25.0
    assert diagnostics.improvement == 15.0
    assert diagnostics.best_observation_index == 1
    assert diagnostics.improvement_ratio == pytest.approx(1.5)


def test_summarize_optimization_rejects_empty_result() -> None:
    result = BayesianOptimizationResult(
        best_decision={},
        best_objective=ObjectiveResult(
            mean_incremental_installs=0.0,
            lower=0.0,
            upper=0.0,
            probability_positive=0.0,
        ),
        observations=(),
    )

    with pytest.raises(ValueError, match="no observations"):
        summarize_optimization(result)


def test_decision_difference() -> None:
    first = make_observation(
        10.0,
        paid_ua_spend=5_000.0,
    ).decision

    second = make_observation(
        20.0,
        paid_ua_spend=10_000.0,
    ).decision

    difference = decision_difference(first, second)

    assert difference[DecisionVariable.PAID_UA_SPEND] == 5_000.0
    assert difference[DecisionVariable.INFLUENCER_SPEND] == 0.0


def test_decision_difference_rejects_different_variables() -> None:
    first = {
        DecisionVariable.PAID_UA_SPEND: 5_000.0,
    }

    second = {
        DecisionVariable.INFLUENCER_SPEND: 1_000.0,
    }

    with pytest.raises(ValueError, match="same variables"):
        decision_difference(first, second)
