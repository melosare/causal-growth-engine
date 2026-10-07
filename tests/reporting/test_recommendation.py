from __future__ import annotations

from cge.optimization.bayesian import (
    BayesianOptimizationResult,
    OptimizationObservation,
)
from cge.optimization.objective import ObjectiveResult
from cge.optimization.problem import DecisionVariable
from cge.reporting.recommendation import (
    build_recommendation,
    format_recommendation,
)


def make_result() -> BayesianOptimizationResult:
    decision = {
        DecisionVariable.PAID_UA_SPEND: 10_000.0,
        DecisionVariable.INFLUENCER_SPEND: 2_000.0,
        DecisionVariable.SOCIAL_MEDIA_POSTS: 5.0,
        DecisionVariable.PRODUCT_TEST_RELEASE: 1.0,
        DecisionVariable.PRODUCT_VERSION_UPDATE: 2.0,
    }

    initial_objective = ObjectiveResult(
        mean_incremental_installs=50.0,
        lower=30.0,
        upper=70.0,
        probability_positive=0.9,
    )

    best_objective = ObjectiveResult(
        mean_incremental_installs=100.0,
        lower=60.0,
        upper=140.0,
        probability_positive=0.95,
    )

    observations = (
        OptimizationObservation(
            decision=decision.copy(),
            objective=initial_objective,
        ),
        OptimizationObservation(
            decision=decision.copy(),
            objective=best_objective,
        ),
    )

    return BayesianOptimizationResult(
        best_decision=decision,
        best_objective=best_objective,
        observations=observations,
    )


def test_build_recommendation() -> None:
    recommendation = build_recommendation(
        make_result()
    )

    assert (
        recommendation.expected_incremental_installs
        == 100.0
    )

    assert recommendation.lower_bound == 60.0
    assert recommendation.upper_bound == 140.0
    assert recommendation.probability_positive == 0.95

    assert (
        recommendation.decision[
            DecisionVariable.PAID_UA_SPEND
        ]
        == 10_000.0
    )

    assert recommendation.diagnostics.n_observations == 2
    assert recommendation.diagnostics.improvement == 50.0


def test_credible_interval() -> None:
    recommendation = build_recommendation(
        make_result()
    )

    assert recommendation.credible_interval == (
        60.0,
        140.0,
    )


def test_format_recommendation() -> None:
    recommendation = build_recommendation(
        make_result()
    )

    formatted = format_recommendation(
        recommendation
    )

    assert "Recommended growth strategy" in formatted
    assert "paid_ua_spend: 10000" in formatted
    assert "Incremental installs: 100.0" in formatted
    assert "Probability of positive impact: 95.0%" in formatted
    assert "Evaluations: 2" in formatted
