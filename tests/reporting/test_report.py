from __future__ import annotations

from cge.optimization.bayesian import (
    BayesianOptimizationResult,
    OptimizationObservation,
)
from cge.optimization.objective import ObjectiveResult
from cge.optimization.problem import DecisionVariable
from cge.reporting.recommendation import build_recommendation
from cge.reporting.report import (
    build_growth_report,
    format_growth_report,
)


def make_recommendation():
    decision = {
        DecisionVariable.PAID_UA_SPEND: 10_000.0,
        DecisionVariable.INFLUENCER_SPEND: 2_000.0,
        DecisionVariable.SOCIAL_MEDIA_POSTS: 5.0,
        DecisionVariable.PRODUCT_TEST_RELEASE: 1.0,
        DecisionVariable.PRODUCT_VERSION_UPDATE: 2.0,
    }

    objective = ObjectiveResult(
        mean_incremental_installs=100.0,
        lower=60.0,
        upper=140.0,
        probability_positive=0.95,
    )

    result = BayesianOptimizationResult(
        best_decision=decision,
        best_objective=objective,
        observations=(
            OptimizationObservation(
                decision=decision.copy(),
                objective=objective,
            ),
        ),
    )

    return build_recommendation(result)


def test_build_growth_report() -> None:
    recommendation = make_recommendation()

    report = build_growth_report(recommendation)

    assert report.total_paid_spend == 12_000.0
    assert report.total_product_actions == 3.0

    assert report.expected_incremental_installs == 100.0
    assert report.credible_interval == (60.0, 140.0)
    assert report.probability_positive == 0.95


def test_format_growth_report() -> None:
    report = build_growth_report(make_recommendation())

    formatted = format_growth_report(report)

    assert "Causal Growth Engine Report" in formatted
    assert "Recommended strategy" in formatted
    assert "Expected incremental installs: 100.0" in formatted
    assert "Credible interval: [60.0, 140.0]" in formatted
    assert "Probability of positive impact: 95.0%" in formatted
    assert "Paid + influencer spend: $12,000" in formatted
    assert "Product actions: 3" in formatted
