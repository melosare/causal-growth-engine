from __future__ import annotations

from dataclasses import dataclass

from cge.optimization.problem import DecisionVariable
from cge.reporting.recommendation import StrategyRecommendation


@dataclass(frozen=True)
class GrowthReport:
    """Structured business report for an optimized growth strategy."""

    recommendation: StrategyRecommendation
    total_paid_spend: float
    total_product_actions: float

    @property
    def expected_incremental_installs(self) -> float:
        """Return the expected incremental installs."""

        return self.recommendation.expected_incremental_installs

    @property
    def credible_interval(self) -> tuple[float, float]:
        """Return the credible interval for incremental installs."""

        return self.recommendation.credible_interval

    @property
    def probability_positive(self) -> float:
        """Return the probability of positive incremental impact."""

        return self.recommendation.probability_positive


def build_growth_report(
    recommendation: StrategyRecommendation,
) -> GrowthReport:
    """Build a business-facing report from a strategy recommendation."""

    decision = recommendation.decision

    total_paid_spend = (
        decision[DecisionVariable.PAID_UA_SPEND]
        + decision[DecisionVariable.INFLUENCER_SPEND]
    )

    total_product_actions = (
        decision[DecisionVariable.PRODUCT_TEST_RELEASE]
        + decision[DecisionVariable.PRODUCT_VERSION_UPDATE]
    )

    return GrowthReport(
        recommendation=recommendation,
        total_paid_spend=float(total_paid_spend),
        total_product_actions=float(total_product_actions),
    )


def format_growth_report(
    report: GrowthReport,
) -> str:
    """Format the growth report for human-readable output."""

    recommendation = report.recommendation

    lines = [
        "Causal Growth Engine Report",
        "============================",
        "",
        "Recommended strategy",
        "--------------------",
    ]

    for variable in DecisionVariable:
        value = recommendation.decision[variable]
        lines.append(
            f"{variable.value}: {value:g}"
        )

    lines.extend(
        [
            "",
            "Expected impact",
            "---------------",
            (
                "Expected incremental installs: "
                f"{report.expected_incremental_installs:,.1f}"
            ),
            (
                "Credible interval: "
                f"[{report.credible_interval[0]:,.1f}, "
                f"{report.credible_interval[1]:,.1f}]"
            ),
            (
                "Probability of positive impact: "
                f"{report.probability_positive:.1%}"
            ),
            "",
            "Resource allocation",
            "-------------------",
            (
                "Paid + influencer spend: "
                f"${report.total_paid_spend:,.0f}"
            ),
            (
                "Product actions: "
                f"{report.total_product_actions:,.0f}"
            ),
            "",
            "Optimization diagnostics",
            "-------------------------",
            (
                "Candidates evaluated: "
                f"{recommendation.diagnostics.n_observations}"
            ),
            (
                "Improvement over initial candidate: "
                f"{recommendation.diagnostics.improvement:,.1f}"
            ),
        ]
    )

    return "\n".join(lines)
