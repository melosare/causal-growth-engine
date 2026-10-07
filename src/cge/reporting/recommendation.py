from __future__ import annotations

from dataclasses import dataclass

from cge.optimization.bayesian import BayesianOptimizationResult
from cge.optimization.diagnostics import (
    OptimizationDiagnostics,
    summarize_optimization,
)
from cge.optimization.problem import DecisionVariable


@dataclass(frozen=True)
class StrategyRecommendation:
    """Human-readable recommendation produced by the optimizer."""

    decision: dict[DecisionVariable, float]
    expected_incremental_installs: float
    lower_bound: float
    upper_bound: float
    probability_positive: float
    diagnostics: OptimizationDiagnostics

    @property
    def credible_interval(self) -> tuple[float, float]:
        """Return the posterior credible interval."""

        return self.lower_bound, self.upper_bound


def build_recommendation(
    result: BayesianOptimizationResult,
) -> StrategyRecommendation:
    """Convert an optimization result into a strategy recommendation."""

    diagnostics = summarize_optimization(result)
    objective = result.best_objective

    return StrategyRecommendation(
        decision=result.best_decision.copy(),
        expected_incremental_installs=(
            objective.mean_incremental_installs
        ),
        lower_bound=objective.lower,
        upper_bound=objective.upper,
        probability_positive=objective.probability_positive,
        diagnostics=diagnostics,
    )


def format_recommendation(
    recommendation: StrategyRecommendation,
) -> str:
    """Format a strategy recommendation as readable text."""

    lines = [
        "Recommended growth strategy",
        "============================",
        "",
        "Decision:",
    ]

    for variable in DecisionVariable:
        value = recommendation.decision[variable]
        lines.append(
            f"  {variable.value}: {value:g}"
        )

    lines.extend(
        [
            "",
            "Expected impact:",
            (
                "  Incremental installs: "
                f"{recommendation.expected_incremental_installs:,.1f}"
            ),
            (
                "  Credible interval: "
                f"[{recommendation.lower_bound:,.1f}, "
                f"{recommendation.upper_bound:,.1f}]"
            ),
            (
                "  Probability of positive impact: "
                f"{recommendation.probability_positive:.1%}"
            ),
            "",
            "Optimization:",
            (
                "  Evaluations: "
                f"{recommendation.diagnostics.n_observations}"
            ),
            (
                "  Improvement over initial candidate: "
                f"{recommendation.diagnostics.improvement:,.1f}"
            ),
        ]
    )

    return "\n".join(lines)
