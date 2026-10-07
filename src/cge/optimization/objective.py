from __future__ import annotations

from dataclasses import dataclass

import arviz as az
import pandas as pd

from cge.optimization.problem import (
    DecisionVariable,
    OptimizationProblem,
)
from cge.simulation.counterfactual import (
    CounterfactualSummary,
    TreatmentIntervention,
    compare_strategy,
)


@dataclass(frozen=True)
class ObjectiveResult:
    """Posterior summary of a candidate intervention's incremental impact."""

    mean_incremental_installs: float
    lower: float
    upper: float
    probability_positive: float

    @classmethod
    def from_summary(
        cls,
        summary: CounterfactualSummary,
    ) -> ObjectiveResult:
        """Create an objective result from a counterfactual summary."""

        return cls(
            mean_incremental_installs=summary.incremental_mean,
            lower=summary.incremental_lower,
            upper=summary.incremental_upper,
            probability_positive=summary.probability_positive,
        )


def _decision_to_intervention(
    decision: dict[DecisionVariable, float],
) -> TreatmentIntervention:
    """Convert an optimization decision into a causal intervention."""

    return TreatmentIntervention(
        paid_ua_spend=decision[DecisionVariable.PAID_UA_SPEND],
        influencer_spend=decision[DecisionVariable.INFLUENCER_SPEND],
        social_media_posts=decision[DecisionVariable.SOCIAL_MEDIA_POSTS],
        product_test_release=decision[DecisionVariable.PRODUCT_TEST_RELEASE],
        product_version_update=decision[DecisionVariable.PRODUCT_VERSION_UPDATE],
    )


def evaluate_objective(
    data: pd.DataFrame,
    trace: az.InferenceData,
    decision: dict[DecisionVariable, float],
    problem: OptimizationProblem | None = None,
) -> ObjectiveResult:
    """Evaluate expected incremental organic installs for a decision."""

    optimization_problem = problem or OptimizationProblem.default()

    optimization_problem.validate_decision(decision)

    intervention = _decision_to_intervention(decision)

    summary = compare_strategy(
        data=data,
        trace=trace,
        intervention=intervention,
    )

    return ObjectiveResult.from_summary(summary)


def objective_value(
    data: pd.DataFrame,
    trace: az.InferenceData,
    decision: dict[DecisionVariable, float],
    problem: OptimizationProblem | None = None,
) -> float:
    """Return the scalar objective value for optimization."""

    result = evaluate_objective(
        data=data,
        trace=trace,
        decision=decision,
        problem=problem,
    )

    return result.mean_incremental_installs
