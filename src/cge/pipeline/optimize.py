from __future__ import annotations

from dataclasses import dataclass

import arviz as az
import pandas as pd

from cge.optimization.bayesian import (
    BayesianOptimizationConfig,
    BayesianOptimizationResult,
    optimize,
)
from cge.optimization.objective import ObjectiveResult, evaluate_objective
from cge.optimization.problem import (
    DecisionVariable,
    OptimizationProblem,
)
from cge.reporting.recommendation import (
    StrategyRecommendation,
    build_recommendation,
)
from cge.simulation.counterfactual import TreatmentIntervention


@dataclass(frozen=True)
class OptimizationPipelineResult:
    """Complete result from the CGE optimization workflow."""

    optimization: BayesianOptimizationResult
    objective: ObjectiveResult
    intervention: TreatmentIntervention
    recommendation: StrategyRecommendation


def _decision_to_intervention(
    decision: dict[DecisionVariable, float],
) -> TreatmentIntervention:
    """Convert an optimizer decision into a causal intervention."""

    return TreatmentIntervention(
        paid_ua_spend=decision[DecisionVariable.PAID_UA_SPEND],
        influencer_spend=decision[DecisionVariable.INFLUENCER_SPEND],
        social_media_posts=decision[
            DecisionVariable.SOCIAL_MEDIA_POSTS
        ],
        product_test_release=decision[
            DecisionVariable.PRODUCT_TEST_RELEASE
        ],
        product_version_update=decision[
            DecisionVariable.PRODUCT_VERSION_UPDATE
        ],
    )


def optimize_growth_strategy(
    data: pd.DataFrame,
    trace: az.InferenceData,
    problem: OptimizationProblem | None = None,
    config: BayesianOptimizationConfig | None = None,
) -> OptimizationPipelineResult:
    """
    Find and evaluate the best growth strategy.

    The workflow:

    1. Defines the optimization problem.
    2. Searches the feasible decision space using Bayesian optimization.
    3. Evaluates the best decision through the causal objective.
    4. Converts the decision into a causal intervention.
    5. Builds a business-facing strategy recommendation.
    """

    optimization_problem = problem or OptimizationProblem.default()

    optimization = optimize(
        data=data,
        trace=trace,
        problem=optimization_problem,
        config=config,
    )

    best_decision = optimization.best_decision

    objective = evaluate_objective(
        data=data,
        trace=trace,
        decision=best_decision,
        problem=optimization_problem,
    )

    intervention = _decision_to_intervention(best_decision)

    recommendation = build_recommendation(
        optimization
    )

    return OptimizationPipelineResult(
        optimization=optimization,
        objective=objective,
        intervention=intervention,
        recommendation=recommendation,
    )
