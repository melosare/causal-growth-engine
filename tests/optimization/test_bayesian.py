from __future__ import annotations

import arviz as az
import pandas as pd

from cge.optimization.bayesian import (
    BayesianOptimizationConfig,
    BayesianOptimizationResult,
    optimize,
)
from cge.optimization.objective import ObjectiveResult
from cge.optimization.problem import (
    DecisionVariable,
    OptimizationProblem,
)


def test_bayesian_optimization_returns_result(
    monkeypatch,
) -> None:
    problem = OptimizationProblem.default()

    def fake_evaluate_objective(
        data: pd.DataFrame,
        trace: az.InferenceData,
        decision: dict[DecisionVariable, float],
        problem: OptimizationProblem,
    ) -> ObjectiveResult:
        score = (
            decision[DecisionVariable.PAID_UA_SPEND]
            + 2.0
            * decision[DecisionVariable.SOCIAL_MEDIA_POSTS]
            + decision[DecisionVariable.PRODUCT_VERSION_UPDATE]
        )

        return ObjectiveResult(
            mean_incremental_installs=score,
            lower=score - 10.0,
            upper=score + 10.0,
            probability_positive=0.9,
        )

    monkeypatch.setattr(
        "cge.optimization.bayesian.evaluate_objective",
        fake_evaluate_objective,
    )

    result = optimize(
        data=pd.DataFrame({"value": [1]}),
        trace=az.InferenceData(),
        problem=problem,
        config=BayesianOptimizationConfig(
            initial_points=3,
            iterations=2,
            random_seed=42,
        ),
    )

    assert isinstance(
        result,
        BayesianOptimizationResult,
    )

    assert len(result.observations) == 5

    problem.validate_decision(
        result.best_decision
    )


def test_bayesian_optimization_is_reproducible(
    monkeypatch,
) -> None:
    problem = OptimizationProblem.default()

    def fake_evaluate_objective(
        data: pd.DataFrame,
        trace: az.InferenceData,
        decision: dict[DecisionVariable, float],
        problem: OptimizationProblem,
    ) -> ObjectiveResult:
        score = (
            decision[DecisionVariable.PAID_UA_SPEND]
            + decision[DecisionVariable.INFLUENCER_SPEND]
        )

        return ObjectiveResult(
            mean_incremental_installs=score,
            lower=score - 10.0,
            upper=score + 10.0,
            probability_positive=0.9,
        )

    monkeypatch.setattr(
        "cge.optimization.bayesian.evaluate_objective",
        fake_evaluate_objective,
    )

    config = BayesianOptimizationConfig(
        initial_points=3,
        iterations=2,
        random_seed=123,
    )

    first = optimize(
        data=pd.DataFrame({"value": [1]}),
        trace=az.InferenceData(),
        problem=problem,
        config=config,
    )

    second = optimize(
        data=pd.DataFrame({"value": [1]}),
        trace=az.InferenceData(),
        problem=problem,
        config=config,
    )

    assert first.best_decision == second.best_decision
    assert (
        first.best_objective.mean_incremental_installs
        == second.best_objective.mean_incremental_installs
    )


def test_config_rejects_invalid_values() -> None:
    try:
        BayesianOptimizationConfig(
            initial_points=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected initial_points validation error."
        )

    try:
        BayesianOptimizationConfig(
            iterations=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected iterations validation error."
        )

    try:
        BayesianOptimizationConfig(
            exploration=-1.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected exploration validation error."
        )
