from __future__ import annotations

import numpy as np

from cge.causal.model import ModelConfig, fit_model
from cge.data import DatasetConfig, SyntheticGrowthGenerator
from cge.optimization.bayesian import BayesianOptimizationConfig
from cge.optimization.problem import DecisionVariable
from cge.pipeline import optimize_growth_strategy


def test_optimization_pipeline_returns_result() -> None:
    data, _ = SyntheticGrowthGenerator(
        DatasetConfig(
            n_titles=2,
            n_days=30,
            seed=42,
        )
    ).generate()

    _, trace = fit_model(
        data,
        config=ModelConfig(
            draws=20,
            tune=20,
            chains=1,
            random_seed=42,
        ),
    )

    result = optimize_growth_strategy(
        data=data,
        trace=trace,
        config=BayesianOptimizationConfig(
            initial_points=2,
            iterations=2,
            random_seed=42,
        ),
    )

    assert result.optimization.best_decision

    assert np.isfinite(
        result.objective.mean_incremental_installs
    )

    assert (
        result.intervention.paid_ua_spend
        == result.optimization.best_decision[
            DecisionVariable.PAID_UA_SPEND
        ]
    )

    assert (
        result.recommendation.decision
        == result.optimization.best_decision
    )

    assert (
        result.recommendation.expected_incremental_installs
        == result.objective.mean_incremental_installs
    )

    assert (
        result.recommendation.probability_positive
        == result.objective.probability_positive
    )

    assert result.recommendation.lower_bound == result.objective.lower
    assert result.recommendation.upper_bound == result.objective.upper
