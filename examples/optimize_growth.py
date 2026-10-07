from __future__ import annotations

from typing import cast

import arviz as az

from cge.causal.model import ModelConfig, fit_model
from cge.data import DatasetConfig, SyntheticGrowthGenerator
from cge.optimization.bayesian import BayesianOptimizationConfig
from cge.pipeline import OptimizationPipelineResult, optimize_growth_strategy


def run_example() -> OptimizationPipelineResult:
    """Run the complete causal growth optimization workflow and return its result."""

    data, _ = SyntheticGrowthGenerator(
        DatasetConfig(
            n_titles=10,
            n_days=60,
            seed=42,
        )
    ).generate()

    _, fitted_trace = fit_model(
        data,
        config=ModelConfig(
            draws=100,
            tune=100,
            chains=2,
            random_seed=42,
        ),
    )

    trace = cast(
        az.InferenceData,
        fitted_trace,
    )

    result = optimize_growth_strategy(
        data=data,
        trace=trace,
        config=BayesianOptimizationConfig(
            initial_points=8,
            iterations=12,
            random_seed=42,
        ),
    )

    return result
