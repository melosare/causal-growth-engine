from __future__ import annotations

from typing import cast

import arviz as az

from cge.causal.model import ModelConfig, fit_model
from cge.data import DatasetConfig, SyntheticGrowthGenerator
from cge.optimization.bayesian import BayesianOptimizationConfig
from cge.pipeline.optimize import optimize_growth_strategy


def main() -> None:
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

    trace = cast(az.InferenceData, fitted_trace)

    result = optimize_growth_strategy(
        data=data,
        trace=trace,
        config=BayesianOptimizationConfig(
            initial_points=8,
            iterations=12,
            random_seed=42,
        ),
    )

    print("Best decision:")
    for variable, value in result.optimization.best_decision.items():
        print(f"  {variable.value}: {value}")

    print()
    print("Expected incremental installs:")
    print(
        f"  {result.optimization.best_objective.mean_incremental_installs:.2f}"
    )


if __name__ == "__main__":
    main()
