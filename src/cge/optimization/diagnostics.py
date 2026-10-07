from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cge.optimization.bayesian import BayesianOptimizationResult
from cge.optimization.problem import DecisionVariable


@dataclass(frozen=True)
class OptimizationDiagnostics:
    """Summary diagnostics for a Bayesian optimization run."""

    n_observations: int
    best_objective: float
    initial_best_objective: float
    improvement: float
    best_observation_index: int

    @property
    def improvement_ratio(self) -> float:
        """Return relative improvement over the initial best objective."""

        if self.initial_best_objective == 0.0:
            return 0.0

        return self.improvement / abs(self.initial_best_objective)


def summarize_optimization(
    result: BayesianOptimizationResult,
) -> OptimizationDiagnostics:
    """Summarize the search history of an optimization result."""

    if not result.observations:
        raise ValueError("Cannot summarize an optimization result with no observations.")

    values = np.asarray(
        [observation.objective.mean_incremental_installs for observation in result.observations],
        dtype=float,
    )

    best_index = int(np.argmax(values))
    initial_best = float(values[0])
    best_value = float(values[best_index])

    return OptimizationDiagnostics(
        n_observations=len(result.observations),
        best_objective=best_value,
        initial_best_objective=initial_best,
        improvement=best_value - initial_best,
        best_observation_index=best_index,
    )


def decision_difference(
    first: dict[DecisionVariable, float],
    second: dict[DecisionVariable, float],
) -> dict[DecisionVariable, float]:
    """Return second-minus-first for every decision variable."""

    if set(first) != set(second):
        raise ValueError("Decisions must contain the same variables.")

    return {variable: second[variable] - first[variable] for variable in first}
