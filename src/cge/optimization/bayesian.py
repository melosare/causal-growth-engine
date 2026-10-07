from __future__ import annotations

from dataclasses import dataclass
from math import erf, pi, sqrt

import arviz as az
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from cge.optimization.objective import (
    ObjectiveResult,
    evaluate_objective,
)
from cge.optimization.problem import (
    CONTROLLABLE_VARIABLES,
    DecisionVariable,
    OptimizationProblem,
)


@dataclass(frozen=True)
class BayesianOptimizationConfig:
    """Configuration for Bayesian optimization."""

    initial_points: int = 8
    iterations: int = 12
    random_seed: int = 42
    exploration: float = 0.01

    def __post_init__(self) -> None:
        if self.initial_points < 1:
            raise ValueError("initial_points must be positive.")

        if self.iterations < 1:
            raise ValueError("iterations must be positive.")

        if self.exploration < 0:
            raise ValueError("exploration must be non-negative.")


@dataclass(frozen=True)
class OptimizationObservation:
    """One evaluated optimization candidate."""

    decision: dict[DecisionVariable, float]
    objective: ObjectiveResult


@dataclass(frozen=True)
class BayesianOptimizationResult:
    """Result returned by Bayesian optimization."""

    best_decision: dict[DecisionVariable, float]
    best_objective: ObjectiveResult
    observations: tuple[OptimizationObservation, ...]


def _decision_to_array(
    decision: dict[DecisionVariable, float],
) -> np.ndarray:
    """Convert a decision dictionary to a model input vector."""

    return np.asarray(
        [decision[variable] for variable in CONTROLLABLE_VARIABLES],
        dtype=float,
    )


def _array_to_decision(
    values: np.ndarray,
    problem: OptimizationProblem,
) -> dict[DecisionVariable, float]:
    """Project a continuous vector onto the valid decision grid."""

    decision: dict[DecisionVariable, float] = {}

    for index, variable in enumerate(CONTROLLABLE_VARIABLES):
        bounds = problem.bounds[variable]

        value = float(values[index])

        value = max(bounds.minimum, min(bounds.maximum, value))

        steps = round((value - bounds.minimum) / bounds.step)

        projected = bounds.minimum + steps * bounds.step

        projected = max(
            bounds.minimum,
            min(bounds.maximum, projected),
        )

        decision[variable] = float(projected)

    return decision


def _random_feasible_decision(
    rng: np.random.Generator,
    problem: OptimizationProblem,
) -> dict[DecisionVariable, float]:
    """Generate one random feasible decision."""

    variables = list(CONTROLLABLE_VARIABLES)

    for _ in range(10_000):
        decision: dict[DecisionVariable, float] = {}

        for variable in variables:
            bounds = problem.bounds[variable]

            n_steps = int(round((bounds.maximum - bounds.minimum) / bounds.step))

            step_index = int(rng.integers(0, n_steps + 1))

            decision[variable] = bounds.minimum + step_index * bounds.step

        try:
            problem.validate_decision(decision)
        except ValueError:
            continue

        return decision

    raise RuntimeError("Unable to generate a feasible optimization candidate.")


def _normal_pdf(values: np.ndarray) -> np.ndarray:
    """Evaluate the standard normal probability density."""

    return np.exp(-0.5 * values**2) / sqrt(2.0 * pi)


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    """Evaluate the standard normal cumulative distribution."""

    result = 0.5 * (1.0 + np.vectorize(erf)(values / sqrt(2.0)))

    return np.asarray(result, dtype=float)


def _expected_improvement(
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    best_value: float,
    exploration: float,
) -> np.ndarray:
    """Calculate expected improvement for maximization."""

    improvement = mean - best_value - exploration

    safe_std = np.maximum(
        standard_deviation,
        1e-12,
    )

    z = improvement / safe_std

    result = improvement * _normal_cdf(z) + safe_std * _normal_pdf(z)

    return np.asarray(result, dtype=float)


def _candidate_pool(
    rng: np.random.Generator,
    problem: OptimizationProblem,
    size: int,
) -> list[dict[DecisionVariable, float]]:
    """Generate a pool of feasible candidates for acquisition search."""

    candidates: list[dict[DecisionVariable, float]] = []
    seen: set[tuple[float, ...]] = set()

    while len(candidates) < size:
        decision = _random_feasible_decision(
            rng=rng,
            problem=problem,
        )

        key = tuple(decision[variable] for variable in CONTROLLABLE_VARIABLES)

        if key in seen:
            continue

        seen.add(key)
        candidates.append(decision)

    return candidates


def optimize(
    data: pd.DataFrame,
    trace: az.InferenceData,
    problem: OptimizationProblem | None = None,
    config: BayesianOptimizationConfig | None = None,
) -> BayesianOptimizationResult:
    """
    Optimize the causal-growth objective with Bayesian optimization.

    The Gaussian-process surrogate is trained on observed objective
    values. Expected Improvement then selects the next feasible
    candidate to evaluate.
    """

    optimization_problem = problem or OptimizationProblem.default()
    optimization_config = config or BayesianOptimizationConfig()

    rng = np.random.default_rng(optimization_config.random_seed)

    observations: list[OptimizationObservation] = []
    seen: set[tuple[float, ...]] = set()

    def evaluate_candidate(
        decision: dict[DecisionVariable, float],
    ) -> None:
        key = tuple(decision[variable] for variable in CONTROLLABLE_VARIABLES)

        if key in seen:
            return

        optimization_problem.validate_decision(decision)

        result = evaluate_objective(
            data=data,
            trace=trace,
            decision=decision,
            problem=optimization_problem,
        )

        observations.append(
            OptimizationObservation(
                decision=decision.copy(),
                objective=result,
            )
        )

        seen.add(key)

    while len(observations) < optimization_config.initial_points:
        evaluate_candidate(
            _random_feasible_decision(
                rng=rng,
                problem=optimization_problem,
            )
        )

    for _ in range(optimization_config.iterations):
        x_train = np.vstack(
            [_decision_to_array(observation.decision) for observation in observations]
        )

        y_train = np.asarray(
            [observation.objective.mean_incremental_installs for observation in observations],
            dtype=float,
        )

        kernel = Matern(
            length_scale=np.ones(len(CONTROLLABLE_VARIABLES)),
            nu=2.5,
        ) + WhiteKernel(
            noise_level=1.0,
            noise_level_bounds=(
                1e-6,
                1e5,
            ),
        )

        surrogate = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=optimization_config.random_seed,
            n_restarts_optimizer=1,
        )

        surrogate.fit(
            x_train,
            y_train,
        )

        candidate_pool = _candidate_pool(
            rng=rng,
            problem=optimization_problem,
            size=2_000,
        )

        x_candidates = np.vstack([_decision_to_array(candidate) for candidate in candidate_pool])

        means, standard_deviations = surrogate.predict(
            x_candidates,
            return_std=True,
        )

        best_value = float(np.max(y_train))

        acquisition = _expected_improvement(
            mean=means,
            standard_deviation=standard_deviations,
            best_value=best_value,
            exploration=optimization_config.exploration,
        )

        order = np.argsort(acquisition)[::-1]

        next_candidate: dict[DecisionVariable, float] | None = None

        for index in order:
            candidate = candidate_pool[int(index)]

            key = tuple(candidate[variable] for variable in CONTROLLABLE_VARIABLES)

            if key not in seen:
                next_candidate = candidate
                break

        if next_candidate is None:
            break

        evaluate_candidate(next_candidate)

    best_observation = max(
        observations,
        key=lambda observation: observation.objective.mean_incremental_installs,
    )

    return BayesianOptimizationResult(
        best_decision=best_observation.decision.copy(),
        best_objective=best_observation.objective,
        observations=tuple(observations),
    )
