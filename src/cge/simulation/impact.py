from __future__ import annotations

from dataclasses import dataclass

import arviz as az
import numpy as np
import pandas as pd

from cge.simulation.counterfactual import (
    TreatmentIntervention,
    compare_strategy,
    simulate_organic_installs,
)


@dataclass(frozen=True)
class ImpactSummary:
    """Summary of posterior incremental business impact."""

    mean_incremental_installs: float
    median_incremental_installs: float
    lower: float
    upper: float
    probability_positive: float


def summarize_incremental_impact(
    incremental_installs: np.ndarray,
    credible_interval: float = 0.90,
) -> ImpactSummary:
    """Summarize a posterior distribution of incremental installs."""

    if incremental_installs.size == 0:
        raise ValueError("incremental_installs must not be empty.")

    if not 0.0 < credible_interval < 1.0:
        raise ValueError("credible_interval must be between 0 and 1.")

    values = np.asarray(incremental_installs, dtype=float).reshape(-1)

    tail_probability = (1.0 - credible_interval) / 2.0

    return ImpactSummary(
        mean_incremental_installs=float(np.mean(values)),
        median_incremental_installs=float(np.median(values)),
        lower=float(np.quantile(values, tail_probability)),
        upper=float(
            np.quantile(
                values,
                1.0 - tail_probability,
            )
        ),
        probability_positive=float(np.mean(values > 0.0)),
    )


def estimate_incremental_impact(
    data: pd.DataFrame,
    trace: az.InferenceData,
    intervention: TreatmentIntervention,
    credible_interval: float = 0.90,
) -> ImpactSummary:
    """Estimate posterior business impact for an intervention."""

    comparison = compare_strategy(
        data=data,
        trace=trace,
        intervention=intervention,
    )

    return summarize_incremental_impact(
        incremental_installs=np.asarray(
            comparison.incremental_mean,
            dtype=float,
        ),
        credible_interval=credible_interval,
    )


def estimate_baseline_installs(
    data: pd.DataFrame,
    trace: az.InferenceData,
) -> float:
    """Estimate the posterior mean of baseline organic installs."""

    simulations = simulate_organic_installs(
        data=data,
        trace=trace,
    )

    return float(np.mean(simulations))
