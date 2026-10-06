from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import arviz as az
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TreatmentIntervention:
    """Counterfactual treatment values for the five controllable drivers."""

    paid_ua_spend: float | None = None
    influencer_spend: float | None = None
    social_media_posts: float | None = None
    product_test_release: float | None = None
    product_version_update: float | None = None

    def apply(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return a copy of the data with intervention values applied."""

        result = data.copy()

        values = {
            "paid_ua_spend": self.paid_ua_spend,
            "influencer_spend": self.influencer_spend,
            "social_media_posts": self.social_media_posts,
            "product_test_release": self.product_test_release,
            "product_version_update": self.product_version_update,
        }

        for column, value in values.items():
            if value is not None:
                if value < 0:
                    raise ValueError(
                        f"{column} intervention cannot be negative."
                    )

                result[column] = value

        return result


@dataclass(frozen=True)
class CounterfactualSummary:
    """Summary statistics for a counterfactual scenario."""

    baseline_mean: float
    counterfactual_mean: float
    incremental_mean: float
    incremental_lower: float
    incremental_upper: float
    probability_positive: float


def _posterior_values(
    trace: az.InferenceData,
    variable: str,
) -> np.ndarray:
    """Flatten posterior chains and draws into one sample dimension."""

    posterior: Any = cast(Any, trace).posterior

    if variable not in posterior:
        raise ValueError(
            f"Posterior does not contain required variable: {variable}"
        )

    values = np.asarray(posterior[variable].values)

    return values.reshape(-1, *values.shape[2:])


def _build_treatment_matrix(
    data: pd.DataFrame,
) -> np.ndarray:
    """Apply the same treatment transformations used by the causal model."""

    return np.column_stack(
        [
            np.log1p(
                data["paid_ua_spend"].to_numpy(dtype=float)
                / 5_000.0
            ),
            np.log1p(
                data["influencer_spend"].to_numpy(dtype=float)
                / 1_000.0
            ),
            data["social_media_posts"].to_numpy(dtype=float),
            data["product_test_release"].to_numpy(dtype=float),
            data["product_version_update"].to_numpy(dtype=float),
        ]
    )


def simulate_organic_installs(
    data: pd.DataFrame,
    trace: az.InferenceData,
    intervention: TreatmentIntervention | None = None,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Simulate organic installs under a treatment strategy.

    Returns an array with shape:

        (posterior_draws, observations)

    Each row represents one posterior draw and each column represents
    one observation in the supplied dataset.
    """

    if data.empty:
        raise ValueError("Cannot simulate an empty dataset.")

    if intervention is not None:
        data = intervention.apply(data)

    title_codes, _ = pd.factorize(data["title_id"])

    organic_baseline = _posterior_values(
        trace,
        "organic_baseline",
    ).reshape(-1)

    title_baseline_sd = _posterior_values(
        trace,
        "title_baseline_sd",
    ).reshape(-1)

    title_baseline_offset = _posterior_values(
        trace,
        "title_baseline_offset",
    )

    title_baseline = (
        organic_baseline[:, None]
        + title_baseline_offset * title_baseline_sd[:, None]
    )

    demand_effect = _posterior_values(
        trace,
        "demand_effect",
    ).reshape(-1)

    paid_ua_effect = _posterior_values(
        trace,
        "paid_ua_effect",
    ).reshape(-1)

    influencer_effect = _posterior_values(
        trace,
        "influencer_effect",
    ).reshape(-1)

    social_posts_effect = _posterior_values(
        trace,
        "social_posts_effect",
    ).reshape(-1)

    test_release_effect = _posterior_values(
        trace,
        "test_release_effect",
    ).reshape(-1)

    version_update_effect = _posterior_values(
        trace,
        "version_update_effect",
    ).reshape(-1)

    dispersion = _posterior_values(
        trace,
        "dispersion",
    ).reshape(-1)

    n_draws = demand_effect.shape[0]

    treatment_matrix = _build_treatment_matrix(data)

    demand = data["market_demand"].to_numpy(dtype=float)

    treatment_signal = (
        treatment_matrix[:, 0][None, :]
        * paid_ua_effect[:, None]
        + treatment_matrix[:, 1][None, :]
        * influencer_effect[:, None]
        + treatment_matrix[:, 2][None, :]
        * social_posts_effect[:, None]
        + treatment_matrix[:, 3][None, :]
        * test_release_effect[:, None]
        + treatment_matrix[:, 4][None, :]
        * version_update_effect[:, None]
    )

    means = (
        title_baseline[:, title_codes]
        + demand_effect[:, None] * demand[None, :]
        + 1_000.0 * treatment_signal
    )

    means = np.maximum(means, 500.0)

    rng = np.random.default_rng(random_seed)

    n_parameter = np.maximum(dispersion, 1e-6)

    probabilities = (
        n_parameter[:, None]
        / (
            n_parameter[:, None]
            + means
        )
    )

    return rng.negative_binomial(
        n=n_parameter[:, None],
        p=probabilities,
        size=(n_draws, data.shape[0]),
    )


def summarize_counterfactual(
    baseline_simulation: np.ndarray,
    counterfactual_simulation: np.ndarray,
) -> CounterfactualSummary:
    """Summarize incremental organic installs between two scenarios."""

    if baseline_simulation.shape != counterfactual_simulation.shape:
        raise ValueError(
            "Baseline and counterfactual simulations must have "
            "the same shape."
        )

    baseline_totals = baseline_simulation.sum(axis=1)
    counterfactual_totals = counterfactual_simulation.sum(axis=1)

    incremental = counterfactual_totals - baseline_totals

    return CounterfactualSummary(
        baseline_mean=float(np.mean(baseline_totals)),
        counterfactual_mean=float(np.mean(counterfactual_totals)),
        incremental_mean=float(np.mean(incremental)),
        incremental_lower=float(np.quantile(incremental, 0.05)),
        incremental_upper=float(np.quantile(incremental, 0.95)),
        probability_positive=float(np.mean(incremental > 0.0)),
    )


def compare_strategy(
    data: pd.DataFrame,
    trace: az.InferenceData,
    intervention: TreatmentIntervention,
    random_seed: int = 42,
) -> CounterfactualSummary:
    """Compare the observed strategy against a counterfactual strategy."""

    baseline_simulation = simulate_organic_installs(
        data=data,
        trace=trace,
        random_seed=random_seed,
    )

    counterfactual_simulation = simulate_organic_installs(
        data=data,
        trace=trace,
        intervention=intervention,
        random_seed=random_seed + 1,
    )

    return summarize_counterfactual(
        baseline_simulation=baseline_simulation,
        counterfactual_simulation=counterfactual_simulation,
    )
