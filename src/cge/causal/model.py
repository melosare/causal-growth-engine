from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pymc as pm

from cge.causal.dag import TREATMENTS


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the Bayesian organic-growth model."""

    draws: int = 500
    tune: int = 500
    chains: int = 2
    target_accept: float = 0.90
    random_seed: int = 42


@dataclass(frozen=True)
class PreparedData:
    """Numerical representation used by the PyMC model."""

    organic_installs: np.ndarray
    market_demand: np.ndarray
    treatment_matrix: np.ndarray
    title_index: np.ndarray
    n_titles: int


def prepare_data(data: pd.DataFrame) -> PreparedData:
    """Prepare a CGE dataset for Bayesian estimation."""

    required_columns = {
        "organic_installs",
        "market_demand",
        "title_id",
        *TREATMENTS,
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing required model columns: {sorted(missing)}"
        )

    if data.empty:
        raise ValueError("Cannot fit a model to an empty dataset.")

    title_codes, _ = pd.factorize(data["title_id"])

    treatment_matrix = np.column_stack(
        [
            np.log1p(data["paid_ua_spend"].to_numpy() / 5_000.0),
            np.log1p(data["influencer_spend"].to_numpy() / 1_000.0),
            data["social_media_posts"].to_numpy(),
            data["product_test_release"].to_numpy(),
            data["product_version_update"].to_numpy(),
        ]
    )

    return PreparedData(
        organic_installs=data["organic_installs"].to_numpy(dtype=float),
        market_demand=data["market_demand"].to_numpy(dtype=float),
        treatment_matrix=treatment_matrix.astype(float),
        title_index=title_codes.astype(int),
        n_titles=int(len(np.unique(title_codes))),
    )


def build_model(
    prepared: PreparedData,
) -> pm.Model:
    """Build the Bayesian causal model for organic installs."""

    if prepared.n_titles < 1:
        raise ValueError("At least one title is required.")

    with pm.Model() as model:
        """Treatment effect"""

        paid_ua_effect = pm.Normal(
            "paid_ua_effect",
            mu=0.05,
            sigma=0.10,
        )

        influencer_effect = pm.Normal(
            "influencer_effect",
            mu=0.10,
            sigma=0.15,
        )

        social_posts_effect = pm.Normal(
            "social_posts_effect",
            mu=0.03,
            sigma=0.08,
        )

        test_release_effect = pm.Normal(
            "test_release_effect",
            mu=0.03,
            sigma=0.08,
        )

        version_update_effect = pm.Normal(
            "version_update_effect",
            mu=0.05,
            sigma=0.10,
        )

        treatment_effects = pm.math.stack(
            [
                paid_ua_effect,
                influencer_effect,
                social_posts_effect,
                test_release_effect,
                version_update_effect,
            ]
        )

        treatment_signal = pm.math.dot(
            prepared.treatment_matrix,
            treatment_effects,
        )

        """Observed Confounder"""

        demand_effect = pm.Normal(
            "demand_effect",
            mu=1_500.0,
            sigma=750.0,
        )

        demand_signal = (
            demand_effect
            * prepared.market_demand
        )

        """Hierarchical baseline"""

        organic_baseline = pm.Normal(
            "organic_baseline",
            mu=8_000.0,
            sigma=2_000.0,
        )

        title_baseline_sd = pm.HalfNormal(
            "title_baseline_sd",
            sigma=1_000.0,
        )

        title_baseline_offset = pm.Normal(
            "title_baseline_offset",
            mu=0.0,
            sigma=1.0,
            shape=prepared.n_titles,
        )

        title_baseline = (
            organic_baseline
            + title_baseline_offset * title_baseline_sd
        )

        """Organic install mean"""

        mu = (
            title_baseline[prepared.title_index]
            + demand_signal
            + 1_000.0 * treatment_signal
        )

        mu = pm.math.maximum(mu, 500.0)

        """Overdispersed likelihood"""

        dispersion = pm.HalfNormal(
            "dispersion",
            sigma=50.0,
        )

        pm.NegativeBinomial(
            "organic_installs",
            mu=mu,
            alpha=dispersion,
            observed=prepared.organic_installs,
        )

    return model


def fit_model(
    data: pd.DataFrame,
    config: ModelConfig | None = None,
) -> tuple[pm.Model, object]:
    """Prepare data, build the model, and sample the posterior."""

    model_config = config or ModelConfig()
    prepared = prepare_data(data)
    model = build_model(prepared)

    with model:
        trace = pm.sample(
            draws=model_config.draws,
            tune=model_config.tune,
            chains=model_config.chains,
            target_accept=model_config.target_accept,
            random_seed=model_config.random_seed,
            return_inferencedata=True,
        )

    return model, trace
