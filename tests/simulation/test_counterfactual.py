from __future__ import annotations

import numpy as np

from cge.causal.model import ModelConfig, fit_model
from cge.data import DatasetConfig, SyntheticGrowthGenerator
from cge.simulation import (
    TreatmentIntervention,
    compare_strategy,
    simulate_organic_installs,
)


def make_fitted_model():
    """Create a small fitted model for simulation tests."""

    config = DatasetConfig(
        n_titles=2,
        n_days=20,
        seed=42,
    )

    data, _ = SyntheticGrowthGenerator(config).generate()

    model_config = ModelConfig(
        draws=30,
        tune=30,
        chains=1,
        target_accept=0.85,
        random_seed=42,
    )

    _, trace = fit_model(data, model_config)

    return data, trace


def test_simulation_returns_expected_shape() -> None:
    data, trace = make_fitted_model()

    simulations = simulate_organic_installs(
        data=data,
        trace=trace,
    )

    assert simulations.ndim == 2
    assert simulations.shape[1] == len(data)
    assert simulations.shape[0] == 30


def test_simulated_installs_are_nonnegative() -> None:
    data, trace = make_fitted_model()

    simulations = simulate_organic_installs(
        data=data,
        trace=trace,
    )

    assert np.all(simulations >= 0)


def test_intervention_changes_treatment_values() -> None:
    config = DatasetConfig(
        n_titles=1,
        n_days=10,
        seed=42,
    )

    data, _ = SyntheticGrowthGenerator(config).generate()

    intervention = TreatmentIntervention(
        paid_ua_spend=15_000.0,
        influencer_spend=8_000.0,
        social_media_posts=10.0,
        product_test_release=5.0,
        product_version_update=5.0,
    )

    counterfactual = intervention.apply(data)

    assert np.all(counterfactual["paid_ua_spend"] == 15_000.0)
    assert np.all(counterfactual["influencer_spend"] == 8_000.0)
    assert np.all(counterfactual["social_media_posts"] == 10.0)
    assert np.all(counterfactual["product_test_release"] == 5.0)
    assert np.all(counterfactual["product_version_update"] == 5.0)


def test_intervention_preserves_unmodified_columns() -> None:
    config = DatasetConfig(
        n_titles=1,
        n_days=10,
        seed=42,
    )

    data, _ = SyntheticGrowthGenerator(config).generate()

    intervention = TreatmentIntervention(
        paid_ua_spend=15_000.0,
    )

    counterfactual = intervention.apply(data)

    np.testing.assert_array_equal(
        counterfactual["organic_installs"],
        data["organic_installs"],
    )

    np.testing.assert_array_equal(
        counterfactual["market_demand"],
        data["market_demand"],
    )

    np.testing.assert_array_equal(
        counterfactual["influencer_spend"],
        data["influencer_spend"],
    )


def test_compare_strategy_returns_summary() -> None:
    data, trace = make_fitted_model()

    intervention = TreatmentIntervention(
        paid_ua_spend=15_000.0,
    )

    summary = compare_strategy(
        data=data,
        trace=trace,
        intervention=intervention,
    )

    assert summary.baseline_mean > 0
    assert summary.counterfactual_mean > 0
    assert summary.incremental_lower <= summary.incremental_mean
    assert summary.incremental_mean <= summary.incremental_upper
    assert 0.0 <= summary.probability_positive <= 1.0


def test_negative_intervention_is_rejected() -> None:
    config = DatasetConfig(
        n_titles=1,
        n_days=10,
        seed=42,
    )

    data, _ = SyntheticGrowthGenerator(config).generate()

    intervention = TreatmentIntervention(
        paid_ua_spend=-1.0,
    )

    try:
        intervention.apply(data)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Negative treatment intervention should raise ValueError."
        )
