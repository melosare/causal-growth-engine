from __future__ import annotations

import numpy as np

from cge.causal.model import ModelConfig, fit_model
from cge.data import DatasetConfig, SyntheticGrowthGenerator
from cge.simulation.counterfactual import TreatmentIntervention
from cge.simulation.impact import (
    estimate_baseline_installs,
    estimate_incremental_impact,
    summarize_incremental_impact,
)


def make_fitted_model():
    """Create a small fitted model for impact tests."""

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


def test_summarize_incremental_impact() -> None:
    values = np.array(
        [-10.0, 0.0, 10.0, 20.0, 30.0]
    )

    summary = summarize_incremental_impact(
        values,
        credible_interval=0.80,
    )

    assert summary.mean_incremental_installs == 10.0
    assert summary.median_incremental_installs == 10.0
    assert np.isclose(summary.lower, -6.0)
    assert np.isclose(summary.upper, 26.0)
    assert summary.probability_positive == 0.6


def test_incremental_impact_rejects_empty_input() -> None:
    try:
        summarize_incremental_impact(
            np.array([]),
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Empty input should raise ValueError."
        )


def test_incremental_impact_rejects_invalid_interval() -> None:
    try:
        summarize_incremental_impact(
            np.array([1.0, 2.0, 3.0]),
            credible_interval=1.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Invalid credible interval should raise ValueError."
        )


def test_estimate_incremental_impact() -> None:
    data, trace = make_fitted_model()

    intervention = TreatmentIntervention(
        paid_ua_spend=15_000.0,
    )

    summary = estimate_incremental_impact(
        data=data,
        trace=trace,
        intervention=intervention,
    )

    assert np.isfinite(
        summary.mean_incremental_installs
    )
    assert np.isfinite(
        summary.median_incremental_installs
    )
    assert summary.lower <= summary.median_incremental_installs
    assert summary.median_incremental_installs <= summary.upper
    assert 0.0 <= summary.probability_positive <= 1.0


def test_estimate_baseline_installs() -> None:
    data, trace = make_fitted_model()

    baseline = estimate_baseline_installs(
        data=data,
        trace=trace,
    )

    assert np.isfinite(baseline)
    assert baseline >= 0.0
