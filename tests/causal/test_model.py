import numpy as np
import pandas as pd

from cge.causal.model import (
    ModelConfig,
    build_model,
    fit_model,
    prepare_data,
)
from cge.data import DatasetConfig, SyntheticGrowthGenerator


def make_test_data() -> pd.DataFrame:
    """Generate a small deterministic dataset for model tests."""

    config = DatasetConfig(
        n_titles=2,
        n_days=20,
        seed=42,
    )

    data, _ = SyntheticGrowthGenerator(config).generate()

    return data


def test_prepare_data() -> None:
    data = make_test_data()

    prepared = prepare_data(data)

    assert prepared.n_titles == 2
    assert prepared.organic_installs.shape == (40,)
    assert prepared.market_demand.shape == (40,)
    assert prepared.treatment_matrix.shape == (40, 5)
    assert prepared.title_index.shape == (40,)


def test_prepare_data_applies_treatment_transformations() -> None:
    data = make_test_data()

    prepared = prepare_data(data)

    expected_paid_ua = np.log1p(data["paid_ua_spend"].to_numpy() / 5_000.0)

    expected_influencer = np.log1p(data["influencer_spend"].to_numpy() / 1_000.0)

    np.testing.assert_allclose(
        prepared.treatment_matrix[:, 0],
        expected_paid_ua,
    )

    np.testing.assert_allclose(
        prepared.treatment_matrix[:, 1],
        expected_influencer,
    )

    np.testing.assert_allclose(
        prepared.treatment_matrix[:, 2],
        data["social_media_posts"].to_numpy(),
    )

    np.testing.assert_allclose(
        prepared.treatment_matrix[:, 3],
        data["product_test_release"].to_numpy(),
    )

    np.testing.assert_allclose(
        prepared.treatment_matrix[:, 4],
        data["product_version_update"].to_numpy(),
    )


def test_build_model_contains_expected_parameters() -> None:
    data = make_test_data()
    prepared = prepare_data(data)

    model = build_model(prepared)

    expected_variables = {
        "paid_ua_effect",
        "influencer_effect",
        "social_posts_effect",
        "test_release_effect",
        "version_update_effect",
        "demand_effect",
        "organic_baseline",
        "title_baseline_sd",
        "title_baseline_offset",
        "dispersion",
        "organic_installs",
    }

    assert expected_variables.issubset(set(model.named_vars))


def test_model_uses_five_treatment_effects() -> None:
    data = make_test_data()
    prepared = prepare_data(data)

    model = build_model(prepared)

    assert "paid_ua_effect" in model.named_vars
    assert "influencer_effect" in model.named_vars
    assert "social_posts_effect" in model.named_vars
    assert "test_release_effect" in model.named_vars
    assert "version_update_effect" in model.named_vars


def test_fit_model_smoke() -> None:
    data = make_test_data()

    config = ModelConfig(
        draws=20,
        tune=20,
        chains=1,
        target_accept=0.80,
        random_seed=42,
    )

    _, trace = fit_model(data, config)

    assert "paid_ua_effect" in trace.posterior
    assert "influencer_effect" in trace.posterior
    assert "social_posts_effect" in trace.posterior
    assert "test_release_effect" in trace.posterior
    assert "version_update_effect" in trace.posterior
