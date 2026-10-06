from __future__ import annotations

import numpy as np

from cge.causal.model import ModelConfig, fit_model
from cge.data import DatasetConfig, GroundTruth, SyntheticGrowthGenerator

EXPECTED_EFFECTS = {
    "paid_ua_effect": GroundTruth.paid_ua_effect,
    "influencer_effect": GroundTruth.influencer_effect,
    "social_posts_effect": GroundTruth.social_posts_effect,
    "test_release_effect": GroundTruth.test_release_effect,
    "version_update_effect": GroundTruth.version_update_effect,
}


def test_posterior_recovers_ground_truth_effects() -> None:
    """Check that the model can approximately recover known treatment effects."""

    config = DatasetConfig(
        n_titles=5,
        n_days=365,
        seed=42,
    )

    data, truth = SyntheticGrowthGenerator(config).generate()

    model_config = ModelConfig(
        draws=300,
        tune=300,
        chains=2,
        target_accept=0.90,
        random_seed=42,
    )

    _, trace = fit_model(data, model_config)

    for parameter_name, expected_value in EXPECTED_EFFECTS.items():
        posterior = trace.posterior[parameter_name].values

        posterior_mean = float(np.mean(posterior))
        posterior_lower = float(np.quantile(posterior, 0.05))
        posterior_upper = float(np.quantile(posterior, 0.95))

        true_value = getattr(
            truth,
            {
                "paid_ua_effect": "paid_ua_effect",
                "influencer_effect": "influencer_effect",
                "social_posts_effect": "social_posts_effect",
                "test_release_effect": "test_release_effect",
                "version_update_effect": "version_update_effect",
            }[parameter_name],
        )

        assert posterior_lower < true_value < posterior_upper, (
            f"{parameter_name}: true value {true_value:.4f} "
            f"is outside posterior interval "
            f"[{posterior_lower:.4f}, {posterior_upper:.4f}]"
        )

        assert np.isclose(
            posterior_mean,
            expected_value,
            rtol=0.75,
            atol=0.02,
        ), (
            f"{parameter_name}: posterior mean {posterior_mean:.4f} "
            f"was not sufficiently close to true value "
            f"{expected_value:.4f}"
        )
