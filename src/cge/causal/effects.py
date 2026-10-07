from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import arviz as az
import numpy as np
import pandas as pd

EFFECT_PARAMETERS = {
    "paid_ua_spend": "paid_ua_effect",
    "influencer_spend": "influencer_effect",
    "social_media_posts": "social_posts_effect",
    "product_test_release": "test_release_effect",
    "product_version_update": "version_update_effect",
}


@dataclass(frozen=True)
class PosteriorEffectSummary:
    """Summary statistics for one posterior causal effect."""

    treatment: str
    parameter: str
    mean: float
    lower: float
    upper: float
    probability_positive: float


def _posterior_values(
    trace: az.InferenceData,
    parameter: str,
) -> np.ndarray:
    """Extract and flatten posterior draws for one parameter."""

    posterior: Any = cast(Any, trace).posterior

    if parameter not in posterior:
        raise ValueError(f"Posterior does not contain required parameter: {parameter}")

    values = np.asarray(posterior[parameter].values)

    return values.reshape(-1)


def posterior_effect_summary(
    trace: az.InferenceData,
    treatment: str,
    parameter: str,
    credible_interval: float = 0.90,
) -> PosteriorEffectSummary:
    """Summarize one treatment's posterior causal effect."""

    if treatment not in EFFECT_PARAMETERS:
        raise ValueError(f"Unknown treatment: {treatment}")

    expected_parameter = EFFECT_PARAMETERS[treatment]

    if parameter != expected_parameter:
        raise ValueError(
            f"Treatment '{treatment}' expects parameter '{expected_parameter}', got '{parameter}'."
        )

    if not 0.0 < credible_interval < 1.0:
        raise ValueError("credible_interval must be between 0 and 1.")

    values = _posterior_values(
        trace,
        parameter,
    )

    tail_probability = (1.0 - credible_interval) / 2.0

    lower = float(
        np.quantile(
            values,
            tail_probability,
        )
    )

    upper = float(
        np.quantile(
            values,
            1.0 - tail_probability,
        )
    )

    return PosteriorEffectSummary(
        treatment=treatment,
        parameter=parameter,
        mean=float(np.mean(values)),
        lower=lower,
        upper=upper,
        probability_positive=float(np.mean(values > 0.0)),
    )


def summarize_all_effects(
    trace: az.InferenceData,
    credible_interval: float = 0.90,
) -> pd.DataFrame:
    """Return posterior summaries for all modeled treatments."""

    summaries = [
        posterior_effect_summary(
            trace=trace,
            treatment=treatment,
            parameter=parameter,
            credible_interval=credible_interval,
        )
        for treatment, parameter in EFFECT_PARAMETERS.items()
    ]

    return pd.DataFrame(
        [
            {
                "treatment": summary.treatment,
                "parameter": summary.parameter,
                "mean": summary.mean,
                "lower": summary.lower,
                "upper": summary.upper,
                "probability_positive": summary.probability_positive,
            }
            for summary in summaries
        ]
    )


def effect_table(
    trace: az.InferenceData,
    credible_interval: float = 0.90,
) -> pd.DataFrame:
    """Return a presentation-ready causal effect table."""

    effects = summarize_all_effects(
        trace=trace,
        credible_interval=credible_interval,
    )

    return effects[
        [
            "treatment",
            "mean",
            "lower",
            "upper",
            "probability_positive",
        ]
    ].copy()
