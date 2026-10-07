from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cge.data.schema import DATA_COLUMNS, DatasetConfig, validate_dataset


@dataclass(frozen=True)
class GroundTruth:
    """True causal parameters used to generate the synthetic world."""

    paid_ua_effect: float = 0.035
    influencer_effect: float = 0.12
    social_posts_effect: float = 0.025
    test_release_effect: float = 0.035
    version_update_effect: float = 0.055
    organic_baseline: float = 8_000.0
    organic_demand_effect: float = 1_500.0
    revenue_per_install: float = 2.75


class SyntheticGrowthGenerator:
    """Generate a realistic synthetic causal-growth dataset."""

    def __init__(
        self,
        config: DatasetConfig | None = None,
        truth: GroundTruth | None = None,
    ) -> None:
        self.config = config or DatasetConfig()
        self.truth = truth or GroundTruth()
        self.rng = np.random.default_rng(self.config.seed)

    def generate(self) -> tuple[pd.DataFrame, GroundTruth]:
        """Generate observations and return the known ground truth."""

        dates = pd.date_range(
            start=self.config.start_date,
            periods=self.config.n_days,
            freq="D",
        )

        rows: list[dict[str, object]] = []

        title_baselines = self.rng.lognormal(
            mean=np.log(1.0),
            sigma=0.18,
            size=self.config.n_titles,
        )

        title_marketing_efficiency = self.rng.lognormal(
            mean=0.0,
            sigma=0.15,
            size=self.config.n_titles,
        )

        previous_rating = np.full(self.config.n_titles, 4.45)

        for title_index in range(self.config.n_titles):
            title_id = f"title_{title_index + 1:02d}"

            for day_index, current_date in enumerate(dates):
                product_age = day_index

                seasonality = (
                    1.0
                    + 0.08 * np.sin(2.0 * np.pi * day_index / 365.0)
                    + 0.035 * np.sin(2.0 * np.pi * day_index / 7.0)
                )

                demand_shock = self.rng.normal(0.0, 0.055)
                market_demand = max(
                    0.5,
                    seasonality * (1.0 + demand_shock),
                )

                baseline = title_baselines[title_index]

                # Marketing decisions are partially endogenous:
                # higher underlying demand increases expected marketing activity.
                demand_pressure = np.clip(
                    market_demand,
                    0.75,
                    1.30,
                )

                paid_ua_spend = np.clip(
                    10_000.0 * demand_pressure * title_marketing_efficiency[title_index]
                    + self.rng.normal(0.0, 700.0),
                    5_000.0,
                    15_000.0,
                )

                influencer_contracts = int(
                    self.rng.binomial(
                        2,
                        np.clip(0.10 * demand_pressure, 0.02, 0.20),
                    )
                )

                if influencer_contracts > 0:
                    contract_value = self.rng.uniform(2_000.0, 4_000.0)
                    influencer_spend = influencer_contracts * contract_value
                else:
                    influencer_spend = 0.0

                social_media_posts = int(self.rng.poisson(np.clip(3.0 * demand_pressure, 0.5, 8.0)))
                social_media_posts = min(social_media_posts, 10)

                product_test_release = int(self.rng.poisson(0.5))
                product_test_release = min(product_test_release, 5)

                product_version_update = int(self.rng.poisson(0.12))
                product_version_update = min(product_version_update, 5)

                paid_installs = np.clip(
                    10_000.0 + 0.55 * (paid_ua_spend - 5_000.0) + self.rng.normal(0.0, 700.0),
                    10_000.0,
                    15_000.0,
                )

                # Product events influence the underlying rating, but rating
                # has temporal inertia and therefore changes slowly.
                rating_target = np.clip(
                    previous_rating[title_index]
                    + 0.045 * product_version_update
                    + 0.025 * product_test_release
                    + self.rng.normal(0.0, 0.025),
                    3.0,
                    5.0,
                )

                current_rating = 0.92 * previous_rating[title_index] + 0.08 * rating_target

                current_rating = float(np.clip(current_rating, 3.0, 5.0))

                previous_rating[title_index] = current_rating

                social_activity = max(
                    0.1,
                    (
                        0.7 * social_media_posts
                        + 0.000015 * paid_installs
                        + 0.0008 * influencer_spend
                    ),
                )

                social_media_likes = int(
                    np.clip(
                        self.rng.poisson(65.0 + 32.0 * social_activity + 40.0 * market_demand),
                        0,
                        1_000,
                    )
                )

                social_media_comments = int(
                    np.clip(
                        self.rng.poisson(7.0 + 0.09 * social_media_likes),
                        0,
                        100,
                    )
                )

                treatment_signal = (
                    self.truth.paid_ua_effect * np.log1p(paid_ua_spend / 5_000.0)
                    + self.truth.influencer_effect * np.log1p(influencer_spend / 1_000.0)
                    + self.truth.social_posts_effect * social_media_posts
                    + self.truth.test_release_effect * product_test_release
                    + self.truth.version_update_effect * product_version_update
                )

                demand_signal = self.truth.organic_demand_effect * market_demand

                organic_mean = (
                    self.truth.organic_baseline * baseline
                    + demand_signal
                    + 1_000.0 * treatment_signal
                    + 180.0 * np.log1p(social_media_likes)
                    + 75.0 * np.log1p(social_media_comments)
                    + 900.0 * (current_rating - 4.0)
                )

                organic_mean = max(500.0, organic_mean)

                organic_installs = float(
                    self.rng.negative_binomial(
                        n=50,
                        p=50 / (50 + organic_mean),
                    )
                )

                engagement = float(
                    np.clip(
                        0.18
                        + 0.000012 * organic_installs
                        + 0.025 * (current_rating - 4.0)
                        + self.rng.normal(0.0, 0.025),
                        0.05,
                        0.95,
                    )
                )

                retention = float(
                    np.clip(
                        0.20
                        + 0.22 * engagement
                        + 0.025 * (current_rating - 4.0)
                        + self.rng.normal(0.0, 0.015),
                        0.05,
                        0.80,
                    )
                )

                revenue_mean = (
                    self.truth.revenue_per_install
                    * (organic_installs + 0.15 * paid_installs)
                    * (0.65 + 0.65 * engagement + 0.35 * retention)
                )

                revenue = float(
                    max(
                        0.0,
                        revenue_mean
                        + self.rng.normal(
                            0.0,
                            max(100.0, 0.08 * revenue_mean),
                        ),
                    )
                )

                rows.append(
                    {
                        "date": current_date.date(),
                        "title_id": title_id,
                        "product_age_days": product_age,
                        "market_demand": market_demand,
                        "paid_ua_spend": paid_ua_spend,
                        "influencer_contracts": influencer_contracts,
                        "influencer_spend": influencer_spend,
                        "social_media_posts": social_media_posts,
                        "social_media_likes": social_media_likes,
                        "social_media_comments": social_media_comments,
                        "product_test_release": product_test_release,
                        "product_version_update": product_version_update,
                        "app_store_rating": current_rating,
                        "paid_installs": paid_installs,
                        "organic_installs": organic_installs,
                        "engagement": engagement,
                        "retention": retention,
                        "revenue": revenue,
                    }
                )

        data = pd.DataFrame(rows, columns=DATA_COLUMNS)
        validate_dataset(data)

        return data, self.truth
