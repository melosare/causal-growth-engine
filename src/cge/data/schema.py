from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

DATA_COLUMNS = [
    "date",
    "title_id",
    "product_age_days",
    "market_demand",
    "paid_ua_spend",
    "influencer_contracts",
    "influencer_spend",
    "social_media_posts",
    "social_media_likes",
    "social_media_comments",
    "product_test_release",
    "product_version_update",
    "app_store_rating",
    "paid_installs",
    "organic_installs",
    "engagement",
    "retention",
    "revenue",
]


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for the synthetic CGE dataset."""

    n_titles: int = 5
    n_days: int = 365
    start_date: date = date(2025, 1, 1)
    seed: int = 42


def validate_dataset(data: pd.DataFrame) -> None:
    """Validate the canonical CGE dataset contract."""

    missing = set(DATA_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if data.empty:
        raise ValueError("Dataset must not be empty.")

    if data["title_id"].nunique() < 1:
        raise ValueError("Dataset must contain at least one title.")

    if data["date"].duplicated().all():
        raise ValueError("Dataset dates appear to be invalid.")

    numeric_columns = [column for column in DATA_COLUMNS if column not in {"date", "title_id"}]

    non_numeric = [
        column for column in numeric_columns if not pd.api.types.is_numeric_dtype(data[column])
    ]

    if non_numeric:
        raise TypeError(f"Expected numeric columns: {non_numeric}")

    bounded_columns = {
        "app_store_rating": (3.0, 5.0),
        "retention": (0.0, 1.0),
    }

    for column, (lower, upper) in bounded_columns.items():
        if ((data[column] < lower) | (data[column] > upper)).any():
            raise ValueError(f"{column} contains values outside [{lower}, {upper}].")

    nonnegative_columns = [
        column for column in numeric_columns if column not in {"app_store_rating"}
    ]

    for column in nonnegative_columns:
        if (data[column] < 0).any():
            raise ValueError(f"{column} contains negative values.")
