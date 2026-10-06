import pandas as pd

from cge.data import DatasetConfig, SyntheticGrowthGenerator


def test_generator_produces_expected_shape() -> None:
    config = DatasetConfig(
        n_titles=5,
        n_days=365,
        seed=42,
    )

    data, truth = SyntheticGrowthGenerator(config).generate()

    assert len(data) == 5 * 365
    assert data["title_id"].nunique() == 5
    assert data["date"].nunique() == 365
    assert truth.paid_ua_effect > 0


def test_generator_has_expected_columns() -> None:
    config = DatasetConfig(n_titles=2, n_days=10)

    data, _ = SyntheticGrowthGenerator(config).generate()

    expected = {
        "date",
        "title_id",
        "paid_ua_spend",
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
    }

    assert expected.issubset(data.columns)


def test_business_ranges_are_realistic() -> None:
    config = DatasetConfig(n_titles=5, n_days=365)

    data, _ = SyntheticGrowthGenerator(config).generate()

    assert data["paid_ua_spend"].between(5_000, 15_000).all()
    assert data["social_media_posts"].between(0, 10).all()
    assert data["social_media_likes"].between(0, 1_000).all()
    assert data["social_media_comments"].between(0, 100).all()
    assert data["product_test_release"].between(0, 5).all()
    assert data["product_version_update"].between(0, 5).all()
    assert data["app_store_rating"].between(3.0, 5.0).all()
    assert data["paid_installs"].between(10_000, 15_000).all()


def test_generator_is_reproducible() -> None:
    config = DatasetConfig(
        n_titles=2,
        n_days=20,
        seed=123,
    )

    data_a, _ = SyntheticGrowthGenerator(config).generate()
    data_b, _ = SyntheticGrowthGenerator(config).generate()

    pd.testing.assert_frame_equal(data_a, data_b)


def test_titles_have_heterogeneous_baselines() -> None:
    config = DatasetConfig(
        n_titles=5,
        n_days=365,
        seed=42,
    )

    data, _ = SyntheticGrowthGenerator(config).generate()

    title_means = data.groupby("title_id")["organic_installs"].mean()

    assert title_means.nunique() > 1
