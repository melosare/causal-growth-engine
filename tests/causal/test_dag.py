from cge.causal import (
    CONFOUNDERS,
    EDGES,
    MEDIATORS,
    OUTCOMES,
    TREATMENTS,
    VARIABLES,
    VariableRole,
    get_children,
    get_parents,
    get_variable,
    validate_dag,
)


def test_dag_is_valid() -> None:
    validate_dag()


def test_expected_treatments() -> None:
    assert TREATMENTS == (
        "paid_ua_spend",
        "influencer_spend",
        "social_media_posts",
        "product_test_release",
        "product_version_update",
    )

    for treatment in TREATMENTS:
        assert get_variable(treatment).role == VariableRole.TREATMENT


def test_expected_mediators() -> None:
    assert MEDIATORS == (
        "influencer_contracts",
        "paid_installs",
        "social_media_likes",
        "social_media_comments",
        "app_store_rating",
    )

    for mediator in MEDIATORS:
        assert get_variable(mediator).role == VariableRole.MEDIATOR


def test_expected_outcomes() -> None:
    assert OUTCOMES == (
        "organic_installs",
        "engagement",
        "retention",
        "revenue",
    )

    for outcome in OUTCOMES:
        assert get_variable(outcome).role == VariableRole.OUTCOME


def test_market_demand_is_a_confounder() -> None:
    assert CONFOUNDERS == ("market_demand",)

    assert "paid_ua_spend" in get_children("market_demand")
    assert "organic_installs" in get_children("market_demand")


def test_paid_ua_has_direct_and_mediated_paths() -> None:
    paid_ua_children = get_children("paid_ua_spend")

    assert "paid_installs" in paid_ua_children
    assert "organic_installs" in paid_ua_children


def test_product_changes_affect_rating() -> None:
    rating_parents = get_parents("app_store_rating")

    assert "product_test_release" in rating_parents
    assert "product_version_update" in rating_parents


def test_rating_affects_organic_installs() -> None:
    assert "app_store_rating" in get_parents("organic_installs")


def test_growth_funnel() -> None:
    assert "organic_installs" in get_parents("engagement")
    assert "engagement" in get_parents("retention")
    assert "engagement" in get_parents("revenue")
    assert "retention" in get_parents("revenue")


def test_all_edge_endpoints_are_defined() -> None:
    variable_names = {variable.name for variable in VARIABLES}

    for edge in EDGES:
        assert edge.source in variable_names
        assert edge.target in variable_names
