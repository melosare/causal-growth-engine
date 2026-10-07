from __future__ import annotations

import pytest

from cge.optimization.problem import (
    CONTROLLABLE_VARIABLES,
    DecisionVariable,
    OptimizationConstraints,
    OptimizationProblem,
    VariableBounds,
)


def test_default_problem_contains_only_controllable_variables() -> None:
    problem = OptimizationProblem.default()

    assert set(problem.bounds) == {
        DecisionVariable.PAID_UA_SPEND,
        DecisionVariable.INFLUENCER_SPEND,
        DecisionVariable.SOCIAL_MEDIA_POSTS,
        DecisionVariable.PRODUCT_TEST_RELEASE,
        DecisionVariable.PRODUCT_VERSION_UPDATE,
    }


def test_controllable_variables_match_default_bounds() -> None:
    problem = OptimizationProblem.default()

    assert set(CONTROLLABLE_VARIABLES) == set(problem.bounds)


def test_default_bounds_match_real_world_ranges() -> None:
    problem = OptimizationProblem.default()

    paid_ua = problem.bounds[DecisionVariable.PAID_UA_SPEND]
    assert paid_ua.minimum == 5_000.0
    assert paid_ua.maximum == 15_000.0
    assert paid_ua.step == 500.0

    influencer = problem.bounds[DecisionVariable.INFLUENCER_SPEND]
    assert influencer.minimum == 0.0
    assert influencer.maximum == 8_000.0

    posts = problem.bounds[DecisionVariable.SOCIAL_MEDIA_POSTS]
    assert posts.minimum == 0.0
    assert posts.maximum == 10.0


def test_valid_decision_passes() -> None:
    problem = OptimizationProblem.default()

    decision = {
        DecisionVariable.PAID_UA_SPEND: 10_000.0,
        DecisionVariable.INFLUENCER_SPEND: 4_000.0,
        DecisionVariable.SOCIAL_MEDIA_POSTS: 5.0,
        DecisionVariable.PRODUCT_TEST_RELEASE: 2.0,
        DecisionVariable.PRODUCT_VERSION_UPDATE: 1.0,
    }

    problem.validate_decision(decision)


def test_decision_rejects_value_above_maximum() -> None:
    problem = OptimizationProblem.default()

    decision = {
        DecisionVariable.PAID_UA_SPEND: 20_000.0,
        DecisionVariable.INFLUENCER_SPEND: 0.0,
        DecisionVariable.SOCIAL_MEDIA_POSTS: 5.0,
        DecisionVariable.PRODUCT_TEST_RELEASE: 2.0,
        DecisionVariable.PRODUCT_VERSION_UPDATE: 1.0,
    }

    with pytest.raises(ValueError, match="exceeds its maximum"):
        problem.validate_decision(decision)


def test_decision_rejects_invalid_step() -> None:
    problem = OptimizationProblem.default()

    decision = {
        DecisionVariable.PAID_UA_SPEND: 10_250.0,
        DecisionVariable.INFLUENCER_SPEND: 0.0,
        DecisionVariable.SOCIAL_MEDIA_POSTS: 5.0,
        DecisionVariable.PRODUCT_TEST_RELEASE: 2.0,
        DecisionVariable.PRODUCT_VERSION_UPDATE: 1.0,
    }

    with pytest.raises(ValueError, match="increments"):
        problem.validate_decision(decision)


def test_decision_rejects_missing_variable() -> None:
    problem = OptimizationProblem.default()

    decision = {
        DecisionVariable.PAID_UA_SPEND: 10_000.0,
        DecisionVariable.INFLUENCER_SPEND: 0.0,
        DecisionVariable.SOCIAL_MEDIA_POSTS: 5.0,
        DecisionVariable.PRODUCT_TEST_RELEASE: 2.0,
    }

    with pytest.raises(ValueError, match="Missing decision variables"):
        problem.validate_decision(decision)


def test_decision_rejects_budget_violation() -> None:
    problem = OptimizationProblem.default()

    decision = {
        DecisionVariable.PAID_UA_SPEND: 15_000.0,
        DecisionVariable.INFLUENCER_SPEND: 8_000.0,
        DecisionVariable.SOCIAL_MEDIA_POSTS: 5.0,
        DecisionVariable.PRODUCT_TEST_RELEASE: 2.0,
        DecisionVariable.PRODUCT_VERSION_UPDATE: 1.0,
    }

    with pytest.raises(ValueError, match="daily budget"):
        problem.validate_decision(decision)


def test_variable_bounds_reject_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        VariableBounds(
            minimum=10.0,
            maximum=5.0,
            step=1.0,
        )


def test_constraints_reject_negative_budget() -> None:
    with pytest.raises(ValueError):
        OptimizationConstraints(
            daily_budget=-1.0,
        )
