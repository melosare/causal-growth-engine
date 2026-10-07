from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DecisionVariable(StrEnum):
    """Variables that the optimizer is allowed to control."""

    PAID_UA_SPEND = "paid_ua_spend"
    INFLUENCER_SPEND = "influencer_spend"
    SOCIAL_MEDIA_POSTS = "social_media_posts"
    PRODUCT_TEST_RELEASE = "product_test_release"
    PRODUCT_VERSION_UPDATE = "product_version_update"


@dataclass(frozen=True)
class VariableBounds:
    """Operational bounds for one decision variable."""

    minimum: float
    maximum: float
    step: float

    def __post_init__(self) -> None:
        if self.minimum < 0:
            raise ValueError("minimum must be non-negative.")

        if self.maximum < self.minimum:
            raise ValueError("maximum must be greater than or equal to minimum.")

        if self.step <= 0:
            raise ValueError("step must be positive.")

        if self.minimum + self.step > self.maximum and self.minimum != self.maximum:
            raise ValueError("step must not exceed the variable range.")


DEFAULT_BOUNDS: dict[DecisionVariable, VariableBounds] = {
    DecisionVariable.PAID_UA_SPEND: VariableBounds(
        minimum=5_000.0,
        maximum=15_000.0,
        step=500.0,
    ),
    DecisionVariable.INFLUENCER_SPEND: VariableBounds(
        minimum=0.0,
        maximum=8_000.0,
        step=1_000.0,
    ),
    DecisionVariable.SOCIAL_MEDIA_POSTS: VariableBounds(
        minimum=0.0,
        maximum=10.0,
        step=1.0,
    ),
    DecisionVariable.PRODUCT_TEST_RELEASE: VariableBounds(
        minimum=0.0,
        maximum=5.0,
        step=1.0,
    ),
    DecisionVariable.PRODUCT_VERSION_UPDATE: VariableBounds(
        minimum=0.0,
        maximum=5.0,
        step=1.0,
    ),
}


@dataclass(frozen=True)
class OptimizationConstraints:
    """Constraints applied to the optimization problem."""

    daily_budget: float = 20_000.0
    influencer_monthly_budget: float = 8_000.0

    def __post_init__(self) -> None:
        if self.daily_budget <= 0:
            raise ValueError("daily_budget must be positive.")

        if self.influencer_monthly_budget < 0:
            raise ValueError("influencer_monthly_budget must be non-negative.")


@dataclass(frozen=True)
class OptimizationProblem:
    """Definition of the controllable decision space."""

    bounds: dict[DecisionVariable, VariableBounds]
    constraints: OptimizationConstraints

    @classmethod
    def default(cls) -> OptimizationProblem:
        """Create the default CGE optimization problem."""

        return cls(
            bounds=DEFAULT_BOUNDS.copy(),
            constraints=OptimizationConstraints(),
        )

    def validate_decision(
        self,
        decision: dict[DecisionVariable, float],
    ) -> None:
        """Validate a proposed decision against the problem definition."""

        expected_variables = set(self.bounds)
        supplied_variables = set(decision)

        missing = expected_variables - supplied_variables
        if missing:
            raise ValueError(
                f"Missing decision variables: {sorted(variable.value for variable in missing)}"
            )

        unexpected = supplied_variables - expected_variables
        if unexpected:
            raise ValueError(
                f"Unexpected decision variables: "
                f"{sorted(variable.value for variable in unexpected)}"
            )

        for variable, value in decision.items():
            variable_bounds = self.bounds[variable]

            if value < variable_bounds.minimum:
                raise ValueError(
                    f"{variable.value} is below its minimum of {variable_bounds.minimum}."
                )

            if value > variable_bounds.maximum:
                raise ValueError(
                    f"{variable.value} exceeds its maximum of {variable_bounds.maximum}."
                )

            scaled_step = (value - variable_bounds.minimum) / variable_bounds.step

            if abs(scaled_step - round(scaled_step)) > 1e-9:
                raise ValueError(f"{variable.value} must use increments of {variable_bounds.step}.")

        paid_ua_spend = decision[DecisionVariable.PAID_UA_SPEND]
        influencer_spend = decision[DecisionVariable.INFLUENCER_SPEND]

        if paid_ua_spend + influencer_spend > self.constraints.daily_budget:
            raise ValueError(
                "Paid UA spend plus influencer spend exceeds "
                f"the daily budget of "
                f"{self.constraints.daily_budget}."
            )

        if influencer_spend > self.constraints.influencer_monthly_budget:
            raise ValueError(
                "Influencer spend exceeds the monthly influencer "
                f"budget of "
                f"{self.constraints.influencer_monthly_budget}."
            )


CONTROLLABLE_VARIABLES = tuple(DEFAULT_BOUNDS)
