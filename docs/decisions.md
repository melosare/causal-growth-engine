# Causal Growth Engine — Architecture Decisions

## ADR-001: Separate interventions from mediators

### Decision

The CGE will distinguish between variables that are directly controllable
interventions and variables that are downstream mechanisms or outcomes.

### Direct interventions

The initial optimization layer will control:

- paid_ua_spend
- influencer_spend
- social_media_posts
- product_test_release
- product_version_update

### Mediators / downstream variables

The initial model will treat the following as mechanisms or state variables:

- paid_installs
- social_media_likes
- social_media_comments
- app_store_rating

### Outcomes

Primary:

- organic_installs

Downstream:

- engagement
- retention
- revenue

### Rationale

Treating every observed variable as an independent treatment would create
causal identification problems.

For example:

    paid_ua_spend → paid_installs → organic_installs

If paid_installs is conditioned on while estimating the total effect of
paid_ua_spend, part of the treatment effect may be removed.

Similarly:

    product_update → app_store_rating → organic_installs

means that app_store_rating should not automatically be considered an
independent intervention.

The architecture therefore explicitly represents causal roles.

---

## ADR-002: Optimize total causal effects

### Decision

Budget optimization will initially use the total causal effect of each
controllable intervention rather than only its direct effect.

### Rationale

The business decision concerns the total incremental value created by an
intervention.

For example, paid UA may create value through:

    paid_ua_spend
        ↓
    paid_installs
        ↓
    social activity
        ↓
    organic installs
        ↓
    revenue

Removing these mediated pathways would underestimate the economic value of
the intervention.

---

## ADR-003: Use Bayesian modeling

### Decision

PyMC will be used for the primary probabilistic causal models.

### Rationale

The decision-support system needs explicit uncertainty around:

- causal effects
- counterfactual outcomes
- incremental revenue
- budget recommendations

Point estimates alone are insufficient for high-impact allocation decisions.

---

## ADR-004: Validate against a known synthetic data-generating process

### Decision

The initial CGE will be validated against synthetic data generated from known
causal parameters.

### Rationale

The synthetic system allows us to determine whether:

- the implementation works
- the model can recover known effects
- uncertainty behaves appropriately
- optimization responds correctly to known changes

This is an implementation validation mechanism, not evidence that the real
world satisfies the same causal assumptions.

---

## ADR-005: Separate causal modeling from the dashboard

### Decision

The dashboard will not contain causal inference or optimization logic.

The architecture will separate:

    UI
     ↓
    application/service layer
     ↓
    causal model / simulation / optimization
     ↓
    data layer

### Rationale

This allows the modeling engine to be tested independently and prevents
business logic from becoming coupled to the presentation layer.

---

## ADR-006: Support heterogeneous product effects

### Decision

The initial synthetic and Bayesian models will support product-level
heterogeneity.

### Rationale

Different products/titles may respond differently to the same growth
intervention.

A hierarchical model provides partial pooling so that products can have
individual effects while still benefiting from information across the
portfolio.
