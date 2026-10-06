# Causal Growth Engine

A Bayesian causal growth and budget-allocation engine for understanding **what drives organic growth, how those drivers affect revenue, and how investment should be allocated across controllable growth levers**.

The project is designed around a product portfolio where marketing, product, social, and business data are combined to estimate causal effects and evaluate counterfactual investment strategies.

## Project Status

**Current stage: Synthetic data generation and validation**

The first implementation milestone is complete:

- 5-product/title synthetic portfolio
- 365 daily observations per title
- 1,825 product-day observations
- Realistic business-scale treatment variables
- Endogenous marketing activity driven partly by latent market demand
- Product-level heterogeneity
- Temporal behavior and seasonality
- Organic installs, engagement, retention, and revenue outcomes
- Known causal ground truth for model validation
- Automated tests
- Ruff linting
- Strict mypy type checking

The next milestone is the causal model: a hierarchical Bayesian model implemented with PyMC and validated against the known synthetic ground truth.

## Objective

The long-term objective is to answer questions such as:

> If we increase paid UA spend by $X, what is the expected incremental organic growth and revenue impact?

> Which controllable growth drivers have the strongest causal effect on organic acquisition?

> Given a fixed investment budget, how should that budget be allocated across titles and growth levers?

> What is the expected range of outcomes rather than a single point estimate?

The system is intended to move from **descriptive analytics** toward **causal decision support**.

## Core Causal Framework

The initial model distinguishes between controllable treatments, downstream mediators, confounders, and business outcomes.

### Primary Controllable Treatments

- `paid_ua_spend`
- `influencer_spend`
- `social_media_posts`
- `product_test_release`
- `product_version_update`

### Downstream Variables / Mediators

- `paid_installs`
- `social_media_likes`
- `social_media_comments`
- `app_store_rating`

These variables may be affected by the primary treatments and therefore should not automatically be treated as ordinary control variables when estimating total treatment effects.

### Outcomes

- `organic_installs`
- `engagement`
- `retention`
- `revenue`

### Important Confounder

The synthetic environment includes latent/observed `market_demand`.

Market demand can influence both business decisions and organic acquisition, creating the type of confounding that the causal model must account for.

Conceptually:

```text
                    Market Demand
                   /             \
                  v               v
        Marketing Decisions    Organic Growth
                  |
                  v
          Downstream Effects
                  |
                  v
             Organic Growth
                  |
                  v
               Revenue
