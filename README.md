# Causal Growth Engine

A Bayesian causal growth and budget-allocation engine for understanding **what drives organic growth, how those drivers affect revenue, and how investment should be allocated across controllable growth levers**.

The project combines synthetic product, marketing, social, and business data to estimate causal effects and evaluate counterfactual investment strategies.

## Project Status

**Current stage: End-to-end synthetic causal optimization application**

The project currently includes:

- Synthetic growth-data generation
- Product-level heterogeneity
- Endogenous marketing activity driven partly by market demand
- Seasonality and temporal behavior
- Organic installs, engagement, retention, and revenue outcomes
- Known causal ground truth for model validation
- Hierarchical Bayesian causal modeling with PyMC
- Posterior treatment-effect estimation
- Constrained Bayesian optimization
- Counterfactual intervention evaluation
- Strategy recommendations
- Optimization diagnostics
- Executive-facing Streamlit dashboard
- Interactive impact and allocation visualizations
- Automated tests
- Ruff linting
- Strict mypy type checking

The application is designed to demonstrate a complete workflow from **synthetic data generation → causal inference → optimization → executive decision support**.

## Objective

The long-term objective is to answer questions such as:

> If we increase paid UA spend by $X, what is the expected incremental organic growth and revenue impact?

> Which controllable growth drivers have the strongest causal effect on organic acquisition?

> Given a fixed investment budget, how should investment be allocated across growth levers?

> What is the expected range of outcomes rather than a single point estimate?

The system is intended to move from **descriptive analytics** toward **causal decision support**.

## Core Causal Framework

The model distinguishes between controllable treatments, downstream mediators, confounders, and business outcomes.

### Primary Controllable Treatments

- `paid_ua_spend`
- `influencer_spend`
- `social_media_posts`
- `product_test_release`
- `product_version_update`

These are the primary growth levers available to the optimizer.

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

The primary optimization objective is incremental organic installs, while the broader dataset supports downstream engagement, retention, and revenue analysis.

### Important Confounder

The synthetic environment includes `market_demand`.

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
