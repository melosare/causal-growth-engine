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

## Running the Streamlit Application

The project includes an executive-facing Streamlit dashboard for exploring the causal model and optimization results interactively.


### 1. Start the Streamlit application

From the project root:

    python -m streamlit run app/streamlit_app.py

Streamlit will start a local web server and display a URL similar to:

    Local URL: http://localhost:8501

Open the local URL in your browser to use the dashboard.

### 2. Stop the application

Press `Ctrl+C` in the terminal running Streamlit.

## Running the Test Suite

Run the complete test suite:

    pytest

Run Ruff linting:

    ruff check .

Run mypy type checking:

    mypy src

A complete local validation run is:

    pytest
    ruff check .
    mypy src

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

    Market Demand
       /     \
      v       v
    Marketing  Organic Growth
    Decisions
       |
       v
    Downstream Effects
       |
       v
    Organic Growth
       |
       v
    Revenue

## Application Workflow

The application follows a complete causal optimization workflow:

    Synthetic Data
          |
          v
    Causal Model
          |
          v
    Posterior Treatment Effects
          |
          v
    Counterfactual Simulation
          |
          v
    Bayesian Optimization
          |
          v
    Recommended Growth Strategy
          |
          v
    Executive Dashboard

### Synthetic Data Generation

The system generates synthetic product-level daily data containing:

- Marketing activity
- Product changes
- Social activity
- Market demand
- Organic acquisition
- Engagement
- Retention
- Revenue

The synthetic environment contains known causal relationships, allowing the model to be validated against ground truth.

### Causal Inference

A hierarchical Bayesian model implemented with PyMC estimates treatment effects while accounting for:

- Product-level heterogeneity
- Market demand
- Temporal effects
- Uncertainty in parameter estimates

Posterior samples are retained as an `ArviZ InferenceData` object and passed into downstream simulation and optimization.

### Counterfactual Simulation

The system evaluates hypothetical changes to controllable growth levers and estimates their expected impact on organic installs.

This allows the application to answer questions such as:

- What happens if paid UA spend increases?
- What happens if social activity increases?
- Which product changes generate the greatest expected lift?
- How uncertain is the estimated impact?

### Bayesian Optimization

A Gaussian-process surrogate model is used to search the feasible decision space.

The optimizer:

1. Generates feasible initial strategies.
2. Evaluates the causal objective.
3. Fits a Gaussian-process surrogate.
4. Uses Expected Improvement to select promising candidates.
5. Evaluates new candidates.
6. Repeats the process for the configured number of iterations.
7. Returns the highest-performing observed strategy.

The optimization respects the decision constraints defined by the optimization problem.

## Executive Dashboard

The Streamlit application provides an executive-facing interface for exploring:

- Growth strategy recommendations
- Expected incremental organic installs
- Optimization performance
- Treatment effects
- Uncertainty
- Allocation across growth levers
- Counterfactual impact
- Optimization diagnostics

The dashboard is designed to expose the decision implications of the model without requiring users to understand the underlying Bayesian implementation.

## Project Structure

    causal-growth-engine/
    ├── app/
    │   └── streamlit_app.py
    ├── examples/
    │   └── optimize_growth.py
    ├── src/
    │   └── cge/
    │       ├── causal/
    │       ├── data/
    │       ├── optimization/
    │       ├── pipeline/
    │       ├── reporting/
    │       └── simulation/
    ├── tests/
    │   ├── causal/
    │   ├── data/
    │   ├── integration/
    │   ├── optimization/
    │   ├── reporting/
    │   └── simulation/
    ├── pyproject.toml
    └── README.md

## Example Workflow

The end-to-end pipeline can also be run programmatically from the example module:

    python examples/optimize_growth.py

The example executes the growth optimization workflow without producing CLI reporting output.

## Development

The project uses:

- Python 3.11
- PyMC
- ArviZ
- NumPy
- pandas
- scikit-learn
- Streamlit
- Plotly
- pytest
- Ruff
- mypy

### Validation

Before committing changes, run:

    pytest
    ruff check .
    mypy src

All three checks should pass before changes are committed.

## Important Modeling Considerations

This project is currently based on synthetic data and is intended as a demonstration and validation environment.

The synthetic dataset provides known causal ground truth, which makes it possible to test whether the causal model can recover the underlying relationships.

In a production environment, additional considerations would include:

- Treatment assignment mechanisms
- Measurement quality
- Missing data
- Time-varying confounding
- Model misspecification
- Experimental validation
- Business constraints
- Budget constraints
- Operational feasibility
- Monitoring of causal-model drift

The optimization output should therefore be interpreted as **decision support under the assumptions of the causal model**, rather than as an unconditional guarantee of future performance.

## License

This project is currently intended as a demonstration project.
