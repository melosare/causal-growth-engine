from __future__ import annotations

from dataclasses import dataclass

import plotly.graph_objects as go
import streamlit as st

from cge.causal import ModelConfig, fit_model
from cge.data import DatasetConfig, SyntheticGrowthGenerator
from cge.optimization import BayesianOptimizationConfig
from cge.optimization.diagnostics import summarize_optimization
from cge.pipeline import OptimizationPipelineResult, optimize_growth_strategy

# Page configuration

st.set_page_config(
    page_title="Causal Growth Engine",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# Theme and styling

st.markdown(
    """
    <style>
        :root {
            --cge-bg: #07111f;
            --cge-panel: #0d1b2a;
            --cge-panel-soft: #101f31;
            --cge-border: #1d3147;
            --cge-text: #f4f7fb;
            --cge-muted: #8ea0b5;
            --cge-accent: #20c997;
            --cge-accent-soft: rgba(32, 201, 151, 0.12);
            --cge-blue: #4da3ff;
        }

        .stApp {
            background:
                radial-gradient(
                    circle at 80% 0%,
                    rgba(32, 201, 151, 0.08),
                    transparent 30%
                ),
                linear-gradient(
                    180deg,
                    #07111f 0%,
                    #091522 100%
                );
            color: var(--cge-text);
        }

        [data-testid="stHeader"] {
            background: rgba(7, 17, 31, 0.88);
        }

        .block-container {
            max-width: 1500px;
            padding-top: 2.5rem;
            padding-bottom: 4rem;
        }

        h1,
        h2,
        h3,
        h4 {
            color: var(--cge-text) !important;
            letter-spacing: -0.025em;
        }

        .hero {
            padding: 0.5rem 0 1.5rem 0;
        }

        .eyebrow {
            color: var(--cge-accent);
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }

        .hero-title {
            color: var(--cge-text);
            font-size: 2.7rem;
            font-weight: 750;
            line-height: 1.05;
            margin: 0;
        }

        .hero-subtitle {
            color: var(--cge-muted);
            font-size: 1rem;
            margin-top: 0.7rem;
            max-width: 760px;
            line-height: 1.6;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            background: rgba(32, 201, 151, 0.10);
            border: 1px solid rgba(32, 201, 151, 0.25);
            border-radius: 999px;
            color: #7ff0ca;
            font-size: 0.76rem;
            font-weight: 650;
            padding: 0.4rem 0.7rem;
            margin-bottom: 1.25rem;
        }

        .status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--cge-accent);
            box-shadow: 0 0 12px rgba(32, 201, 151, 0.7);
        }

        .configuration-card {
            background: linear-gradient(
                145deg,
                rgba(16, 31, 49, 0.98),
                rgba(11, 25, 40, 0.98)
            );
            border: 1px solid var(--cge-border);
            border-radius: 18px;
            padding: 1.35rem 1.45rem 0.9rem 1.45rem;
            margin-bottom: 1.6rem;
            box-shadow: 0 14px 35px rgba(0, 0, 0, 0.14);
        }

        .configuration-title {
            color: var(--cge-text);
            font-size: 1rem;
            font-weight: 750;
            margin-bottom: 0.15rem;
        }

        .configuration-description {
            color: var(--cge-muted);
            font-size: 0.78rem;
            line-height: 1.5;
            margin-bottom: 1rem;
        }

        .configuration-section {
            color: #71869d;
            font-size: 0.69rem;
            font-weight: 750;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin: 0.25rem 0 0.45rem 0;
        }

        .section-label {
            color: #71869d;
            font-size: 0.72rem;
            font-weight: 750;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin: 1.4rem 0 0.65rem 0;
        }

        .kpi-card {
            background: linear-gradient(
                145deg,
                rgba(16, 31, 49, 0.98),
                rgba(11, 25, 40, 0.98)
            );
            border: 1px solid var(--cge-border);
            border-radius: 16px;
            padding: 1.15rem 1.2rem 1.1rem 1.2rem;
            min-height: 132px;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.12);
        }

        .kpi-label {
            color: var(--cge-muted);
            font-size: 0.76rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0.07em;
        }

        .kpi-value {
            color: var(--cge-text);
            font-size: 1.8rem;
            font-weight: 750;
            line-height: 1.15;
            margin-top: 0.55rem;
        }

        .kpi-value.accent {
            color: var(--cge-accent);
        }

        .kpi-detail {
            color: #687d94;
            font-size: 0.78rem;
            margin-top: 0.35rem;
        }

        .strategy-card {
            background: rgba(13, 27, 42, 0.9);
            border: 1px solid var(--cge-border);
            border-radius: 16px;
            padding: 1.15rem;
            margin-bottom: 0.8rem;
        }

        .strategy-name {
            color: var(--cge-muted);
            font-size: 0.78rem;
            margin-bottom: 0.3rem;
        }

        .strategy-value {
            color: var(--cge-text);
            font-size: 1.35rem;
            font-weight: 700;
        }

        .strategy-unit {
            color: #70849a;
            font-size: 0.75rem;
            margin-left: 0.2rem;
        }

        .insight-card {
            background: linear-gradient(
                135deg,
                rgba(32, 201, 151, 0.09),
                rgba(13, 27, 42, 0.95)
            );
            border: 1px solid rgba(32, 201, 151, 0.2);
            border-radius: 16px;
            padding: 1.2rem 1.3rem;
            color: #cbd8e5;
            line-height: 1.65;
        }

        .footer-note {
            color: #53687e;
            font-size: 0.73rem;
            margin-top: 2rem;
            text-align: center;
        }

        .stAlert {
            border-radius: 12px;
        }

        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input {
            background-color: #0d1b2a;
            border-color: var(--cge-border);
        }

        div[data-testid="stRadio"] label {
            color: var(--cge-text);
        }

        div[data-testid="stButton"] > button {
            width: 100%;
            border-radius: 10px;
            border: 1px solid rgba(32, 201, 151, 0.35);
            background: var(--cge-accent);
            color: #06131b;
            font-weight: 750;
            min-height: 2.7rem;
        }

        div[data-testid="stButton"] > button:hover {
            border-color: #4de0b4;
            background: #39d6a6;
            color: #041017;
        }

        div[data-testid="stMetric"] {
            background: transparent;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Application configuration

@dataclass(frozen=True)
class AppRunConfig:
    """Configuration for one CGE analysis run."""

    mode: str
    n_titles: int
    n_days: int
    draws: int
    tune: int
    chains: int
    target_accept: float
    initial_points: int
    iterations: int
    random_seed: int


DEMO_DEFAULTS = AppRunConfig(
    mode="Demo",
    n_titles=10,
    n_days=60,
    draws=100,
    tune=100,
    chains=2,
    target_accept=0.90,
    initial_points=8,
    iterations=12,
    random_seed=42,
)

ANALYSIS_DEFAULTS = AppRunConfig(
    mode="Analysis",
    n_titles=10,
    n_days=90,
    draws=500,
    tune=500,
    chains=2,
    target_accept=0.92,
    initial_points=12,
    iterations=20,
    random_seed=42,
)


# Cached CGE execution


@st.cache_data(show_spinner=False)
def run_cge(config: AppRunConfig) -> OptimizationPipelineResult:
    """Execute the complete CGE workflow."""

    dataset_config = DatasetConfig(
        n_titles=config.n_titles,
        n_days=config.n_days,
        seed=config.random_seed,
    )

    generator = SyntheticGrowthGenerator(
        config=dataset_config,
    )

    data, _ = generator.generate()

    model_config = ModelConfig(
        draws=config.draws,
        tune=config.tune,
        chains=config.chains,
        target_accept=config.target_accept,
        random_seed=config.random_seed,
    )

    _, trace = fit_model(
        data=data,
        config=model_config,
    )

    optimization_config = BayesianOptimizationConfig(
        initial_points=config.initial_points,
        iterations=config.iterations,
        random_seed=config.random_seed,
    )

    return optimize_growth_strategy(
        data=data,
        trace=trace,
        config=optimization_config,
    )



# Formatting helpers


def format_number(value: float) -> str:
    """Format a numeric KPI."""

    return f"{value:,.0f}"


def format_currency(value: float) -> str:
    """Format a currency value."""

    return f"${value:,.0f}"


def format_probability(value: float) -> str:
    """Format a probability as a percentage."""

    return f"{value * 100:.1f}%"


def render_kpi(
    label: str,
    value: str,
    detail: str,
    *,
    accent: bool = False,
) -> None:
    """Render a styled KPI card."""

    accent_class = " accent" if accent else ""

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value{accent_class}">{value}</div>
            <div class="kpi-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Plotly visualization helpers

PLOTLY_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {
        "color": "#b9c7d6",
        "family": "Inter, ui-sans-serif, system-ui, sans-serif",
    },
    "margin": {
        "l": 20,
        "r": 20,
        "t": 30,
        "b": 20,
    },
}


def make_impact_chart(
    lower: float,
    mean: float,
    upper: float,
) -> go.Figure:
    """Create the expected-impact uncertainty visualization."""

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=[lower, upper],
            y=[
                "Expected incremental installs",
                "Expected incremental installs",
            ],
            mode="lines",
            line={
                "color": "rgba(32, 201, 151, 0.35)",
                "width": 18,
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )

    figure.add_trace(
        go.Scatter(
            x=[mean],
            y=["Expected incremental installs"],
            mode="markers",
            marker={
                "size": 16,
                "color": "#20c997",
                "line": {
                    "width": 3,
                    "color": "#d7fff2",
                },
            },
            name="Expected impact",
            hovertemplate=(
                "<b>Expected impact</b><br>"
                "%{x:,.0f} incremental installs"
                "<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=[lower, upper],
            y=[
                "Expected incremental installs",
                "Expected incremental installs",
            ],
            mode="markers",
            marker={
                "size": 8,
                "color": "#7e91a7",
            },
            name="Credible interval",
            hovertemplate=(
                "%{x:,.0f} incremental installs"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        **PLOTLY_LAYOUT,
        height=230,
        xaxis={
            "title": "Incremental installs",
            "gridcolor": "rgba(142, 160, 181, 0.10)",
            "zeroline": False,
            "tickformat": ",.0f",
        },
        yaxis={
            "showgrid": False,
            "showticklabels": False,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )

    return figure


def make_allocation_chart(
    paid_ua_spend: float,
    influencer_spend: float,
) -> go.Figure:
    """Create the cash allocation chart."""

    labels = ["Paid UA", "Influencer"]
    values = [paid_ua_spend, influencer_spend]
    colors = ["#20c997", "#4da3ff"]

    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={
                "color": colors,
                "line": {
                    "color": "rgba(255,255,255,0.08)",
                    "width": 1,
                },
            },
            text=[format_currency(value) for value in values],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "$%{x:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        **PLOTLY_LAYOUT,
        height=260,
        xaxis={
            "title": "Spend",
            "tickprefix": "$",
            "tickformat": ",.0f",
            "gridcolor": "rgba(142, 160, 181, 0.10)",
            "zeroline": False,
        },
        yaxis={
            "showgrid": False,
        },
        showlegend=False,
    )

    return figure


def make_actions_chart(
    social_media_posts: float,
    product_test_release: float,
    product_version_update: float,
) -> go.Figure:
    """Create the product and social action chart."""

    labels = [
        "Social posts",
        "Test releases",
        "Version updates",
    ]

    values = [
        social_media_posts,
        product_test_release,
        product_version_update,
    ]

    figure = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker={
                "color": ["#20c997", "#4da3ff", "#7d8cff"],
                "line": {
                    "color": "rgba(255,255,255,0.08)",
                    "width": 1,
                },
            },
            text=[f"{value:.0f}" for value in values],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "%{y:.0f} actions"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        **PLOTLY_LAYOUT,
        height=260,
        yaxis={
            "title": "Actions",
            "dtick": 1,
            "gridcolor": "rgba(142, 160, 181, 0.10)",
            "zeroline": False,
        },
        xaxis={
            "showgrid": False,
        },
        showlegend=False,
    )

    return figure


# Main dashboard header

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Causal intelligence · Bayesian optimization</div>
        <div class="hero-title">Growth strategy, quantified.</div>
        <div class="hero-subtitle">
            Estimate causal growth effects, explore uncertainty, and identify
            the most promising investment strategy under real resource
            constraints.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <span class="status-pill">
        <span class="status-dot"></span>
        ENGINE READY
    </span>
    """,
    unsafe_allow_html=True,
)

# Top configuration panel

st.markdown(
    """
    <div class="configuration-card">
        <div class="configuration-title">Analysis Settings</div>
        <div class="configuration-description">
            Configure the scenario growth environment, model, and
            optimization before running the engine.
        </div>
    """,
    unsafe_allow_html=True,
)

mode_column, scenario_column, model_column, optimization_column = st.columns(
    [1, 1.1, 1.35, 1.35]
)

with mode_column:
    st.markdown(
        '<div class="configuration-section">Run mode</div>',
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Select analysis mode",
        options=["Demo", "Analysis"],
        index=0,
        label_visibility="collapsed",
    )

defaults = (
    DEMO_DEFAULTS
    if mode == "Demo"
    else ANALYSIS_DEFAULTS
)

with scenario_column:
    st.markdown(
        '<div class="configuration-section">Scenario Size</div>',
        unsafe_allow_html=True,
    )

    n_titles = st.number_input(
        "Number of products/titles",
        min_value=1,
        max_value=50,
        value=defaults.n_titles,
        step=1,
    )

    n_days = st.number_input(
        "Number of historical days",
        min_value=7,
        max_value=730,
        value=defaults.n_days,
        step=1,
    )

with model_column:
    st.markdown(
        '<div class="configuration-section">Model Control</div>',
        unsafe_allow_html=True,
    )

    draws = st.number_input(
        "Analysis Samples",
        min_value=50,
        max_value=2_000,
        value=defaults.draws,
        step=50,
    )

    tune = st.number_input(
        "Calibration Steps",
        min_value=50,
        max_value=2_000,
        value=defaults.tune,
        step=50,
    )

    chains = st.number_input(
        "Independent Analysis Steps",
        min_value=1,
        max_value=4,
        value=defaults.chains,
        step=1,
    )

    target_accept = st.slider(
        "Minimum Accuracy",
        min_value=0.80,
        max_value=0.99,
        value=defaults.target_accept,
        step=0.01,
    )

with optimization_column:
    st.markdown(
        '<div class="configuration-section">Optimization</div>',
        unsafe_allow_html=True,
    )

    initial_points = st.number_input(
        "Starting Strategy Number",
        min_value=2,
        max_value=50,
        value=defaults.initial_points,
        step=1,
    )

    iterations = st.number_input(
        "Additional Tests",
        min_value=1,
        max_value=100,
        value=defaults.iterations,
        step=1,
    )

    random_seed = st.number_input(
        "Reproducibility Setting",
        min_value=0,
        max_value=1_000_000,
        value=defaults.random_seed,
        step=1,
    )

st.markdown(
    "</div>",
    unsafe_allow_html=True,
)

run_column, spacer_column = st.columns([1, 3])

with run_column:
    run_clicked = st.button(
        "Run Optimization",
        type="primary",
        use_container_width=True,
    )


# Execute analysis

config = AppRunConfig(
    mode=mode,
    n_titles=int(n_titles),
    n_days=int(n_days),
    draws=int(draws),
    tune=int(tune),
    chains=int(chains),
    target_accept=float(target_accept),
    initial_points=int(initial_points),
    iterations=int(iterations),
    random_seed=int(random_seed),
)

if run_clicked:
    with st.status(
        "Running the causal growth engine...",
        expanded=True,
    ) as status:
        st.write("Generating synthetic growth data...")
        st.write("Fitting Bayesian causal model...")
        st.write("Searching the constrained strategy space...")

        try:
            result = run_cge(config)
        except Exception as exc:
            status.update(
                label="Optimization failed",
                state="error",
            )
            st.error(
                "The CGE run failed. Check the configuration and terminal "
                "output for details."
            )
            st.exception(exc)
            st.stop()

        status.update(
            label="Optimization complete",
            state="complete",
        )

    st.session_state["cge_result"] = result
    st.session_state["cge_config"] = config


# Empty state


if "cge_result" not in st.session_state:
    st.markdown(
        """
        <div class="insight-card">
            <strong>Ready to analyze.</strong><br>
            Configure the analysis parameters above and run the optimization.
            CGE will generate a synthetic growth environment, fit its Bayesian
            causal model, and search for a constrained growth strategy.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="footer-note">'
        "Causal Growth Engine · Synthetic-data analysis environment"
        "</div>",
        unsafe_allow_html=True,
    )

    st.stop()


# Results


result = st.session_state["cge_result"]

if not isinstance(result, OptimizationPipelineResult):
    st.error("Invalid optimization result in session state.")
    st.stop()

recommendation = result.recommendation
optimization = result.optimization

expected_impact = recommendation.expected_incremental_installs
lower_bound = recommendation.lower_bound
upper_bound = recommendation.upper_bound
probability_positive = recommendation.probability_positive

candidates_evaluated = len(optimization.observations)

diagnostics = summarize_optimization(
    optimization
)

improvement = diagnostics.improvement


# KPI section

st.markdown(
    '<div class="section-label">Recommended outcome</div>',
    unsafe_allow_html=True,
)

kpi_columns = st.columns(4)

with kpi_columns[0]:
    render_kpi(
        "Expected incremental installs",
        format_number(expected_impact),
        "Posterior expected impact",
        accent=True,
    )

with kpi_columns[1]:
    render_kpi(
        "Probability of positive impact",
        format_probability(probability_positive),
        "Posterior probability",
        accent=True,
    )

with kpi_columns[2]:
    render_kpi(
        "Credible interval",
        f"{format_number(lower_bound)} – {format_number(upper_bound)}",
        "Lower / upper uncertainty bounds",
    )

with kpi_columns[3]:
    render_kpi(
        "Candidates evaluated",
        f"{candidates_evaluated}",
        f"+{format_number(improvement)} vs initial candidate",
    )


# Recommended strategy

st.markdown(
    '<div class="section-label">Recommended strategy</div>',
    unsafe_allow_html=True,
)

strategy = recommendation.decision

strategy_columns = st.columns(5)

strategy_items = [
    (
        "Paid UA spend",
        format_currency(strategy["paid_ua_spend"]),
        "daily spend",
    ),
    (
        "Influencer spend",
        format_currency(strategy["influencer_spend"]),
        "daily spend",
    ),
    (
        "Social media posts",
        f"{strategy['social_media_posts']:.0f}",
        "posts",
    ),
    (
        "Product test releases",
        f"{strategy['product_test_release']:.0f}",
        "releases",
    ),
    (
        "Version updates",
        f"{strategy['product_version_update']:.0f}",
        "updates",
    ),
]

for column, (name, value, unit) in zip(
    strategy_columns,
    strategy_items,
    strict=True,
):
    with column:
        st.markdown(
            f"""
            <div class="strategy-card">
                <div class="strategy-name">{name}</div>
                <div class="strategy-value">
                    {value}
                    <span class="strategy-unit">{unit}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# Impact and allocation

st.markdown(
    '<div class="section-label">Impact & allocation</div>',
    unsafe_allow_html=True,
)

impact_column, allocation_column = st.columns([1.15, 1])

with impact_column:
    st.markdown("#### Expected impact")

    st.plotly_chart(
        make_impact_chart(
            lower=lower_bound,
            mean=expected_impact,
            upper=upper_bound,
        ),
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )

with allocation_column:
    st.markdown("#### Cash allocation")

    st.plotly_chart(
        make_allocation_chart(
            paid_ua_spend=strategy["paid_ua_spend"],
            influencer_spend=strategy["influencer_spend"],
        ),
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )



# Product and social actions

actions_column, diagnostics_column = st.columns([1, 1])

with actions_column:
    st.markdown("#### Product & social actions")

    st.plotly_chart(
        make_actions_chart(
            social_media_posts=strategy["social_media_posts"],
            product_test_release=strategy["product_test_release"],
            product_version_update=strategy["product_version_update"],
        ),
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )

with diagnostics_column:
    st.markdown("#### Optimization diagnostics")

    st.markdown(
        f"""
        <div class="strategy-card">
            <div class="strategy-name">Candidates evaluated</div>
            <div class="strategy-value">{candidates_evaluated}</div>
        </div>

        <div class="strategy-card">
            <div class="strategy-name">Improvement over initial candidate</div>
            <div class="strategy-value">
                +{format_number(improvement)}
                <span class="strategy-unit">installs</span>
            </div>
        </div>

        <div class="insight-card">
            The recommended strategy is the best feasible decision found by
            the Bayesian optimizer under the configured resource constraints.
            The uncertainty interval reflects the causal model's posterior
            uncertainty rather than a guaranteed forecast.
        </div>
        """,
        unsafe_allow_html=True,
    )


# Run configuration

with st.expander("Run configuration"):
    config_columns = st.columns(4)

    with config_columns[0]:
        st.metric("Mode", config.mode)
        st.metric("Titles", config.n_titles)

    with config_columns[1]:
        st.metric("Days", config.n_days)
        st.metric("Draws", config.draws)

    with config_columns[2]:
        st.metric("Tune", config.tune)
        st.metric("Chains", config.chains)

    with config_columns[3]:
        st.metric("Initial candidates", config.initial_points)
        st.metric("Iterations", config.iterations)



# Footer


st.markdown(
    '<div class="footer-note">'
    "Causal Growth Engine · Results are based on synthetic growth data and "
    "should not be interpreted as production forecasts."
    "</div>",
    unsafe_allow_html=True,
)
