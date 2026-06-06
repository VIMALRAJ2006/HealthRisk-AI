"""HealthRisk Lab: a Streamlit healthcare investment simulation game.

Run with:
    streamlit run app/healthrisk_lab.py
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


INITIAL_PORTFOLIO_VALUE = 1_000_000.0
SESSION_KEY = "healthrisk_lab"


@dataclass(frozen=True)
class HealthcareEvent:
    """A market-moving healthcare scenario."""

    name: str
    description: str
    portfolio_impact_pct: float
    risk_level: str


@dataclass(frozen=True)
class ActionEffect:
    """A player action and the way it changes event impact."""

    name: str
    description: str
    impact_adjuster: Callable[[float, str], float]


HEALTHCARE_EVENTS: list[HealthcareEvent] = [
    HealthcareEvent(
        name="Pandemic Outbreak",
        description=(
            "A fast-moving infectious disease increases demand for care while "
            "disrupting hospital operations and elective procedures."
        ),
        portfolio_impact_pct=-14.0,
        risk_level="Critical",
    ),
    HealthcareEvent(
        name="New Drug Approval",
        description=(
            "A major therapy receives approval, lifting pharmaceutical revenue "
            "expectations and investor confidence."
        ),
        portfolio_impact_pct=10.5,
        risk_level="Medium",
    ),
    HealthcareEvent(
        name="Hospital Capacity Crisis",
        description=(
            "Regional hospital utilization exceeds safe operating thresholds, "
            "raising labor costs and delaying profitable procedures."
        ),
        portfolio_impact_pct=-8.0,
        risk_level="High",
    ),
    HealthcareEvent(
        name="Insurance Claim Surge",
        description=(
            "Unexpectedly high claim volumes pressure payer margins and raise "
            "reserve requirements across the sector."
        ),
        portfolio_impact_pct=-6.5,
        risk_level="High",
    ),
    HealthcareEvent(
        name="AI Diagnostic Breakthrough",
        description=(
            "A validated diagnostic AI platform improves throughput and lowers "
            "screening costs, boosting health-tech valuations."
        ),
        portfolio_impact_pct=12.0,
        risk_level="Medium",
    ),
    HealthcareEvent(
        name="Vaccine Discovery",
        description=(
            "A successful vaccine candidate shows strong trial results, creating "
            "a sector-wide rally in biotech and public-health suppliers."
        ),
        portfolio_impact_pct=13.5,
        risk_level="Medium",
    ),
    HealthcareEvent(
        name="Regulatory Fine",
        description=(
            "A large compliance penalty hits a leading provider network, "
            "dragging sentiment across managed-care and hospital equities."
        ),
        portfolio_impact_pct=-7.5,
        risk_level="High",
    ),
    HealthcareEvent(
        name="Disease Containment Success",
        description=(
            "Public-health intervention succeeds faster than expected, reducing "
            "acute care strain and stabilizing investor expectations."
        ),
        portfolio_impact_pct=6.5,
        risk_level="Low",
    ),
    HealthcareEvent(
        name="Medical Device Recall",
        description=(
            "A major device recall creates replacement costs, legal exposure, "
            "and near-term revenue uncertainty for manufacturers."
        ),
        portfolio_impact_pct=-9.5,
        risk_level="High",
    ),
    HealthcareEvent(
        name="Healthcare Policy Reform",
        description=(
            "A broad reimbursement and coverage reform reshapes expected margins "
            "across providers, insurers, and pharmaceutical companies."
        ),
        portfolio_impact_pct=4.0,
        risk_level="Medium",
    ),
]


def buy_healthcare_stocks(base_impact: float, risk_level: str) -> float:
    """Amplify healthcare exposure, increasing both upside and downside."""
    risk_multiplier = {"Low": 1.15, "Medium": 1.25, "High": 1.35, "Critical": 1.45}
    return base_impact * risk_multiplier.get(risk_level, 1.25)


def sell_risky_assets(base_impact: float, risk_level: str) -> float:
    """Reduce volatility by trimming risky positions."""
    if base_impact < 0:
        hedge_multiplier = {"Low": 0.75, "Medium": 0.62, "High": 0.50, "Critical": 0.42}
        return base_impact * hedge_multiplier.get(risk_level, 0.6)
    return base_impact * 0.62


def increase_insurance_reserve(base_impact: float, risk_level: str) -> float:
    """Pay a small opportunity cost to absorb adverse shocks."""
    reserve_cost = 1.1
    if base_impact < 0:
        reserve_buffer = {"Low": 2.0, "Medium": 3.5, "High": 5.0, "Critical": 7.0}
        return min(base_impact + reserve_buffer.get(risk_level, 3.5), 0.0) - reserve_cost
    return base_impact * 0.78 - reserve_cost


def hold_position(base_impact: float, risk_level: str) -> float:
    """Keep current allocation unchanged."""
    _ = risk_level
    return base_impact


ACTION_EFFECTS: list[ActionEffect] = [
    ActionEffect(
        name="Buy Healthcare Stocks",
        description="Increase exposure to healthcare equities before the event resolves.",
        impact_adjuster=buy_healthcare_stocks,
    ),
    ActionEffect(
        name="Sell Risky Assets",
        description="De-risk the portfolio and dampen event volatility.",
        impact_adjuster=sell_risky_assets,
    ),
    ActionEffect(
        name="Increase Insurance Reserve",
        description="Build a reserve buffer to reduce downside from adverse shocks.",
        impact_adjuster=increase_insurance_reserve,
    ),
    ActionEffect(
        name="Hold Position",
        description="Accept the event's full market impact with no allocation change.",
        impact_adjuster=hold_position,
    ),
]


def _new_event(excluded_event_name: str | None = None) -> HealthcareEvent:
    """Select a random event, avoiding an immediate repeat when possible."""
    available_events = [
        event for event in HEALTHCARE_EVENTS if event.name != excluded_event_name
    ]
    return random.choice(available_events or HEALTHCARE_EVENTS)


def reset_simulation() -> None:
    """Reset the game state to a fresh portfolio."""
    st.session_state[SESSION_KEY] = {
        "portfolio_value": INITIAL_PORTFOLIO_VALUE,
        "current_event": asdict(_new_event()),
        "history": [],
        "round_number": 1,
        "last_decision": "No decision yet",
        "last_profit_loss": 0.0,
        "last_adjusted_impact_pct": 0.0,
    }


def initialize_state() -> None:
    """Create session state once so the simulator persists across reruns."""
    if SESSION_KEY not in st.session_state:
        reset_simulation()


def get_state() -> dict:
    initialize_state()
    return st.session_state[SESSION_KEY]


def current_event_from_state() -> HealthcareEvent:
    event_data = get_state()["current_event"]
    return HealthcareEvent(**event_data)


def selected_action(action_name: str) -> ActionEffect:
    for action in ACTION_EFFECTS:
        if action.name == action_name:
            return action
    raise ValueError(f"Unknown action: {action_name}")


def resolve_round(action_name: str) -> None:
    """Apply the selected decision, update portfolio value, and record history."""
    state = get_state()
    event = current_event_from_state()
    action = selected_action(action_name)

    starting_value = float(state["portfolio_value"])
    adjusted_impact_pct = round(
        action.impact_adjuster(event.portfolio_impact_pct, event.risk_level),
        2,
    )
    profit_loss = round(starting_value * (adjusted_impact_pct / 100.0), 2)
    ending_value = round(max(starting_value + profit_loss, 0.0), 2)

    state["history"].append(
        {
            "Round": state["round_number"],
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Event": event.name,
            "Risk Level": event.risk_level,
            "Base Impact %": event.portfolio_impact_pct,
            "Decision Taken": action.name,
            "Adjusted Impact %": adjusted_impact_pct,
            "Profit/Loss": profit_loss,
            "Portfolio Value": ending_value,
        }
    )

    state["portfolio_value"] = ending_value
    state["round_number"] += 1
    state["last_decision"] = action.name
    state["last_profit_loss"] = profit_loss
    state["last_adjusted_impact_pct"] = adjusted_impact_pct
    state["current_event"] = asdict(_new_event(event.name))


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def impact_color(value: float) -> str:
    return "#16a34a" if value >= 0 else "#dc2626"


def render_page_styles() -> None:
    st.markdown(
        """
        <style>
            .main .block-container {
                padding-top: 2rem;
                max-width: 1220px;
            }
            div[data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 0.75rem 0.9rem;
                box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
            }
            .lab-panel {
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 1rem;
                background: #ffffff;
                box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
            }
            .risk-pill {
                display: inline-block;
                border-radius: 999px;
                padding: 0.2rem 0.7rem;
                font-size: 0.82rem;
                font-weight: 700;
                color: #111827;
                background: #e5e7eb;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.title("HealthRisk Lab")
    st.caption(
        "A healthcare investment simulation where market events test allocation, "
        "reserve, and risk-management decisions."
    )


def render_scoreboard() -> None:
    state = get_state()
    profit_loss = float(state["last_profit_loss"])
    cumulative_return = (
        (float(state["portfolio_value"]) - INITIAL_PORTFOLIO_VALUE)
        / INITIAL_PORTFOLIO_VALUE
        * 100.0
    )

    cols = st.columns(4)
    cols[0].metric(
        "Current Portfolio Value",
        format_currency(float(state["portfolio_value"])),
        delta=f"{cumulative_return:+.2f}% total",
    )
    cols[1].metric("Current Round", f"{state['round_number']}")
    cols[2].metric("Decision Taken", state["last_decision"])
    cols[3].metric(
        "Profit/Loss",
        format_currency(profit_loss),
        delta=f"{state['last_adjusted_impact_pct']:+.2f}% last round",
    )


def render_event_panel() -> str:
    event = current_event_from_state()
    action_names = [action.name for action in ACTION_EFFECTS]

    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Current Event")
        st.markdown(
            f"""
            <div class="lab-panel">
                <h3 style="margin-top:0;">{event.name}</h3>
                <span class="risk-pill">Risk Level: {event.risk_level}</span>
                <p style="margin-top:0.9rem;">{event.description}</p>
                <p style="font-size:1.05rem;">
                    Event Impact:
                    <strong style="color:{impact_color(event.portfolio_impact_pct)};">
                        {event.portfolio_impact_pct:+.2f}%
                    </strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.subheader("Choose Action")
        action_name = st.radio(
            "Decision",
            options=action_names,
            index=3,
            label_visibility="collapsed",
        )
        action = selected_action(action_name)
        st.info(action.description)

        apply_col, reset_col = st.columns([1, 1])
        if apply_col.button("Apply Decision", type="primary", use_container_width=True):
            resolve_round(action_name)
            st.rerun()

        if reset_col.button("Reset Simulation", use_container_width=True):
            reset_simulation()
            st.rerun()

    return action_name


def history_dataframe() -> pd.DataFrame:
    history = get_state()["history"]
    if not history:
        return pd.DataFrame(
            columns=[
                "Round",
                "Timestamp",
                "Event",
                "Risk Level",
                "Base Impact %",
                "Decision Taken",
                "Adjusted Impact %",
                "Profit/Loss",
                "Portfolio Value",
            ]
        )
    return pd.DataFrame(history)


def portfolio_chart(history_df: pd.DataFrame) -> go.Figure:
    if history_df.empty:
        chart_df = pd.DataFrame(
            {
                "Round": [0],
                "Portfolio Value": [INITIAL_PORTFOLIO_VALUE],
                "Decision Taken": ["Starting Portfolio"],
            }
        )
    else:
        chart_df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "Round": [0],
                        "Portfolio Value": [INITIAL_PORTFOLIO_VALUE],
                        "Decision Taken": ["Starting Portfolio"],
                    }
                ),
                history_df[["Round", "Portfolio Value", "Decision Taken"]],
            ],
            ignore_index=True,
        )

    fig = px.line(
        chart_df,
        x="Round",
        y="Portfolio Value",
        markers=True,
        hover_data=["Decision Taken"],
        labels={"Portfolio Value": "Portfolio Value ($)"},
    )
    fig.add_hline(
        y=INITIAL_PORTFOLIO_VALUE,
        line_dash="dash",
        line_color="#64748b",
        annotation_text="Initial capital",
    )
    fig.update_traces(line_color="#0f766e", marker=dict(size=8))
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=30, b=20))
    return fig


def event_impact_chart(history_df: pd.DataFrame) -> go.Figure:
    if history_df.empty:
        event = current_event_from_state()
        chart_df = pd.DataFrame(
            {
                "Event": [event.name],
                "Adjusted Impact %": [event.portfolio_impact_pct],
                "Risk Level": [event.risk_level],
            }
        )
        title = "Current Event Impact"
    else:
        chart_df = history_df[["Event", "Adjusted Impact %", "Risk Level"]].copy()
        title = "Resolved Event Impacts"

    fig = px.bar(
        chart_df,
        x="Event",
        y="Adjusted Impact %",
        color="Risk Level",
        title=title,
        color_discrete_map={
            "Low": "#16a34a",
            "Medium": "#ca8a04",
            "High": "#dc2626",
            "Critical": "#7f1d1d",
        },
    )
    fig.add_hline(y=0, line_width=1, line_color="#111827")
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=50, b=80),
        xaxis_tickangle=-35,
    )
    return fig


def profit_loss_chart(history_df: pd.DataFrame) -> go.Figure:
    if history_df.empty:
        chart_df = pd.DataFrame({"Round": [0], "Profit/Loss": [0.0], "Event": ["Start"]})
    else:
        chart_df = history_df[["Round", "Profit/Loss", "Event"]].copy()

    chart_df["Outcome"] = chart_df["Profit/Loss"].apply(
        lambda value: "Profit" if value >= 0 else "Loss"
    )
    fig = px.bar(
        chart_df,
        x="Round",
        y="Profit/Loss",
        color="Outcome",
        hover_data=["Event"],
        color_discrete_map={"Profit": "#16a34a", "Loss": "#dc2626"},
        labels={"Profit/Loss": "Profit/Loss ($)"},
    )
    fig.add_hline(y=0, line_width=1, line_color="#111827")
    fig.update_layout(height=330, margin=dict(l=20, r=20, t=30, b=20))
    return fig


def action_distribution_chart(history_df: pd.DataFrame) -> go.Figure:
    if history_df.empty:
        chart_df = pd.DataFrame({"Decision Taken": ["No decisions yet"], "Count": [1]})
    else:
        chart_df = (
            history_df["Decision Taken"]
            .value_counts()
            .rename_axis("Decision Taken")
            .reset_index(name="Count")
        )

    fig = px.pie(
        chart_df,
        names="Decision Taken",
        values="Count",
        hole=0.48,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(height=330, margin=dict(l=20, r=20, t=30, b=20))
    return fig


def render_charts() -> None:
    history_df = history_dataframe()

    st.subheader("Simulation Analytics")
    first_row = st.columns(2)
    with first_row[0]:
        st.plotly_chart(portfolio_chart(history_df), use_container_width=True)
    with first_row[1]:
        st.plotly_chart(event_impact_chart(history_df), use_container_width=True)

    second_row = st.columns(2)
    with second_row[0]:
        st.plotly_chart(profit_loss_chart(history_df), use_container_width=True)
    with second_row[1]:
        st.plotly_chart(action_distribution_chart(history_df), use_container_width=True)


def render_history_table() -> None:
    history_df = history_dataframe()
    st.subheader("Event History Table")
    if history_df.empty:
        st.write("No decisions have been applied yet.")
        return

    display_df = history_df.copy()
    display_df["Profit/Loss"] = display_df["Profit/Loss"].map(format_currency)
    display_df["Portfolio Value"] = display_df["Portfolio Value"].map(format_currency)
    display_df["Base Impact %"] = display_df["Base Impact %"].map(lambda value: f"{value:+.2f}%")
    display_df["Adjusted Impact %"] = display_df["Adjusted Impact %"].map(
        lambda value: f"{value:+.2f}%"
    )

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_event_catalog() -> None:
    with st.expander("Healthcare Event Catalog", expanded=False):
        catalog_df = pd.DataFrame(
            [
                {
                    "Event": event.name,
                    "Risk Level": event.risk_level,
                    "Portfolio Impact %": f"{event.portfolio_impact_pct:+.2f}%",
                    "Description": event.description,
                }
                for event in HEALTHCARE_EVENTS
            ]
        )
        st.dataframe(catalog_df, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="HealthRisk Lab", layout="wide")
    initialize_state()
    render_page_styles()
    render_header()
    render_scoreboard()
    st.divider()
    render_event_panel()
    st.divider()
    render_charts()
    render_history_table()
    render_event_catalog()


if __name__ == "__main__":
    main()
