"""Streamlit entry point: sidebar controls plus tab routing.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow `streamlit run app/streamlit_app.py` from a clone with no install step.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import (  # noqa: E402
    ui_assumptions,
    ui_export,
    ui_intro,
    ui_population,
    ui_rationale,
    ui_results,
)
from app.state import (  # noqa: E402
    current_scenario,
    has_result,
    init_state,
    is_stale,
    request_run,
)

st.set_page_config(page_title="EV charging agent-based model", page_icon="🔌", layout="wide")


def sidebar() -> None:
    with st.sidebar:
        st.title("EV charging agent-based model")
        st.caption(
            "Agent-based model of EV home charging across six driver archetypes, "
            "simulated over a single 24-hour window anchored at noon."
        )

        st.number_input(
            "Population size",
            min_value=50,
            max_value=100_000,
            step=250,
            key="n_agents",
            help=(
                "The State of Charge percentiles get their sample size from the agent "
                "count, so a few thousand agents gives a stable band."
            ),
        )
        st.number_input(
            "Random seed",
            min_value=0,
            max_value=1_000_000,
            step=1,
            key="seed",
            help="Same seed and same scenario reproduce the run exactly.",
        )

        total_pct = sum(a.population_pct for a in current_scenario().archetypes)
        runnable = total_pct > 0
        if not runnable:
            st.error("Give at least one archetype a non-zero population share.")

        st.button(
            "Run simulation",
            type="primary",
            use_container_width=True,
            disabled=not runnable,
            on_click=request_run,
        )

        if is_stale():
            st.caption("⚠️ Scenario edited since the last run.")
        elif has_result():
            st.caption("✅ Results are up to date.")

        st.divider()
        st.caption(
            "Charge duration is always derived from the State of Charge gap, never "
            "set directly. Managed archetypes charge inside a cheap-rate window "
            "rather than against the driver's deadline."
        )


def main() -> None:
    init_state()
    sidebar()

    intro, population, assumptions, rationale, results, export = st.tabs(
        [
            "Introduction",
            "Population",
            "Assumptions",
            "Rationale",
            "Results",
            "Export",
        ]
    )
    with intro:
        ui_intro.render()
    with population:
        ui_population.render()
    with assumptions:
        ui_assumptions.render()
    with results:
        ui_results.render()
    with rationale:
        ui_rationale.render()
    with export:
        ui_export.render()


main()
