"""Tab 4 -- scenario JSON download, reset, and CSV downloads.
"""

from __future__ import annotations

import json

import streamlit as st

from app.state import current_scenario, latest_result, request_scenario_load
from model.aggregate import (
    compute_archetype_breakdown,
    compute_charging_demand,
    compute_occupancy_bars,
    compute_soc_band,
)
from model.config import ScenarioConfig, default_archetypes
from model.population import population_summary
from model.validate import validate_io_average


def render() -> None:
    _config_section()
    st.divider()
    _data_section()


def _config_section() -> None:
    st.subheader("Scenario configuration")
    scenario = current_scenario()
    payload = json.dumps(scenario.to_dict(), indent=2)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.download_button(
            "Download scenario JSON",
            data=payload,
            file_name="model_scenario.json",
            mime="application/json",
            use_container_width=True,
        )
        st.caption(
            "The same file the headless runner accepts: "
            "`python scripts/run_headless.py --config model_scenario.json`"
        )
    with col2:
        if st.button("Reset to defaults", use_container_width=True):
            # Queued rather than applied here: the Tab 1 and 2 widgets already exist
            # in this script run, so their keys cannot be written until the next one.
            request_scenario_load(ScenarioConfig(archetypes=default_archetypes()))
            st.rerun()

    with st.expander("Current scenario JSON"):
        st.code(payload, language="json")


def _data_section() -> None:
    st.subheader("Results data")
    result = latest_result()
    if result is None:
        st.info("Run a simulation to download its results.")
        return

    tables = {
        "occupancy.csv": compute_occupancy_bars(result),
        "soc_band.csv": compute_soc_band(result),
        "charging_demand.csv": compute_charging_demand(result),
        "archetype_breakdown.csv": compute_archetype_breakdown(result),
        "event_log.csv": result.events,
        "population_summary.csv": population_summary(result.agents, result.archetypes),
        "validation_io_average.csv": validate_io_average(result),
    }

    st.caption(
        f"One row per 15-minute step, except the event log which has one row per "
        f"agent ({result.n_agents:,} rows)."
    )
    columns = st.columns(2, gap="medium")
    for i, (filename, frame) in enumerate(tables.items()):
        with columns[i % 2]:
            st.download_button(
                f"{filename}  ({len(frame):,} rows)",
                data=frame.to_csv(index=False),
                file_name=filename,
                mime="text/csv",
                use_container_width=True,
                key=f"dl_{filename}",
            )

    with st.expander("Preview the event log"):
        st.dataframe(result.events.head(200), hide_index=True, use_container_width=True)
