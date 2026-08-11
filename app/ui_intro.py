"""Tab 0 -- objective, method and assumptions."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.state import current_archetypes
from model import timegrid as tg


def render() -> None:
    st.subheader("Objective")
    st.markdown(
        "Estimate when a fleet of electric vehicles is plugged in at home, and what "
        "state of charge it holds, across a 24-hour period. Each driver is simulated "
        "individually and the population is then aggregated, so the effect of changing "
        "the mix of driver types or any single behavioural assumption can be read off "
        "directly."
    )

    st.subheader("Method")
    st.markdown(
        """
Six driver archetypes are defined, each with its own average plug-in frequency, plug-in
times, battery capacity and charging demand. Agents are allocated to archetypes in the
proportions set on the Population tab.

For every agent the model samples a plug-in time, a state of charge at plug-in and a
plug-out time — including a ready-by time for Intelligent Octopus — from that
archetype's distributions. The energy needed to reach the target follows from the state
of charge gap, the battery capacity and the chargepoint rating, which fixes how long the
charge takes. Where that charge sits in the night depends on the type of charging:

- **Unmanaged charging** begins the moment the car is plugged in.
- **Scheduled charging** runs to a timer the driver sets once, so it starts at much the
  same time every night regardless of the tariff or of when the car is next needed.
- **Managed charging** is placed by the platform inside a cheap-rate window, against
  wholesale prices rather than against the driver's ready-by time.

Scheduled and managed charging share the same placement mechanism here; what differs is
what sets the window and how tightly the start clusters within it.

After plug-out, state of charge falls by an amount derived from the archetype's annual
mileage, efficiency and driving frequency, spread over the drive to the next plug-in.

Each agent's trace is sampled onto a 15-minute grid, then aggregated across the fleet.
"""
    )

    st.subheader("Population as configured")
    archetypes = current_archetypes()
    total = sum(c.population_pct for c in archetypes) or 1.0
    rows = []
    for cfg in archetypes:
        if cfg.is_continuous:
            window = "continuous"
        else:
            start, end = tg.window_offsets(cfg.window_start_hr, cfg.window_end_hr)
            window = f"{tg.format_clock(start)}–{tg.format_clock(end)}"
        rows.append(
            {
                "Archetype": cfg.name,
                "Share": f"{100.0 * cfg.population_pct / total:.1f}%",
                "Battery": f"{cfg.battery_kwh:g} kWh",
                "Charger": f"{cfg.charger_kw:g} kW",
                "Plug-in window": window,
                "Plug-in SoC": f"{cfg.plugin_soc_mean:.0f}% ± {cfg.plugin_soc_std:.0f}",
                "Target": (
                    f"{cfg.mean_target_soc:.0f}% mean"
                    if len(cfg.target_soc_distribution) > 1
                    else f"{cfg.mean_target_soc:.0f}%"
                ),
                "Charge duration": tg.format_hours(cfg.mean_required_duration_hrs),
                "Charging": cfg.charging_kind,
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        "Charge duration is the time implied by each archetype's mean plug-in state of "
        "charge, battery and chargepoint. It is derived, not entered, and is sampled "
        "per agent at run time."
    )

    st.caption(
        "One archetype, Intelligent Octopus Average, is based on published telemetry "
        "from customers on an automated smart-charging tariff. The others come from the "
        "project brief. All values on every tab are editable; set the population size "
        "and seed in the sidebar, then press Run simulation."
    )
