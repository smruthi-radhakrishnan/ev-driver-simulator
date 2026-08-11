"""Tab 3 -- the combined chart, supporting views, agent explorer and validation panel."""

from __future__ import annotations

import streamlit as st

from app.state import is_stale, latest_result
from model import timegrid as tg
from model.aggregate import MIN_CONNECTED_FOR_BAND, compute_agent_trace, headline_metrics
from model.plotting import (
    build_agent_trace_chart,
    build_breakdown_chart,
    build_combined_chart,
    build_demand_chart,
    build_demand_per_agent_box,
    build_plugin_soc_histogram,
)
from model.population import population_summary
from model.validate import validate_energy_closure, validate_io_average


def render() -> None:
    result = latest_result()
    if result is None:
        st.info("Set up the population and assumptions, then press **Run simulation**.")
        return

    if is_stale():
        st.warning(
            "The scenario has changed since this run. Press **Run simulation** to refresh.",
            icon="⚠️",
        )

    _metrics(result)
    st.divider()

    st.subheader("Occupancy and State of Charge")
    st.caption(
        "Grey bars: share of the fleet plugged in at each 15-minute step. They are "
        "already a population aggregate, so they carry no uncertainty band. The three "
        "State of Charge lines sit on a shared scale — light for the 5th percentile, mid "
        "for the mean, dark for the 95th — because they are one quantity read at three "
        "points in its distribution."
    )
    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        scope = st.radio(
            "State of Charge measured across",
            options=["Whole fleet", "Connected agents"],
            horizontal=True,
            help=(
                "The whole fleet includes agents that have driven away, whose State of "
                "Charge is being depleted by the driving model. Agents with no plug-in "
                "event this window have no modelled State of Charge and are excluded "
                "from both."
            ),
        )
    with c2:
        min_connected = st.slider(
            "Suppress the State of Charge lines below this many agents",
            min_value=1,
            max_value=200,
            value=MIN_CONNECTED_FOR_BAND,
            help="Avoids reporting a percentile from a handful of agents.",
        )
    st.plotly_chart(
        build_combined_chart(
            result,
            min_connected=min_connected,
            connected_only=(scope == "Connected agents"),
        ),
        use_container_width=True,
    )

    st.divider()
    st.subheader("By archetype")
    normalise = st.toggle(
        "Show each archetype's own plug-in rate instead of its share of the fleet",
        value=False,
        help="Fairer comparison when one archetype is 40% of the fleet and another is 1%.",
    )
    st.plotly_chart(
        build_breakdown_chart(result, normalise=normalise), use_container_width=True
    )

    st.divider()
    st.subheader("Charging demand")
    per_agent = st.toggle("Show per-agent mean instead of fleet total", value=False)
    st.plotly_chart(build_demand_chart(result, per_agent=per_agent), use_container_width=True)

    st.caption(
        "Per-agent demand as a distribution. Energy rather than power, since an "
        "individual agent's power is either zero or its chargepoint rating."
    )
    metric = st.radio(
        "Measure",
        options=["Energy delivered (kWh)", "Charging duration (hrs)"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.plotly_chart(
        build_demand_per_agent_box(
            result, metric="energy" if metric.startswith("Energy") else "duration"
        ),
        use_container_width=True,
    )

    st.plotly_chart(build_plugin_soc_histogram(result), use_container_width=True)

    st.divider()
    _agent_explorer(result)

    st.divider()
    with st.expander("Model checks"):
        _validation_panel(result)

    st.divider()
    with st.expander("Realised population mix"):
        st.dataframe(
            population_summary(result.agents, result.archetypes),
            hide_index=True,
            use_container_width=True,
        )


def _metrics(result) -> None:
    m = headline_metrics(result)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Agents", f"{m['agents']:,.0f}")
    c2.metric("Peak plugged in", f"{m['peak_pct_plugged_in']:.1f}%")
    c3.metric(
        "Peak demand",
        f"{m['peak_demand_kw']:,.0f} kW",
        help=f"Reached at {m['peak_demand_time']}",
    )
    c4.metric("Energy delivered", f"{m['total_energy_kwh']:,.0f} kWh")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Agents plugging in", f"{m['pct_with_event']:.1f}%")
    c6.metric(
        "Median State of Charge",
        f"{m['median_plugin_soc']:.1f}% → {m['median_plugout_soc']:.1f}%",
        help="Plug-in to plug-out.",
    )
    c7.metric("Median charge duration", tg.format_hours(m["median_charge_duration_hrs"]))
    c8.metric(
        "Left below target",
        f"{m['pct_left_below_target']:.2f}%",
        help="Drivers whose sampled ready-by time arrived before charging finished.",
    )


def _agent_explorer(result) -> None:
    st.subheader("Agent explorer")
    events = result.events

    archetype = st.selectbox(
        "Archetype", options=[c.name for c in result.archetypes], key="explorer_archetype"
    )
    subset = events[events["archetype"] == archetype]
    if subset.empty:
        st.info(f"No agents assigned to {archetype} in this run.")
        return

    only_with_event = st.checkbox(
        "Only agents that plugged in", value=True, key="explorer_with_event"
    )
    if only_with_event:
        subset = subset[subset["has_event"]]
        if subset.empty:
            st.info(f"No {archetype} agent plugged in during this window.")
            return

    ids = subset["agent_id"].tolist()
    agent_id = st.select_slider("Agent", options=ids, value=ids[0], key="explorer_agent")

    st.plotly_chart(build_agent_trace_chart(result, agent_id), use_container_width=True)

    row = events[events["agent_id"] == agent_id].iloc[0]
    if not row["has_event"]:
        st.caption(
            "This agent has no plug-in event in this window: its renewal phase places "
            "the next event on another day (§4.7)."
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Plugged in", row["plugin_time"])
        c2.metric("Plugged out", row["plugout_time"])
        c3.metric("Charged", f"{row['charge_start_time']} – {row['charge_end_time']}")
        c4.metric("Energy", f"{row['energy_kwh']:.1f} kWh")
        st.caption(
            f"Plug-in State of Charge {row['plugin_soc']:.1f}% → target "
            f"{row['target_soc']:.0f}%, a {row['charge_duration_hrs']:.2f} hr charge."
        )
        if row["charge_window_clipped"]:
            st.caption(
                "The charge block did not fit its cheap window, so it was re-anchored "
                "to finish as close to the deadline as physically possible (§4.4)."
            )

    with st.expander("Raw trace"):
        trace = compute_agent_trace(result, agent_id).copy()
        trace["time"] = trace["time"].dt.strftime("%H:%M")
        st.dataframe(trace, hide_index=True, use_container_width=True)


def _validation_panel(result) -> None:
    """Two sanity checks on the model itself, kept collapsed.

    Neither is a result of the scenario -- they answer "is the model behaving", not
    "what does this fleet do" -- so they sit behind a fold rather than alongside the
    charts. Both are also printed by the headless runner and asserted in the tests.
    """
    st.markdown("**Intelligent Octopus Average against the published report**")
    st.caption(
        "The report states a 72.5 kWh battery, 7 kW, an 80% target, a 52% median plug-in "
        "state of charge and a 2.5 hour median charge duration. Those figures do not "
        "reconcile: the first four imply 2.90 hours. The likely cause is that 72.5 kWh is "
        "an *average* while the other two are *medians* — a right-skewed battery mix would "
        "put the median near 62.5 kWh, which reconciles them exactly. Using the mean is "
        "the right choice here because fleet energy depends on it, so the check below is "
        "against the duration the report's own inputs imply, with its stated 2.5 hours "
        "shown alongside as context."
    )
    table = validate_io_average(result)
    if table.empty:
        st.info(
            "The Intelligent Octopus Average archetype has no agents in this run, so "
            "there is nothing to compare."
        )
        return

    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Within tolerance": st.column_config.CheckboxColumn(
                "Within tolerance", help="Blank where the row is informational only."
            )
        },
    )
    st.caption(
        "The duration tolerance scales with the number of agents: the median of a few "
        "hundred samples carries real noise, and a fixed bound either failed at random or "
        "was loose enough to miss a genuine error. The simulated interquartile range is "
        "narrower than the report's because plug-in State of Charge is drawn from a single "
        "normal distribution, whereas real behaviour is more dispersed."
    )

    st.markdown("**Energy closure**")
    st.caption(
        "The energy taken out driving should equal the energy put back in overnight. "
        "The driving drop is derived from mileage and efficiency while the charge "
        "requirement comes from the sampled plug-in State of Charge, so their agreement "
        "is a check rather than an identity."
    )
    closure = validate_energy_closure(result)
    if closure.empty:
        st.info("No archetype in this run has a post-plug-out drive to check.")
    else:
        st.dataframe(closure, hide_index=True, use_container_width=True)
