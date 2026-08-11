"""Tab 2 -- every editable assumption, one expander per archetype.

Two framing decisions keep this readable:

* Battery capacity and charge rate are **stated, not editable**. They define the
  vehicle, and the report's 72.5 kWh figure is a measured property of its dataset.
* A **ready-by setting** is something a driver types into a smart-charging app, so it
  only appears for the archetypes that have one. For everyone else the same underlying
  quantity is simply *when the car leaves*, and it is labelled that way.

Every derived caption re-reads the archetype from session state *after* its widgets have
been created, so the text tracks the controls within the same interaction rather than
lagging a rerun behind.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.state import (
    arch_key,
    archetype_at,
    current_archetypes,
    list_baseline_key,
    list_editor_key,
)
from model import timegrid as tg
from model.config import (
    CNZ_PLUGIN_DAYS_PER_WEEK,
    INFREQUENT_CHARGING,
    MANAGED,
)

QUARTER_HOUR = 15 * 60  # st.time_input step, in seconds


def render() -> None:
    st.subheader("Assumptions")
    st.caption(
        "Timing and State of Charge are sampled per agent per event. Charge duration is "
        "never set directly — it follows from the State of Charge gap, the battery and "
        "the charge rate. See the Population tab for the distributions these produce."
    )

    for i, cfg in enumerate(current_archetypes()):
        with st.expander(cfg.name, expanded=(i == 0)):
            _vehicle(i, cfg)
            if cfg.is_continuous:
                _trip(i)
            else:
                _plugin_frequency(i, cfg)
                _plugin_timing(i)
                _departure(i, cfg)
            _plugin_soc(i)
            if cfg.behaviour == MANAGED:
                _cheap_window(i)
            if cfg.behaviour == INFREQUENT_CHARGING:
                _cadence(i)
            _driving(i)


def _vehicle(index: int, cfg) -> None:
    st.markdown("**Vehicle**")
    live = archetype_at(index)
    c1, c2, c3 = st.columns(3)
    c1.metric("Battery", f"{live.battery_kwh:g} kWh")
    c2.metric("Charge rate", f"{live.charger_kw:g} kW")
    c3.metric(
        "Implied charge duration",
        tg.format_hours(live.mean_required_duration_hrs),
        help="From the mean plug-in State of Charge. Sampled per agent at run time.",
    )
    st.caption("Battery and charge rate are fixed constants for this archetype.")

    if not cfg.is_continuous:
        st.time_input(
            "Earliest plug-in",
            step=QUARTER_HOUR,
            key=arch_key(index, "window_start_hr"),
            help="The earliest the driver could get home and plug in.",
        )

    st.markdown("**Target state of charge**")
    st.caption(
        "A weighted choice, not one figure: the report finds the population split roughly "
        "evenly across 80%, 90% and 100%. Two drivers arriving at the same state of charge "
        "but holding different targets need very different charge lengths, which a single "
        "target cannot express."
    )
    _target_soc_editor(index)


def _target_soc_editor(index: int) -> None:
    """Target state of charge as a weighted choice. Same mechanism as ready-by."""
    field_name = "target_soc_choices"
    key = arch_key(index, field_name)
    baseline = (
        st.session_state.get(list_baseline_key(index, field_name))
        or st.session_state[key]
        or []
    )

    frame = pd.DataFrame(
        {
            "Target (%)": [float(t) for t, _ in baseline],
            "Weight": [float(w) for _, w in baseline],
        }
    )
    edited = st.data_editor(
        frame,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        key=list_editor_key(index, field_name),
        column_config={
            "Target (%)": st.column_config.NumberColumn(
                "Target (%)", min_value=10.0, max_value=100.0, step=5.0
            ),
            "Weight": st.column_config.NumberColumn(
                "Weight", min_value=0.0, step=0.01, help="Relative; need not total 1."
            ),
        },
    )

    parsed = [
        [float(t), float(w)]
        for t, w in zip(edited["Target (%)"], edited["Weight"])
        if pd.notna(t) and pd.notna(w) and float(w) > 0
    ]
    if parsed:
        st.session_state[key] = parsed
    else:
        st.warning("At least one target with a positive weight is required.")

    live = archetype_at(index)
    if len(live.target_soc_distribution) > 1:
        mix = ", ".join(f"{t:.0f}% {w:.0%}" for t, w in live.target_soc_distribution)
        st.caption(
            f"Mix: {mix}. Mode {live.modal_target_soc:.0f}%, weighted mean "
            f"{live.mean_target_soc:.1f}% — the mean is what the energy figures should be "
            "read against."
        )


def _plugin_frequency(index: int, cfg) -> None:
    """How often the driver plugs in -- distinct from how often they drive."""
    st.markdown("**Plug-in frequency**")
    if cfg.behaviour == INFREQUENT_CHARGING:
        st.caption(
            "Set by the cadence parameters below rather than here, since this archetype "
            "is defined by charging every few days."
        )
        return

    st.slider(
        "Plug-in days per week",
        min_value=1.0,
        max_value=7.0,
        step=0.1,
        key=arch_key(index, "plugin_days_per_week"),
        help=(
            "Agents that do not plug in on a given day simply do not appear in that "
            "window. Separate from driving frequency: a driver can drive daily and still "
            "plug in every other day."
        ),
    )
    live = archetype_at(index)
    st.caption(
        f"{live.expected_plugins_per_day:.2f} plug-ins per day, so about "
        f"{100 * live.expected_plugins_per_day:.0f}% of these agents connect in any one "
        f"window. The Intelligent Octopus study implies about "
        f"{CNZ_PLUGIN_DAYS_PER_WEEK:.1f} days a week; the default is daily because the "
        "mileage figures were derived on that basis."
    )
    if live.plugin_days_per_week < 7.0:
        st.info(
            "Lowering this raises the energy each session has to deliver for the same "
            "annual mileage. Check the energy closure under **Model checks** on the "
            "Results tab — mileage may need re-deriving to match."
        )


def _plugin_timing(index: int) -> None:
    st.markdown("**Plug-in time**")
    c1, c2 = st.columns(2)
    with c1:
        st.time_input(
            "Mean plug-in time", step=QUARTER_HOUR, key=arch_key(index, "plugin_time_mean_hr")
        )
    with c2:
        st.slider(
            "Spread",
            min_value=0,
            max_value=240,
            step=5,
            format="%d min",
            key=arch_key(index, "plugin_time_std_hr"),
        )

    live = archetype_at(index)
    win_start, win_end = tg.window_offsets(live.window_start_hr, live.window_end_hr)
    mean = tg.to_window_offset(live.plugin_time_mean_hr)
    st.caption(
        f"Mean {tg.format_clock(mean)} ± {tg.format_hours(live.plugin_time_std_hr)}, "
        f"truncated to {tg.format_clock(win_start)}–{tg.format_clock(win_end)}."
    )
    if not (win_start <= mean <= win_end):
        st.warning(
            "The mean plug-in time sits outside the plug-in window, so sampled plug-ins "
            "will pile up on whichever edge of the window is nearer."
        )


def _departure(index: int, cfg) -> None:
    """Departure times, framed as a ready-by setting only where one actually exists."""
    managed = cfg.behaviour == MANAGED
    if managed:
        st.markdown("**Ready-by setting and departure**")
        st.caption(
            "The ready-by time is what the driver asks the charger for, and what "
            "scheduling aims at. Departure is when the car actually leaves: the "
            "ready-by time plus an offset plus noise. A driver leaving before the "
            "charge finishes departs below target."
        )
    else:
        st.markdown("**Departure time**")
        st.caption(
            "When the car leaves in the morning. This archetype has no smart-charging "
            "schedule, so there is no ready-by setting to distinguish from it."
        )

    _departure_editor(index, managed)

    c1, c2 = st.columns(2)
    with c1:
        if managed:
            st.slider(
                "Departure offset from ready-by",
                min_value=-60,
                max_value=180,
                step=5,
                format="%d min",
                key=arch_key(index, "plugout_offset_hr"),
                help="Positive means drivers typically leave after the time they set.",
            )
    with c2:
        st.slider(
            "Departure spread",
            min_value=0,
            max_value=180,
            step=5,
            format="%d min",
            key=arch_key(index, "plugout_time_std_hr"),
        )

    live = archetype_at(index)
    mean_offset = (
        tg.window_offsets(live.window_start_hr, live.mean_readyby_hr)[1] + live.plugout_offset_hr
    )
    st.caption(
        f"Mean departure {tg.format_clock(mean_offset)} ± "
        f"{tg.format_hours(live.plugout_time_std_hr)}."
    )


def _departure_editor(index: int, managed: bool) -> None:
    """A weighted choice among times, not a continuous density.

    Both a ready-by setting and a habitual departure time land on times people actually
    pick, so the same discrete mechanism serves both -- only the label differs.

    The editor writes back to a plain session-state key rather than binding a widget to
    it, because the value is a list and ``st.data_editor`` holds its edits as deltas
    against the frame it was handed. See ``load_archetypes`` for why that frame has to
    stay fixed.
    """
    key = arch_key(index, "readyby_choices")
    baseline = (
        st.session_state.get(list_baseline_key(index, "readyby_choices"))
        or st.session_state[key]
        or []
    )
    column = "Ready-by" if managed else "Departs"

    frame = pd.DataFrame(
        {
            column: [tg.hours_to_time(h) for h, _ in baseline],
            "Weight": [float(w) for _, w in baseline],
        }
    )
    edited = st.data_editor(
        frame,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        key=list_editor_key(index, "readyby_choices"),
        column_config={
            column: st.column_config.TimeColumn(column, format="HH:mm", step=900),
            "Weight": st.column_config.NumberColumn(
                "Weight", min_value=0.0, step=0.1, help="Relative; need not total 1."
            ),
        },
    )

    parsed = [
        [tg.time_to_hours(t), float(w)]
        for t, w in zip(edited[column], edited["Weight"])
        if t is not None and pd.notna(w) and float(w) > 0
    ]
    if parsed:
        st.session_state[key] = parsed
    else:
        st.warning("At least one time with a positive weight is required.")


def _plugin_soc(index: int) -> None:
    st.markdown("**State of Charge at plug-in**")
    c1, c2 = st.columns(2)
    with c1:
        st.slider(
            "Mean (%)",
            min_value=0.0,
            max_value=100.0,
            step=0.5,
            key=arch_key(index, "plugin_soc_mean"),
        )
    with c2:
        st.slider(
            "Spread (percentage points)",
            min_value=0.0,
            max_value=30.0,
            step=0.5,
            key=arch_key(index, "plugin_soc_std"),
        )

    live = archetype_at(index)
    energy = max(live.mean_target_soc - live.plugin_soc_mean, 0.0) / 100.0 * live.battery_kwh
    duration = energy / live.charger_kw if live.charger_kw else 0.0
    st.caption(
        f"Mean requirement {live.mean_target_soc - live.plugin_soc_mean:.1f} percentage "
        f"points, {energy:.1f} kWh, {tg.format_hours(duration)}."
    )
    if live.plugin_soc_mean >= live.mean_target_soc:
        st.warning(
            f"Mean plug-in State of Charge ({live.plugin_soc_mean:.0f}%) is at or above "
            f"the {live.mean_target_soc:.0f}% mean target, so most agents will barely charge."
        )


def _cheap_window(index: int) -> None:
    st.markdown("**Cheap-rate charging window**")
    st.caption(
        "Charging is scheduled inside this window, set by the tariff rather than by "
        "when the driver needs the car."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.time_input("Opens", step=QUARTER_HOUR, key=arch_key(index, "cheap_window_start_hr"))
    with c2:
        st.time_input("Closes", step=QUARTER_HOUR, key=arch_key(index, "cheap_window_end_hr"))
    with c3:
        st.slider(
            "Placement spread",
            min_value=0,
            max_value=240,
            step=5,
            format="%d min",
            key=arch_key(index, "charge_placement_std_hr"),
            help="Spread of where in the window the charge block starts.",
        )

    live = archetype_at(index)
    cheap_start, cheap_end = tg.window_offsets(
        live.cheap_window_start_hr, live.cheap_window_end_hr
    )
    st.caption(
        f"{tg.format_clock(cheap_start)}–{tg.format_clock(cheap_end)} "
        f"({tg.format_hours(cheap_end - cheap_start)}), placement spread "
        f"{tg.format_hours(live.charge_placement_std_hr or 0.0)}."
    )


def _cadence(index: int) -> None:
    st.markdown("**Plug-in cadence**")
    st.caption(
        "Days between plug-in events, from a triangular distribution. Each agent gets a "
        "random phase within its own gap, so no warm-up period is needed."
    )
    columns = st.columns(3)
    fields = (
        ("Minimum (days)", "interplug_gap_min_days", 30.0),
        ("Most likely (days)", "interplug_gap_mode_days", 30.0),
        ("Maximum (days)", "interplug_gap_max_days", 60.0),
    )
    for column, (label, field, ceiling) in zip(columns, fields):
        with column:
            st.number_input(
                label,
                min_value=0.5,
                max_value=ceiling,
                step=0.5,
                key=arch_key(index, field),
            )

    live = archetype_at(index)
    lo, mode, hi = (
        live.interplug_gap_min_days,
        live.interplug_gap_mode_days,
        live.interplug_gap_max_days,
    )
    if not (lo <= mode <= hi):
        st.error("The gap parameters must satisfy minimum ≤ most likely ≤ maximum.")
    else:
        st.caption(
            f"Mean gap {live.expected_plugin_gap_days:.2f} days, so about "
            f"{100.0 * live.expected_plugins_per_day:.0f}% of these agents plug in on "
            "any given day."
        )


def _trip(index: int) -> None:
    st.markdown("**Daily trip**")
    st.caption(
        "This archetype is connected for the whole window apart from one trip, placed at "
        "random between 6 and 18 hours in. Charging resumes on reconnection, so it has no "
        "plug-in or departure distribution."
    )
    st.slider(
        "Trip duration",
        min_value=15,
        max_value=480,
        step=15,
        format="%d min",
        key=arch_key(index, "trip_duration_hrs"),
    )


def _driving(index: int) -> None:
    st.markdown("**Driving**")
    st.caption(
        "Mileage, efficiency and frequency set how far State of Charge falls after the "
        "car leaves. Reducing the frequency lengthens each trip rather than shrinking "
        "it, since annual mileage is unchanged. How long the drive takes is a fixed "
        "assumption: it changes how steeply the line falls, not how far."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input(
            "Annual mileage",
            min_value=0.0,
            max_value=60_000.0,
            step=100.0,
            key=arch_key(index, "miles_per_year"),
        )
    with c2:
        st.number_input(
            "Efficiency (mi/kWh)",
            min_value=1.0,
            max_value=8.0,
            step=0.1,
            key=arch_key(index, "efficiency_mi_per_kwh"),
        )
    with c3:
        st.slider(
            "Driving days per week",
            min_value=0.0,
            max_value=7.0,
            step=0.5,
            key=arch_key(index, "driving_days_per_week"),
        )

    live = archetype_at(index)
    requirement = live.mean_target_soc - live.plugin_soc_mean
    if live.driving_probability_per_day <= 0:
        st.caption("Never drives, so State of Charge does not fall after plug-out.")
        return
    st.caption(
        f"{live.trip_miles:.1f} miles per trip on "
        f"{100 * live.driving_probability_per_day:.0f}% of days "
        f"({live.daily_miles:.1f} miles/day averaged), {live.trip_energy_kwh:.1f} kWh — a "
        f"{live.trip_soc_drop_pct:.1f} point drop per trip, "
        f"{live.expected_daily_soc_drop_pct:.1f} points a day. Over a "
        f"{live.expected_plugin_gap_days:.2f} day plug-in gap that is "
        f"{live.driving_soc_drop_between_plugins_pct:.1f} points, against a "
        f"{requirement:.1f} point charge requirement."
    )
    if abs(live.driving_soc_drop_between_plugins_pct - requirement) > 5.0:
        st.warning(
            "Depletion between plug-ins and the charge requirement differ by more than "
            "5 percentage points, so State of Charge will drift between successive days "
            "rather than returning to where it started."
        )
