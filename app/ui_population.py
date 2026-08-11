"""Tab 1 -- the archetype mix, and what each archetype's distributions look like.

The structural numbers (battery, chargepoint, window, target) are summarised on the
Introduction tab and edited on the Assumptions tab, so they are not repeated here.
This tab answers "who is in the fleet, and how do they behave".
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.state import arch_key, archetype_at, current_archetypes
from model import timegrid as tg
from model.config import (
    ALWAYS_PLUGGED_IN,
    INFREQUENT_CHARGING,
    MANAGED,
)
from model.plotting import build_soc_distribution_preview, build_timing_preview
from model.population import allocate_counts

BEHAVIOUR_LABELS = {
    MANAGED: "managed (charges in a cheap window)",
    INFREQUENT_CHARGING: "unmanaged, plugs in every few days",
    ALWAYS_PLUGGED_IN: "unmanaged, connected except for one trip",
}
DEFAULT_BEHAVIOUR_LABEL = "unmanaged (charges on plug-in)"


def render() -> None:
    st.subheader("Population mix")
    st.caption(
        "Shares are treated as weights, so they need not total 100%. Agents are "
        "allocated by largest remainder, which keeps small archetypes exact rather "
        "than losing them to sampling noise."
    )

    archetypes = current_archetypes()
    left, right = st.columns([3, 2], gap="large")

    with left:
        for i, cfg in enumerate(archetypes):
            behaviour = BEHAVIOUR_LABELS.get(cfg.behaviour, DEFAULT_BEHAVIOUR_LABEL)
            st.slider(
                f"{cfg.name} — {behaviour}",
                min_value=0.0,
                max_value=100.0,
                step=0.5,
                key=arch_key(i, "population_pct"),
            )

    # Re-read after the sliders so the summary reflects this run's values.
    archetypes = current_archetypes()
    total = sum(c.population_pct for c in archetypes)
    n_agents = int(st.session_state["n_agents"])

    with right:
        if total <= 0:
            st.error("At least one archetype needs a non-zero share.")
            return
        st.metric("Total share entered", f"{total:.1f}%")
        if abs(total - 100.0) > 0.01:
            st.info(
                f"Shares total {total:.1f}%, so they will be normalised to 100% "
                "when the population is built."
            )

        counts = allocate_counts([c.population_pct for c in archetypes], n_agents)
        st.dataframe(
            pd.DataFrame(
                {
                    "Archetype": [c.name for c in archetypes],
                    "Share (%)": [100.0 * c.population_pct / total for c in archetypes],
                    "Agents": counts,
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.divider()
    st.subheader("Resulting distributions")
    st.caption(
        "Computed in closed form from the parameters on the Assumptions tab, so they "
        "respond immediately without needing a simulation run. The connection window is "
        "derived from the plug-in and plug-out distributions rather than set directly."
    )

    for i, cfg in enumerate(archetypes):
        if cfg.population_pct <= 0:
            continue
        with st.expander(cfg.name, expanded=(i == 0)):
            live = archetype_at(i)
            _distribution_previews(live, i)
            _summary_caption(live)


def _distribution_previews(live, index: int) -> None:
    """Timing and State of Charge previews, omitting rows that do not apply."""
    timing = build_timing_preview(live)
    if timing is not None:
        st.plotly_chart(timing, use_container_width=True, key=f"pop_timing_{index}")
    else:
        st.caption(
            "Connected for the whole window apart from one trip, so there is no "
            "plug-in, plug-out or ready-by distribution to show."
        )
    st.plotly_chart(
        build_soc_distribution_preview(live),
        use_container_width=True,
        key=f"pop_soc_{index}",
    )


def _summary_caption(live) -> None:
    # Reads the *mean* of the target distribution, not the legacy `target_soc` scalar.
    # That scalar only seeds the single-value default, and quoting it here understated the
    # requirement for every archetype carrying a spread of targets.
    energy = max(live.mean_target_soc - live.plugin_soc_mean, 0.0) / 100.0 * live.battery_kwh
    duration = energy / live.charger_kw if live.charger_kw else 0.0

    targets = live.target_soc_distribution
    if len(targets) == 1:
        target_text = f"target {targets[0][0]:.0f}%"
    else:
        mix = ", ".join(f"{t:.0f}% ({w:.0%})" for t, w in targets)
        target_text = f"targets {mix} — mean {live.mean_target_soc:.0f}%"

    parts = [
        f"{live.battery_kwh:g} kWh battery on a {live.charger_kw:g} kW chargepoint",
        target_text,
        f"mean requirement {energy:.1f} kWh, {tg.format_hours(duration)}",
    ]
    if not live.is_continuous:
        win_start, win_end = tg.window_offsets(live.window_start_hr, live.window_end_hr)
        parts.insert(
            1,
            f"plug-in window {tg.format_clock(win_start)}–{tg.format_clock(win_end)}",
        )
    st.caption(". ".join(parts) + ".")
