"""Plotly figures (§4.9). Shared unchanged by the headless script and the Streamlit app.

Nothing here imports Streamlit; every function takes a
:class:`~model.simulator.SimulationResult` and returns a ``go.Figure``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model import timegrid as tg
from model.aggregate import (
    compute_agent_trace,
    compute_archetype_breakdown,
    compute_charging_demand,
    compute_demand_per_agent,
    compute_occupancy_bars,
    compute_soc_band,
)
from model.distributions import (
    connected_probability,
    truncated_normal_pdf,
)
from model.events import SOC_CEILING, SOC_FLOOR
from model.simulator import SimulationResult


TRANSPARENT = "rgba(0,0,0,0)"
GRID_COLOUR = "rgba(128,128,128,0.20)"
MUTED_TEXT = "rgba(128,128,128,0.95)"

OCCUPANCY_COLOUR = "rgba(130,136,148,0.55)"

SOC_COLOUR = "#f1682e"  # the mean, and the scale's mid stop
SOC_P5_COLOUR = "#fdd870"
SOC_P95_COLOUR = "#b3182b"
SOC_BAND_FILL = "rgba(228, 87, 46, 0.16)"
DEMAND_COLOUR = "#4c78a8"

ARCHETYPE_COLOURS = ["#4c78a8", "#f58518", "#72b7b2", "#54a24b", "#b279a2", "#9d755d"]

_BAR_WIDTH_MS = tg.RESOLUTION_MINUTES * 60 * 1000  # Plotly bar width is in ms on a date axis


def _style(
    fig: go.Figure,
    title: str | None = None,
    height: int | None = None,
    legend: bool = False,
) -> go.Figure:
    """Apply the shared, theme-neutral chart styling."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)) if title else None,
        paper_bgcolor=TRANSPARENT,
        plot_bgcolor=TRANSPARENT,
        font=dict(size=12),
        hovermode="x unified",
        hoverlabel=dict(font_size=12),
        showlegend=legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            font=dict(size=11),
            bgcolor=TRANSPARENT,
        ),
        margin=dict(t=64 if title else 40, r=64, l=64, b=52),
        height=height,
        bargap=0,  # 96 slots read as a silhouette, not a comb
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False, linecolor=GRID_COLOUR, ticks="outside", ticklen=4
    )
    fig.update_yaxes(gridcolor=GRID_COLOUR, zeroline=False, showline=False)
    return fig


def _time_axis(fig: go.Figure, row: int | None = None, col: int | None = None) -> None:
    kwargs = dict(
        title_text="Time of day",
        tickformat="%H:%M",
        dtick=2 * 60 * 60 * 1000,  # a tick every 2 hours
        showgrid=False,
    )
    if row is None:
        fig.update_xaxes(**kwargs)
    else:
        fig.update_xaxes(row=row, col=col, **kwargs)


def build_combined_chart(
    result: SimulationResult, min_connected: int = 30, connected_only: bool = True
) -> go.Figure:
    """The primary deliverable: % plugged in (black bars, left) + SoC (red, right)."""
    occupancy = compute_occupancy_bars(result)
    band = compute_soc_band(result, min_connected=min_connected, connected_only=connected_only)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    scope = "connected agents" if connected_only else "whole fleet"

    fig.add_trace(
        go.Bar(
            x=occupancy["time"],
            y=occupancy["pct_plugged_in"],
            name="% plugged in",
            marker_color=OCCUPANCY_COLOUR,
            marker_line_width=0,
            width=_BAR_WIDTH_MS,
            hovertemplate="%{y:.1f}% plugged in<extra></extra>",
        ),
        secondary_y=False,
    )

    # Percentiles as separate lines rather than a shaded band, all solid and separated by
    # colour alone. Slightly thinner than the mean, so it still reads as the main series.
    for column, label, colour in (
        ("soc_p95", "95th percentile", SOC_P95_COLOUR),
        ("soc_p5", "5th percentile", SOC_P5_COLOUR),
    ):
        fig.add_trace(
            go.Scatter(
                x=band["time"],
                y=band[column],
                name=label,
                mode="lines",
                line=dict(color=colour, width=1.75, shape="spline", smoothing=0.4),
                hovertemplate=f"{label} %{{y:.1f}}%<extra></extra>",
            ),
            secondary_y=True,
        )
    fig.add_trace(
        go.Scatter(
            x=band["time"],
            # Named for the scope actually plotted. Hardcoding "(connected)" here
            # contradicted the title whenever the whole-fleet view was selected.
            y=band["soc_mean"],
            name=f"Mean SoC, {scope}",
            mode="lines",
            line=dict(color=SOC_COLOUR, width=2.5, shape="spline", smoothing=0.4),
            hovertemplate="mean SoC %{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )

    _style(fig, title=f"Plug-in occupancy and State of Charge — {scope}", legend=True)
    _time_axis(fig)
    fig.update_yaxes(
        title_text="% of fleet plugged in",
        secondary_y=False,
        range=[0, 100],
        gridcolor=GRID_COLOUR,
    )
    fig.update_yaxes(
        title_text="State of Charge (%)",
        secondary_y=True,
        range=[0, 100],
        color=SOC_COLOUR,
        showgrid=False,
    )
    _annotate_suppressed_band(fig, band)
    return fig


def _annotate_suppressed_band(fig: go.Figure, band) -> None:
    """Flag the §4.9 edge case rather than leaving an unexplained gap in the red line."""
    suppressed = int(band["band_suppressed"].sum())
    if not suppressed:
        return
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1.0,
        y=-0.16,
        showarrow=False,
        xanchor="right",
        font=dict(size=10, color=MUTED_TEXT),
        text=(
            f"SoC band omitted at {suppressed} of {len(band)} timesteps "
            "(fewer than 30 agents connected)"
        ),
    )


def build_breakdown_chart(result: SimulationResult, normalise: bool = False) -> go.Figure:
    """Stacked % plugged in by archetype.

    ``normalise=False`` stacks each archetype's share of the whole fleet, so the
    stack total equals the combined chart's occupancy curve. ``normalise=True``
    instead shows each archetype's own plug-in rate as separate lines, which is the
    fairer comparison when one archetype is 40% of the fleet and another is 1%.
    """
    breakdown = compute_archetype_breakdown(result)
    fig = go.Figure()
    if breakdown.empty:
        return fig

    for i, cfg in enumerate(result.archetypes):
        subset = breakdown[breakdown["archetype"] == cfg.name]
        if subset.empty:
            continue
        colour = ARCHETYPE_COLOURS[i % len(ARCHETYPE_COLOURS)]
        if normalise:
            fig.add_trace(
                go.Scatter(
                    x=subset["time"],
                    y=subset["pct_of_archetype"],
                    name=cfg.name,
                    mode="lines",
                    line=dict(color=colour, width=2),
                    hovertemplate=f"{cfg.name}: %{{y:.1f}}%<extra></extra>",
                )
            )
        else:
            fig.add_trace(
                go.Bar(
                    x=subset["time"],
                    y=subset["pct_of_fleet"],
                    name=cfg.name,
                    marker_color=colour,
                    marker_line_width=0,
                    width=_BAR_WIDTH_MS,
                    hovertemplate=f"{cfg.name}: %{{y:.1f}}%<extra></extra>",
                )
            )

    _style(
        fig,
        title="Plug-in rate within each archetype" if normalise else "Plugged in by archetype",
        legend=True,
    )
    fig.update_layout(barmode="stack")
    _time_axis(fig)
    fig.update_yaxes(
        title_text="% of archetype plugged in" if normalise else "% of fleet plugged in",
        rangemode="tozero",
    )
    return fig


def build_demand_chart(result: SimulationResult, per_agent: bool = False) -> go.Figure:
    """Aggregate charging power demand across the fleet."""
    demand = compute_charging_demand(result)
    column = "mean_kw_per_agent" if per_agent else "total_kw"
    label = "Mean charging power per agent (kW)" if per_agent else "Total charging power (kW)"

    fig = go.Figure(
        go.Scatter(
            x=demand["time"],
            y=demand[column],
            name=label,
            mode="lines",
            line=dict(color=DEMAND_COLOUR, width=2, shape="spline", smoothing=0.4),
            fill="tozeroy",
            fillcolor="rgba(76, 120, 168, 0.22)",
            hovertemplate="%{x|%H:%M}<br>%{y:,.1f} kW<extra></extra>",
        )
    )
    _style(fig, title="Aggregate charging demand")
    _time_axis(fig)
    fig.update_yaxes(title_text=label, rangemode="tozero")
    _shade_evening_peak(fig, demand[column].max())
    return fig


def _shade_evening_peak(fig: go.Figure, y_max: float) -> None:
    """Mark the 17:00-20:00 grid peak -- report Figure 14 shows IO charging avoids it."""
    if not np.isfinite(y_max) or y_max <= 0:
        return
    fig.add_vrect(
        x0=tg.offset_to_timestamp(tg.to_window_offset(17.0)),
        x1=tg.offset_to_timestamp(tg.to_window_offset(20.0)),
        fillcolor="rgba(228, 87, 46, 0.08)",
        line_width=0,
        layer="below",
        annotation_text="grid peak 17:00-20:00",
        annotation_position="top left",
        annotation_font_size=10,
        annotation_font_color=MUTED_TEXT,
    )


def build_agent_trace_chart(result: SimulationResult, agent_id: int) -> go.Figure:
    """One agent's plug status and SoC over the window."""
    trace = compute_agent_trace(result, agent_id)
    row = result.events[result.events["agent_id"] == agent_id].iloc[0]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=trace["time"],
            y=trace["plugged_in"].astype(float) * 100.0,
            name="Plugged in",
            marker_color="rgba(130,136,148,0.30)",
            width=_BAR_WIDTH_MS,
            hovertemplate="%{x|%H:%M}<br>plugged in<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=trace["time"],
            y=trace["charging"].astype(float) * 100.0,
            name="Charging",
            marker_color="rgba(76,120,168,0.60)",
            width=_BAR_WIDTH_MS,
            hovertemplate="%{x|%H:%M}<br>charging<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=trace["time"],
            y=trace["soc"],
            name="State of Charge",
            mode="lines",
            line=dict(color=SOC_COLOUR, width=2.5),
            hovertemplate="%{x|%H:%M}<br>SoC %{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )

    _style(fig, title=f"Agent {agent_id} — {row['archetype']}", legend=True)
    fig.update_layout(barmode="overlay")
    _time_axis(fig)
    fig.update_yaxes(
        title_text="Connection status",
        secondary_y=False,
        range=[0, 105],
        showticklabels=False,
        showgrid=False,
    )
    fig.update_yaxes(
        title_text="State of Charge (%)",
        secondary_y=True,
        range=[0, 100],
        color=SOC_COLOUR,
        showgrid=False,
    )
    return fig


def build_demand_per_agent_box(result: SimulationResult, metric: str = "energy") -> go.Figure:
    """Box plot of per-agent charging demand, one box per archetype.

    ``metric`` is "energy" (kWh delivered) or "duration" (hours charging). Per-agent
    *power* is not plotted because it is either zero or the charger rating.
    """
    per_agent = compute_demand_per_agent(result)
    column, label, unit = (
        ("energy_kwh", "Energy delivered per agent (kWh)", "kWh")
        if metric == "energy"
        else ("charge_duration_hrs", "Charging duration per agent (hrs)", "hrs")
    )

    fig = go.Figure()
    for i, cfg in enumerate(result.archetypes):
        subset = per_agent[per_agent["archetype"] == cfg.name]
        if subset.empty:
            continue
        fig.add_trace(
            go.Box(
                y=subset[column],
                name=cfg.name,
                marker_color=ARCHETYPE_COLOURS[i % len(ARCHETYPE_COLOURS)],
                boxmean=True,
                boxpoints="outliers",
                hovertemplate=f"{cfg.name}<br>%{{y:.2f}} {unit}<extra></extra>",
            )
        )

    _style(fig, title="Charging demand per agent")
    fig.update_layout(hovermode="closest", margin=dict(t=64, r=64, l=64, b=96))
    fig.update_xaxes(tickangle=-20)
    fig.update_yaxes(title_text=label, rangemode="tozero")
    return fig


def build_soc_distribution_preview(cfg) -> go.Figure:
    """Assumptions preview: the plug-in SoC distribution implied by the parameters.

    Drawn analytically rather than from a run, so it responds to the sliders immediately.
    A *truncated* normal, matching :func:`model.events.sample_plugin_soc` -- an earlier
    version drew clipping atoms at the bounds because the sampler clipped, and both have
    been corrected together.
    """
    grid = np.linspace(SOC_FLOOR, SOC_CEILING, 401)
    density = truncated_normal_pdf(
        grid, cfg.plugin_soc_mean, cfg.plugin_soc_std, SOC_FLOOR, SOC_CEILING
    )
    # A probability density on a percentage axis is not a number anyone can check, so the
    # axis carries units instead: share of drivers per percentage point of state of charge.
    # The same curve scaled by 100, which is what makes tick labels worth showing at all.
    share = density * 100.0
    headroom = float(share.max()) * 1.35

    fig = go.Figure(
        go.Scatter(
            x=grid,
            y=share,
            mode="lines",
            name="Plug-in SoC",
            line=dict(color=SOC_COLOUR, width=2, shape="spline", smoothing=0.4),
            fill="tozeroy",
            fillcolor=SOC_BAND_FILL,
            hovertemplate="%{x:.0f}%: %{y:.2f}% of drivers<extra></extra>",
        )
    )

    target = cfg.mean_target_soc
    requirement = target - cfg.plugin_soc_mean
    if requirement > 0:
        fig.add_vrect(
            x0=cfg.plugin_soc_mean,
            x1=target,
            fillcolor="rgba(128,128,128,0.10)",
            line_width=0,
            layer="below",
        )
        # Horizontal, and pinned low. The two line labels are rotated at the top, so
        # separating this one vertically is what keeps all three legible when the means sit
        # close together -- the case that made the original horizontal labels collide.
        fig.add_annotation(
            x=(cfg.plugin_soc_mean + target) / 2.0,
            y=0.08,
            yref="paper",
            text=f"requirement {requirement:.0f}pp",
            showarrow=False,
            font=dict(size=10, color=MUTED_TEXT),
        )

    # Rotated labels beside each line, following _mark_departures. Vertical text is a dozen
    # pixels wide, so two lines a few percentage points apart no longer overwrite each other.
    # They also sit on opposite sides of their own line -- plug-in to the right, target to
    # the left -- so a slider that drives plug-in SoC up onto the target degrades into two
    # adjacent labels rather than two stacked on the same pixels.
    for value, colour, dash, anchor, shift, label in (
        (cfg.plugin_soc_mean, SOC_COLOUR, "dash", "left", 3,
         f"mean plug-in {cfg.plugin_soc_mean:.0f}%"),
        (target, MUTED_TEXT, "solid", "right", -3, f"mean target {target:.0f}%"),
    ):
        fig.add_vline(x=value, line=dict(color=colour, width=1.5, dash=dash))
        fig.add_annotation(
            x=value,
            y=0.97,
            yref="paper",
            text=label,
            textangle=-90,
            showarrow=False,
            xanchor=anchor,
            yanchor="top",
            xshift=shift,
            font=dict(size=10, color=colour),
        )

    _style(fig, title="Plug-in State of Charge distribution", height=300)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(
        title_text="State of Charge at plug-in (%)", range=[0, 100], dtick=10, ticksuffix="%"
    )
    fig.update_yaxes(
        title_text="Share of drivers (per pp)",
        range=[0, headroom],
        showticklabels=True,
        showgrid=True,
        ticksuffix="%",
        tickformat=".1f",
    )
    return fig


def plugin_distribution(cfg) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Plug-in density on the grid, plus the parameters that generated it."""
    offsets = tg.offsets()
    win_start, win_end = tg.window_offsets(cfg.window_start_hr, cfg.window_end_hr)
    mean = tg.to_window_offset(cfg.plugin_time_mean_hr)
    if mean < win_start:
        mean += tg.WINDOW_HOURS
    params = (mean, cfg.plugin_time_std_hr, win_start, win_end)
    return truncated_normal_pdf(offsets, *params), params


def plugout_mixture(cfg) -> list[tuple[float, float, float]]:
    """Ready-by mixture as (mean_offset, std, weight) components.

    One component per configured ready-by time, each shifted by the plug-out offset,
    so the whole plug-out distribution stays closed form however many deadlines the
    archetype offers.
    """
    return [
        (
            tg.window_offsets(cfg.window_start_hr, hour)[1] + cfg.plugout_offset_hr,
            cfg.plugout_time_std_hr,
            weight,
        )
        for hour, weight in cfg.readyby_distribution
    ]


def build_timing_preview(cfg) -> Optional[go.Figure]:
    """Assumptions preview: the share of this archetype plugged in, over the window.

    A single derived curve rather than separate views of its inputs. The two sampled
    distributions are already legible in it -- the rising edge *is* the plug-in
    distribution and the falling edge *is* the departure distribution, each integrated --
    so plotting them again as heat strips added rows without adding information.

        P(plugged in at t) = P(plug-in <= t) x P(departure > t)

    Closed form, so it responds to the controls without a simulation run. Returns
    ``None`` for an always-connected archetype, which has no plug-in or departure
    distribution at all.
    """
    if cfg.is_continuous:
        return None

    offsets = tg.offsets()
    times = tg.make_timegrid()
    _, plugin_params = plugin_distribution(cfg)
    components = plugout_mixture(cfg)
    connected = connected_probability(offsets, plugin_params, components) * 100.0

    connected = connected * min(cfg.expected_plugins_per_day, 1.0)

    fig = go.Figure(
        go.Scatter(
            x=times,
            y=connected,
            mode="lines",
            name="Plugged in",
            line=dict(color="rgba(130,136,148,0.95)", width=2, shape="spline", smoothing=0.4),
            fill="tozeroy",
            fillcolor="rgba(130,136,148,0.22)",
            hovertemplate="%{x|%H:%M}<br>%{y:.1f}% plugged in<extra></extra>",
        )
    )

    _mark_windows(fig, cfg)
    _mark_departures(fig, cfg)

    _style(fig, title="Share of this archetype plugged in", height=290)
    _time_axis(fig)
    fig.update_yaxes(title_text="% plugged in", range=[0, 105])
    return fig


def _mark_departures(fig: go.Figure, cfg) -> None:
    """Mark the times the driver leaves, labelled by share when there is more than one.

    For a managed archetype these are the ready-by settings the charger optimises
    against; for an unmanaged one they are simply when the car tends to leave.

    Labels are rotated and sit inside the plot.
    """
    distribution = cfg.readyby_distribution
    kind = "ready-by" if cfg.managed else "departure"
    for hour, weight in distribution:
        offset = tg.window_offsets(cfg.window_start_hr, hour)[1] + cfg.plugout_offset_hr
        text = (
            tg.format_clock(offset)
            if len(distribution) == 1
            else f"{tg.format_clock(offset)} · {weight:.0%}"
        )
        timestamp = tg.offset_to_timestamp(offset)
        # Line and label added separately: add_vline's own annotation support does
        # arithmetic on the x value, which fails for a Timestamp on this pandas version.
        fig.add_vline(x=timestamp, line=dict(color=SOC_COLOUR, width=1.5, dash="dash"))
        fig.add_annotation(
            x=timestamp,
            y=0.97,
            yref="paper",
            text=text,
            textangle=-90,
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(size=10, color=SOC_COLOUR),
        )

    # Named once, in the corner, rather than prefixed onto every label.
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.0,
        y=1.02,
        xanchor="left",
        yanchor="bottom",
        showarrow=False,
        text=f"dashed: {kind} times",
        font=dict(size=10, color=SOC_COLOUR),
    )


def _mark_windows(fig: go.Figure, cfg) -> None:
    """Plug-in window bounds on every row; cheap window shaded on the derived row."""
    win_start, win_end = tg.window_offsets(cfg.window_start_hr, cfg.window_end_hr)
    for offset in (win_start, win_end):
        fig.add_vline(
            x=tg.offset_to_timestamp(offset),
            line=dict(color=GRID_COLOUR, width=1, dash="dot"),
        )

    if cfg.cheap_window_start_hr is None:
        return
    cheap_start, cheap_end = tg.window_offsets(cfg.cheap_window_start_hr, cfg.cheap_window_end_hr)
    fig.add_vrect(
        x0=tg.offset_to_timestamp(cheap_start),
        x1=tg.offset_to_timestamp(cheap_end),
        fillcolor="rgba(31,119,180,0.10)",
        line_width=0,
        layer="below",
        row=3,
        col=1,
        annotation_text="cheap window",
        annotation_position="top left",
        annotation_font_size=10,
        annotation_font_color="#777777",
    )


def build_plugin_soc_histogram(result: SimulationResult) -> go.Figure:
    """Realised plug-in SoC, stacked by archetype.

    Stacked on preference, having tried both. The stack reads as one fleet distribution with
    its composition shown, which is the question this chart answers; six translucent
    overlapping histograms make the fleet total impossible to see and the archetypes hard to
    compare against each other anyway, since their populations differ by 40x.

    Stacking is no longer *hiding* anything, which it briefly was: the sampler used to clip
    to [5, 95] and pile atoms at both bounds, and the stack obscured the resulting edge
    spikes. It truncates now (see sample_plugin_soc), so the tails are honest either way.
    """
    events = result.events[result.events["has_event"]]
    fig = go.Figure()
    for i, cfg in enumerate(result.archetypes):
        subset = events[events["archetype"] == cfg.name]
        if subset.empty:
            continue
        fig.add_trace(
            go.Histogram(
                x=subset["plugin_soc"],
                name=cfg.name,
                marker_color=ARCHETYPE_COLOURS[i % len(ARCHETYPE_COLOURS)],
                marker_line_width=0,
                xbins=dict(start=0, end=100, size=2.5),
                hovertemplate=f"{cfg.name}<br>%{{x}}%: %{{y}} agents<extra></extra>",
            )
        )
    _style(fig, title="Realised plug-in State of Charge, stacked by archetype", legend=True)
    fig.update_layout(barmode="stack", hovermode="closest")
    fig.update_xaxes(title_text="Plug-in State of Charge (%)", range=[0, 100])
    fig.update_yaxes(title_text="Agents")
    return fig
