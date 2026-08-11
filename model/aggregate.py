from __future__ import annotations

import numpy as np
import pandas as pd

from model import timegrid as tg
from model.simulator import SimulationResult

MIN_CONNECTED_FOR_BAND = 30  # §4.9 edge case -- below this, omit the SoC band


def compute_occupancy_bars(result: SimulationResult) -> pd.DataFrame:
    """% of the whole fleet plugged in at each timestep."""
    pct = result.plugged_in.mean(axis=0) * 100.0
    return pd.DataFrame(
        {
            "time": result.timegrid,
            "offset_hrs": tg.offsets(),
            "pct_plugged_in": pct,
            "n_plugged_in": result.plugged_in.sum(axis=0),
        }
    )


def compute_soc_band(
    result: SimulationResult,
    min_connected: int = MIN_CONNECTED_FOR_BAND,
    connected_only: bool = True,
) -> pd.DataFrame:
    """Mean / p5 / p95 SoC, masked when the sample is thin.

    ``connected_only`` restricts the cross-section to agents currently plugged in,
    which is the home-charging view and the one §4.9 specifies. Setting it False
    includes agents that have driven away, whose SoC is being depleted by the driving
    model -- useful for seeing the whole fleet's state of charge around the clock
    rather than only while it is at home.
    """
    soc = np.where(result.plugged_in, result.soc, np.nan) if connected_only else result.soc
    n_connected = np.sum(np.isfinite(soc), axis=0)

    with np.errstate(invalid="ignore"):
        # All-NaN columns are expected (nobody connected at midday), so the warnings
        # numpy would emit here are noise rather than signal.
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            mean = np.nanmean(soc, axis=0)
            p5 = np.nanpercentile(soc, 5, axis=0)
            p95 = np.nanpercentile(soc, 95, axis=0)

    thin = n_connected < min_connected
    frame = pd.DataFrame(
        {
            "time": result.timegrid,
            "offset_hrs": tg.offsets(),
            "soc_mean": np.where(thin, np.nan, mean),
            "soc_p5": np.where(thin, np.nan, p5),
            "soc_p95": np.where(thin, np.nan, p95),
            "n_connected": n_connected,
            "band_suppressed": thin,
        }
    )
    return frame


def compute_charging_demand(result: SimulationResult) -> pd.DataFrame:
    """Aggregate charging power, summed across agents charging at each timestep.

    ``total_kw`` is the slot-*average* power, so a block that starts or ends
    mid-slot contributes a partial slot. ``n_charging`` counts agents drawing any
    power during the slot -- the interval measure that matches ``total_kw`` rather
    than the instantaneous ``result.charging`` snapshot, so ``total_kw`` can never
    exceed ``n_charging`` times the charger rating.
    """
    total_kw = result.charge_power_kw.sum(axis=0)
    n = max(result.n_agents, 1)
    return pd.DataFrame(
        {
            "time": result.timegrid,
            "offset_hrs": tg.offsets(),
            "total_kw": total_kw,
            "mean_kw_per_agent": total_kw / n,
            "n_charging": (result.charge_power_kw > 0).sum(axis=0),
        }
    )


def compute_archetype_breakdown(result: SimulationResult) -> pd.DataFrame:
    """Long-format % plugged in, split by archetype.

    ``pct_of_fleet`` sums across archetypes to the overall occupancy curve, so the
    bars stack; ``pct_of_archetype`` is each archetype's own plug-in rate, which is
    the more useful read when comparing a 1% archetype against a 40% one.
    """
    frames = []
    n_total = max(result.n_agents, 1)
    for cfg in result.archetypes:
        mask = result.archetype_mask(cfg.name)
        n_arch = int(mask.sum())
        if n_arch == 0:
            continue
        plugged = result.plugged_in[mask].sum(axis=0)
        frames.append(
            pd.DataFrame(
                {
                    "time": result.timegrid,
                    "offset_hrs": tg.offsets(),
                    "archetype": cfg.name,
                    "n_plugged_in": plugged,
                    "pct_of_fleet": plugged / n_total * 100.0,
                    "pct_of_archetype": plugged / n_arch * 100.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_demand_per_agent(result: SimulationResult) -> pd.DataFrame:
    """Per-agent charging demand, one row per agent that plugged in.

    Energy rather than power, because per-agent power is either zero or the charger
    rating -- the distribution worth looking at is how much each agent draws.
    """
    events = result.events
    subset = events[events["has_event"]].copy()
    return subset[
        [
            "agent_id",
            "archetype",
            "energy_kwh",
            "charge_duration_hrs",
            "plugin_soc",
            "plugout_soc",
        ]
    ]


def compute_agent_trace(result: SimulationResult, agent_id: int) -> pd.DataFrame:
    """One agent's plug status, SoC and charging power over the window."""
    idx = int(np.flatnonzero(result.events["agent_id"].to_numpy() == agent_id)[0])
    return pd.DataFrame(
        {
            "time": result.timegrid,
            "offset_hrs": tg.offsets(),
            "plugged_in": result.plugged_in[idx],
            "charging": result.charging[idx],
            "soc": result.soc[idx],
            "charge_power_kw": result.charge_power_kw[idx],
        }
    )


def headline_metrics(result: SimulationResult) -> dict[str, float]:
    """A few scalars worth showing above the charts."""
    occupancy = compute_occupancy_bars(result)
    demand = compute_charging_demand(result)
    events = result.events
    peak_idx = int(demand["total_kw"].idxmax())
    return {
        "agents": float(result.n_agents),
        "peak_pct_plugged_in": float(occupancy["pct_plugged_in"].max()),
        "peak_demand_kw": float(demand["total_kw"].max()),
        "peak_demand_time": tg.format_clock(float(demand["offset_hrs"].iloc[peak_idx])),
        "total_energy_kwh": float(result.charge_power_kw.sum() * tg.STEP_HOURS),
        "pct_with_event": float(events["has_event"].mean() * 100.0),
        "median_plugin_soc": float(events["plugin_soc"].median(skipna=True)),
        "median_plugout_soc": float(events["plugout_soc"].median(skipna=True)),
        "median_charge_duration_hrs": float(events["charge_duration_hrs"].median(skipna=True)),
        "pct_left_below_target": float(
            (1.0 - events.loc[events["has_event"], "reached_target"].astype(float).mean()) * 100.0
        ),
    }
