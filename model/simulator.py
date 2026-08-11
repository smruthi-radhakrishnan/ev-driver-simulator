"""Discretise per-agent events onto the 15-minute grid.

The event generators in :mod:`model.events` work in continuous time; this module
turns them into the ``(n_agents, n_timesteps)`` arrays the aggregation layer needs,
plus an agent-level event log.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from model import timegrid as tg
from model.config import ArchetypeConfig, ScenarioConfig
from model.events import AgentEvent, generate_event
from model.population import Agent, build_population


@dataclass
class SimulationResult:
    """Everything one run produces."""

    plugged_in: np.ndarray  # (n_agents, n_timesteps) bool
    soc: np.ndarray  # (n_agents, n_timesteps) float, NaN where the agent has no event
    charging: np.ndarray  # (n_agents, n_timesteps) bool
    charge_power_kw: np.ndarray  # (n_agents, n_timesteps) float, slot-average kW
    events: pd.DataFrame  # one row per agent
    timegrid: pd.DatetimeIndex
    agents: list[Agent]
    archetypes: list[ArchetypeConfig]

    @property
    def n_agents(self) -> int:
        return self.plugged_in.shape[0]

    @property
    def archetype_names(self) -> list[str]:
        return [a.name for a in self.archetypes]

    def archetype_mask(self, name: str) -> np.ndarray:
        """Boolean row-mask selecting the agents belonging to one archetype."""
        return (self.events["archetype"] == name).to_numpy()


class Simulator:
    """Runs a scenario. Stateless -- ``run`` is the whole API."""

    @staticmethod
    def run(
        scenario: ScenarioConfig | Sequence[ArchetypeConfig],
        n_agents: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> SimulationResult:
        if isinstance(scenario, ScenarioConfig):
            archetypes = list(scenario.archetypes)
            n_agents = scenario.n_agents if n_agents is None else n_agents
            seed = scenario.seed if seed is None else seed
        else:
            archetypes = list(scenario)
            if n_agents is None:
                raise ValueError("n_agents is required when passing a bare archetype sequence")

        agents = build_population(archetypes, n_agents=n_agents, seed=seed)
        # A separate stream from the population builder so that changing population
        # size does not reshuffle the behavioural draws of unrelated agents.
        rng = np.random.default_rng(None if seed is None else seed + 1)

        by_index = {i: cfg for i, cfg in enumerate(archetypes)}
        events = [generate_event(by_index[a.archetype_index], a, rng) for a in agents]

        plugged_in, soc, charging, power = _discretise(events)
        log = _event_log(agents, events)

        return SimulationResult(
            plugged_in=plugged_in,
            soc=soc,
            charging=charging,
            charge_power_kw=power,
            events=log,
            timegrid=tg.make_timegrid(),
            agents=agents,
            archetypes=archetypes,
        )


def _stack(events: Sequence[AgentEvent]) -> dict[str, np.ndarray]:
    """Flatten the event objects into per-agent scalar arrays for vectorised work."""
    n = len(events)
    nan = np.full(n, np.nan)
    cols = {
        "c1s": nan.copy(),
        "c1e": nan.copy(),
        "c2s": nan.copy(),
        "c2e": nan.copy(),
        "cs": nan.copy(),
        "ce": nan.copy(),
        "plugin_soc": nan.copy(),
        "target_soc": nan.copy(),
        "duration": np.zeros(n),
        "charger_kw": np.zeros(n),
        "battery_kwh": np.zeros(n),
        "initial_soc": nan.copy(),
        "plugout": nan.copy(),
        "plugout_soc": nan.copy(),
        "drop_pct": np.zeros(n),
        "drive_hrs": np.zeros(n),
    }
    for i, ev in enumerate(events):
        if ev.connections:
            cols["c1s"][i], cols["c1e"][i] = ev.connections[0]
            if len(ev.connections) > 1:
                cols["c2s"][i], cols["c2e"][i] = ev.connections[1]
            if len(ev.connections) > 2:  # pragma: no cover -- not produced today
                raise ValueError("cannot discretise more than two connection intervals")
        if ev.charge_start is not None:
            cols["cs"][i] = ev.charge_start
            cols["ce"][i] = ev.charge_end
        cols["plugin_soc"][i] = ev.plugin_soc
        cols["target_soc"][i] = ev.target_soc
        cols["duration"][i] = ev.charge_duration_hrs
        cols["charger_kw"][i] = ev.charger_kw
        cols["battery_kwh"][i] = ev.battery_kwh
        if ev.initial_soc is not None:
            cols["initial_soc"][i] = ev.initial_soc
        if ev.plugout_offset is not None:
            cols["plugout"][i] = ev.plugout_offset
        cols["plugout_soc"][i] = ev.plugout_soc
        cols["drop_pct"][i] = ev.driving_soc_drop_pct
        cols["drive_hrs"][i] = ev.drive_duration_hrs
    return cols


def _contains(t: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Half-open interval test, NaN-safe (a NaN bound means "no such interval")."""
    with np.errstate(invalid="ignore"):
        inside = (t >= start) & (t < end)
    return np.where(np.isnan(start) | np.isnan(end), False, inside)


def _discretise(
    events: Sequence[AgentEvent],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample the continuous-time events onto the grid.

    Two deliberately different semantics live here. ``plugged_in``, ``charging`` and
    ``soc`` are *instantaneous* samples at each timestep, which is what §4.9's
    ``mean(plugged_in[:, t])`` asks for and what a state variable should be.
    ``charge_power_kw`` is instead a slot *average*, because power only means
    anything over an interval -- that also makes the delivered energy exactly equal
    the event log's ``energy_kwh``, which an instantaneous sample would not.
    """
    c = _stack(events)
    t = tg.offsets()[None, :]  # (1, T)
    col = lambda key: c[key][:, None]  # noqa: E731 -- (n, 1) for broadcasting

    plugged_in = _contains(t, col("c1s"), col("c1e")) | _contains(t, col("c2s"), col("c2e"))

    cs, ce = col("cs"), col("ce")
    charging = _contains(t, cs, ce) & plugged_in

    # Slot-average power: the fraction of the 15-minute slot actually spent charging,
    # so a block that ends mid-slot contributes a partial slot rather than a full one.
    with np.errstate(invalid="ignore"):
        overlap = np.clip(np.minimum(t + tg.STEP_HOURS, ce) - np.maximum(t, cs), 0.0, None)
    overlap = np.nan_to_num(overlap, nan=0.0)
    charge_power_kw = col("charger_kw") * overlap / tg.STEP_HOURS

    # SoC: plug-in SoC plus whatever has been delivered by time t, capped at target.
    with np.errstate(invalid="ignore"):
        delivered_hrs = np.clip(t - cs, 0.0, None)
    delivered_hrs = np.nan_to_num(delivered_hrs, nan=0.0)
    delivered_hrs = np.minimum(delivered_hrs, col("duration"))
    gained_pct = delivered_hrs * col("charger_kw") / col("battery_kwh") * 100.0
    soc = np.minimum(col("plugin_soc") + gained_pct, col("target_soc"))

    soc = _apply_pre_trip_soc(soc, c, t)
    soc = _apply_driving_depletion(soc, c, t)

    # Agents with no event this window have no meaningful SoC trace.
    no_event = np.isnan(c["c1s"])[:, None]
    soc = np.where(no_event, np.nan, soc)

    return plugged_in, soc, charging, charge_power_kw


def _apply_driving_depletion(
    soc: np.ndarray, c: dict[str, np.ndarray], t: np.ndarray
) -> np.ndarray:
    """Deplete SoC after plug-out, over the drive to the next plug-in.

    The drop is derived from the archetype's mileage and efficiency rather than
    assumed, so the SoC reached at the end of the drive lands back near the next
    plug-in SoC (see :func:`model.validate.validate_energy_closure`). The decline is
    linear over ``drive_duration_hrs`` and then flat -- the day's driving happens as a
    trip after departure, not spread across the whole afternoon.
    """
    plugout = c["plugout"][:, None]
    plugout_soc = c["plugout_soc"][:, None]
    drop = c["drop_pct"][:, None]
    drive_hrs = c["drive_hrs"][:, None]

    driving = np.where(drive_hrs > 0, drive_hrs, np.nan)
    with np.errstate(invalid="ignore"):
        progress = np.clip((t - plugout) / driving, 0.0, 1.0)
        after_plugout = t >= plugout
    progress = np.nan_to_num(progress, nan=1.0)
    depleted = np.clip(plugout_soc - drop * progress, 0.0, None)

    after_plugout = np.where(np.isnan(plugout), False, after_plugout)
    return np.where(after_plugout, depleted, soc)


def _apply_pre_trip_soc(
    soc: np.ndarray, c: dict[str, np.ndarray], t: np.ndarray
) -> np.ndarray:
    """Always Plugged-In (§4.8): sit at target before the trip, ramp down across it.

    The ramp is cosmetic -- the agent is disconnected for its duration and so is
    excluded from the connected-fleet aggregates -- but it keeps the agent-level
    trace readable instead of showing a discontinuity.
    """
    has_initial = ~np.isnan(c["initial_soc"])
    if not has_initial.any():
        return soc

    initial = c["initial_soc"][:, None]
    trip_start = c["c1e"][:, None]
    trip_end = c["c2s"][:, None]
    plugin_soc = c["plugin_soc"][:, None]
    rows = has_initial[:, None]

    pre_trip = rows & (t < trip_start)
    soc = np.where(pre_trip, initial, soc)

    span = np.where(trip_end > trip_start, trip_end - trip_start, np.nan)
    with np.errstate(invalid="ignore"):
        frac = np.clip((t - trip_start) / span, 0.0, 1.0)
    in_trip = rows & (t >= trip_start) & (t < trip_end)
    ramp = initial + (plugin_soc - initial) * np.nan_to_num(frac, nan=1.0)
    return np.where(in_trip, ramp, soc)


def _event_log(agents: Sequence[Agent], events: Sequence[AgentEvent]) -> pd.DataFrame:
    """One row per agent, in both window-offset and clock-time terms."""
    rows = []
    for agent, ev in zip(agents, events):
        plugin, plugout = ev.plugin_offset, ev.plugout_offset
        rows.append(
            {
                "agent_id": agent.agent_id,
                "archetype": agent.archetype,
                "archetype_index": agent.archetype_index,
                "has_event": ev.has_event,
                "plugin_offset_hrs": plugin,
                "plugout_offset_hrs": plugout,
                "plugin_time": tg.format_clock(plugin) if plugin is not None else None,
                "plugout_time": tg.format_clock(plugout) if plugout is not None else None,
                "deadline_offset_hrs": ev.deadline if ev.has_event else None,
                "deadline_time": (
                    tg.format_clock(ev.deadline)
                    if ev.has_event and np.isfinite(ev.deadline)
                    else None
                ),
                "deadline_offset_hrs": ev.deadline if ev.has_event else None,
                "deadline_time": (
                    tg.format_clock(ev.deadline)
                    if ev.has_event and np.isfinite(ev.deadline)
                    else None
                ),
                "plugin_soc": ev.plugin_soc,
                "plugout_soc": ev.plugout_soc if ev.has_event else np.nan,
                "target_soc": ev.target_soc,
                "reached_target": ev.reached_target if ev.has_event else None,
                "energy_kwh": ev.energy_kwh if ev.has_event else np.nan,
                "charge_duration_hrs": ev.charge_duration_hrs if ev.has_event else np.nan,
                "required_duration_hrs": ev.required_duration_hrs if ev.has_event else np.nan,
                "drove_after_plugout": ev.drove_after_plugout if ev.has_event else None,
                "driving_soc_drop_pct": ev.driving_soc_drop_pct if ev.has_event else np.nan,
                "soc_after_driving": (
                    max(ev.plugout_soc - ev.driving_soc_drop_pct, 0.0)
                    if ev.has_event and not ev.initial_soc
                    else np.nan
                ),
                "charge_start_offset_hrs": ev.charge_start if ev.has_event else None,
                "charge_end_offset_hrs": ev.charge_end if ev.has_event else None,
                "charge_start_time": (
                    tg.format_clock(ev.charge_start)
                    if ev.has_event and ev.charge_start is not None
                    else None
                ),
                "charge_end_time": (
                    tg.format_clock(ev.charge_end)
                    if ev.has_event and ev.charge_end is not None
                    else None
                ),
                "charge_window_clipped": ev.charge_window_clipped,
                "days_since_last_plugin": agent.days_since_last_plugin,
                "renewal_gap_days": agent.next_gap_days,
            }
        )
    return pd.DataFrame(rows)
