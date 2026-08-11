"""Per-agent event generation: when the car is connected and when it charges.

All times are window offsets (hours since noon, see :mod:`model.timegrid`).

Charge duration is never a stored parameter -- it is derived per event from the
sampled plug-in SoC, the battery and the charger (§4.3):

    energy_required_kwh = (target_soc - plugin_soc)/100 * battery_kwh
    charge_duration_hrs = energy_required_kwh / charger_kw
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from model import timegrid as tg
from model.config import ArchetypeConfig
from model.population import Agent

# Physical bounds on state of charge, and nothing tighter. §4.6 suggested clipping to a
# "sensible range" such as [5, 95], but the report's Figure 7 shows real plug-in events
# both below 5% (0.8%) and above 95% (0.5%), so a narrower bound is not sensible, it is
# wrong -- and combined with clipping it produced visible artefacts. See sample_plugin_soc.
SOC_FLOOR = 0.0
SOC_CEILING = 100.0

# §4.8 -- when the daily disconnect happens, as *clock* hours. Converted to window offsets
# at the point of use, which is the whole reason for stating them this way: an earlier
# version used 6.0 and 18.0 directly as offsets, and on a noon-anchored window those mean
# 18:00 and 06:00, so the car made its only trip of the day in the middle of the night.
#
# The window's daylight sits either side of the noon anchor -- 12:00-18:00 at offsets 0-6,
# then 06:00-12:00 at offsets 18-24 -- so a single contiguous daytime trip has to pick a
# side. The afternoon is the right side: charging resumes when the car gets back, and that
# needs the rest of the window still ahead of it.
TRIP_PLACEMENT_START_HR = 13.0
TRIP_PLACEMENT_END_HR = 18.0

# How long the drive after plug-out takes. A fixed modelling assumption rather than a
# tunable input: it only controls how steeply the post-plug-out SoC line falls, not how
# far, and nothing downstream depends on its value.
DRIVE_DURATION_HRS = 1.0


@dataclass
class AgentEvent:
    """What one agent does over the simulated window.

    ``connections`` holds up to two (start, end) intervals in window offsets --
    two only for Always Plugged-In, which is connected either side of its trip.
    An empty list means the agent has no plug-in event in this window (an
    Infrequent Charging agent that is not due).
    """

    archetype: str
    connections: list[tuple[float, float]] = field(default_factory=list)
    plugin_soc: float = float("nan")
    target_soc: float = float("nan")
    charge_start: Optional[float] = None
    charge_end: Optional[float] = None
    charge_duration_hrs: float = 0.0  # actually delivered
    required_duration_hrs: float = 0.0  # needed to reach target (§4.3)
    energy_kwh: float = 0.0  # actually delivered
    charger_kw: float = 0.0
    battery_kwh: float = 0.0
    initial_soc: Optional[float] = None  # SoC before the trip, Always Plugged-In only
    charge_window_clipped: bool = False  # §4.4.3 safety constraint bound
    plugout_soc: float = float("nan")  # SoC at the moment the driver disconnects
    reached_target: bool = True  # False when the driver left mid-charge
    driving_soc_drop_pct: float = 0.0  # depletion over the drive after plug-out
    drive_duration_hrs: float = 0.0
    drove_after_plugout: bool = False
    deadline: float = float("nan")  # this driver's ready-by time, as a window offset

    @property
    def has_event(self) -> bool:
        return bool(self.connections)

    @property
    def plugin_offset(self) -> Optional[float]:
        return self.connections[0][0] if self.connections else None

    @property
    def plugout_offset(self) -> Optional[float]:
        return self.connections[-1][1] if self.connections else None


# --- sampling helpers ---------------------------------------------------------


def truncated_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    low: float,
    high: float,
    max_tries: int = 50,
) -> float:
    """Draw from a normal truncated to [low, high], falling back to a clip.

    Rejection sampling keeps the shape honest for the usual case where the bounds
    are a couple of standard deviations out; the clip only bites when the caller
    has asked for a mean sitting outside (or hard against) the bounds.
    """
    if high <= low:
        return float(low)
    if std <= 0:
        return float(np.clip(mean, low, high))
    for _ in range(max_tries):
        draw = rng.normal(mean, std)
        if low <= draw <= high:
            return float(draw)
    return float(np.clip(rng.normal(mean, std), low, high))


def sample_plugin_soc(cfg: ArchetypeConfig, rng: np.random.Generator) -> float:
    """Draw this driver's plug-in state of charge from a normal 
    *truncated* to [SOC_FLOOR, SOC_CEILING].
    """
    return truncated_normal(
        rng, cfg.plugin_soc_mean, cfg.plugin_soc_std, SOC_FLOOR, SOC_CEILING
    )


def sample_target_soc(cfg: ArchetypeConfig, rng: np.random.Generator) -> float:
    """Draw this driver's target state of charge from the archetype's weighted choices.

    A per-agent draw rather than an archetype constant: the report finds the population
    split roughly evenly across 80/90/100 rather than agreeing on one figure, and the
    target is what decides how long a given plug-in SoC takes to charge. Two agents
    arriving at the same 52% need 2.9h and 5.0h on an 80% and a 100% target.
    """
    pairs = cfg.target_soc_distribution
    if len(pairs) == 1:
        return pairs[0][0]
    return float(rng.choice([t for t, _ in pairs], p=[w for _, w in pairs]))


def charge_requirement(
    cfg: ArchetypeConfig, plugin_soc: float, target_soc: float | None = None
) -> tuple[float, float]:
    """§4.3 -- (energy_kwh, duration_hrs) implied by the SoC gap. Never negative.

    Falls back to the archetype's mean target when no per-agent draw is supplied, which
    is what the diagnostic helpers want.
    """
    target = cfg.mean_target_soc if target_soc is None else target_soc
    deficit_pct = max(target - plugin_soc, 0.0)
    energy_kwh = deficit_pct / 100.0 * cfg.battery_kwh
    duration_hrs = energy_kwh / cfg.charger_kw if cfg.charger_kw > 0 else 0.0
    return energy_kwh, duration_hrs


def sample_plugin_time(cfg: ArchetypeConfig, rng: np.random.Generator) -> float:
    """§4.5 -- truncated normal, held inside the archetype's plug-in window."""
    win_start, win_end = tg.window_offsets(cfg.window_start_hr, cfg.window_end_hr)
    mean = tg.to_window_offset(cfg.plugin_time_mean_hr)
    # The mean is given as a clock hour; lift it into the same day as the window
    # so a 22:30 mean is compared against a 22:00-09:00 window correctly.
    if mean < win_start:
        mean += tg.WINDOW_HOURS
    return truncated_normal(rng, mean, cfg.plugin_time_std_hr, win_start, win_end)


def sample_deadline(cfg: ArchetypeConfig, rng: np.random.Generator, plugin: float) -> float:
    """Draw this driver's ready-by deadline, as a window offset.

    A weighted choice among the archetype's configured times rather than a draw from
    a continuous density: ready-by is a value the driver enters, so it lands on the
    times people actually pick. Clamped to fall after the plug-in, which only bites
    if a very wide set of choices is configured against a late plug-in window.
    """
    pairs = cfg.readyby_distribution
    hours = [h for h, _ in pairs]
    weights = [w for _, w in pairs]
    chosen = float(rng.choice(hours, p=weights)) if len(hours) > 1 else hours[0]
    offset = tg.window_offsets(cfg.window_start_hr, chosen)[1]
    return float(max(offset, plugin + tg.STEP_HOURS))


def sample_plugout_time(
    cfg: ArchetypeConfig, rng: np.random.Generator, plugin: float, deadline: float
) -> float:
    """Sample when the driver actually disconnects.

    Departure is the driver's own deadline plus a configurable offset plus noise. The
    deadline is what they asked the charger for and what managed scheduling optimises
    against; this is when the car physically leaves, and the two are separate
    quantities. A driver who leaves before charging finishes departs below target
    rather than the charge being allowed to overrun -- see :func:`_finalise_charge`.
    """
    plugout = deadline + cfg.plugout_offset_hr + rng.normal(0.0, cfg.plugout_time_std_hr)
    return float(max(plugout, plugin + tg.STEP_HOURS))


def _finalise_charge(
    cfg: ArchetypeConfig,
    plugin: float,
    plugout: float,
    plugin_soc: float,
    charge_start: float,
    required_duration: float,
    clipped: bool,
    rng: np.random.Generator,
    deadline: float = float("nan"),
    target_soc: float | None = None,
) -> AgentEvent:
    """Truncate the charge at the actual plug-out and account for what was delivered."""
    target = cfg.mean_target_soc if target_soc is None else target_soc
    drove = bool(rng.random() < cfg.driving_probability_per_day)
    charge_end = min(charge_start + required_duration, plugout)
    delivered_hrs = max(charge_end - charge_start, 0.0)
    delivered_kwh = delivered_hrs * cfg.charger_kw
    gained_pct = delivered_kwh / cfg.battery_kwh * 100.0 if cfg.battery_kwh > 0 else 0.0
    plugout_soc = min(plugin_soc + gained_pct, target)

    return AgentEvent(
        archetype=cfg.name,
        connections=[(plugin, plugout)],
        plugin_soc=plugin_soc,
        target_soc=target,
        charge_start=charge_start,
        charge_end=charge_end,
        charge_duration_hrs=delivered_hrs,
        required_duration_hrs=required_duration,
        energy_kwh=delivered_kwh,
        charger_kw=cfg.charger_kw,
        battery_kwh=cfg.battery_kwh,
        charge_window_clipped=clipped,
        plugout_soc=plugout_soc,
        reached_target=delivered_hrs >= required_duration - 1e-9,
        # One trip, or none, rather than a smeared daily average: an infrequent driver
        # takes a normal-length trip on some days and stays home on others.
        drove_after_plugout=drove,
        driving_soc_drop_pct=cfg.trip_soc_drop_pct if drove else 0.0,
        drive_duration_hrs=DRIVE_DURATION_HRS if drove else 0.0,
        deadline=deadline,
    )


# --- event generators ---------------------------------------------------------


def unmanaged_event(cfg: ArchetypeConfig, rng: np.random.Generator) -> AgentEvent:
    """§4.4 unmanaged: one contiguous block starting the moment the car is plugged in."""
    plugin = sample_plugin_time(cfg, rng)
    plugin_soc = sample_plugin_soc(cfg, rng)
    target_soc = sample_target_soc(cfg, rng)
    _, duration = charge_requirement(cfg, plugin_soc, target_soc)
    deadline = sample_deadline(cfg, rng, plugin)
    plugout = sample_plugout_time(cfg, rng, plugin, deadline)
    return _finalise_charge(
        cfg, plugin, plugout, plugin_soc, plugin, duration, False, rng,
        deadline=deadline, target_soc=target_soc,
    )


def managed_event(cfg: ArchetypeConfig, rng: np.random.Generator) -> AgentEvent:
    """§4.4 managed: the charge block is placed inside the cheap-rate window.

    Deliberately *not* anchored to the driver's deadline. Real Intelligent Octopus
    charging is scheduled against wholesale prices within a fixed cheap window
    (23:30-05:30, report Table 1), independent of the ready-by time. Plug-out is
    unaffected -- only *when charging happens* moves.
    """
    plugin = sample_plugin_time(cfg, rng)
    plugin_soc = sample_plugin_soc(cfg, rng)
    target_soc = sample_target_soc(cfg, rng)
    _, duration = charge_requirement(cfg, plugin_soc, target_soc)

    # The charger optimises against *this driver's* deadline, not an archetype-wide
    # one. A driver whose deadline falls before the cheap window closes therefore has
    # their charge pulled forward out of it, which is correct -- they cannot wait for
    # cheap rate -- but it does mean they behave like an unmanaged driver.
    deadline = sample_deadline(cfg, rng, plugin)
    cheap_start, cheap_end = tg.window_offsets(cfg.cheap_window_start_hr, cfg.cheap_window_end_hr)

    # Charging cannot begin before the car is actually connected.
    earliest_start = max(cheap_start, plugin)
    # Latest start that still keeps the block inside the cheap window.
    latest_start = cheap_end - duration
    clipped = False

    if latest_start < earliest_start:
        # The block does not fit in the cheap window (a very depleted battery, or a
        # late plug-in). Start as early as we can and accept the overrun.
        charge_start = earliest_start
        clipped = True
    elif not cfg.charge_placement_std_hr:
        # A timer with no spread fires the moment its window opens -- or the moment the car
        # arrives, if that is later. Falling through to the branch below would instead put
        # the charge deterministically in the *centre* of the feasible range, which is not
        # what zero spread means and would give every agent a different start time, since
        # the centre depends on their own plug-in time and charge duration.
        charge_start = earliest_start
    else:
        # §4.4.2 -- centre the mean so the sampled duration sits inside the window.
        mean_start = earliest_start + (latest_start - earliest_start) / 2.0
        charge_start = truncated_normal(
            rng, mean_start, cfg.charge_placement_std_hr, earliest_start, latest_start
        )

    # §4.4.3 safety constraint -- aim to finish by the driver's stated deadline. This
    # targets the deadline, not the sampled departure: the charger only knows what it
    # was asked for.
    #
    # The floor here is the plug-in, not the cheap window's start. The cheap window is
    # a price preference, not a physical constraint, so a driver whose deadline falls
    # before it opens charges from plug-in instead -- effectively unmanaged. Flooring
    # at the window start would pin their charge to a time after they had already left.
    if charge_start + duration > deadline:
        charge_start = max(plugin, deadline - duration)
        clipped = True

    plugout = sample_plugout_time(cfg, rng, plugin, deadline)
    return _finalise_charge(
        cfg, plugin, plugout, plugin_soc, charge_start, duration, clipped, rng,
        deadline=deadline, target_soc=target_soc,
    )


def always_plugged_in_event(cfg: ArchetypeConfig, rng: np.random.Generator) -> AgentEvent:
    """§4.8 -- connected all window bar one trip; charging resumes on reconnection.

    The car starts the window sitting at its target SoC, leaves on a trip, and comes
    back at the sampled plug-in SoC. There is no driving-phase depletion model (§6):
    the SoC simply ramps down across the disconnected gap, during which the agent is
    excluded from the connected-fleet aggregates anyway.
    """
    trip_duration = float(cfg.trip_duration_hrs or 0.0)
    earliest = tg.to_window_offset(TRIP_PLACEMENT_START_HR)
    latest = tg.to_window_offset(TRIP_PLACEMENT_END_HR)
    latest_trip_start = max(earliest, latest - trip_duration)
    trip_start = float(rng.uniform(earliest, latest_trip_start))
    trip_end = trip_start + trip_duration

    plugin_soc = sample_plugin_soc(cfg, rng)
    target_soc = sample_target_soc(cfg, rng)
    energy_kwh, duration = charge_requirement(cfg, plugin_soc, target_soc)

    charge_start = trip_end
    charge_end = min(charge_start + duration, tg.WINDOW_HOURS)
    delivered_hrs = max(charge_end - charge_start, 0.0)
    gained_pct = delivered_hrs * cfg.charger_kw / cfg.battery_kwh * 100.0

    return AgentEvent(
        archetype=cfg.name,
        connections=[(0.0, trip_start), (trip_end, tg.WINDOW_HOURS)],
        plugin_soc=plugin_soc,
        target_soc=target_soc,
        charge_start=charge_start,
        charge_end=charge_end,
        charge_duration_hrs=delivered_hrs,
        required_duration_hrs=duration,
        energy_kwh=delivered_hrs * cfg.charger_kw,
        charger_kw=cfg.charger_kw,
        battery_kwh=cfg.battery_kwh,
        # Start the window wherever the car must have been for the trip to leave it at
        # the sampled plug-in SoC. Anchoring to target_soc instead would let the trip
        # *raise* SoC whenever the sampled plug-in SoC came out above target.
        initial_soc=min(plugin_soc + cfg.trip_soc_drop_pct, 100.0),
        # Still connected when the window closes, so there is no plug-out SoC and no
        # post-plug-out drive; the depletion for this archetype is the trip above.
        plugout_soc=min(plugin_soc + gained_pct, target_soc),
        reached_target=delivered_hrs >= duration - 1e-9,
        driving_soc_drop_pct=0.0,
        drive_duration_hrs=0.0,
    )


def infrequent_charging_gate(agent: Agent, rng: np.random.Generator) -> bool:
    """§4.7 -- is this Infrequent Charging agent due to plug in during this window?

    True when the agent's phase has reached the final day of its renewal gap. The
    phase was drawn at population-build time, so agents are not synchronised and no
    burn-in is needed. ``rng`` is unused but kept in the signature for symmetry with
    the event generators.
    """
    if agent.days_since_last_plugin is None or agent.next_gap_days is None:
        return True
    return agent.days_since_last_plugin >= agent.next_gap_days - 1.0


def no_event(cfg: ArchetypeConfig) -> AgentEvent:
    """An agent that simply does not plug in during this window."""
    return AgentEvent(
        archetype=cfg.name,
        connections=[],
        plugin_soc=float("nan"),
        target_soc=cfg.mean_target_soc,
        charger_kw=cfg.charger_kw,
        battery_kwh=cfg.battery_kwh,
    )


def generate_event(cfg: ArchetypeConfig, agent: Agent, rng: np.random.Generator) -> AgentEvent:
    """Dispatch to the right generator for the agent's archetype."""
    from model.config import ALWAYS_PLUGGED_IN, INFREQUENT_CHARGING, MANAGED

    if cfg.behaviour == ALWAYS_PLUGGED_IN:
        return always_plugged_in_event(cfg, rng)
    if cfg.behaviour == INFREQUENT_CHARGING:
        # Cadence comes from the renewal phase set at population-build time.
        if not infrequent_charging_gate(agent, rng):
            return no_event(cfg)
        return unmanaged_event(cfg, rng)

    # Every other archetype gets a Bernoulli gate on its plug-in frequency. Short-circuited
    # at the daily default so no random draw is consumed, which keeps the random stream
    # identical to a model without the gate at all.
    rate = cfg.expected_plugins_per_day
    if rate < 1.0 and rng.random() >= rate:
        return no_event(cfg)

    if cfg.behaviour == MANAGED:
        return managed_event(cfg, rng)
    return unmanaged_event(cfg, rng)
