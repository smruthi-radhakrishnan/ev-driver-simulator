"""Unmanaged mechanics (§4.4-§4.6), the Always Plugged-In trip gap (§4.8),
and the aggregation rules of §4.9."""

import numpy as np
import pytest

from model import ScenarioConfig, Simulator, default_archetypes
from model import timegrid as tg
from model.aggregate import (
    MIN_CONNECTED_FOR_BAND,
    compute_agent_trace,
    compute_archetype_breakdown,
    compute_occupancy_bars,
    compute_soc_band,
)
from model.config import ALWAYS_PLUGGED_IN


def _solo_run(name: str, n_agents: int = 3000, seed: int = 5):
    archetypes = default_archetypes()
    cfg = next(c for c in archetypes if c.name == name)
    cfg.population_pct = 100.0
    return cfg, Simulator.run([cfg], n_agents=n_agents, seed=seed)


@pytest.fixture(scope="module")
def full_run():
    return Simulator.run(ScenarioConfig(n_agents=4000, seed=42))


# --- sampling helpers ---------------------------------------------------------


def test_unmanaged_charging_starts_at_plug_in(full_run):
    events = full_run.events
    unmanaged = events[
        events["archetype"].isin(["Average (UK)", "Infrequent Driving"]) & events["has_event"]
    ]
    # At the daily default every archetype plugs in except the gated Infrequent Charging
    # cohort, which is 10% of the fleet.
    assert events["has_event"].mean() > 0.9
    assert np.allclose(
        unmanaged["charge_start_offset_hrs"], unmanaged["plugin_offset_hrs"], atol=1e-9
    )


def test_plugin_times_stay_inside_each_window(full_run):
    for cfg in full_run.archetypes:
        if cfg.is_continuous:
            continue
        start, end = tg.window_offsets(cfg.window_start_hr, cfg.window_end_hr)
        subset = full_run.events[
            (full_run.events["archetype"] == cfg.name) & full_run.events["has_event"]
        ]
        assert subset["plugin_offset_hrs"].min() >= start - 1e-9, cfg.name
        assert subset["plugin_offset_hrs"].max() <= end + 1e-9, cfg.name


def test_soc_never_exceeds_target_and_never_falls_while_connected(full_run):
    soc = np.where(full_run.plugged_in, full_run.soc, np.nan)
    targets = full_run.events["target_soc"].to_numpy()[:, None]

    # Always Plugged-In legitimately starts above target: it begins the window wherever
    # it must have been for its trip to leave it at the sampled plug-in SoC.
    continuous = np.array(
        [c.is_continuous for c in full_run.archetypes]
    )[full_run.events["archetype_index"].to_numpy()][:, None]
    overshoot = np.nan_to_num(soc, nan=0.0) - targets
    assert np.all(np.where(continuous, 0.0, overshoot) <= 1e-9)
    assert np.nanmax(full_run.soc) <= 100.0 + 1e-9

    # Within a single connection SoC is monotonic; the only falls are the trip gap of
    # Always Plugged-In and the post-plug-out drive, both of which are disconnected and
    # therefore masked out above.
    diffs = np.diff(np.nan_to_num(soc, nan=0.0), axis=1)
    connected_both = full_run.plugged_in[:, :-1] & full_run.plugged_in[:, 1:]
    assert np.all(diffs[connected_both] >= -1e-9)


def test_soc_falls_after_plugout_and_never_rises_while_disconnected(full_run):
    """The driving depletion model: SoC only ever declines away from the chargepoint."""
    soc = np.where(~full_run.plugged_in, full_run.soc, np.nan)
    diffs = np.diff(np.nan_to_num(soc, nan=0.0), axis=1)
    both_away = ~full_run.plugged_in[:, :-1] & ~full_run.plugged_in[:, 1:]
    finite = np.isfinite(soc[:, :-1]) & np.isfinite(soc[:, 1:])
    assert np.all(diffs[both_away & finite] <= 1e-9)

    # And the drop actually happens: plug-out SoC exceeds SoC after driving.
    events = full_run.events[full_run.events["has_event"]].dropna(subset=["soc_after_driving"])
    assert (events["soc_after_driving"] <= events["plugout_soc"] + 1e-9).all()
    assert (events["soc_after_driving"] < events["plugout_soc"] - 1e-9).mean() > 0.9


def test_delivered_energy_matches_the_event_log(full_run):
    """Ties the slot-average power grid back to §4.3's per-event energy."""
    delivered = full_run.charge_power_kw.sum() * tg.STEP_HOURS
    assert delivered == pytest.approx(full_run.events["energy_kwh"].sum(skipna=True), rel=1e-9)


# --- Always Plugged-In (§4.8) -------------------------------------------------


def test_always_plugged_in_has_exactly_one_disconnect():
    cfg, result = _solo_run("Always Plugged-In", n_agents=800)
    plugged = result.plugged_in
    assert plugged[:, 0].all(), "should start the window connected"

    # One contiguous gap per agent.
    for row in plugged:
        transitions = np.diff(row.astype(int))
        assert list(transitions[transitions != 0]) == [-1, 1], "expected one disconnect"

    # And it has to fall in daylight. The trip placement constants are clock hours while the
    # connection intervals are window offsets, and reading one as the other placed every trip
    # at 18:00-06:00 on a noon-anchored window -- cars driving through the night, on a chart
    # that looked plausible. Asserted in clock time so the anchor cannot hide the mistake.
    disconnected_clock = [
        tg.to_clock_hour(offset)
        for offset, any_away in zip(tg.offsets(), (~plugged).any(axis=0))
        if any_away
    ]
    assert disconnected_clock, "expected at least one disconnected timestep"
    assert min(disconnected_clock) >= 6.0, f"trip starts before 06:00: {min(disconnected_clock)}"
    assert max(disconnected_clock) < 20.0, f"trip runs past 20:00: {max(disconnected_clock)}"


def test_occupancy_bars_are_a_plain_population_mean(full_run):
    bars = compute_occupancy_bars(full_run)
    assert len(bars) == tg.N_TIMESTEPS
    assert (bars["pct_plugged_in"] >= 0).all() and (bars["pct_plugged_in"] <= 100).all()
    assert np.allclose(
        bars["pct_plugged_in"], full_run.plugged_in.mean(axis=0) * 100.0
    )


def test_soc_band_is_suppressed_when_too_few_agents_are_connected():
    """§4.9 edge case -- do not compute a percentile from a handful of points."""
    # A fleet with no Always Plugged-In agents is empty at midday.
    archetypes = [c for c in default_archetypes() if c.behaviour != ALWAYS_PLUGGED_IN]
    result = Simulator.run(archetypes, n_agents=1000, seed=4)
    band = compute_soc_band(result)

    midday = band[band["offset_hrs"] < 4.0]
    assert midday["band_suppressed"].all()
    assert midday["soc_mean"].isna().all()
    assert (band.loc[~band["band_suppressed"], "n_connected"] >= MIN_CONNECTED_FOR_BAND).all()


def test_archetype_breakdown_sums_to_overall_occupancy(full_run):
    breakdown = compute_archetype_breakdown(full_run)
    stacked = breakdown.groupby("offset_hrs")["pct_of_fleet"].sum().to_numpy()
    overall = compute_occupancy_bars(full_run)["pct_plugged_in"].to_numpy()
    assert np.allclose(stacked, overall)


def test_agent_trace_matches_the_grid(full_run):
    trace = compute_agent_trace(full_run, agent_id=0)
    assert len(trace) == tg.N_TIMESTEPS
    assert np.array_equal(trace["plugged_in"].to_numpy(), full_run.plugged_in[0])


def test_scenario_roundtrips_through_json():
    import json

    scenario = ScenarioConfig(n_agents=1234, seed=7)
    restored = ScenarioConfig.from_dict(json.loads(json.dumps(scenario.to_dict())))
    assert restored.n_agents == 1234 and restored.seed == 7
    assert [a.to_dict() for a in restored.archetypes] == [
        a.to_dict() for a in scenario.archetypes
    ]


