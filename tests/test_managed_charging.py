"""§4.4 / §7.3 -- managed charge blocks sit in the cheap window, NOT at the deadline.

This is the piece of the spec that most recently changed, so it gets the most direct
tests: the previous design anchored the block to the driver's ready-by time, and the
tests below are written to fail if anything drifts back to that.
"""

import numpy as np
import pytest

from model import Simulator, default_archetypes
from model import timegrid as tg
from model.config import UNMANAGED

MANAGED_NAMES = ["Intelligent Octopus Average", "Scheduled Charging"]


def _solo_run(name: str, n_agents: int = 4000, seed: int = 21):
    """Run one archetype on its own so the event log is unambiguous."""
    archetypes = default_archetypes()
    cfg = next(c for c in archetypes if c.name == name)
    cfg.population_pct = 100.0
    result = Simulator.run([cfg], n_agents=n_agents, seed=seed)
    return cfg, result


@pytest.mark.parametrize("name", MANAGED_NAMES)
def test_charge_blocks_fall_within_the_cheap_window(name):
    cfg, result = _solo_run(name)
    cheap_start, cheap_end = tg.window_offsets(
        cfg.cheap_window_start_hr, cfg.cheap_window_end_hr
    )
    events = result.events[result.events["has_event"]]
    fits = ~events["charge_window_clipped"]
    # About 16% of IO charges are too long for the 6-hour cheap window, from two
    # report-calibrated inputs together: the 25pp plug-in SoC spread and the 80/90/100
    # target mix, whose mean target of ~84% lengthens every charge. The report's own
    # figure is the mirror image -- "in 85% of overnight plug events, we could move the
    # charging to complementary times and still meet customer preferences".
    assert fits.mean() > 0.80, "unexpectedly many blocks failed to fit the cheap window"

    # Blocks that fit sit entirely inside the cheap window.
    assert events.loc[fits, "charge_start_offset_hrs"].min() >= cheap_start - 1e-9
    assert events.loc[fits, "charge_end_offset_hrs"].max() <= cheap_end + 1e-9

    # The flagged tail is allowed to start earlier: a battery needing longer than the
    # cheap window cannot fit inside it, and meeting the driver's deadline takes
    # priority over price. Those blocks still overlap the window rather than sitting
    # somewhere unrelated to it.
    clipped = events.loc[~fits]
    if not clipped.empty:
        assert (clipped["charge_end_offset_hrs"] > cheap_start).all()
        # Each still starts no earlier than its own plug-in, and the bulk of them are
        # pinned to the window's start. A near-empty battery on an early ready-by has to
        # begin before the window opens to finish in time, so no absolute lower bound
        # holds -- only the physical one.
        assert (clipped["charge_start_offset_hrs"] >= clipped["plugin_offset_hrs"] - 1e-9).all()
        assert clipped["charge_start_offset_hrs"].median() >= cheap_start - 1e-9


@pytest.mark.parametrize("name", MANAGED_NAMES)
def test_charging_is_not_anchored_to_the_deadline(name):
    """The regression guard: block ends must not pile up against ``window_end``."""
    cfg, result = _solo_run(name)
    _, win_end = tg.window_offsets(cfg.window_start_hr, cfg.window_end_hr)
    ends = result.events.loc[result.events["has_event"], "charge_end_offset_hrs"]

    # A deadline-anchored model would put essentially every charge_end at win_end. The
    # tail that cannot fit the cheap window *is* re-anchored to the deadline, and that
    # tail grew with the report-calibrated SoC spread, so allow a few percent.
    assert (ends >= win_end - 1e-6).mean() < 0.05
    # And the median end should sit clearly before the deadline.
    assert ends.median() < win_end - 1.0


@pytest.mark.parametrize("name", MANAGED_NAMES)
def test_no_charging_during_the_evening_grid_peak(name):
    """Report Figure 14 -- managed charging is essentially absent from 17:00-20:00."""
    _, result = _solo_run(name)
    offsets = tg.offsets()
    peak = (offsets >= tg.to_window_offset(17.0)) & (offsets < tg.to_window_offset(20.0))
    assert result.charge_power_kw[:, peak].sum() == 0.0


def test_zero_placement_spread_behaves_like_a_timer():
    """Scheduled Charging's defining feature: one fixed start time, not a centred block.

    Zero spread is a separate branch in the placement code because the centred placement
    does not converge on the window start as the spread falls -- it converges on the middle
    of each agent's own feasible range. Without the branch this run produced roughly one
    distinct start time per two agents.
    """
    cfg = next(c for c in default_archetypes() if c.name == "Scheduled Charging")
    cfg.population_pct = 100.0
    assert cfg.charge_placement_std_hr == 0.0, "this archetype is meant to be a timer"

    result = Simulator.run([cfg], n_agents=2000, seed=7)
    events = result.events[result.events["has_event"]]
    timer_offset = tg.to_window_offset(cfg.cheap_window_start_hr)

    # The timer has to open while the car is already plugged in, or it can never be what
    # starts the charge -- placement floors at the plug-in.
    plugged_in_first = events["plugin_offset_hrs"] < timer_offset
    assert plugged_in_first.mean() > 0.95, "most agents should be home before the timer"

    on_time = events.loc[plugged_in_first, "charge_start_offset_hrs"]
    assert (on_time - timer_offset).abs().max() < 1e-6, "every one of them starts at the timer"

    # Agents arriving after it opened charge on arrival instead, never before.
    late = events.loc[~plugged_in_first]
    if not late.empty:
        assert (late["charge_start_offset_hrs"] >= late["plugin_offset_hrs"] - 1e-9).all()


def test_cheap_window_is_configurable():
    """The window is a parameter, not a constant -- moving it moves the charging."""
    archetypes = default_archetypes()
    cfg = next(c for c in archetypes if c.name == "Intelligent Octopus Average")
    cfg.population_pct = 100.0
    cfg.cheap_window_start_hr = 1.0  # 01:00-04:00 instead of 23:30-05:30
    cfg.cheap_window_end_hr = 4.0
    result = Simulator.run([cfg], n_agents=1500, seed=23)

    events = result.events[result.events["has_event"]]
    start, end = tg.window_offsets(1.0, 4.0)
    fits = ~events["charge_window_clipped"]
    assert events.loc[fits, "charge_start_offset_hrs"].min() >= start - 1e-9
    assert events.loc[fits, "charge_start_offset_hrs"].max() <= end + 1e-9

    # A 3-hour window fits far fewer charges than the 6-hour default, so the tail that
    # has to start early is much larger here -- but the bulk of the energy still lands
    # inside the window.
    inside = (tg.offsets() >= start) & (tg.offsets() < end)
    share_inside = result.charge_power_kw[:, inside].sum() / result.charge_power_kw.sum()
    assert share_inside > 0.6, f"only {share_inside:.1%} of energy landed in the window"


def test_managed_shifts_demand_later_than_unmanaged():
    """The behavioural point of the whole managed/unmanaged split."""
    archetypes = default_archetypes()
    io = next(c for c in archetypes if c.name == "Intelligent Octopus Average")
    io.population_pct = 100.0
    managed_result = Simulator.run([io], n_agents=2000, seed=31)

    unmanaged = next(c for c in default_archetypes() if c.name == "Intelligent Octopus Average")
    unmanaged.population_pct = 100.0
    unmanaged.behaviour = UNMANAGED
    unmanaged.managed = False
    unmanaged_result = Simulator.run([unmanaged], n_agents=2000, seed=31)

    def demand_weighted_hour(result):
        power = result.charge_power_kw.sum(axis=0)
        return float(np.average(tg.offsets(), weights=power))

    assert demand_weighted_hour(managed_result) > demand_weighted_hour(unmanaged_result) + 4.0

    # Same energy, just moved. Not bit-identical: the managed generator draws the extra
    # placement sample, so the two runs consume the shared RNG stream differently. The
    # totals still agree to sampling error because energy depends only on plug-in SoC.
    # Wider than before: both runs now also draw a Bernoulli plug-in gate, so they differ
    # in how many agents plug in at all, not just in their SoC draws.
    assert managed_result.charge_power_kw.sum() == pytest.approx(
        unmanaged_result.charge_power_kw.sum(), rel=0.06
    )


