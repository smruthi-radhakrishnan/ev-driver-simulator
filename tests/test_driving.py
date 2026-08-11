"""Driving frequency and trip length, and the closure between them and charging.

Annual mileage is split into how often the driver drives and how far each trip is,
rather than being flattened into a daily average. A low-mileage driver takes a normal
trip less often, not a token trip every day: the mean depletion is the same but the
distribution is not.
"""

from __future__ import annotations

import pytest

from model import ScenarioConfig, Simulator, default_archetypes
from model.validate import validate_energy_closure


def _solo(name: str, n_agents: int = 20000, seed: int = 5, **overrides):
    cfg = next(c for c in default_archetypes() if c.name == name)
    cfg.population_pct = 100.0
    for field, value in overrides.items():
        setattr(cfg, field, value)
    return cfg, Simulator.run([cfg], n_agents=n_agents, seed=seed)


def test_reducing_frequency_lengthens_the_trip_and_holds_the_mean():
    """The core of the fix: the same annual mileage, redistributed."""
    daily = next(c for c in default_archetypes() if c.name == "Average (UK)")
    infrequent = next(c for c in default_archetypes() if c.name == "Average (UK)")
    infrequent.driving_days_per_week = 2.0

    assert infrequent.miles_per_year == daily.miles_per_year
    # Fewer driving days, so each trip is longer...
    assert infrequent.trip_miles > daily.trip_miles * 3
    assert infrequent.trip_soc_drop_pct > daily.trip_soc_drop_pct * 3
    # ...but the average daily depletion is untouched.
    assert infrequent.expected_daily_soc_drop_pct == pytest.approx(
        daily.expected_daily_soc_drop_pct
    )
    assert infrequent.daily_miles == pytest.approx(daily.daily_miles)


def test_never_driving_means_no_depletion():
    cfg, result = _solo("Average (UK)", n_agents=2000, driving_days_per_week=0.0)
    assert cfg.trip_miles == 0.0
    assert cfg.trip_soc_drop_pct == 0.0
    events = result.events[result.events["has_event"]]
    assert not events["drove_after_plugout"].any()
    assert (events["soc_after_driving"] == events["plugout_soc"]).all()


def test_closure_holds_across_frequency_and_cadence():
    """Every archetype's driving must reconcile with its charge requirement.

    A daily rate multiplied by the plug-in cadence, so this validates driving frequency,
    trip length and plug-in frequency against each other rather than checking a total.
    """
    closure = validate_energy_closure(Simulator.run(ScenarioConfig(n_agents=8000, seed=42)))
    assert not closure.empty
    assert closure["Closes"].all(), closure.to_string(index=False)
