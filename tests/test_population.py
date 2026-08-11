"""§7 -- archetype proportions are exact, and the §4.7 phase offset gives ~20% due."""

import numpy as np
import pytest

from model import ScenarioConfig, Simulator, build_population, default_archetypes
from model.config import ALWAYS_PLUGGED_IN, INFREQUENT_CHARGING
from model.events import infrequent_charging_gate
from model.population import allocate_counts


def test_allocate_counts_is_exact_and_proportional():
    counts = allocate_counts([40, 30, 10, 10, 9, 1], 1000)
    assert sum(counts) == 1000
    assert counts == [400, 300, 100, 100, 90, 10]


def test_population_matches_requested_mix():
    archetypes = default_archetypes()
    agents = build_population(archetypes, n_agents=2000, seed=3)
    assert len(agents) == 2000
    for cfg in archetypes:
        share = sum(a.archetype == cfg.name for a in agents) / len(agents) * 100.0
        assert share == pytest.approx(cfg.population_pct, abs=0.05), cfg.name


def test_phase_offset_produces_roughly_one_in_five_due_today():
    """§4.7 -- ~1/mode ~= 20% of Infrequent Charging agents should have an event due."""
    archetypes = default_archetypes()
    cfg = next(c for c in archetypes if c.behaviour == INFREQUENT_CHARGING)
    rng = np.random.default_rng(0)
    agents = [
        a
        for a in build_population(archetypes, n_agents=40000, seed=11)
        if a.archetype == cfg.name
    ]
    due = np.mean([infrequent_charging_gate(a, rng) for a in agents])
    assert due == pytest.approx(0.20, abs=0.03), f"due rate {due:.3f}"


def test_plugin_frequency_gates_every_archetype():
    """The configurable plug-in frequency, not just the Infrequent Charging cadence."""
    archetypes = default_archetypes()
    for cfg in archetypes:
        if cfg.behaviour not in (INFREQUENT_CHARGING, ALWAYS_PLUGGED_IN):
            cfg.plugin_days_per_week = 3.5  # half of the week
    events = Simulator.run(archetypes, n_agents=20000, seed=19).events

    gated = events[
        ~events["archetype"].isin(["Infrequent Charging", "Always Plugged-In"])
    ]
    assert gated["has_event"].mean() == pytest.approx(0.5, abs=0.02)
    # The always-connected archetype is never gated.
    assert events[events["archetype"] == "Always Plugged-In"]["has_event"].all()


def test_run_is_reproducible_and_seed_sensitive():
    a = Simulator.run(ScenarioConfig(n_agents=800, seed=99))
    b = Simulator.run(ScenarioConfig(n_agents=800, seed=99))
    c = Simulator.run(ScenarioConfig(n_agents=800, seed=100))
    assert np.array_equal(a.plugged_in, b.plugged_in)
    assert np.allclose(a.soc, b.soc, equal_nan=True)
    assert not np.allclose(
        np.nan_to_num(a.soc), np.nan_to_num(c.soc)
    ), "different seeds produced identical SoC traces"
