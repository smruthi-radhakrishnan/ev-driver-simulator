"""Charts build, and the closed-form previews agree with the simulation.

Deliberately thin. An earlier version asserted colours, luminance ordering, dash
patterns, annotation strings and bar widths -- 34 assertions on presentation detail.
They encoded taste rather than behaviour and were a net cost: every visual change broke
several of them without ever catching a modelling error. What is worth guarding is that
the figures render at all, and that a preview drawn analytically actually describes what
the simulator does -- because that one can drift silently.
"""

from __future__ import annotations

import numpy as np

from model import ScenarioConfig, Simulator, default_archetypes
from model import timegrid as tg
from model.distributions import connected_probability
from model.plotting import (
    build_agent_trace_chart,
    build_breakdown_chart,
    build_combined_chart,
    build_demand_chart,
    build_demand_per_agent_box,
    build_plugin_soc_histogram,
    build_soc_distribution_preview,
    build_timing_preview,
    plugin_distribution,
    plugout_mixture,
)


def test_every_chart_builds_and_serialises():
    """Streamlit and the headless writer both serialise to JSON, so this must hold."""
    result = Simulator.run(ScenarioConfig(n_agents=1500, seed=42))
    figures = [
        build_combined_chart(result),
        build_combined_chart(result, connected_only=False),
        build_breakdown_chart(result),
        build_breakdown_chart(result, normalise=True),
        build_demand_chart(result),
        build_demand_per_agent_box(result),
        build_demand_per_agent_box(result, metric="duration"),
        build_agent_trace_chart(result, 0),
        build_plugin_soc_histogram(result),
    ]
    for cfg in result.archetypes:
        figures.append(build_soc_distribution_preview(cfg))
        timing = build_timing_preview(cfg)
        # None for the always-connected archetype, which has no plug-in event at all.
        assert (timing is None) == cfg.is_continuous
        if timing is not None:
            figures.append(timing)

    for fig in figures:
        assert fig.to_json()


def test_connection_preview_matches_the_simulation():
    """The Population tab's curve is closed form; it has to be what the model does."""
    for name in ("Average (UK)", "Intelligent Octopus Average"):
        cfg = next(c for c in default_archetypes() if c.name == name)
        cfg.population_pct = 100.0
        simulated = Simulator.run([cfg], n_agents=20000, seed=11).plugged_in.mean(axis=0)

        _, params = plugin_distribution(cfg)
        closed = connected_probability(tg.offsets(), params, plugout_mixture(cfg))
        closed = closed * min(cfg.expected_plugins_per_day, 1.0)

        # Tolerance is the simulation's own sampling noise at this population size.
        assert np.abs(closed - simulated).max() < 0.02, name
