"""Building the agent population from an archetype mix (§4.7 phase offsets included)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from model.config import INFREQUENT_CHARGING, ArchetypeConfig, ScenarioConfig


@dataclass
class Agent:
    """One simulated driver. Behaviour lives in :mod:`model.events`; this is state."""

    agent_id: int
    archetype_index: int
    archetype: str

    # §4.7 -- Infrequent Charging only. ``next_gap_days`` is this agent's draw from the
    # triangular renewal distribution; ``days_since_last_plugin`` is its random phase
    # within that gap, so no warm-up period is needed.
    next_gap_days: Optional[float] = None
    days_since_last_plugin: Optional[float] = None


def allocate_counts(proportions: Sequence[float], n_agents: int) -> list[int]:
    """Split ``n_agents`` across archetypes by proportion, exactly.

    Uses largest-remainder rather than multinomial sampling so the realised mix
    matches the requested mix exactly -- important when Always Plugged-In is only
    1% of the fleet and sampling noise would swamp it.
    """
    total = float(sum(proportions))
    if total <= 0:
        raise ValueError("archetype proportions must sum to something positive")

    exact = [n_agents * p / total for p in proportions]
    counts = [int(np.floor(x)) for x in exact]
    remainder = n_agents - sum(counts)
    if remainder:
        # Hand the leftover agents to the largest fractional parts.
        order = sorted(range(len(exact)), key=lambda i: exact[i] - counts[i], reverse=True)
        for i in order[:remainder]:
            counts[i] += 1
    return counts


def build_population(
    scenario: ScenarioConfig | Sequence[ArchetypeConfig],
    n_agents: Optional[int] = None,
    seed: Optional[int] = None,
) -> list[Agent]:
    """Create the agent list for a scenario.

    Accepts either a :class:`ScenarioConfig` or a bare sequence of archetypes plus
    ``n_agents``/``seed``.
    """
    if isinstance(scenario, ScenarioConfig):
        archetypes = scenario.archetypes
        n_agents = scenario.n_agents if n_agents is None else n_agents
        seed = scenario.seed if seed is None else seed
    else:
        archetypes = list(scenario)
        if n_agents is None:
            raise ValueError("n_agents is required when passing a bare archetype sequence")

    rng = np.random.default_rng(seed)
    counts = allocate_counts([a.population_pct for a in archetypes], n_agents)

    agents: list[Agent] = []
    agent_id = 0
    for idx, (cfg, count) in enumerate(zip(archetypes, counts)):
        for _ in range(count):
            agent = Agent(agent_id=agent_id, archetype_index=idx, archetype=cfg.name)
            if cfg.behaviour == INFREQUENT_CHARGING:
                gap, phase = _sample_renewal_phase(cfg, rng)
                agent.next_gap_days = gap
                agent.days_since_last_plugin = phase
            agents.append(agent)
            agent_id += 1
    return agents


def _sample_renewal_phase(
    cfg: ArchetypeConfig, rng: np.random.Generator
) -> tuple[float, float]:
    """Draw an Infrequent Charging agent's renewal gap and its phase within it (§4.7).

    The gap comes from the configured triangular distribution. The phase --
    "days since last plug-in" -- is drawn uniformly across that gap, which is the
    stationary (length-biased) position of a renewal process observed at a random
    instant. An event is then due whenever the phase falls in the final day of the
    gap, which makes the share of agents charging on any one day ``E[1/gap]``
    ~= 1/mode ~= 20% for the 3/5/8 defaults, as §4.7 requires, and keeps all three
    gap parameters live rather than only the mode.
    """
    gap = float(
        rng.triangular(
            cfg.interplug_gap_min_days,
            cfg.interplug_gap_mode_days,
            cfg.interplug_gap_max_days,
        )
    )
    phase = float(rng.uniform(0.0, gap))
    return gap, phase


def population_summary(agents: Sequence[Agent], archetypes: Sequence[ArchetypeConfig]):
    """Realised archetype mix, for display in the app."""
    import pandas as pd

    counts = {cfg.name: 0 for cfg in archetypes}
    for agent in agents:
        counts[agent.archetype] += 1
    n = max(len(agents), 1)
    return pd.DataFrame(
        {
            "Archetype": list(counts),
            "Agents": list(counts.values()),
            "Share of fleet (%)": [100.0 * c / n for c in counts.values()],
            "Requested (%)": [cfg.population_pct for cfg in archetypes],
        }
    )
