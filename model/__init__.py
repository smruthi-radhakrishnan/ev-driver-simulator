"""Agent-based model of EV home charging behaviour across six driver archetypes.

Usable headlessly -- nothing in this package imports Streamlit.

    from model import ScenarioConfig, Simulator
    result = Simulator.run(ScenarioConfig())
"""

from model.config import (
    ALWAYS_PLUGGED_IN,
    DEFAULT_ARCHETYPES,
    INFREQUENT_CHARGING,
    MANAGED,
    UNMANAGED,
    ArchetypeConfig,
    ScenarioConfig,
    default_archetypes,
)
from model.population import Agent, build_population
from model.simulator import SimulationResult, Simulator

__all__ = [
    "ALWAYS_PLUGGED_IN",
    "DEFAULT_ARCHETYPES",
    "INFREQUENT_CHARGING",
    "MANAGED",
    "UNMANAGED",
    "Agent",
    "ArchetypeConfig",
    "ScenarioConfig",
    "SimulationResult",
    "Simulator",
    "build_population",
    "default_archetypes",
]
