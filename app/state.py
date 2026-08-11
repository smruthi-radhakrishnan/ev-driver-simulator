"""Session-state helpers.

``st.session_state`` is the single source of truth for the scenario. Every editable
field is stored under a stable key (``arch0__battery_kwh``), which lets the widgets
own their own values -- there is no separate copy of the config to keep in sync.
:func:`current_scenario` reads those keys back into a :class:`ScenarioConfig`.
"""

from __future__ import annotations

import json
from dataclasses import fields
from typing import Any

import streamlit as st

from model import timegrid as tg
from model.config import ArchetypeConfig, ScenarioConfig, default_archetypes
from model.simulator import SimulationResult, Simulator

ARCH_FIELDS = [f.name for f in fields(ArchetypeConfig)]
RUN_TOKEN = "run_token"  # scenario JSON of the last run the user asked for
PENDING_SCENARIO = "_pending_scenario"

# Fields the config stores as float clock hours but the UI edits as HH:MM, and fields
# stored as float hours but edited in whole minutes. Session state holds the widget's
# native type; the conversion happens on the way in and out of ArchetypeConfig.
CLOCK_FIELDS = frozenset(
    {
        "window_start_hr",
        "window_end_hr",
        "plugin_time_mean_hr",
        "cheap_window_start_hr",
        "cheap_window_end_hr",
    }
)
MINUTE_FIELDS = frozenset(
    {
        "plugin_time_std_hr",
        "plugout_time_std_hr",
        "charge_placement_std_hr",
        "trip_duration_hrs",
    }
)


LIST_EDITOR_VERSION = "_list_editor_version"

# Fields edited through a data editor rather than a single widget. Each needs a stable
# baseline frame, for the reason documented on load_archetypes.
LIST_FIELDS = ("readyby_choices", "target_soc_choices")


def arch_key(index: int, field_name: str) -> str:
    return f"arch{index}__{field_name}"


def list_baseline_key(index: int, field_name: str) -> str:
    """The frame handed to a data editor, held stable across reruns."""
    return f"arch{index}__{field_name}__baseline"


def list_editor_key(index: int, field_name: str) -> str:
    """Widget key for a data editor, versioned so a reset discards stale edits."""
    version = st.session_state.get(LIST_EDITOR_VERSION, 0)
    return f"{field_name}_editor_{index}_{version}"


def to_widget_value(field_name: str, value):
    """Config value -> the type the widget for that field expects."""
    if value is None:
        return None
    if field_name in CLOCK_FIELDS:
        return tg.hours_to_time(value)
    if field_name in MINUTE_FIELDS:
        return int(round(value * 60))
    return value


def from_widget_value(field_name: str, value):
    """Widget value -> the float the config stores."""
    if value is None:
        return None
    if field_name in CLOCK_FIELDS:
        return tg.time_to_hours(value)
    if field_name in MINUTE_FIELDS:
        return float(value) / 60.0
    return value


def init_state() -> None:
    """Seed session state, and apply any queued scenario load.

    Must run before any widget is created: Streamlit refuses to let a widget-backed
    key be written once that widget exists in the current script run, which is why
    resets and imports queue themselves rather than writing state directly.
    """
    if not st.session_state.get("_initialised"):
        load_archetypes(default_archetypes())
        st.session_state.setdefault("n_agents", 2000)
        st.session_state.setdefault("seed", 42)
        st.session_state["_initialised"] = True

    queued = st.session_state.pop(PENDING_SCENARIO, None)
    if queued is not None:
        load_scenario(ScenarioConfig.from_dict(queued))


def request_scenario_load(scenario: ScenarioConfig) -> None:
    """Queue a scenario to be applied at the top of the next script run."""
    st.session_state[PENDING_SCENARIO] = scenario.to_dict()


def load_archetypes(archetypes: list[ArchetypeConfig]) -> None:
    """Write an archetype list into session state, replacing what is there.

    Also records a *baseline* copy of each ready-by table and bumps a version counter.
    ``st.data_editor`` stores its edits as deltas against the frame it was handed, so
    it must keep being handed the same frame for the life of a config generation --
    feeding it the edited result would re-apply those deltas and duplicate rows. The
    version is part of the editor's widget key, so resetting the scenario retires the
    old editor and its stale deltas along with it.
    """
    st.session_state["n_archetypes"] = len(archetypes)
    for i, cfg in enumerate(archetypes):
        for field_name, value in cfg.to_dict().items():
            st.session_state[arch_key(i, field_name)] = to_widget_value(field_name, value)
        for field_name in LIST_FIELDS:
            st.session_state[list_baseline_key(i, field_name)] = [
                list(entry) for entry in getattr(cfg, field_name)
            ]
    st.session_state[LIST_EDITOR_VERSION] = st.session_state.get(LIST_EDITOR_VERSION, 0) + 1


def load_scenario(scenario: ScenarioConfig) -> None:
    load_archetypes(scenario.archetypes)
    st.session_state["n_agents"] = scenario.n_agents
    st.session_state["seed"] = scenario.seed


def archetype_at(index: int) -> ArchetypeConfig:
    """Build one archetype from session state as it stands *right now*.

    Call this again after rendering an archetype's widgets: within a single script run
    a widget writes its new value to session state as it is created, so re-reading here
    is what keeps derived captions and warnings in step with the controls above them
    rather than lagging a rerun behind.
    """
    data: dict[str, Any] = {
        f: from_widget_value(f, st.session_state[arch_key(index, f)]) for f in ARCH_FIELDS
    }
    return ArchetypeConfig.from_dict(data)


def current_archetypes() -> list[ArchetypeConfig]:
    return [archetype_at(i) for i in range(st.session_state.get("n_archetypes", 0))]


def current_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        archetypes=current_archetypes(),
        n_agents=int(st.session_state["n_agents"]),
        seed=int(st.session_state["seed"]),
    )


def scenario_key(scenario: ScenarioConfig) -> str:
    """A stable string identity for a scenario -- the cache key and the run token."""
    return json.dumps(scenario.to_dict(), sort_keys=True)


@st.cache_data(show_spinner=False, max_entries=8)
def run_simulation(scenario_json: str) -> SimulationResult:
    """Cached simulation run.

    Keyed on the scenario JSON rather than the config object so that identical
    scenarios hit the cache regardless of how the objects were constructed.
    """
    return Simulator.run(ScenarioConfig.from_dict(json.loads(scenario_json)))


def request_run() -> None:
    """Record that the user pressed Run against the scenario as it stands now."""
    st.session_state[RUN_TOKEN] = scenario_key(current_scenario())


def has_result() -> bool:
    return bool(st.session_state.get(RUN_TOKEN))


def latest_result() -> SimulationResult | None:
    token = st.session_state.get(RUN_TOKEN)
    return run_simulation(token) if token else None


def is_stale() -> bool:
    """True when the scenario has been edited since the last run."""
    if not has_result():
        return False
    return st.session_state[RUN_TOKEN] != scenario_key(current_scenario())
