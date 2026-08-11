"""§4.3 exception / §7.7 -- IO Average's simulated stats land near the report figures."""

import numpy as np
import pytest

from model import ScenarioConfig, Simulator, default_archetypes
from model.validate import (
    IO_AVERAGE_NAME,
    REPORT_MEDIAN_PLUGIN_SOC,
    SOC_ABS_TOLERANCE_PP,
    io_average_stats,
    validate_io_average,
)


@pytest.fixture(scope="module")
def result():
    return Simulator.run(ScenarioConfig(n_agents=6000, seed=42))


def test_median_plugin_soc_near_report(result):
    stats = io_average_stats(result)
    assert abs(stats["median_plugin_soc"] - REPORT_MEDIAN_PLUGIN_SOC) <= SOC_ABS_TOLERANCE_PP


def test_validation_table_shape_and_verdicts(result):
    table = validate_io_average(result)
    assert list(table.columns) == [
        "Metric",
        "Simulated",
        "Report",
        "Tolerance",
        "Within tolerance",
    ]
    verdicts = table.set_index("Metric")["Within tolerance"]
    assert verdicts["Median plug-in SoC"] is True
    assert verdicts["Mean energy per plug event"] is None
    # Every duration row is informational: the report's stated 2h30 does not follow from
    # the other figures it states, and a median cannot be checked against an analytic mean.
    assert verdicts["Median charge duration"] is None
    assert verdicts["Charge duration lower quartile"] is None
    assert verdicts["Plugged in already above target"] is None


def test_validation_is_empty_when_io_average_absent():
    """The panel must degrade gracefully, not raise, if the archetype is zeroed out."""
    archetypes = [c for c in default_archetypes() if c.name != IO_AVERAGE_NAME]
    result = Simulator.run(archetypes, n_agents=500, seed=7)
    assert validate_io_average(result).empty
    assert io_average_stats(result) == {}


def test_duration_follows_the_soc_formula_not_a_stored_parameter(result):
    """§4.3 -- duration is derived per event, so it must reconcile exactly."""
    cfg = next(c for c in result.archetypes if c.name == IO_AVERAGE_NAME)
    events = result.events
    subset = events[(events["archetype"] == IO_AVERAGE_NAME) & events["has_event"]]
    # Against each agent's *own* sampled target, now that the target is a distribution.
    expected = (
        (subset["target_soc"] - subset["plugin_soc"]).clip(lower=0)
        / 100.0
        * cfg.battery_kwh
        / cfg.charger_kw
    )
    assert np.allclose(subset["required_duration_hrs"], expected)
    # Delivered duration matches the requirement except where the driver left early.
    finished = subset["reached_target"].astype(bool)
    assert np.allclose(subset.loc[finished, "charge_duration_hrs"], expected[finished])


