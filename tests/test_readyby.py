"""Ready-by as a weighted choice among times, and its effect on managed charging.

Ready-by is a value the driver enters, so it is modelled as a discrete choice rather
than a continuous density. It is distinct from actual plug-out (which is ready-by plus
an offset plus noise) and from the archetype's plug-in window.
"""

from __future__ import annotations

import pytest

from model import Simulator, default_archetypes
from model import timegrid as tg
from model.config import ArchetypeConfig


def _solo(name: str, n_agents: int = 20000, seed: int = 5, **overrides):
    cfg = next(c for c in default_archetypes() if c.name == name)
    cfg.population_pct = 100.0
    for field, value in overrides.items():
        setattr(cfg, field, value)
    return cfg, Simulator.run([cfg], n_agents=n_agents, seed=seed)


def test_defaults_carry_the_reports_ready_by_mix_where_it_applies():
    """The four archetypes on the evening/morning cycle take report Figure 2's mix.

    Scheduled Charging and Always Plugged-In keep a single time: the report measures
    Intelligent Octopus customers, and neither of those two is one.
    """
    mixes = {c.name: c.readyby_distribution for c in default_archetypes()}
    multi = {n: d for n, d in mixes.items() if len(d) > 1}
    assert len(multi) == 4
    for name, distribution in multi.items():
        assert sum(w for _, w in distribution) == pytest.approx(1.0)
        # 07:00 is the report's most popular deadline.
        assert max(distribution, key=lambda p: p[1])[0] == 7.0
        assert dict(distribution)[7.0] > dict(distribution).get(6.0, 0.0), name

    for cfg in default_archetypes():
        if len(cfg.readyby_choices) == 1:
            assert cfg.readyby_choices == [[cfg.window_end_hr, 1.0]]


def test_sampled_deadlines_match_the_configured_weights():
    cfg, result = _solo("Average (UK)", readyby_choices=[[6.0, 0.2], [7.0, 0.5], [8.0, 0.3]])
    observed = result.events["deadline_time"].value_counts(normalize=True)
    assert observed["06:00"] == pytest.approx(0.2, abs=0.02)
    assert observed["07:00"] == pytest.approx(0.5, abs=0.02)
    assert observed["08:00"] == pytest.approx(0.3, abs=0.02)
    # Deadlines land only on the configured times -- no continuous spread.
    assert set(observed.index) == {"06:00", "07:00", "08:00"}


def test_plugout_is_centred_on_each_agents_own_deadline():
    cfg, result = _solo(
        "Average (UK)", readyby_choices=[[6.0, 0.5], [9.0, 0.5]], plugout_offset_hr=0.0
    )
    events = result.events
    for label, expected_hr in (("06:00", 6.0), ("09:00", 9.0)):
        subset = events[events["deadline_time"] == label]
        expected = tg.window_offsets(cfg.window_start_hr, expected_hr)[1]
        assert subset["plugout_offset_hrs"].mean() == pytest.approx(expected, abs=0.05)
        assert subset["plugout_offset_hrs"].std() == pytest.approx(
            cfg.plugout_time_std_hr, abs=0.05
        )


def test_deadline_inside_the_cheap_window_keeps_charging_inside_it():
    """A deadline within the window curtails it; it does not push charging out.

    Guards a diagnostic that was originally wrong: 04:00 sits inside 23:30-05:30, so
    the safety constraint charges earlier within the window rather than leaving it.
    """
    cfg, result = _solo(
        "Intelligent Octopus Average", readyby_choices=[[4.0, 0.2], [7.0, 0.8]]
    )
    cheap_start, cheap_end = tg.window_offsets(
        cfg.cheap_window_start_hr, cfg.cheap_window_end_hr
    )
    events = result.events[result.events["has_event"]]
    inside = (
        (events["charge_start_offset_hrs"] >= cheap_start - 1e-9)
        & (events["charge_end_offset_hrs"] <= cheap_end + 1e-9)
    ).mean()
    # Lower than it once was: the report-calibrated SoC spread and target mix together
    # leave about a sixth of charges too long for the 6-hour window, so they have to start
    # before it opens.
    assert inside > 0.70


def test_readyby_choices_are_validated():
    def build(**kwargs):
        return ArchetypeConfig(
            name="x",
            population_pct=100.0,
            battery_kwh=60.0,
            charger_kw=7.0,
            window_start_hr=18.0,
            window_end_hr=7.0,
            default_target_soc=80.0,
            managed=False,
            **kwargs,
        )

    with pytest.raises(ValueError, match="hour, weight"):
        build(readyby_choices=[[7.0]])
    with pytest.raises(ValueError, match="positive"):
        build(readyby_choices=[[7.0, 0.0]])
    assert build().readyby_choices == [[7.0, 1.0]]


def test_target_soc_defaults_match_the_reports_measured_shares():
    """Report Figure 2: 80%/90%/100% at 25%/24%/23%, with 80% the mode."""
    for cfg in default_archetypes():
        distribution = dict(cfg.target_soc_distribution)
        if len(distribution) == 1:
            continue
        assert distribution[80.0] == pytest.approx(0.25)
        assert distribution[90.0] == pytest.approx(0.24)
        assert distribution[100.0] == pytest.approx(0.23)
        # 80% must remain the single most common choice, as the report states.
        assert cfg.modal_target_soc == 80.0
        assert sum(distribution.values()) == pytest.approx(1.0)
        # The mean sits above the modal 80% because 90 and 100 together outweigh it.
        assert cfg.mean_target_soc == pytest.approx(84.2, abs=0.5)


def test_sampled_targets_match_the_configured_weights():
    cfg, result = _solo("Intelligent Octopus Average")
    observed = result.events[result.events["has_event"]]["target_soc"].value_counts(
        normalize=True
    )
    for target, weight in cfg.target_soc_distribution:
        assert observed[target] == pytest.approx(weight, abs=0.02)


def test_target_spread_widens_the_charge_duration_for_identical_plug_in_soc():
    """The point of the distribution: duration is no longer a function of SoC alone."""
    cfg, result = _solo("Intelligent Octopus Average")
    events = result.events[result.events["has_event"]]

    # Agents arriving within a narrow SoC band still span a wide range of durations.
    band = events[(events["plugin_soc"] > 50) & (events["plugin_soc"] < 54)]
    assert len(band) > 100
    spread = band["required_duration_hrs"].max() - band["required_duration_hrs"].min()
    assert spread > 3.0, f"identical plug-in SoC should still span durations, got {spread:.2f}h"

    # A single target collapses that spread to nearly nothing.
    single, single_result = _solo(
        "Intelligent Octopus Average", target_soc_choices=[[80.0, 1.0]]
    )
    single_events = single_result.events[single_result.events["has_event"]]
    single_band = single_events[
        (single_events["plugin_soc"] > 50) & (single_events["plugin_soc"] < 54)
    ]
    single_spread = (
        single_band["required_duration_hrs"].max() - single_band["required_duration_hrs"].min()
    )
    assert single_spread < 0.5
