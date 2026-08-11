"""Smoke tests for the Streamlit wrapper via ``AppTest``.

These exercise the session-state plumbing in :mod:`app.state` -- the part most likely
to break silently, since a mis-keyed widget shows up as a stale chart rather than an
error. The charts themselves are covered by the ``model`` tests.
"""

from __future__ import annotations

from datetime import time
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
TIMEOUT = 120  # the default 3s is not enough for a real simulation run


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    return at


def _set_readyby(at, index: int, choices: list[list[float]]) -> None:
    """Seed a ready-by table the way a scenario import does.

    Both the config value and the editor's baseline frame have to move together: the
    data editor is authoritative over the baseline, so setting the config value alone
    would be overwritten on the next rerun.
    """
    at.session_state[f"arch{index}__readyby_choices"] = choices
    at.session_state[f"arch{index}__readyby_choices__baseline"] = [list(c) for c in choices]


def test_app_starts_without_exceptions(app):
    assert not app.exception


def test_run_button_produces_results():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.session_state["n_agents"] = 400  # keep the test quick
    at.run()
    at.sidebar.button[0].click().run()

    assert not at.exception
    assert at.session_state["run_token"]
    # Headline metrics appear once there is a result.
    labels = [m.label for m in at.metric]
    assert "Agents" in labels
    assert "Peak plugged in" in labels
    # Four charts (combined, breakdown, demand, agent trace) plus the histogram.
    assert len(at.get("plotly_chart")) >= 4


def test_zeroing_the_whole_mix_disables_the_run_button():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    for i in range(at.session_state["n_archetypes"]):
        at.session_state[f"arch{i}__population_pct"] = 0.0
    at.run()

    assert not at.exception
    assert at.sidebar.button[0].disabled
    assert any("non-zero population share" in e.value for e in at.error)


def test_derived_captions_track_the_controls_in_the_same_run():
    """Regression guard: captions must re-read state after their widgets are created.

    Reading the archetype once at the top of the tab left every derived caption a
    rerun behind the slider that fed it.
    """
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    before = " ".join(c.value for c in at.caption)
    # Several archetypes share the 17:00-07:00 window and it is echoed on more than one
    # tab, so compare counts rather than pinning an exact number.
    before_count = before.count("17:00\u201307:00")
    assert before_count >= 3
    assert "17:00\u201308:30" not in before

    at.session_state["arch0__window_end_hr"] = time(8, 30)
    at.run()
    after = " ".join(c.value for c in at.caption)
    assert "17:00\u201308:30" in after, "caption did not follow the window input"
    assert after.count("17:00\u201307:00") < before_count, "unedited archetypes also changed"


