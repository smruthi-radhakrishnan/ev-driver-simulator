"""§7.1 / §4.1 -- the noon-anchored window must contain every archetype's full cycle."""

import numpy as np
import pytest

from model import timegrid as tg
from model.config import MANAGED, default_archetypes


def test_grid_shape_and_bounds():
    grid = tg.make_timegrid()
    assert len(grid) == tg.N_TIMESTEPS == 96
    assert grid[0].hour == 12 and grid[0].minute == 0
    assert grid[-1].hour == 11 and grid[-1].minute == 45
    # 15-minute resolution throughout
    deltas = np.diff(grid.values).astype("timedelta64[m]").astype(int)
    assert set(deltas) == {tg.RESOLUTION_MINUTES}


@pytest.mark.parametrize(
    "clock_hour,expected_offset",
    [(12.0, 0.0), (18.0, 6.0), (23.5, 11.5), (0.0, 12.0), (5.5, 17.5), (7.0, 19.0), (9.0, 21.0)],
)
def test_to_window_offset(clock_hour, expected_offset):
    assert tg.to_window_offset(clock_hour) == pytest.approx(expected_offset)


def test_every_plugin_window_fits_inside_the_window_untruncated():
    """The core §4.1 claim: no archetype's plug-in..deadline cycle is clipped."""
    for cfg in default_archetypes():
        start, end = tg.window_offsets(cfg.window_start_hr, cfg.window_end_hr)
        assert start >= 0.0, f"{cfg.name}: window starts before the simulation window"
        assert end - start <= tg.WINDOW_HOURS, f"{cfg.name}: cycle longer than 24hr"
        if not cfg.is_continuous:
            # The overnight archetypes must sit strictly inside [0, 24) -- this is
            # what noon-anchoring buys us and what a midnight anchor would break.
            assert end <= tg.WINDOW_HOURS, f"{cfg.name}: deadline falls outside the window"
            assert start < end


def test_managed_cheap_windows_sit_inside_their_drivers_window():
    """§4.4.3 should rarely bind: the cheap window ends before each deadline."""
    for cfg in default_archetypes():
        if cfg.behaviour != MANAGED:
            continue
        win_start, win_end = tg.window_offsets(cfg.window_start_hr, cfg.window_end_hr)
        cheap_start, cheap_end = tg.window_offsets(
            cfg.cheap_window_start_hr, cfg.cheap_window_end_hr
        )
        assert win_start <= cheap_start < cheap_end <= win_end, (
            f"{cfg.name}: cheap window {cheap_start}-{cheap_end} escapes "
            f"driver window {win_start}-{win_end}"
        )
