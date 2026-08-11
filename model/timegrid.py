"""The simulation time grid: a single 24-hour window anchored at 12:00 noon (§4.1).

Noon-anchoring means every archetype's overnight cycle (earliest plug-in 18:00,
latest deadline 09:00) sits fully inside one window with no wrap-around at the
boundary. Internally the simulator works in *window offsets*: hours elapsed since
noon, so 18:00 -> 6.0 and 07:00 -> 19.0.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd

ANCHOR_DATE = _dt.date(2024, 1, 1)  # arbitrary; only the clock time matters
ANCHOR_HOUR = 12  # noon
WINDOW_HOURS = 24.0
RESOLUTION_MINUTES = 15
STEP_HOURS = RESOLUTION_MINUTES / 60.0
N_TIMESTEPS = int(WINDOW_HOURS / STEP_HOURS)  # 96


def make_timegrid() -> pd.DatetimeIndex:
    """The 96 timestamps of the window, 12:00 noon through 11:45 the next day."""
    start = _dt.datetime.combine(ANCHOR_DATE, _dt.time(ANCHOR_HOUR))
    return pd.date_range(start=start, periods=N_TIMESTEPS, freq=f"{RESOLUTION_MINUTES}min")


def offsets() -> np.ndarray:
    """Hours since window start for each timestep: [0.0, 0.25, ..., 23.75]."""
    return np.arange(N_TIMESTEPS) * STEP_HOURS


def to_window_offset(clock_hour: float) -> float:
    """Map a 24-hour clock hour onto hours-since-noon, in [0, 24)."""
    return float((clock_hour - ANCHOR_HOUR) % WINDOW_HOURS)


def to_clock_hour(offset_hrs: float) -> float:
    """Inverse of :func:`to_window_offset`, in [0, 24)."""
    return float((offset_hrs + ANCHOR_HOUR) % WINDOW_HOURS)


def format_clock(offset_hrs: float) -> str:
    """Render a window offset as an HH:MM clock label."""
    clock = to_clock_hour(offset_hrs)
    hours = int(clock)
    minutes = int(round((clock - hours) * 60))
    if minutes == 60:  # rounding spill
        hours, minutes = (hours + 1) % 24, 0
    return f"{hours:02d}:{minutes:02d}"


def window_offsets(start_hr: float, end_hr: float) -> tuple[float, float]:
    """Convert a clock-hour window to a monotonic offset pair.

    The end is pushed forward a day when it would otherwise land at or before the
    start, so an 18:00-07:00 window becomes (6.0, 19.0) rather than (6.0, 19.0-24).
    """
    start = to_window_offset(start_hr)
    end = to_window_offset(end_hr)
    if end <= start:
        end += WINDOW_HOURS
    return start, end


def hours_to_time(clock_hour: float) -> _dt.time:
    """Clock hour as a float -> ``datetime.time``, for HH:MM inputs. 18.25 -> 18:15."""
    clock = float(clock_hour) % WINDOW_HOURS
    total_minutes = int(round(clock * 60))
    if total_minutes >= 24 * 60:  # a value that rounds up to midnight
        total_minutes = 24 * 60 - 1
    return _dt.time(total_minutes // 60, total_minutes % 60)


def time_to_hours(value: _dt.time) -> float:
    """``datetime.time`` -> clock hour as a float. 18:15 -> 18.25."""
    return value.hour + value.minute / 60.0 + value.second / 3600.0


def format_hours(hours: float) -> str:
    """A duration in hours as a human label: 0.5 -> '30 min', 2.25 -> '2h 15m'."""
    total = int(round(float(hours) * 60))
    if total < 60:
        return f"{total} min"
    h, m = divmod(total, 60)
    return f"{h}h" if m == 0 else f"{h}h {m:02d}m"


def offset_to_timestamp(offset_hrs: float) -> pd.Timestamp:
    """Window offset -> wall-clock timestamp on the grid's anchor day."""
    start = _dt.datetime.combine(ANCHOR_DATE, _dt.time(ANCHOR_HOUR))
    return pd.Timestamp(start) + pd.Timedelta(hours=float(offset_hrs))
