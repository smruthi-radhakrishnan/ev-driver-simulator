"""Analytic densities for the distributions the model samples from.

Used by the assumptions preview charts, which show the *resultant* distribution for a
set of parameters without needing to run the simulation. Kept free of scipy -- the
normal CDF comes from :func:`math.erf`.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

SQRT_2PI = math.sqrt(2.0 * math.pi)


def normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std <= 0:
        return np.zeros_like(x, dtype=float)
    z = (np.asarray(x, dtype=float) - mean) / std
    return np.exp(-0.5 * z * z) / (std * SQRT_2PI)


def normal_cdf(x: float, mean: float, std: float) -> float:
    if std <= 0:
        return 1.0 if x >= mean else 0.0
    return 0.5 * (1.0 + math.erf((x - mean) / (std * math.sqrt(2.0))))


def truncated_normal_pdf(
    x: np.ndarray, mean: float, std: float, low: float, high: float
) -> np.ndarray:
    """Density of a normal truncated to [low, high]: zero outside, renormalised inside.

    Matches :func:`model.events.truncated_normal`, which rejection-samples, and is what
    both the plug-in time and the plug-in state of charge previews draw. There is no
    clipped counterpart: clipping piles the excluded tails onto the two bounds as
    single-value spikes, which the model no longer does anywhere.
    """
    x = np.asarray(x, dtype=float)
    if high <= low:
        return np.zeros_like(x)
    if std <= 0:
        return np.where(np.isclose(x, np.clip(mean, low, high)), 1.0, 0.0)

    mass = normal_cdf(high, mean, std) - normal_cdf(low, mean, std)
    inside = (x >= low) & (x <= high)
    if mass <= 0:
        # The bounds sit far into a tail, so sampling collapses onto the nearer bound.
        return np.where(np.isclose(x, np.clip(mean, low, high)), 1.0, 0.0)
    return np.where(inside, normal_pdf(x, mean, std) / mass, 0.0)


def truncated_normal_cdf(
    x: np.ndarray, mean: float, std: float, low: float, high: float
) -> np.ndarray:
    """CDF of a normal truncated to [low, high]."""
    x = np.asarray(x, dtype=float)
    if high <= low:
        return (x >= low).astype(float)
    if std <= 0:
        return (x >= np.clip(mean, low, high)).astype(float)
    mass = normal_cdf(high, mean, std) - normal_cdf(low, mean, std)
    if mass <= 0:
        return (x >= np.clip(mean, low, high)).astype(float)
    lower = normal_cdf(low, mean, std)
    raw = np.array([normal_cdf(v, mean, std) for v in x.ravel()]).reshape(x.shape)
    return np.clip((raw - lower) / mass, 0.0, 1.0)


def mixture_normal_pdf(
    x: np.ndarray, components: Sequence[tuple[float, float, float]]
) -> np.ndarray:
    """Weighted sum of normal densities. ``components`` is (mean, std, weight)."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for mean, std, weight in components:
        out = out + weight * normal_pdf(x, mean, std)
    return out


def mixture_normal_cdf(
    x: np.ndarray, components: Sequence[tuple[float, float, float]]
) -> np.ndarray:
    """Weighted sum of normal CDFs, matching :func:`mixture_normal_pdf`."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for mean, std, weight in components:
        component = np.array([normal_cdf(v, mean, std) for v in x.ravel()]).reshape(x.shape)
        out = out + weight * component
    return np.clip(out, 0.0, 1.0)


def connected_probability(
    grid: np.ndarray,
    plugin: tuple[float, float, float, float],
    plugout_components: Sequence[tuple[float, float, float]],
) -> np.ndarray:
    """P(plugged in at t) = P(plug-in <= t) x P(plug-out > t).

    ``plugin`` is (mean, std, window_low, window_high) for the truncated plug-in
    distribution; ``plugout_components`` is the ready-by mixture. The two are
    independent in the sampler apart from a guard forcing plug-out after plug-in,
    which never binds for realistic windows -- verified against simulation to within
    the simulation's own sampling noise.
    """
    mean, std, low, high = plugin
    f_in = truncated_normal_cdf(grid, mean, std, low, high)
    f_out = mixture_normal_cdf(grid, plugout_components)
    return f_in * (1.0 - f_out)
