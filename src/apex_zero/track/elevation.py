"""Elevation, grade, vertical curvature and banking (P2.6).

- ``z(s)``: the line's z plus the per-bin **median** height offset ``h`` of all accurate-lap
  samples, smoothed heavily (z noise in the feed is larger than x/y noise).
- ``grade(s) = dz/ds`` and vertical curvature ``d2z/ds2``: periodic central differences on the
  1 m grid.
- Banking (optional): per ``window_m`` window of ``s``, the least-squares slope of ``h`` on ``n``
  gives ``bank = atan(dh/dn)``. Positive bank means the left edge (``n > 0``) is higher. Windows
  where the samples span less than ``min_spread_m`` laterally get 0. The result is clipped to
  ±``clip_deg`` and smoothed. The feed's z noise makes this a coarse estimate: treat it as
  "is there notable banking here", not as a survey.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from apex_zero.track.centerline import bin_medians, periodic_smooth

__all__ = ["Elevation", "estimate_bank", "fit_elevation", "periodic_derivative"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Elevation:
    """Elevation channels on the line's grid."""

    z: FloatArray  # (N,) [m]
    grade: FloatArray  # (N,) dz/ds [-]
    vertical_curvature: FloatArray  # (N,) d2z/ds2 [1/m]


def periodic_derivative(values: FloatArray, ds: float, order: int = 1) -> FloatArray:
    """Central-difference derivative of a periodic channel on a uniform grid."""
    v = np.asarray(values, dtype=np.float64)
    for _ in range(order):
        v = (np.roll(v, -1) - np.roll(v, 1)) / (2 * ds)
    return v


def fit_elevation(
    s_line: FloatArray,
    z_line: FloatArray,
    length: float,
    s: FloatArray,
    h: FloatArray,
    *,
    bin_m: float,
    ell_z_m: float,
) -> Elevation:
    """Smoothed elevation from samples' height offsets ``h`` relative to the line."""
    bc, h_med, cnt = bin_medians(s, h, length, bin_m)
    dz = periodic_smooth(bc, h_med, length, ell_z_m, s_line, weights=cnt)
    z_raw = z_line + dz
    z = periodic_smooth(s_line, z_raw, length, ell_z_m, s_line)
    ds = length / len(s_line)
    return Elevation(z, periodic_derivative(z, ds), periodic_derivative(z, ds, order=2))


def estimate_bank(
    s_line: FloatArray,
    length: float,
    s: FloatArray,
    n: FloatArray,
    h: FloatArray,
    *,
    window_m: float = 50.0,
    min_spread_m: float = 4.0,
    min_count: int = 30,
    clip_deg: float = 10.0,
    ell_m: float = 50.0,
) -> FloatArray:
    """Bank angle [rad] per grid point (see module docstring)."""
    ok = np.isfinite(s) & np.isfinite(n) & np.isfinite(h)
    s, n, h = s[ok], n[ok], h[ok]
    n_win = max(int(round(length / window_m)), 1)
    idx = np.minimum((s / (length / n_win)).astype(np.int64), n_win - 1)
    centres, angles = [], []
    for b in range(n_win):
        sel = idx == b
        nn, hh = n[sel], h[sel]
        centres.append((b + 0.5) * length / n_win)
        if len(nn) < min_count or np.percentile(nn, 95) - np.percentile(nn, 5) < min_spread_m:
            angles.append(0.0)
            continue
        slope = np.polyfit(nn, hh, 1)[0]
        angles.append(float(np.arctan(slope)))
    clip = np.deg2rad(clip_deg)
    raw = np.clip(np.array(angles), -clip, clip)
    return np.clip(periodic_smooth(np.array(centres), raw, length, ell_m, s_line), -clip, clip)
