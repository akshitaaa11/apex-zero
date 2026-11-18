"""Left/right track boundaries from the lateral spread of the cars (P2.5).

Per ``bin_m`` bin of ``s``, the ``lo_pct`` and ``hi_pct`` percentiles of every accurate-lap
sample's lateral offset ``n`` bound the band the cars actually use. Pit and out laps are
excluded upstream (only ``IsAccurate`` laps are passed in), and samples with ``|n| > max_abs_n_m``
(pit lane, mis-projections) are dropped.

The band's middle becomes the asset's centerline (``recentre=True``): a median racing line is
biased towards the inside of corners, the middle of the used band is not. Half the band plus
``margin_m`` is each side's width, clipped to the config prior ``[w_min/2, w_max/2]`` and
smoothed (periodic spline, length scale ``ell_m``).

Smoothing can pull a boundary slightly inside where the observed spread peaks briefly. That is
deliberate: an upper envelope would let a single car's excursion into the run-off widen the
limits, and loose limits are what an RL agent exploits. ``margin_m`` should be about half a
car's width (~1 m for an F1 car), since samples are car centres.

Assumption: the cars, summed over a session, use the track symmetrically about its middle.
This is a statement about the data, not about the asphalt; tracks with wide run-off that cars
use only on one side will have their centre biased towards that side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from apex_zero.errors import TrackReconstructionError
from apex_zero.track.centerline import periodic_smooth

__all__ = ["WidthEstimate", "estimate_width"]


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class WidthEstimate:
    """Boundary estimate on a line's grid.

    Attributes:
        centre_n: ``(N,)`` lateral offset of the band middle from the input line [m].
        w_left: ``(N,)`` width left of the (re-centred) line [m].
        w_right: ``(N,)`` width right of the (re-centred) line [m].
        bin_s: ``(B,)`` bin centres used [m].
        bin_lo: ``(B,)`` ``lo_pct`` percentile of ``n`` per bin [m].
        bin_hi: ``(B,)`` ``hi_pct`` percentile of ``n`` per bin [m].
    """

    centre_n: FloatArray
    w_left: FloatArray
    w_right: FloatArray
    bin_s: FloatArray
    bin_lo: FloatArray
    bin_hi: FloatArray


def estimate_width(
    s_line: FloatArray,
    length: float,
    s: FloatArray,
    n: FloatArray,
    *,
    bin_m: float,
    margin_m: float,
    w_min: float,
    w_max: float,
    ell_m: float,
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
    min_count: int = 10,
    max_abs_n_m: float = 30.0,
    recentre: bool = True,
) -> WidthEstimate:
    """Estimate boundaries from samples ``(s, n)`` relative to a line with grid ``s_line``.

    Raises:
        TrackReconstructionError: Fewer than 80 % of bins have ``min_count`` samples.
    """
    ok = np.isfinite(s) & np.isfinite(n) & (np.abs(n) <= max_abs_n_m)
    s, n = s[ok], n[ok]
    n_bins = max(int(round(length / bin_m)), 1)
    idx = np.minimum((s / (length / n_bins)).astype(np.int64), n_bins - 1)
    order = np.argsort(idx, kind="stable")
    idx, n = idx[order], n[order]
    bounds = np.searchsorted(idx, np.arange(n_bins + 1))
    bs, lo, hi, cnt = [], [], [], []
    for b in range(n_bins):
        v = n[bounds[b] : bounds[b + 1]]
        if len(v) >= min_count:
            bs.append((b + 0.5) * length / n_bins)
            lo.append(float(np.percentile(v, lo_pct)))
            hi.append(float(np.percentile(v, hi_pct)))
            cnt.append(len(v))
    if len(bs) < 0.8 * n_bins:
        raise TrackReconstructionError(f"width: only {len(bs)} of {n_bins} bins have samples")
    b_s, b_lo, b_hi = np.array(bs), np.array(lo), np.array(hi)
    w = np.array(cnt, dtype=np.float64)
    if recentre:
        centre = periodic_smooth(b_s, (b_hi + b_lo) / 2, length, ell_m, s_line, weights=w)
        half = periodic_smooth(b_s, (b_hi - b_lo) / 2, length, ell_m, s_line, weights=w)
        left = right = half
    else:
        centre = np.zeros(len(s_line))
        left = periodic_smooth(b_s, b_hi, length, ell_m, s_line, weights=w)
        right = periodic_smooth(b_s, -b_lo, length, ell_m, s_line, weights=w)
    w_left = np.clip(left + margin_m, w_min / 2, w_max / 2)
    w_right = np.clip(right + margin_m, w_min / 2, w_max / 2)
    return WidthEstimate(centre, w_left, w_right, b_s, b_lo, b_hi)
