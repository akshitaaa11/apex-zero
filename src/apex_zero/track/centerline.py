"""Centerline from fused multi-lap positions (P2.3).

Pipeline (``SKILLS.md`` -> *reconstruct-track*):

1. **Seed**: the fastest accurate lap's raw samples (~4 Hz, ~20 m apart at speed) as a closed
   polyline, smoothed (``smoothing.seed``, default 25 m) and resampled to 1 m. Refinement only
   adds *smooth* offsets, so wiggles left in the seed would survive into the curvature; the
   seed's corner-cutting bias, by contrast, is long-wavelength and refinement removes it.
2. **Refine** (``iterations`` times): project every point onto the current line with an ``s``
   continuity window (each lap's previous-iteration ``s`` as the hint), take the **median**
   lateral offset ``n`` and height offset ``h`` per ``bin_m`` bin, move the line by those
   medians along its normal and vertically, and refit.
3. **Smooth**: every fit is a periodic cubic smoothing spline of ``x(s), y(s)`` (and ``z(s)``)
   with a length scale in metres from ``configs/tracks/<name>.yaml``.

Smoothing assumption: ``scipy.interpolate.make_smoothing_spline`` minimises
``sum w (y - f)^2 + lam * integral f''^2``. With samples of spacing ``h`` and
``lam = h * ell**4`` its transfer function is ``H(omega) = 1 / (1 + (ell * omega)**4)``: a
sinusoid of wavelength ``2 pi ell`` is halved, longer ones pass almost untouched. ``ell`` is what
the config gives. Smoothing the *coordinates* of a curve of radius ``r`` therefore pulls it
inwards by about ``r * (ell / r)**4 = ell**4 / r**3`` (tested): negligible on a 150 m bend
(0.015 m at ``ell = 15``) but ~1.3 m in a 20 m hairpin at ``ell = 10``. That is why refinement
smooths the lateral *offsets* and never the line's coordinates. Periodicity is obtained by
fitting on the data tiled one third of a lap beyond each end and evaluating on ``[0, L)``.

The result follows the *median* racing line of the cars; ``width.py`` re-centres it between
the observed boundaries.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import make_smoothing_spline

from apex_zero.errors import TrackReconstructionError
from apex_zero.track.frenet import RefLine, frenet_to_world, world_to_frenet

__all__ = [
    "LapPoints",
    "bin_medians",
    "periodic_smooth",
    "project_laps",
    "refine_line",
    "seed_line",
    "smooth_line",
]

FloatArray = NDArray[np.float64]

#: Fraction of a lap the data is tiled beyond each end for periodic fitting.
_PAD_FRACTION = 1.0 / 3.0


@dataclass(frozen=True)
class LapPoints:
    """Raw position samples of one lap, in time order.

    Attributes:
        lap_id: Unique id (e.g. ``"VER/11"``).
        t: ``(M,)`` session time [s].
        xyz: ``(M, 3)`` world position [m].
        t_start: Lap start time [s] (timing-line crossing).
        t_end: Lap end time [s].
    """

    lap_id: str
    t: FloatArray
    xyz: FloatArray
    t_start: float
    t_end: float


def periodic_smooth(
    s: FloatArray,
    values: FloatArray,
    length: float,
    ell_m: float,
    s_out: FloatArray,
    weights: FloatArray | None = None,
) -> FloatArray:
    """Periodic smoothing spline of samples ``values(s)``, evaluated at ``s_out``.

    Args:
        s: ``(K,)`` sample positions in ``[0, length)`` (any order; duplicates averaged).
        values: ``(K,)`` or ``(K, D)`` samples.
        length: Period [m].
        ell_m: Smoothing length scale [m] (see module docstring); 0 interpolates.
        s_out: ``(M,)`` evaluation points in ``[0, length)``.
        weights: ``(K,)`` optional sample weights (e.g. points per bin).

    Returns:
        ``(M,)`` or ``(M, D)`` smoothed values.
    """
    s = np.mod(np.asarray(s, dtype=np.float64), length)
    v = np.asarray(values, dtype=np.float64)
    order = np.argsort(s, kind="stable")
    s, v = s[order], v[order]
    w = np.ones(len(s)) if weights is None else np.asarray(weights, dtype=np.float64)[order]
    keep = np.concatenate([[True], np.diff(s) > 1e-9])
    if not keep.all():  # merge duplicate abscissae (weighted mean)
        groups = np.cumsum(keep) - 1
        wsum = np.bincount(groups, weights=w)
        vv = v.reshape(len(s), -1)
        vm = np.column_stack(
            [np.bincount(groups, weights=w * vv[:, k]) / wsum for k in range(vv.shape[1])]
        )
        s, w = s[keep], wsum
        v = vm[:, 0] if v.ndim == 1 else vm
    if len(s) < 8:
        raise TrackReconstructionError(f"too few samples ({len(s)}) for a periodic fit")
    pad = _PAD_FRACTION * length
    lo, hi = s >= length - pad, s < pad
    s_ext = np.concatenate([s[lo] - length, s, s[hi] + length])
    v_ext = np.concatenate([v[lo], v, v[hi]], axis=0)
    w_ext = np.concatenate([w[lo], w, w[hi]])
    h = length / len(s)
    lam = h * ell_m**4 if ell_m > 0 else 0.0
    spline = make_smoothing_spline(s_ext, v_ext, w=w_ext / w_ext.mean(), lam=lam)
    return np.asarray(spline(np.asarray(s_out, dtype=np.float64)), dtype=np.float64)


def smooth_line(line: RefLine, ell_xy_m: float, ell_z_m: float, ds: float = 1.0) -> RefLine:
    """Smooth a line's x, y (scale ``ell_xy_m``) and z (``ell_z_m``) and resample to ``ds``.

    The smoothed curve is evaluated densely (``ds / 4``) and resampled by arc length, so the
    new line's ``s`` is its own arc length.
    """
    s_dense = np.arange(0.0, line.length, ds / 4.0)
    xy = periodic_smooth(line.s, line.xyz[:, :2], line.length, ell_xy_m, s_dense)
    z = periodic_smooth(line.s, line.xyz[:, 2], line.length, ell_z_m, s_dense)
    return RefLine.resample(np.column_stack([xy, z]), ds)


def seed_line(lap: LapPoints, ds: float = 1.0, ell_m: float = 25.0) -> RefLine:
    """Closed seed line from one lap's samples within ``[t_start, t_end]``.

    Raises:
        TrackReconstructionError: Too few samples.
    """
    inside = (lap.t >= lap.t_start) & (lap.t <= lap.t_end)
    pts = lap.xyz[inside]
    if len(pts) < 50:
        raise TrackReconstructionError(f"seed lap {lap.lap_id} has only {len(pts)} samples")
    raw = RefLine.resample(pts, ds)
    return smooth_line(raw, ell_m, ell_m, ds)


def project_laps(
    line: RefLine,
    laps: Sequence[LapPoints],
    hints: Sequence[FloatArray] | None,
    window_m: float,
) -> list[tuple[FloatArray, FloatArray, FloatArray]]:
    """Project each lap's samples onto ``line`` keeping ``s`` continuous along the lap.

    Args:
        line: Current reference line.
        laps: Laps to project.
        hints: Per-lap ``(M,)`` approximate ``s`` (e.g. from the previous iteration), or
            ``None`` to derive hints from lap time: the fraction of the lap elapsed times the
            line length, which is good to a few hundred metres, then a second, tight pass.
        window_m: Search half-width around the hint [m].

    Returns:
        Per lap ``(s, n, h)`` arrays; invalid projections are NaN.
    """
    out: list[tuple[FloatArray, FloatArray, FloatArray]] = []
    for i, lap in enumerate(laps):
        if hints is not None:
            hint = hints[i]
        else:
            frac = (lap.t - lap.t_start) / max(lap.t_end - lap.t_start, 1e-9)
            coarse = world_to_frenet(line, lap.xyz, frac * line.length, window_m=400.0)
            hint = _continuous(coarse.s, line.length)
        f = world_to_frenet(line, lap.xyz, hint, window_m=window_m)
        s = np.where(f.valid, f.s, np.nan)
        out.append((s, np.where(f.valid, f.n, np.nan), np.where(f.valid, f.h, np.nan)))
    return out


def _continuous(s: FloatArray, length: float) -> FloatArray:
    """Replace jumps (wrong-branch projections) by a running median of the unwrapped ``s``."""
    unwrapped = np.unwrap(s * (2 * np.pi / length)) * (length / (2 * np.pi))
    k = 5
    if len(unwrapped) <= k:
        return s
    padded = np.pad(unwrapped, k // 2, mode="edge")
    med = np.median(np.lib.stride_tricks.sliding_window_view(padded, k), axis=1)
    return np.mod(med, length)


def bin_medians(
    s: FloatArray, values: FloatArray, length: float, bin_m: float, min_count: int = 3
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Median of ``values`` per ``bin_m`` bin of ``s``.

    Returns:
        ``(s_centre, median, count)`` for bins with at least ``min_count`` samples.
    """
    ok = np.isfinite(s) & np.isfinite(values)
    s, values = s[ok], values[ok]
    n_bins = max(int(round(length / bin_m)), 1)
    idx = np.minimum((s / (length / n_bins)).astype(np.int64), n_bins - 1)
    order = np.argsort(idx, kind="stable")
    idx, vals = idx[order], values[order]
    bounds = np.searchsorted(idx, np.arange(n_bins + 1))
    centres, meds, counts = [], [], []
    for b in range(n_bins):
        lo, hi = bounds[b], bounds[b + 1]
        if hi - lo >= min_count:
            centres.append((b + 0.5) * length / n_bins)
            meds.append(float(np.median(vals[lo:hi])))
            counts.append(hi - lo)
    return np.array(centres), np.array(meds), np.array(counts, dtype=np.float64)


def refine_line(
    line: RefLine,
    laps: Sequence[LapPoints],
    *,
    iterations: int,
    bin_m: float,
    ell_xy_m: float,
    ell_z_m: float,
    window_m: float = 50.0,
    ds: float = 1.0,
) -> tuple[RefLine, list[tuple[FloatArray, FloatArray, FloatArray]]]:
    """Iterative median refinement (see module docstring).

    Returns:
        The refined line and the final per-lap projections ``(s, n, h)`` onto it.

    Raises:
        TrackReconstructionError: Coverage too sparse to refine.
    """
    proj = project_laps(line, laps, None, window_m)
    for _ in range(iterations):
        s_all = np.concatenate([p[0] for p in proj])
        n_all = np.concatenate([p[1] for p in proj])
        h_all = np.concatenate([p[2] for p in proj])
        sc, n_med, cnt = bin_medians(s_all, n_all, line.length, bin_m)
        _, h_med, _ = bin_medians(s_all, h_all, line.length, bin_m)
        if len(sc) < 0.8 * line.length / bin_m:
            raise TrackReconstructionError(
                f"only {len(sc)} of {int(line.length / bin_m)} bins have enough samples"
            )
        n_line = periodic_smooth(sc, n_med, line.length, ell_xy_m, line.s, weights=cnt)
        h_line = periodic_smooth(sc, h_med, line.length, ell_z_m, line.s, weights=cnt)
        # Smooth the *offsets*, never x(s), y(s) themselves: smoothing coordinates pulls a
        # curve inwards by ~ell^4 / r^3 (module docstring), which matters in hairpins.
        moved = frenet_to_world(line, line.s, n_line, h_line)
        new = RefLine.resample(moved, ds)
        # carry each lap's s over to the new line as the hint for the next projection
        scale = new.length / line.length
        hints = [np.mod(np.nan_to_num(p[0], nan=0.0) * scale, new.length) for p in proj]
        hints = [_fill_nan_hint(h, p[0], new.length) for h, p in zip(hints, proj, strict=True)]
        line = new
        proj = project_laps(line, laps, hints, window_m)
    return line, proj


def _fill_nan_hint(hint: FloatArray, s_prev: FloatArray, length: float) -> FloatArray:
    """Hints for samples whose previous projection failed: interpolate neighbours' ``s``."""
    bad = ~np.isfinite(s_prev)
    if not bad.any() or bad.all():
        return hint
    idx = np.arange(len(hint))
    good = np.unwrap(hint[~bad] * (2 * np.pi / length)) * (length / (2 * np.pi))
    out = hint.copy()
    out[bad] = np.mod(np.interp(idx[bad], idx[~bad], good), length)
    return out
