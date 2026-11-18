"""Frenet frame of a closed reference line: ``s, n`` <-> world ``x, y, z`` (P2.2).

The reference line is a closed polyline sampled on a uniform ``s`` grid (1 m in a
``TrackAsset``). Its Frenet frame is defined so that the two conversions are exact inverses:

- Position: ``c(s)`` is linear between grid points.
- Lateral axis: ``N(s)`` is the unit left-normal (horizontal) at each grid point, linearly
  interpolated between grid points and renormalised. ``n > 0`` is to the left of travel.
- Height: ``h`` is the vertical offset from the line, ``z = z_line(s) + h``.

``frenet_to_world(s, n, h) = (c_xy(s) + n N(s), z_line(s) + h)``. ``world_to_frenet`` inverts it
by solving, on each candidate segment, the quadratic ``cross(p - c(t), N(t)) = 0`` for the
segment parameter ``t`` in ``[0, 1]``. The inverse is unique while ``|n|`` is smaller than the
local turning radius.

Self-intersecting layouts (Suzuka's figure-eight): pass ``s_hint`` (e.g. the previous sample's
``s`` along a lap) so only segments within ``window_m`` of it are considered. Without a hint the
nearest grid point in **3D** seeds the search, which separates a bridge crossing only if the two
levels differ in z by more than the lateral search distance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

__all__ = [
    "FrenetPoints",
    "RefLine",
    "frenet_to_world",
    "interp_angle",
    "interp_periodic",
    "wrap_s",
    "world_to_frenet",
]

FloatArray = NDArray[np.float64]

#: Half-width [m] of the segment search around the 3D-nearest grid point when no hint is given.
_NO_HINT_WINDOW_M = 15.0


def wrap_s(s: FloatArray | float, length: float) -> FloatArray:
    """Wrap arc length into ``[0, length)``."""
    return np.mod(np.asarray(s, dtype=np.float64), length)


def interp_periodic(
    s_grid: FloatArray, values: FloatArray, s: FloatArray, length: float
) -> FloatArray:
    """Linear interpolation of a periodic channel.

    Args:
        s_grid: ``(N,)`` increasing grid in ``[0, length)``.
        values: ``(N,)`` or ``(N, D)`` samples on the grid.
        s: ``(M,)`` query arc lengths (any real value; wrapped).
        length: Period [m].

    Returns:
        ``(M,)`` or ``(M, D)`` interpolated values.
    """
    sq = wrap_s(s, length)
    s_ext = np.append(s_grid, length)
    v = np.asarray(values, dtype=np.float64)
    v_ext = np.concatenate([v, v[:1]], axis=0)
    if v.ndim == 1:
        out: FloatArray = np.interp(sq, s_ext, v_ext)
        return out
    return np.column_stack([np.interp(sq, s_ext, v_ext[:, k]) for k in range(v.shape[1])])


def interp_angle(s_grid: FloatArray, angle: FloatArray, s: FloatArray, length: float) -> FloatArray:
    """Interpolate a periodic angle channel [rad] without jumps at the ±π wrap."""
    c = interp_periodic(s_grid, np.cos(angle), s, length)
    sn = interp_periodic(s_grid, np.sin(angle), s, length)
    return np.arctan2(sn, c)


@dataclass(frozen=True)
class RefLine:
    """Closed reference line on a uniform grid.

    Attributes:
        s: ``(N,)`` arc length [m], ``s[k] = k * ds``.
        xyz: ``(N, 3)`` world position [m]; the segment from the last point closes the loop.
        length: Loop length [m] (``N * ds``).
    """

    s: FloatArray
    xyz: FloatArray
    length: float

    def __post_init__(self) -> None:
        if self.xyz.shape != (len(self.s), 3):
            raise ValueError("xyz must be (N, 3) with N == len(s)")
        if len(self.s) < 4:
            raise ValueError("a reference line needs at least 4 points")
        expected = np.arange(len(self.s)) * (self.length / len(self.s))
        if not np.allclose(self.s, expected, rtol=0.0, atol=1e-6 * max(self.length, 1.0)):
            raise ValueError("s must be the uniform grid k * length / N (use RefLine.resample)")

    @classmethod
    def resample(cls, xyz: FloatArray, ds: float = 1.0) -> RefLine:
        """Uniform line from a closed polyline (the last point connects back to the first).

        Args:
            xyz: ``(M, 3)`` polyline vertices [m], not repeating the first point at the end.
            ds: Target spacing [m]; the actual spacing is ``length / round(length / ds)``.

        Returns:
            A :class:`RefLine` whose length is the polyline's (closed) length.
        """
        pts = np.asarray(xyz, dtype=np.float64)
        closed = np.vstack([pts, pts[:1]])
        seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        s_poly = np.concatenate([[0.0], np.cumsum(seg)])
        length = float(s_poly[-1])
        n = max(int(round(length / ds)), 4)
        s = np.arange(n, dtype=np.float64) * (length / n)
        out = np.column_stack([np.interp(s, s_poly, closed[:, k]) for k in range(3)])
        return cls(s=s, xyz=out, length=length)

    @property
    def ds(self) -> float:
        """Grid spacing [m]."""
        return self.length / len(self.s)

    @property
    def normals(self) -> FloatArray:
        """``(N, 2)`` unit left-normals at the grid points (central-difference tangents)."""
        xy = self.xyz[:, :2]
        t = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        return np.column_stack([-t[:, 1], t[:, 0]])

    def kdtree(self) -> cKDTree:
        """3D KD-tree over the grid points (for hint-free projection)."""
        return cKDTree(self.xyz)


@dataclass(frozen=True)
class FrenetPoints:
    """Result of :func:`world_to_frenet`.

    Attributes:
        s: ``(M,)`` arc length in ``[0, length)`` [m].
        n: ``(M,)`` lateral offset, positive to the left [m].
        h: ``(M,)`` height above the line (``z - z_line(s)``) [m]; zeros for 2D input.
        valid: ``(M,)`` False where no segment in the search window admitted a solution.
    """

    s: FloatArray
    n: FloatArray
    h: FloatArray
    valid: NDArray[np.bool_]


def frenet_to_world(
    line: RefLine, s: FloatArray, n: FloatArray, h: FloatArray | None = None
) -> FloatArray:
    """Frenet ``(s, n, h)`` -> world ``(M, 3)``.

    Args:
        line: Reference line.
        s: ``(M,)`` arc length [m] (wrapped).
        n: ``(M,)`` lateral offset [m], positive left.
        h: ``(M,)`` height offset [m]; default 0.
    """
    s = np.asarray(s, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    c = interp_periodic(line.s, line.xyz, s, line.length)
    nrm = interp_periodic(line.s, line.normals, s, line.length)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    out = c.copy()
    out[:, :2] += nrm * n[:, None]
    if h is not None:
        out[:, 2] += np.asarray(h, dtype=np.float64)
    return out


def _cross(ax: FloatArray, ay: FloatArray, bx: FloatArray, by: FloatArray) -> FloatArray:
    return ax * by - ay * bx


#: Points per vectorised block (bounds memory to roughly ``_CHUNK * window`` floats per array).
_CHUNK = 2048


def world_to_frenet(
    line: RefLine,
    points: FloatArray,
    s_hint: FloatArray | None = None,
    window_m: float = 50.0,
) -> FrenetPoints:
    """World points -> Frenet coordinates, vectorised.

    Args:
        line: Reference line.
        points: ``(M, 2)`` or ``(M, 3)`` world positions [m].
        s_hint: ``(M,)`` approximate arc length of each point; only segments within
            ``window_m`` of it are searched. ``None``: seed from the nearest grid point in 3D
            (2D input: in 2D) and search ±15 m around it.
        window_m: Half-width of the hinted search window [m].

    Returns:
        :class:`FrenetPoints`. Among valid solutions the one with the smallest ``|n|`` wins.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] not in (2, 3):
        raise ValueError("points must be (M, 2) or (M, 3)")
    if s_hint is not None and np.shape(s_hint) != (len(p),):
        raise ValueError("s_hint must be (M,)")
    if len(p) > _CHUNK:
        parts = [
            world_to_frenet(
                line,
                p[i : i + _CHUNK],
                None if s_hint is None else np.asarray(s_hint)[i : i + _CHUNK],
                window_m,
            )
            for i in range(0, len(p), _CHUNK)
        ]
        return FrenetPoints(
            s=np.concatenate([q.s for q in parts]),
            n=np.concatenate([q.n for q in parts]),
            h=np.concatenate([q.h for q in parts]),
            valid=np.concatenate([q.valid for q in parts]),
        )
    m = len(p)
    n_grid = len(line.s)
    ds = line.ds
    if s_hint is None:
        if p.shape[1] == 3:
            _, idx0 = line.kdtree().query(p)
        else:
            _, idx0 = cKDTree(line.xyz[:, :2]).query(p)
        centre = np.asarray(idx0, dtype=np.int64)
        half = int(np.ceil(_NO_HINT_WINDOW_M / ds))
    else:
        hint = np.asarray(s_hint, dtype=np.float64)
        centre = np.floor(wrap_s(hint, line.length) / ds).astype(np.int64)
        half = int(np.ceil(window_m / ds))

    offsets = np.arange(-half, half + 1)
    seg = np.mod(centre[:, None] + offsets[None, :], n_grid)  # (M, K) segment start indices
    nxt = np.mod(seg + 1, n_grid)
    xy = line.xyz[:, :2]
    nrm = line.normals
    c0, c1 = xy[seg], xy[nxt]  # (M, K, 2)
    n0, n1 = nrm[seg], nrm[nxt]
    d = c1 - c0
    dn = n1 - n0
    a = p[:, None, :2] - c0
    # cross(a - t d, n0 + t dn) = A + B t + C t^2
    big_a = _cross(a[..., 0], a[..., 1], n0[..., 0], n0[..., 1])
    big_b = _cross(a[..., 0], a[..., 1], dn[..., 0], dn[..., 1]) - _cross(
        d[..., 0], d[..., 1], n0[..., 0], n0[..., 1]
    )
    big_c = -_cross(d[..., 0], d[..., 1], dn[..., 0], dn[..., 1])

    t = np.full(big_a.shape + (2,), np.nan)
    lin = np.abs(big_c) < 1e-12 * np.maximum(np.abs(big_b), 1e-30)
    with np.errstate(divide="ignore", invalid="ignore"):
        t[..., 0] = np.where(lin, -big_a / big_b, np.nan)
        disc = big_b**2 - 4 * big_a * big_c
        sq = np.sqrt(np.where(disc >= 0, disc, np.nan))
        q = -0.5 * (big_b + np.copysign(sq, big_b))
        r1 = q / big_c
        r2 = big_a / q
        t[..., 0] = np.where(lin, t[..., 0], r1)
        t[..., 1] = np.where(lin, np.nan, r2)
    eps = 1e-9
    ok = (t >= -eps) & (t <= 1 + eps)
    t = np.clip(np.where(ok, t, np.nan), 0.0, 1.0)

    foot = c0[..., None, :] + t[..., None] * d[..., None, :]  # (M, K, 2, 2)
    nt = n0[..., None, :] + t[..., None] * dn[..., None, :]
    nt /= np.linalg.norm(nt, axis=-1, keepdims=True)
    off = p[:, None, None, :2] - foot
    n_val = np.sum(off * nt, axis=-1)  # (M, K, 2)
    score = np.where(ok, np.abs(n_val), np.inf).reshape(m, -1)
    best = np.argmin(score, axis=1)
    valid = np.isfinite(score[np.arange(m), best])

    k_idx, root = np.divmod(best, 2)
    rows = np.arange(m)
    t_best = np.where(valid, t[rows, k_idx, root], 0.0)
    seg_best = seg[rows, k_idx]
    s_out = wrap_s((seg_best + t_best) * ds, line.length)
    n_out = np.where(valid, n_val[rows, k_idx, root], np.nan)
    if p.shape[1] == 3:
        z_line = interp_periodic(line.s, line.xyz[:, 2], s_out, line.length)
        h_out = p[:, 2] - z_line
    else:
        h_out = np.zeros(m)
    return FrenetPoints(s=s_out, n=n_out, h=h_out, valid=valid)
