"""Inspection plots of a built track (P2.9): ``apex track inspect``.

Writes ``<out>/<name>_{xy,curvature,width,elevation}.png``; with raw samples (``--debug``) also
``<name>_debug_raw.png`` (samples over the fitted line, zoomed on the tightest corner) and
``<name>_debug_lateral.png`` (histogram of every sample's lateral offset). Uses the
non-interactive Agg backend.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from apex_zero.track.asset import TrackAsset  # noqa: E402

__all__ = ["plot_debug", "plot_track"]


def _title(asset: TrackAsset, what: str) -> str:
    return f"{asset.name}: {what} (L = {asset.length:.1f} m, build {asset.build_hash[:10]})"


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_track(asset: TrackAsset, out_dir: Path) -> list[Path]:
    """The four standard inspection plots. Returns the written paths."""
    paths = []
    nrm = asset.line.normals
    xy = asset.xyz[:, :2]
    left = xy + nrm * asset.w_left[:, None]
    right = xy - nrm * asset.w_right[:, None]
    width_src = asset.report.get("width_source", "?")

    fig, ax = plt.subplots(figsize=(9, 7))
    for edge in (left, right):
        ax.plot(*np.vstack([edge, edge[:1]]).T, color="0.4", lw=0.8)
    ax.plot(*np.vstack([xy, xy[:1]]).T, color="tab:orange", lw=1.0, label="centerline")
    ax.plot(*xy[0], "o", color="tab:red", label="s = 0 (timing line)")
    k = len(xy) // 40
    ax.annotate("", xy=xy[k], xytext=xy[0], arrowprops={"arrowstyle": "->", "color": "tab:red"})
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="best")
    ax.set_title(_title(asset, f"layout, width source: {width_src}"))
    paths.append(_save(fig, out_dir / f"{asset.name}_xy.png"))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(asset.s, asset.kappa, lw=0.8)
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel("s [m]")
    ax.set_ylabel("curvature [1/m] (> 0 left)")
    ax.set_title(_title(asset, "curvature"))
    paths.append(_save(fig, out_dir / f"{asset.name}_curvature.png"))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(asset.s, asset.w_left, label="w_left")
    ax.plot(asset.s, -asset.w_right, label="-w_right")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel("s [m]")
    ax.set_ylabel("boundary offset [m]")
    ax.legend()
    note = " (ASSUMED: not observable in the data)" if width_src == "config_nominal" else ""
    ax.set_title(_title(asset, f"width{note}"))
    paths.append(_save(fig, out_dir / f"{asset.name}_width.png"))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    a1.plot(asset.s, asset.z)
    a1.set_ylabel("z [m]")
    a1.set_title(_title(asset, "elevation"))
    a2.plot(asset.s, 100 * asset.grade, color="tab:green")
    a2.axhline(0, color="0.6", lw=0.5)
    a2.set_ylabel("grade [%]")
    a2.set_xlabel("s [m]")
    paths.append(_save(fig, out_dir / f"{asset.name}_elevation.png"))
    return paths


def plot_debug(
    asset: TrackAsset, points: NDArray[np.float64], n: NDArray[np.float64], out_dir: Path
) -> list[Path]:
    """Raw samples over the fitted line, and the lateral-offset histogram.

    Args:
        asset: Built track.
        points: ``(M, 3)`` raw samples [m].
        n: ``(M,)`` their lateral offsets from the centerline [m].
        out_dir: Output directory.
    """
    paths = []
    i = int(np.argmax(np.abs(asset.kappa)))
    cx, cy = asset.x[i], asset.y[i]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, half in zip(axes, (None, 80.0), strict=True):
        sel = (
            slice(None)
            if half is None
            else ((np.abs(points[:, 0] - cx) < half) & (np.abs(points[:, 1] - cy) < half))
        )
        ax.scatter(
            points[sel, 0], points[sel, 1], s=1 if half is None else 4, c="0.5", label="raw samples"
        )
        ax.plot(
            *np.vstack([asset.xyz[:, :2], asset.xyz[:1, :2]]).T,
            color="tab:orange",
            lw=1.0,
            label="fitted centerline",
        )
        if half is not None:
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
            ax.set_title(
                f"tightest corner (s = {asset.s[i]:.0f} m, R = {1 / abs(asset.kappa[i]):.1f} m)"
            )
        else:
            ax.set_title(_title(asset, f"{len(points)} raw samples"))
        ax.set_aspect("equal")
        ax.legend(loc="best")
    paths.append(_save(fig, out_dir / f"{asset.name}_debug_raw.png"))

    fig, ax = plt.subplots(figsize=(8, 4))
    ok = np.isfinite(n)
    ax.hist(n[ok], bins=np.linspace(-6, 6, 121).tolist(), color="tab:blue")
    ax.set_xlabel("lateral offset n of raw samples [m]")
    ax.set_ylabel("count")
    p1, p99 = np.percentile(n[ok], [1, 99])
    ax.set_title(f"{asset.name}: lateral spread p1 = {p1:.2f} m, p99 = {p99:.2f} m")
    paths.append(_save(fig, out_dir / f"{asset.name}_debug_lateral.png"))
    return paths
