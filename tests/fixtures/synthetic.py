"""Synthetic sessions with known ground truth (P1.7).

Two closed tracks with elevation:

- ``ellipse_track``: an ellipse with a gentle sinusoidal elevation profile.
- ``figure_eight_track``: a lemniscate of Gerono whose two branches cross at different heights
  (a bridge, like Suzuka), so 2D nearest-point projection is ambiguous at the crossing.

``make_session`` drives several "drivers" round a track for a few laps, each with its own
lateral-offset profile and a curvature-dependent speed profile, and samples position and car
data at ~4 Hz with timestamp jitter and position noise, like the real feed. The output has the
shape of a loaded FastF1 session (``laps`` frame with timedeltas, ``pos_data``/``car_data`` dicts
keyed by car number, raw position units of 1/10 m, speed in km/h), so it can be fed to
``apex_zero.data.fastf1_loader`` unchanged.

Every function takes an explicit ``np.random.Generator``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from apex_zero.utils.units import FASTF1_POS_TO_M

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SyntheticTrack:
    """Ground-truth centerline resampled every ``ds`` metres (periodic, ``s`` in [0, L))."""

    name: str
    s: FloatArray  # (N,) arc length [m]
    xyz: FloatArray  # (N, 3) world position [m]
    length_m: float
    half_width_m: float

    def at(self, s: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Centerline position ``(M, 3)`` and unit left-normal ``(M, 2)`` at arc lengths ``s``."""
        s = np.mod(s, self.length_m)
        s_ext = np.append(self.s, self.length_m)
        pts = np.vstack([self.xyz, self.xyz[:1]])
        xyz = np.column_stack([np.interp(s, s_ext, pts[:, k]) for k in range(3)])
        d = np.gradient(pts[:, :2], s_ext, axis=0)
        tx = np.interp(s, s_ext, d[:, 0])
        ty = np.interp(s, s_ext, d[:, 1])
        norm = np.hypot(tx, ty)
        normal = np.column_stack([-ty / norm, tx / norm])
        return xyz, normal

    def curvature(self, s: FloatArray) -> FloatArray:
        """Unsigned planar curvature [1/m] at ``s`` (finite differences)."""
        pts = self.xyz[:, :2]
        d1 = np.gradient(pts, self.s, axis=0)
        d2 = np.gradient(d1, self.s, axis=0)
        k = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.hypot(*d1.T) ** 3
        return np.interp(np.mod(s, self.length_m), self.s, k)


def _resample(name: str, xyz_dense: FloatArray, half_width_m: float, ds: float) -> SyntheticTrack:
    closed = np.vstack([xyz_dense, xyz_dense[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s_dense = np.concatenate([[0.0], np.cumsum(seg)])
    length = float(s_dense[-1])
    s = np.arange(0.0, length, ds)
    xyz = np.column_stack([np.interp(s, s_dense, closed[:, k]) for k in range(3)])
    return SyntheticTrack(name, s, xyz, length, half_width_m)


def ellipse_track(
    a_m: float = 600.0, b_m: float = 300.0, z_amp_m: float = 6.0, ds: float = 1.0
) -> SyntheticTrack:
    """Ellipse with semi-axes ``a_m``, ``b_m`` and elevation ``z = z_amp sin(2t)``."""
    t = np.linspace(0.0, 2 * np.pi, 20000, endpoint=False)
    xyz = np.column_stack([a_m * np.cos(t), b_m * np.sin(t), z_amp_m * np.sin(2 * t)])
    return _resample("synthetic_ellipse", xyz, 6.0, ds)


def figure_eight_track(
    scale_m: float = 500.0, bridge_m: float = 8.0, ds: float = 1.0
) -> SyntheticTrack:
    """Lemniscate of Gerono; the branches cross at the origin ``2 * bridge_m`` apart in z."""
    t = np.linspace(0.0, 2 * np.pi, 20000, endpoint=False)
    xyz = np.column_stack(
        [scale_m * np.sin(t), scale_m * np.sin(t) * np.cos(t), bridge_m * np.cos(t)]
    )
    return _resample("synthetic_figure8", xyz, 6.0, ds)


@dataclass
class FakeSession:
    """Implements ``apex_zero.data.fastf1_loader.FastF1Session`` for tests."""

    name: str
    event: pd.Series
    laps: pd.DataFrame
    pos_data: dict[str, pd.DataFrame]
    car_data: dict[str, pd.DataFrame]
    track: SyntheticTrack = field(repr=False)
    #: Ground-truth lap times [s] per driver code, in lap order.
    lap_times_s: dict[str, list[float]] = field(default_factory=dict)

    def load(self, **_: object) -> None:
        """No-op (data is already 'loaded')."""


def _sample_times(
    t_start: float, t_end: float, rate_hz: float, rng: np.random.Generator
) -> FloatArray:
    n = int((t_end - t_start) * rate_hz * 1.2) + 2
    dt = (1.0 / rate_hz) * (1.0 + rng.uniform(-0.15, 0.15, n))
    t = t_start + rng.uniform(0, 1.0 / rate_hz) + np.cumsum(dt)
    return t[t < t_end]


def make_session(
    track: SyntheticTrack,
    rng: np.random.Generator,
    drivers: tuple[str, ...] = ("AAA", "BBB", "CCC"),
    n_laps: int = 3,
    rate_hz: float = 4.0,
    pos_noise_m: float = 0.3,
    v_max_mps: float = 80.0,
    t0_s: float = 600.0,
    session_name: str = "Qualifying",
    lateral_frac: tuple[float, float] = (0.3, 0.8),
) -> FakeSession:
    """Simulate ``n_laps`` consecutive flying laps per driver.

    Args:
        track: Ground-truth track.
        rng: Random generator (offsets, noise, timestamp jitter).
        drivers: Three-letter codes; car numbers are 1, 2, 3, ...
        n_laps: Laps per driver. Lap 1 is flagged inaccurate (as an out-lap would be).
        rate_hz: Mean sample rate of position and car data.
        pos_noise_m: Gaussian position noise per axis [m].
        v_max_mps: Speed on straights; corners slow down with curvature.
        t0_s: Session time of the first driver's lap-1 start.
        session_name: FastF1 session name.
        lateral_frac: Range of each driver's lateral-offset amplitude as a fraction of the
            half-width. ``(0, 0)`` makes every car follow the centerline exactly, like
            map-matched feed positions.
    """
    ds = 0.5
    s_grid = np.arange(0.0, track.length_m * n_laps + ds, ds)
    v = v_max_mps / (1.0 + 60.0 * track.curvature(s_grid))
    t_grid = np.concatenate([[0.0], np.cumsum(ds / v[:-1])])

    laps_rows: list[dict[str, object]] = []
    pos_data: dict[str, pd.DataFrame] = {}
    car_data: dict[str, pd.DataFrame] = {}
    lap_times: dict[str, list[float]] = {}
    for i, drv in enumerate(drivers):
        num = str(i + 1)
        start = t0_s + 90.0 * i
        amp = rng.uniform(*lateral_frac) * track.half_width_m
        phase = rng.uniform(0, 2 * np.pi)
        k = int(rng.integers(2, 6))

        # time -> s along the (multi-lap) run
        def s_of(t: FloatArray, start: float = start) -> FloatArray:
            return np.interp(t - start, t_grid, s_grid)

        t_end = start + float(t_grid[-1])
        tp = _sample_times(start - 2.0, t_end + 2.0, rate_hz, rng)
        sp = s_of(tp)
        center, normal = track.at(sp)
        n_off = amp * np.sin(2 * np.pi * k * sp / track.length_m + phase)
        xyz = center.copy()
        xyz[:, :2] += normal * n_off[:, None]
        xyz += rng.normal(0.0, pos_noise_m, xyz.shape)
        pos_data[num] = pd.DataFrame(
            {
                "Date": pd.Timestamp("2024-01-01") + pd.to_timedelta(tp, unit="s"),
                "Status": "OnTrack",
                "X": np.round(xyz[:, 0] / FASTF1_POS_TO_M),
                "Y": np.round(xyz[:, 1] / FASTF1_POS_TO_M),
                "Z": np.round(xyz[:, 2] / FASTF1_POS_TO_M),
                "Source": "pos",
                "SessionTime": pd.to_timedelta(tp, unit="s"),
            }
        )

        tc = _sample_times(start - 2.0, t_end + 2.0, rate_hz, rng)
        vc = np.interp(np.clip(tc - start, 0, t_grid[-1]), t_grid, v)
        accel = np.gradient(vc, tc)
        car_data[num] = pd.DataFrame(
            {
                "Date": pd.Timestamp("2024-01-01") + pd.to_timedelta(tc, unit="s"),
                "RPM": 8000.0 + 50.0 * vc,
                "Speed": np.round(vc * 3.6),
                "nGear": np.clip((vc / 12).astype(int) + 1, 1, 8),
                "Throttle": np.where(accel >= 0, 100.0, 0.0),
                "Brake": accel < -2.0,
                "DRS": 8,
                "Source": "car",
                "SessionTime": pd.to_timedelta(tc, unit="s"),
            }
        )

        bounds = [float(np.interp(track.length_m * j, s_grid, t_grid)) for j in range(n_laps + 1)]
        lap_times[drv] = [b - a for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
        for j in range(n_laps):
            lap_start, lap_end = start + bounds[j], start + bounds[j + 1]
            lt = lap_end - lap_start
            laps_rows.append(
                {
                    "Time": pd.Timedelta(lap_end, unit="s"),
                    "Driver": drv,
                    "DriverNumber": num,
                    "LapTime": pd.Timedelta(lt, unit="s"),
                    "LapNumber": float(j + 1),
                    "Stint": 1.0,
                    "PitOutTime": pd.NaT,
                    "PitInTime": pd.NaT,
                    "Sector1Time": pd.Timedelta(lt * 0.3, unit="s"),
                    "Sector2Time": pd.Timedelta(lt * 0.4, unit="s"),
                    "Sector3Time": pd.Timedelta(lt * 0.3, unit="s"),
                    "Compound": "SOFT",
                    "TyreLife": float(j + 1),
                    "LapStartTime": pd.Timedelta(lap_start, unit="s"),
                    "IsAccurate": j > 0,
                    "Deleted": False,
                }
            )

    event = pd.Series(
        {
            "EventName": f"{track.name.replace('_', ' ').title()} Grand Prix",
            "RoundNumber": 99,
            "Country": "Testland",
            "Location": "Testville",
        }
    )
    return FakeSession(
        name=session_name,
        event=event,
        laps=pd.DataFrame(laps_rows),
        pos_data=pos_data,
        car_data=car_data,
        track=track,
        lap_times_s=lap_times,
    )
