"""Build a ``TrackAsset`` from processed sessions (P2.4, P2.8).

``build_track(config, sessions)`` loads every **accurate** lap (``IsAccurate``: no pit or out
laps) of the given sessions from the processed cache and calls :func:`reconstruct`:

1. Seed from the fastest accurate lap; iterative median refinement (``centerline.py``).
2. Boundaries from the lateral spread; the line moves to the middle of the band (``width.py``).
3. Start/finish (P2.4): every lap's position at its start time (interpolated between the two
   samples around the timing-line crossing) is projected onto the line; the circular mean of
   those ``s`` becomes ``s = 0``.
4. All samples are re-projected onto the final line; final boundaries (not re-centred),
   elevation, grade and banking from those projections (``elevation.py``).
5. Heading and curvature from the final line; validation report.

The report records the length error against ``official_length_m`` and, for each driver's
fastest accurate lap, the fraction of samples inside ``[-w_right, w_left]``.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from apex_zero import __version__
from apex_zero.config.paths import Paths, resolve_paths
from apex_zero.config.track import SessionRef, TrackConfig
from apex_zero.data import cache
from apex_zero.errors import TrackReconstructionError
from apex_zero.track.asset import ASSET_VERSION, JSONScalar, TrackAsset
from apex_zero.track.centerline import (
    LapPoints,
    periodic_smooth,
    project_laps,
    refine_line,
    seed_line,
)
from apex_zero.track.elevation import estimate_bank, fit_elevation
from apex_zero.track.frenet import RefLine, frenet_to_world, interp_periodic, world_to_frenet
from apex_zero.track.width import estimate_width
from apex_zero.utils.logging import get_logger

__all__ = [
    "build_track",
    "heading_curvature",
    "load_laps",
    "load_track_config",
    "reconstruct",
    "resolve_sessions",
    "shift_start",
    "track_dir",
]

log = get_logger(__name__)
FloatArray = NDArray[np.float64]


def load_track_config(path: Path) -> TrackConfig:
    """Parse ``configs/tracks/<name>.yaml``."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TrackConfig.model_validate(data)


def track_dir(paths: Paths, name: str) -> Path:
    """Directory of a built track: ``<APEX_PROCESSED_DIR>/tracks/<name>/``."""
    return paths.processed_dir / "tracks" / name


def resolve_sessions(paths: Paths, refs: Sequence[SessionRef]) -> list[cache.SessionKey]:
    """Processed-cache keys of the configured sessions (they must have been fetched).

    Raises:
        TrackReconstructionError: A session is not in the processed cache (or not intact).
    """
    keys = []
    for ref in refs:
        key = cache.alias_lookup(paths, ref.source, ref.year, ref.event, ref.session)
        if key is None:
            key = cache.SessionKey(
                ref.year, cache.slugify(ref.event), cache.normalize_session(ref.session)
            )
        meta = cache.read_meta(paths, key)
        if meta is None or meta.source != ref.source or not cache.is_complete(paths, key):
            raise TrackReconstructionError(
                f"{ref.year} {ref.event} {ref.session} ({ref.source}) is not in the processed "
                f"cache; run `apex data fetch --year {ref.year} --event {ref.event!r} "
                f"--session {ref.session} --source {ref.source}`"
            )
        keys.append(key)
    return keys


def load_laps(paths: Paths, keys: Sequence[cache.SessionKey]) -> tuple[list[LapPoints], list[str]]:
    """Accurate laps of the sessions as :class:`LapPoints`, plus the sessions' input hashes."""
    laps: list[LapPoints] = []
    hashes: list[str] = []
    for key in keys:
        meta = cache.read_meta(paths, key)
        assert meta is not None
        hashes.extend(f"{key}/{f.path}:{f.sha256}" for f in meta.files if f.path == "laps.parquet")
        table = pd.read_parquet(cache.laps_path(paths, key))
        acc = table[table["IsAccurate"].fillna(False).astype(bool)]
        for _, lap in acc.iterrows():
            f = cache.pos_path(paths, key, str(lap["Driver"]), int(lap["LapNumber"]))
            if not f.is_file():
                continue
            rec = next(
                r
                for r in meta.files
                if r.path == f.relative_to(cache.session_dir(paths, key)).as_posix()
            )
            hashes.append(f"{key}/{rec.path}:{rec.sha256}")
            pos = pd.read_parquet(f)
            laps.append(
                LapPoints(
                    lap_id=f"{key}/{lap['Driver']}/{int(lap['LapNumber'])}",
                    t=pos["SessionTime"].to_numpy(dtype=np.float64),
                    xyz=pos[["X", "Y", "Z"]].to_numpy(dtype=np.float64),
                    t_start=float(lap["LapStartTime"]),
                    t_end=float(lap["LapEndTime"]),
                )
            )
    return laps, hashes


def heading_curvature(line: RefLine, ell_kappa_m: float = 0.0) -> tuple[FloatArray, FloatArray]:
    """Heading ``psi`` [rad] and signed curvature ``kappa = dpsi/ds`` [1/m] (central diffs).

    ``kappa`` uses the wrapped heading change between neighbours, so it is continuous across
    the ±π wrap and the loop seam. ``ell_kappa_m > 0`` smooths it further.
    """
    xy = line.xyz[:, :2]
    t = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    psi = np.arctan2(t[:, 1], t[:, 0])
    dpsi = np.angle(np.exp(1j * (np.roll(psi, -1) - np.roll(psi, 1))))
    kappa = dpsi / (2 * line.ds)
    if ell_kappa_m > 0:
        kappa = periodic_smooth(line.s, kappa, line.length, ell_kappa_m, line.s)
    return psi, kappa


def shift_start(line: RefLine, s0: float) -> RefLine:
    """The same loop with ``s = 0`` moved to ``s0`` of ``line``."""
    xyz = interp_periodic(line.s, line.xyz, line.s + s0, line.length)
    return RefLine(s=line.s.copy(), xyz=xyz, length=line.length)


def _start_positions(laps: Sequence[LapPoints]) -> FloatArray:
    out = []
    for lap in laps:
        i = int(np.searchsorted(lap.t, lap.t_start))
        if i == 0 or i >= len(lap.t):
            continue
        w = (lap.t_start - lap.t[i - 1]) / (lap.t[i] - lap.t[i - 1])
        out.append(lap.xyz[i - 1] + w * (lap.xyz[i] - lap.xyz[i - 1]))
    return np.array(out)


def _circular_mean(s: FloatArray, length: float) -> float:
    ang = s * (2 * np.pi / length)
    return float(
        np.mod(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) * length / (2 * np.pi), length)
    )


def reconstruct(
    config: TrackConfig,
    laps: Sequence[LapPoints],
    source_sessions: Sequence[str],
    input_hashes: Sequence[str] = (),
) -> TrackAsset:
    """Reconstruct a track from accurate laps (see module docstring).

    Raises:
        TrackReconstructionError: Too few laps or samples, or a degenerate result.
    """
    t0 = time.perf_counter()
    if len(laps) < 3:
        raise TrackReconstructionError(f"need at least 3 accurate laps, got {len(laps)}")
    sm = config.smoothing
    fastest = min(laps, key=lambda lap: lap.t_end - lap.t_start)
    line = seed_line(fastest, ell_m=sm.seed)
    line, proj = refine_line(
        line,
        laps,
        iterations=config.refine_iterations,
        bin_m=config.bin_m,
        ell_xy_m=sm.xy,
        ell_z_m=sm.z,
        window_m=config.window_m,
    )

    s_all = np.concatenate([p[0] for p in proj])
    n_all = np.concatenate([p[1] for p in proj])
    prior = config.width_prior_m
    if config.recentre:
        band = estimate_width(
            line.s,
            line.length,
            s_all,
            n_all,
            bin_m=config.width_bin_m,
            margin_m=config.width_margin_m,
            w_min=prior.w_min,
            w_max=prior.w_max,
            ell_m=sm.xy,
        )
        line = RefLine.resample(frenet_to_world(line, line.s, band.centre_n), line.ds)

    # P2.4: s = 0 at the timing line
    starts = _start_positions(laps)
    if len(starts) < 3:
        raise TrackReconstructionError("too few laps with samples around the start crossing")
    s_start = world_to_frenet(line, starts).s
    s0 = _circular_mean(s_start, line.length)
    line = shift_start(line, s0)

    hints = [np.mod(np.nan_to_num(p[0], nan=0.0) - s0, line.length) for p in proj]
    proj = project_laps(line, laps, hints, config.window_m)
    s_all = np.concatenate([p[0] for p in proj])
    n_all = np.concatenate([p[1] for p in proj])
    h_all = np.concatenate([p[2] for p in proj])

    width = estimate_width(
        line.s,
        line.length,
        s_all,
        n_all,
        bin_m=config.width_bin_m,
        margin_m=config.width_margin_m,
        w_min=prior.w_min,
        w_max=prior.w_max,
        ell_m=sm.width,
        recentre=False,
    )
    spread = float(np.median(width.bin_hi - width.bin_lo))
    width_source = "observed"
    w_left, w_right = width.w_left, width.w_right
    if spread < config.min_observed_spread_m:
        if config.nominal_width_m is None:
            raise TrackReconstructionError(
                f"median lateral spread of the cars is {spread:.2f} m (< "
                f"{config.min_observed_spread_m} m): the positions look map-matched to one "
                "reference line, so the track width is not observable. Set nominal_width_m "
                "in the track config (it is recorded as an assumption)."
            )
        w_left = w_right = np.full(len(line.s), config.nominal_width_m / 2)
        width_source = "config_nominal"
        log.warning(
            "width_not_observable", lateral_spread_m=spread, nominal_width_m=config.nominal_width_m
        )
    elev = fit_elevation(
        line.s, line.xyz[:, 2], line.length, s_all, h_all, bin_m=config.bin_m, ell_z_m=sm.z
    )
    bank_source = "disabled"
    if config.estimate_bank and spread < config.min_observed_spread_m:
        bank = np.zeros(len(line.s))
        bank_source = "not_observable"
    elif config.estimate_bank:
        bank_source = "observed"
        h_resid = h_all - (
            interp_periodic(line.s, elev.z, s_all, line.length)
            - interp_periodic(line.s, line.xyz[:, 2], s_all, line.length)
        )
        bank = estimate_bank(line.s, line.length, s_all, n_all, h_resid, ell_m=sm.bank)
    else:
        bank = np.zeros(len(line.s))
    psi, kappa = heading_curvature(line, sm.kappa)

    # validation: each driver's fastest lap inside the boundaries
    best: dict[str, int] = {}
    for i, lap in enumerate(laps):
        drv = lap.lap_id.rsplit("/", 1)[0]
        if (
            drv not in best
            or lap.t_end - lap.t_start < laps[best[drv]].t_end - laps[best[drv]].t_start
        ):
            best[drv] = i
    inside, total = 0, 0
    for i in best.values():
        s_l, n_l, _ = proj[i]
        lap = laps[i]
        sel = (lap.t >= lap.t_start) & (lap.t <= lap.t_end) & np.isfinite(s_l)
        wl = interp_periodic(line.s, w_left, s_l[sel], line.length)
        wr = interp_periodic(line.s, w_right, s_l[sel], line.length)
        inside += int(np.sum((n_l[sel] <= wl) & (n_l[sel] >= -wr)))
        total += int(sel.sum())
    valid_frac = float(np.mean(np.isfinite(s_all)))
    params = config.model_dump(mode="json", by_alias=True)
    build_hash = hashlib.sha256(
        json.dumps(
            {
                "asset_version": ASSET_VERSION,
                "apex_zero": __version__,
                "config": params,
                "inputs": sorted(input_hashes),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    report: dict[str, JSONScalar] = {
        "length_m": line.length,
        "official_length_m": config.official_length_m,
        "length_error_pct": 100 * (line.length / config.official_length_m - 1),
        "fastest_laps_inside_frac": inside / total if total else float("nan"),
        "fastest_laps_samples": total,
        "n_laps": len(laps),
        "n_samples": int(len(s_all)),
        "projected_frac": valid_frac,
        "seed_lap": fastest.lap_id,
        "start_offset_spread_m": float(np.std(np.mod(s_start - s0 + line.length / 2, line.length))),
        "max_abs_kappa": float(np.abs(kappa).max()),
        "z_range_m": float(elev.z.max() - elev.z.min()),
        "max_abs_bank_deg": float(np.rad2deg(np.abs(bank).max())),
        "lateral_spread_median_m": spread,
        "width_source": width_source,
        "bank_source": bank_source,
        "build_seconds": time.perf_counter() - t0,
    }
    log.info(
        "track_reconstructed",
        name=config.name,
        **{k: v for k, v in report.items() if isinstance(v, float | int)},
    )
    return TrackAsset(
        name=config.name,
        length=line.length,
        s=line.s,
        x=line.xyz[:, 0],
        y=line.xyz[:, 1],
        z=elev.z,
        psi=psi,
        kappa=kappa,
        grade=elev.grade,
        bank=bank,
        w_left=w_left,
        w_right=w_right,
        source_sessions=tuple(source_sessions),
        build_hash=build_hash,
        params=params,
        report=report,
    )


def build_track(
    config: TrackConfig,
    sessions: Sequence[cache.SessionKey] | None = None,
    paths: Paths | None = None,
) -> TrackAsset:
    """Build a track from processed sessions (default: the config's sessions)."""
    paths = resolve_paths() if paths is None else paths
    keys = list(sessions) if sessions is not None else resolve_sessions(paths, config.sessions.all)
    laps, hashes = load_laps(paths, keys)
    return reconstruct(config, laps, [str(k) for k in keys], hashes)
