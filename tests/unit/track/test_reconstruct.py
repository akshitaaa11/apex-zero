"""P2.11 synthetic ground-truth tests, plus the orchestrator and config."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apex_zero.config.paths import Paths
from apex_zero.config.track import TrackConfig
from apex_zero.data import cache
from apex_zero.data.fastf1_loader import _handle_from_session, export_processed
from apex_zero.errors import TrackReconstructionError
from apex_zero.track.asset import TrackAsset
from apex_zero.track.centerline import LapPoints, project_laps
from apex_zero.track.frenet import RefLine, world_to_frenet
from apex_zero.track.reconstruct import (
    build_track,
    heading_curvature,
    load_track_config,
    reconstruct,
    resolve_sessions,
    shift_start,
    track_dir,
)
from tests.fixtures.synthetic import SyntheticTrack, make_session
from tests.unit.track.conftest import DRIVERS, laps_of, synthetic_config

REPO = Path(__file__).resolve().parents[3]


def circ(a: np.ndarray, length: float) -> np.ndarray:
    return np.mod(a + length / 2, length) - length / 2


@pytest.fixture(scope="module")
def ellipse_asset(ellipse: SyntheticTrack, ellipse_laps: list[LapPoints]) -> TrackAsset:
    return reconstruct(synthetic_config(ellipse), ellipse_laps, ["synthetic/ellipse"])


@pytest.fixture(scope="module")
def figure8_asset(figure8: SyntheticTrack, figure8_laps: list[LapPoints]) -> TrackAsset:
    return reconstruct(synthetic_config(figure8), figure8_laps, ["synthetic/figure8"])


def test_ellipse_ground_truth(ellipse: SyntheticTrack, ellipse_asset: TrackAsset) -> None:
    """P2.11: centerline RMS < 0.5 m, length error < 0.2 %, curvature error < 5 %."""
    truth = RefLine.resample(ellipse.xyz, 1.0)
    f = world_to_frenet(truth, ellipse_asset.xyz)
    rms = float(np.sqrt(np.mean(f.n**2)))
    length_err = abs(ellipse_asset.length / truth.length - 1)
    _, k_true = heading_curvature(truth)
    k_err = np.linalg.norm(ellipse_asset.kappa - np.interp(f.s, truth.s, k_true)) / np.linalg.norm(
        k_true
    )
    print(
        f"ellipse: xy RMS {rms:.3f} m, length error {100 * length_err:.3f} %, "
        f"curvature error {100 * k_err:.2f} %"
    )
    assert rms < 0.5
    assert length_err < 0.002
    assert k_err < 0.05


def test_figure8_crossing_and_elevation(
    figure8: SyntheticTrack, figure8_asset: TrackAsset, figure8_laps: list[LapPoints]
) -> None:
    """P2.11: s continuity through the crossing; z RMS < 0.5 m."""
    truth = RefLine.resample(figure8.xyz, 1.0)
    f = world_to_frenet(truth, figure8_asset.xyz)
    z_rms = float(np.sqrt(np.mean(f.h**2)))
    print(f"figure-eight: z RMS {z_rms:.3f} m, xy RMS {np.sqrt(np.mean(f.n**2)):.3f} m")
    assert z_rms < 0.5
    # the asset's s runs along the truth in one direction, with no jump at the crossing
    step = circ(np.diff(f.s), truth.length)
    assert ((step > 0.9) & (step < 1.1)).all()
    # every lap projects onto the asset with continuous s (no branch switch at the crossing)
    line = figure8_asset.line
    for s, _, _ in project_laps(line, figure8_laps, None, 50.0):
        ds = circ(np.diff(s), line.length)
        assert np.isfinite(s).all() and (ds > 0).all() and ds.max() < 40.0


def test_start_line_at_lap_start(ellipse: SyntheticTrack, ellipse_asset: TrackAsset) -> None:
    """P2.4: synthetic laps start at the generator's s = 0, so must the asset."""
    gap = np.linalg.norm(ellipse_asset.xyz[0, :2] - ellipse.xyz[0, :2])
    assert gap < 3.0
    assert ellipse_asset.report["start_offset_spread_m"] < 3.0  # type: ignore[operator]


def test_widths_and_report(ellipse_asset: TrackAsset) -> None:
    r = ellipse_asset.report
    assert r["width_source"] == "observed" and r["bank_source"] == "observed"
    # cars use up to ±4.8 m of a ±6 m road: limits must be inside the true edge, not looser
    assert 4.0 < ellipse_asset.w_left.mean() < 6.0
    assert float(r["fastest_laps_inside_frac"]) >= 0.99  # type: ignore[arg-type]
    assert np.rad2deg(np.abs(ellipse_asset.bank)).max() < 1.0  # the synthetic road is flat
    assert abs(float(r["length_error_pct"])) < 0.2  # type: ignore[arg-type]
    assert len(ellipse_asset.build_hash) == 64


def test_map_matched_data_needs_nominal_width(ellipse: SyntheticTrack) -> None:
    snapped = laps_of(
        make_session(
            ellipse,
            np.random.default_rng(2),
            drivers=DRIVERS,
            n_laps=3,
            lateral_frac=(0.0, 0.0),
            pos_noise_m=0.05,
        )
    )
    cfg = synthetic_config(ellipse)
    with pytest.raises(TrackReconstructionError, match="not observable"):
        reconstruct(cfg, snapped, ["x"])
    asset = reconstruct(synthetic_config(ellipse, nominal_width_m=12.0), snapped, ["x"])
    assert asset.report["width_source"] == "config_nominal"
    assert asset.report["bank_source"] == "not_observable"
    assert np.allclose(asset.w_left, 6.0) and np.allclose(asset.bank, 0.0)


def test_reconstruct_needs_laps(ellipse: SyntheticTrack, ellipse_laps: list[LapPoints]) -> None:
    with pytest.raises(TrackReconstructionError, match="at least 3"):
        reconstruct(synthetic_config(ellipse), ellipse_laps[:2], ["x"])


def test_heading_curvature_of_circle() -> None:
    r = 80.0
    t = np.linspace(0, 2 * np.pi, 3000, endpoint=False)
    line = RefLine.resample(np.column_stack([r * np.cos(t), r * np.sin(t), 0 * t]), 1.0)
    psi, kappa = heading_curvature(line)
    np.testing.assert_allclose(kappa, 1 / r, rtol=1e-3)  # counter-clockwise: left turn, > 0
    np.testing.assert_allclose(np.cos(psi[0]), 0.0, atol=0.01)
    cw = RefLine.resample(line.xyz[::-1].copy(), 1.0)
    assert (heading_curvature(cw, 5.0)[1] < 0).all()


def test_shift_start() -> None:
    sq = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], dtype=float)
    line = RefLine.resample(sq, 1.0)
    moved = shift_start(line, 10.0)
    np.testing.assert_allclose(moved.xyz[0], [10, 0, 0], atol=1e-9)
    assert moved.length == line.length


def test_repo_track_configs_parse() -> None:
    cfg = load_track_config(REPO / "configs/tracks/suzuka.yaml")
    assert cfg.name == "suzuka" and cfg.self_intersecting and cfg.official_length_m == 5807
    assert cfg.width_prior_m.w_min < cfg.width_prior_m.w_max
    assert cfg.sessions.reference.source == "openf1"
    with pytest.raises(ValueError):
        TrackConfig.model_validate(
            {**cfg.model_dump(by_alias=True), "width_prior_m": {"min": 10, "max": 5}}
        )


def test_build_track_from_processed_cache(tmp_path: Path, ellipse: SyntheticTrack) -> None:
    """End to end through the cache: export a synthetic session, then build from disk."""
    paths = Paths(tmp_path / "c", tmp_path / "p", tmp_path / "r")
    session = make_session(ellipse, np.random.default_rng(1), drivers=DRIVERS, n_laps=3)
    handle = _handle_from_session(2024, "Q", session)
    export_processed(handle, paths)
    cache.alias_record(paths, "fastf1", 2024, "Synthetic", "Q", handle.key)
    cfg = synthetic_config(
        ellipse,
        sessions={
            "reference": {"year": 2024, "event": "Synthetic", "session": "Q", "source": "fastf1"}
        },
    )
    asset = build_track(cfg, paths=paths)
    assert asset.source_sessions == ("2024/synthetic_ellipse_grand_prix/Q",)
    asset.save(track_dir(paths, "synthetic"))
    again = build_track(cfg, paths=paths)
    assert again.build_hash == asset.build_hash  # deterministic inputs -> same hash
    assert TrackAsset.load(track_dir(paths, "synthetic")).length == pytest.approx(asset.length)
    missing = cfg.model_copy(
        update={
            "sessions": cfg.sessions.model_copy(
                update={"reference": cfg.sessions.reference.model_copy(update={"year": 2023})}
            )
        }
    )
    with pytest.raises(TrackReconstructionError, match="apex data fetch"):
        resolve_sessions(paths, missing.sessions.all)
