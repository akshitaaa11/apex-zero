"""Unit tests of centerline, width, elevation and asset building blocks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from apex_zero.errors import TrackReconstructionError
from apex_zero.track.asset import ASSET_VERSION, CHANNELS, TrackAsset
from apex_zero.track.centerline import (
    LapPoints,
    bin_medians,
    periodic_smooth,
    seed_line,
    smooth_line,
)
from apex_zero.track.elevation import estimate_bank, fit_elevation, periodic_derivative
from apex_zero.track.frenet import RefLine
from apex_zero.track.width import estimate_width

L = 1000.0
GRID = np.arange(1000, dtype=np.float64)


# --- centerline -------------------------------------------------------------------------------


def test_periodic_smooth_recovers_long_wave_and_is_periodic() -> None:
    rng = np.random.default_rng(0)
    s = rng.uniform(0, L, 4000)
    truth = lambda x: 2.0 * np.sin(2 * np.pi * x / L)  # noqa: E731
    y = truth(s) + rng.normal(0, 0.3, len(s))
    fit = periodic_smooth(s, y, L, 20.0, GRID)
    assert np.sqrt(np.mean((fit - truth(GRID)) ** 2)) < 0.05
    seam = periodic_smooth(s, y, L, 20.0, np.array([0.0, L - 1e-9]))
    assert abs(seam[0] - seam[1]) < 1e-3


def test_periodic_smooth_zero_ell_interpolates_and_merges_duplicates() -> None:
    s = np.linspace(0, L, 50, endpoint=False)
    y = np.cos(2 * np.pi * s / L)
    np.testing.assert_allclose(periodic_smooth(s, y, L, 0.0, s), y, atol=1e-9)
    s_dup = np.concatenate([s, s[:5]])
    y_dup = np.concatenate([y, y[:5] + 1.0])  # duplicates average to y + 0.5
    out = periodic_smooth(s_dup, y_dup, L, 0.0, s[:5])
    np.testing.assert_allclose(out, y[:5] + 0.5, atol=1e-9)
    two = periodic_smooth(s, np.column_stack([y, -y]), L, 0.0, s)
    np.testing.assert_allclose(two[:, 1], -y, atol=1e-9)
    with pytest.raises(TrackReconstructionError, match="too few"):
        periodic_smooth(s[:5], y[:5], L, 0.0, s)


def test_smoothing_coordinates_shrinks_curves_as_documented() -> None:
    """Why refinement smooths offsets, not x(s), y(s): coordinate smoothing pulls inwards."""
    r = 50.0
    t = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    circle = RefLine.resample(np.column_stack([r * np.cos(t), r * np.sin(t), 0 * t]), 1.0)
    smooth = smooth_line(circle, ell_xy_m=8.0, ell_z_m=8.0)
    shrink = r - np.hypot(smooth.xyz[:, 0], smooth.xyz[:, 1]).mean()
    # H(omega) = 1 / (1 + (ell omega)^4) at omega = 1/r: shrink ~ ell^4 / r^3 = 0.0328 m
    assert shrink == pytest.approx(8.0**4 / r**3, rel=0.1)


def test_bin_medians() -> None:
    s = np.array([1.0, 2.0, 3.0, 12.0, 13.0, np.nan, 25.0])
    v = np.array([1.0, 2.0, 30.0, 5.0, 7.0, 1.0, 9.0])
    centres, med, cnt = bin_medians(s, v, 30.0, 10.0, min_count=2)
    np.testing.assert_allclose(centres, [5.0, 15.0])
    np.testing.assert_allclose(med, [2.0, 6.0])  # median ignores the 30 outlier
    np.testing.assert_allclose(cnt, [3, 2])


def test_seed_line_rejects_short_lap() -> None:
    lap = LapPoints("x", np.arange(10.0), np.zeros((10, 3)), 0.0, 9.0)
    with pytest.raises(TrackReconstructionError, match="only 10 samples"):
        seed_line(lap)


# --- width ------------------------------------------------------------------------------------


def _band(lo: float, hi: float, n: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    return rng.uniform(0, L, n), rng.uniform(lo, hi, n)


def test_width_recentres_on_band() -> None:
    s, n = _band(-2.0, 6.0)
    w = estimate_width(GRID, L, s, n, bin_m=10.0, margin_m=0.5, w_min=4.0, w_max=30.0, ell_m=20.0)
    np.testing.assert_allclose(w.centre_n, 2.0, atol=0.15)  # middle of [-2, 6]
    np.testing.assert_allclose(w.w_left, 4.0 * 0.98 + 0.5, atol=0.15)  # p1..p99 half-width + margin
    np.testing.assert_array_equal(w.w_left, w.w_right)


def test_width_without_recentre_and_prior_clip() -> None:
    s, n = _band(-2.0, 6.0)
    w = estimate_width(
        GRID, L, s, n, bin_m=10.0, margin_m=0.0, w_min=4.0, w_max=10.0, ell_m=20.0, recentre=False
    )
    np.testing.assert_allclose(w.centre_n, 0.0)
    assert np.allclose(w.w_left, 5.0)  # 6 m clipped to w_max / 2
    np.testing.assert_allclose(w.w_right, 1.92, atol=0.15)
    narrow_s, narrow_n = _band(-0.1, 0.1)
    w2 = estimate_width(
        GRID, L, narrow_s, narrow_n, bin_m=10.0, margin_m=0.0, w_min=4.0, w_max=10.0, ell_m=20.0
    )
    assert np.allclose(w2.w_left, 2.0)  # clipped up to w_min / 2


def test_width_ignores_pit_lane_outliers_and_needs_coverage() -> None:
    s, n = _band(-3.0, 3.0)
    n[:40] = 25.0  # 0.2 % of samples within max_abs_n_m: counted, but beyond the p99
    n[40:4000] = 80.0  # pit lane, 20 % of samples: dropped, so it cannot move the p99
    w = estimate_width(GRID, L, s, n, bin_m=10.0, margin_m=0.0, w_min=1.0, w_max=30.0, ell_m=20.0)
    assert w.w_left.max() < 5.0
    with pytest.raises(TrackReconstructionError, match="bins"):
        estimate_width(
            GRID, L, s[:50], n[:50], bin_m=10.0, margin_m=0.0, w_min=1.0, w_max=30.0, ell_m=20.0
        )


# --- elevation --------------------------------------------------------------------------------


def test_periodic_derivative_matches_analytic() -> None:
    z = 5.0 * np.sin(2 * np.pi * GRID / L)
    k = 2 * np.pi / L
    np.testing.assert_allclose(periodic_derivative(z, 1.0), 5.0 * k * np.cos(k * GRID), atol=1e-6)
    np.testing.assert_allclose(
        periodic_derivative(z, 1.0, 2), -5.0 * k**2 * np.sin(k * GRID), atol=1e-7
    )


def test_grade_of_constant_slope_section() -> None:
    """A 5 % climb over half the lap (and descent back) gives grade ±0.05 away from the ends."""
    z_true = np.where(GRID < 500, 0.05 * GRID, 0.05 * (1000 - GRID))
    rng = np.random.default_rng(0)
    s = rng.uniform(0, L, 20000)
    h = np.interp(s, GRID, z_true) + rng.normal(0, 0.3, len(s))
    elev = fit_elevation(GRID, np.zeros(1000), L, s, h, bin_m=5.0, ell_z_m=15.0)
    mid = (GRID > 100) & (GRID < 400)
    np.testing.assert_allclose(elev.grade[mid], 0.05, atol=0.005)
    np.testing.assert_allclose(elev.z[mid], z_true[mid], atol=0.3)


def test_bank_estimate_recovers_slope_and_clips() -> None:
    rng = np.random.default_rng(1)
    s, n = rng.uniform(0, L, 30000), rng.uniform(-5, 5, 30000)
    for deg, expected in ((4.0, 4.0), (20.0, 10.0)):
        h = np.tan(np.deg2rad(deg)) * n + rng.normal(0, 0.05, len(n))
        bank = estimate_bank(GRID, L, s, n, h, ell_m=50.0)
        np.testing.assert_allclose(np.rad2deg(bank), expected, atol=0.3)
    narrow = estimate_bank(GRID, L, s, n * 0.1, np.tan(0.1) * n * 0.1)
    assert np.allclose(narrow, 0.0)  # not enough lateral spread: no estimate


# --- asset ------------------------------------------------------------------------------------


def make_asset(**overrides: object) -> TrackAsset:
    n = 100
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    fields: dict[str, object] = {c: np.zeros(n) for c in CHANNELS}
    fields.update(
        s=np.arange(n) * 6.283185307179586,
        x=100 * np.cos(t),
        y=100 * np.sin(t),
        psi=t + np.pi / 2,
        kappa=np.full(n, 0.01),
        w_left=np.full(n, 6.0),
        w_right=np.full(n, 6.0),
    )
    fields.update(overrides)
    return TrackAsset(
        name="circle",
        length=n * 6.283185307179586,
        source_sessions=("a/b/Q",),
        build_hash="abc",
        params={"k": 1},
        report={"r": 0.5},
        **fields,
    )  # type: ignore[arg-type]


def test_asset_round_trip_and_readonly(tmp_path: Path) -> None:
    a = make_asset()
    a.save(tmp_path / "t")
    b = TrackAsset.load(tmp_path / "t")
    for c in CHANNELS:
        np.testing.assert_array_equal(getattr(a, c), getattr(b, c))
    assert (b.name, b.source_sessions, b.build_hash, dict(b.params), dict(b.report)) == (
        "circle",
        ("a/b/Q",),
        "abc",
        {"k": 1},
        {"r": 0.5},
    )
    with pytest.raises(ValueError):
        a.x[0] = 1.0
    meta = json.loads((tmp_path / "t" / "meta.json").read_text())
    assert meta["asset_version"] == ASSET_VERSION and meta["n_points"] == 100


def test_asset_load_refuses_tampering_and_old_versions(tmp_path: Path) -> None:
    d = make_asset().save(tmp_path / "t")
    with pytest.raises(TrackReconstructionError, match="no track asset"):
        TrackAsset.load(tmp_path / "missing")
    meta = json.loads((d / "meta.json").read_text())
    (d / "meta.json").write_text(json.dumps({**meta, "asset_version": 0}))
    with pytest.raises(TrackReconstructionError, match="version"):
        TrackAsset.load(d)
    (d / "meta.json").write_text(json.dumps({**meta, "asset_sha256": "0" * 64}))
    with pytest.raises(TrackReconstructionError, match="checksum"):
        TrackAsset.load(d)


def test_asset_validation_and_interpolation() -> None:
    a = make_asset()
    assert a.ds == pytest.approx(6.283185307179586)
    np.testing.assert_allclose(a.at("kappa", np.array([1.0, 1e6])), 0.01)
    psi = a.at("psi", np.array([a.length - 1e-6]))
    assert abs(np.angle(np.exp(1j * (psi[0] - np.pi / 2)))) < 0.1
    assert a.line.length == a.length
    with pytest.raises(KeyError):
        a.at("s", np.array([0.0]))
    with pytest.raises(ValueError, match="shape"):
        make_asset(z=np.zeros(3))
    with pytest.raises(ValueError, match="non-finite"):
        make_asset(z=np.full(100, np.nan))
    with pytest.raises(ValueError, match="positive"):
        make_asset(w_left=np.zeros(100))
    with pytest.raises(ValueError, match="uniform"):
        make_asset(s=np.arange(100.0))
