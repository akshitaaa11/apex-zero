"""The synthetic session generator must itself be right, since later phases test against it."""

from __future__ import annotations

import numpy as np
import pytest

from apex_zero.utils.units import FASTF1_POS_TO_M
from tests.fixtures.synthetic import ellipse_track, figure_eight_track, make_session


def test_ellipse_length_matches_ramanujan() -> None:
    a, b = 600.0, 300.0
    h = ((a - b) / (a + b)) ** 2
    planar = np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))
    tr = ellipse_track(a, b, z_amp_m=0.0)
    assert tr.length_m == pytest.approx(planar, rel=1e-4)
    assert np.allclose(np.diff(tr.s), 1.0)


def test_figure_eight_crosses_itself_at_different_heights() -> None:
    tr = figure_eight_track(scale_m=500.0, bridge_m=8.0)
    near_origin = np.hypot(tr.xyz[:, 0], tr.xyz[:, 1]) < 2.0
    z = tr.xyz[near_origin, 2]
    assert z.min() == pytest.approx(-8.0, abs=0.1)
    assert z.max() == pytest.approx(8.0, abs=0.1)


def test_session_is_deterministic_and_fastf1_shaped() -> None:
    tr = ellipse_track()
    a = make_session(tr, np.random.default_rng(3), n_laps=2)
    b = make_session(tr, np.random.default_rng(3), n_laps=2)
    assert a.laps.equals(b.laps)
    assert a.pos_data["1"].equals(b.pos_data["1"])
    assert set(a.pos_data) == {"1", "2", "3"}
    assert {"SessionTime", "X", "Y", "Z", "Status"} <= set(a.pos_data["1"].columns)
    assert {"Speed", "RPM", "nGear", "Throttle", "Brake", "DRS"} <= set(a.car_data["1"].columns)
    dt = np.diff(a.pos_data["1"]["SessionTime"].dt.total_seconds())
    assert 0.2 < dt.mean() < 0.3  # ~4 Hz
    assert (dt > 0).all()


def test_positions_follow_the_track_within_width() -> None:
    tr = figure_eight_track()
    sess = make_session(tr, np.random.default_rng(5), n_laps=1, pos_noise_m=0.0)
    pos = sess.pos_data["2"]
    xy = pos[["X", "Y"]].to_numpy() * FASTF1_POS_TO_M
    d = np.min(np.linalg.norm(xy[:, None, :] - tr.xyz[None, ::2, :2], axis=2), axis=1)
    assert d.max() < tr.half_width_m + 1.0


def test_lap_times_consistent_with_lap_table() -> None:
    sess = make_session(ellipse_track(), np.random.default_rng(1), n_laps=3)
    aaa = sess.laps[sess.laps["Driver"] == "AAA"]
    np.testing.assert_allclose(aaa["LapTime"].dt.total_seconds(), sess.lap_times_s["AAA"])
    assert aaa["IsAccurate"].tolist() == [False, True, True]
