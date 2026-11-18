from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from apex_zero.track.frenet import (
    RefLine,
    frenet_to_world,
    interp_angle,
    interp_periodic,
    world_to_frenet,
    wrap_s,
)
from tests.fixtures.synthetic import ellipse_track, figure_eight_track

ELLIPSE = RefLine.resample(ellipse_track().xyz, 1.0)
FIG8 = RefLine.resample(figure_eight_track().xyz, 1.0)


def circ_diff(a: np.ndarray, b: np.ndarray, length: float) -> np.ndarray:
    return np.abs(np.mod(a - b + length / 2, length) - length / 2)


def test_min_radius_of_test_lines_exceeds_20m() -> None:
    from apex_zero.track.reconstruct import heading_curvature

    for line in (ELLIPSE, FIG8):
        assert 1 / np.abs(heading_curvature(line)[1]).max() > 25.0


@settings(max_examples=60, deadline=None)
@given(
    s=st.lists(st.floats(0, 1, allow_nan=False), min_size=1, max_size=40),
    n=st.floats(-20.0, 20.0, allow_nan=False),
    h=st.floats(-2.0, 2.0, allow_nan=False),
    hint_err=st.floats(-30.0, 30.0, allow_nan=False),
    which=st.sampled_from(["ellipse", "figure8"]),
)
def test_round_trip_property(
    s: list[float], n: float, h: float, hint_err: float, which: str
) -> None:
    """P2.11: frenet_to_world(world_to_frenet(p)) == p for points within 20 m of the line."""
    line = ELLIPSE if which == "ellipse" else FIG8
    s_arr = np.array(s) * line.length
    p = frenet_to_world(line, s_arr, np.full(len(s_arr), n), np.full(len(s_arr), h))
    f = world_to_frenet(line, p, s_hint=s_arr + hint_err, window_m=50.0)
    assert f.valid.all()
    np.testing.assert_allclose(frenet_to_world(line, f.s, f.n, f.h), p, atol=1e-6)
    np.testing.assert_allclose(f.n, n, atol=1e-6)
    assert circ_diff(f.s, s_arr, line.length).max() < 1e-6


def test_self_intersection_needs_hint() -> None:
    """At the figure-eight crossing the hint picks the right branch."""
    crossing = np.argsort(np.hypot(FIG8.xyz[:, 0], FIG8.xyz[:, 1]))[:2]
    s_true = FIG8.s[crossing]
    p = frenet_to_world(FIG8, s_true, np.array([3.0, -3.0]))
    f = world_to_frenet(FIG8, p, s_hint=s_true, window_m=50.0)
    assert circ_diff(f.s, s_true, FIG8.length).max() < 1e-6
    # the two branches are > 1 km apart along s but meet in the plane
    assert circ_diff(s_true[:1], s_true[1:], FIG8.length)[0] > 1000


def test_no_hint_uses_3d_nearest() -> None:
    rng = np.random.default_rng(0)
    s = rng.uniform(0, FIG8.length, 500)
    n = rng.uniform(-8, 8, 500)
    p = frenet_to_world(FIG8, s, n)
    f = world_to_frenet(FIG8, p)
    assert circ_diff(f.s, s, FIG8.length).max() < 1e-6  # bridge is 16 m high: z separates
    f2 = world_to_frenet(FIG8, p[:, :2][:50])
    assert f2.valid.all() and (f2.h == 0).all()


def test_chunking_matches_unchunked() -> None:
    rng = np.random.default_rng(1)
    s = rng.uniform(0, ELLIPSE.length, 5000)
    p = frenet_to_world(ELLIPSE, s, rng.uniform(-5, 5, 5000))
    full = world_to_frenet(ELLIPSE, p, s_hint=s)
    part = world_to_frenet(ELLIPSE, p[:100], s_hint=s[:100])
    np.testing.assert_array_equal(full.s[:100], part.s)


def test_points_outside_window_are_invalid() -> None:
    p = frenet_to_world(ELLIPSE, np.array([100.0]), np.array([0.0]))
    f = world_to_frenet(ELLIPSE, p, s_hint=np.array([1000.0]), window_m=20.0)
    assert not f.valid[0] and np.isnan(f.n[0])


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="points"):
        world_to_frenet(ELLIPSE, np.zeros((3, 4)))
    with pytest.raises(ValueError, match="s_hint"):
        world_to_frenet(ELLIPSE, np.zeros((3, 3)), s_hint=np.zeros(2))
    with pytest.raises(ValueError, match="uniform"):
        RefLine(s=np.array([0.0, 1.0, 2.0, 4.0]), xyz=np.zeros((4, 3)), length=5.0)
    with pytest.raises(ValueError, match="at least 4"):
        RefLine(s=np.array([0.0, 1.0]), xyz=np.zeros((2, 3)), length=2.0)
    with pytest.raises(ValueError, match="xyz"):
        RefLine(s=np.arange(4.0), xyz=np.zeros((4, 2)), length=4.0)


def test_resample_is_uniform_and_keeps_length() -> None:
    square = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], dtype=float)
    line = RefLine.resample(square, 0.7)
    assert line.length == pytest.approx(40.0)
    assert np.allclose(np.diff(line.s), line.ds)
    assert line.ds == pytest.approx(40.0 / round(40 / 0.7))


def test_periodic_interpolation_helpers() -> None:
    grid = np.arange(4.0)
    vals = np.array([0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        interp_periodic(grid, vals, np.array([3.5, -0.5, 4.0]), 4.0), [1.5, 1.5, 0.0]
    )
    two = interp_periodic(grid, np.column_stack([vals, -vals]), np.array([0.5]), 4.0)
    np.testing.assert_allclose(two, [[0.5, -0.5]])
    ang = np.array([3.0, -3.0, -3.0, 3.0])  # crosses the ±pi wrap between 0 and 1
    mid = interp_angle(grid, ang, np.array([0.5]), 4.0)[0]
    assert abs(abs(mid) - np.pi) < 0.2
    np.testing.assert_allclose(wrap_s(np.array([-1.0, 5.0]), 4.0), [3.0, 1.0])
