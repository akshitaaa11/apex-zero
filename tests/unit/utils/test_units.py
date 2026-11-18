from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from apex_zero.utils.units import MPS_PER_KPH, deg_to_rad, kph_to_mps, mps_to_kph, rad_to_deg

finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False)


def test_known_speed_values() -> None:
    assert kph_to_mps(3.6) == pytest.approx(1.0)
    assert kph_to_mps(360.0) == pytest.approx(100.0)
    assert mps_to_kph(100.0) == pytest.approx(360.0)
    assert pytest.approx(1.0 / 3.6) == MPS_PER_KPH


def test_known_angle_values() -> None:
    assert deg_to_rad(180.0) == pytest.approx(math.pi)
    assert rad_to_deg(math.pi / 2) == pytest.approx(90.0)


def test_arrays_keep_shape_and_dtype() -> None:
    v_kph = np.array([0.0, 36.0, 360.0], dtype=np.float64)
    v_mps = kph_to_mps(v_kph)
    assert v_mps.shape == (3,)
    assert v_mps.dtype == np.float64
    np.testing.assert_allclose(v_mps, [0.0, 10.0, 100.0])
    np.testing.assert_allclose(rad_to_deg(deg_to_rad(np.array([-90.0, 45.0]))), [-90.0, 45.0])


@given(finite)
def test_speed_round_trip(x: float) -> None:
    assert mps_to_kph(kph_to_mps(x)) == pytest.approx(x, rel=1e-12, abs=1e-9)


@given(finite)
def test_angle_round_trip(x: float) -> None:
    assert rad_to_deg(deg_to_rad(x)) == pytest.approx(x, rel=1e-12, abs=1e-9)
