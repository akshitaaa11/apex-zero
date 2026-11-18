"""Unit conversions. Everything inside APEX ZERO is SI; convert only at the edges.

Non-SI quantities carry a suffix in their name (``speed_kph``, ``angle_deg``).

Telemetry unit constants (P1.4). Both FastF1 and OpenF1 relay the F1 live-timing position feed,
whose X/Y/Z are integers in decimetres:

- ``OPENF1_POS_TO_M`` is **verified**: ``tests/network/test_units.py`` measures the fastest-lap
  polyline at two circuits against their official lengths (results in ``docs/data.md``).
- ``FASTF1_POS_TO_M`` is taken from FastF1's own documentation (``fastf1.core.Telemetry``:
  "X position [1/10 m]") and agrees with the OpenF1 measurement, but the FastF1 half of the
  network test could not run yet (live timing requires an F1 account; see ``docs/data.md``).
- The OpenF1 vs FastF1 axis comparison is **not yet verified** for the same reason. Until it is,
  code must not mix positions from the two sources in one track build.
"""

from __future__ import annotations

import math
from typing import overload

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "FASTF1_POS_TO_M",
    "MPS_PER_KPH",
    "OPENF1_POS_TO_M",
    "deg_to_rad",
    "kph_to_mps",
    "mps_to_kph",
    "rad_to_deg",
]

#: Metres per second in one kilometre per hour (1000 m / 3600 s).
MPS_PER_KPH: float = 1000.0 / 3600.0

#: Metres per FastF1 position unit (X/Y/Z are in 1/10 m). See module docstring.
FASTF1_POS_TO_M: float = 0.1

#: Metres per OpenF1 position unit (``location`` x/y/z). Verified by tests/network/test_units.py.
OPENF1_POS_TO_M: float = 0.1

FloatArray = NDArray[np.float64]


@overload
def kph_to_mps(speed_kph: float) -> float: ...
@overload
def kph_to_mps(speed_kph: FloatArray) -> FloatArray: ...
def kph_to_mps(speed_kph: float | FloatArray) -> float | FloatArray:
    """Convert a speed from km/h to m/s.

    Args:
        speed_kph: Speed in km/h; a scalar or an ``(N,) float64`` array.

    Returns:
        Speed in m/s, same shape as the input.
    """
    return speed_kph * MPS_PER_KPH


@overload
def mps_to_kph(speed_mps: float) -> float: ...
@overload
def mps_to_kph(speed_mps: FloatArray) -> FloatArray: ...
def mps_to_kph(speed_mps: float | FloatArray) -> float | FloatArray:
    """Convert a speed from m/s to km/h.

    Args:
        speed_mps: Speed in m/s; a scalar or an ``(N,) float64`` array.

    Returns:
        Speed in km/h, same shape as the input.
    """
    return speed_mps / MPS_PER_KPH


@overload
def deg_to_rad(angle_deg: float) -> float: ...
@overload
def deg_to_rad(angle_deg: FloatArray) -> FloatArray: ...
def deg_to_rad(angle_deg: float | FloatArray) -> float | FloatArray:
    """Convert an angle from degrees to radians.

    Args:
        angle_deg: Angle in degrees; a scalar or an ``(N,) float64`` array.

    Returns:
        Angle in radians, same shape as the input.
    """
    return angle_deg * (math.pi / 180.0)


@overload
def rad_to_deg(angle_rad: float) -> float: ...
@overload
def rad_to_deg(angle_rad: FloatArray) -> FloatArray: ...
def rad_to_deg(angle_rad: float | FloatArray) -> float | FloatArray:
    """Convert an angle from radians to degrees.

    Args:
        angle_rad: Angle in radians; a scalar or an ``(N,) float64`` array.

    Returns:
        Angle in degrees, same shape as the input.
    """
    return angle_rad * (180.0 / math.pi)
