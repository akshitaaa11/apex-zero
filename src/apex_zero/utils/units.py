"""Unit conversions. Everything inside APEX ZERO is SI; convert only at the edges.

Non-SI quantities carry a suffix in their name (``speed_kph``, ``angle_deg``).

Telemetry unit constants (e.g. the scale of FastF1/OpenF1 position X/Y/Z) are deliberately
**not** defined yet: they are pinned against real data, with tests, in Phase 1 (P1). Do not
assume them.
"""

from __future__ import annotations

import math
from typing import overload

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MPS_PER_KPH",
    "deg_to_rad",
    "kph_to_mps",
    "mps_to_kph",
    "rad_to_deg",
]

#: Metres per second in one kilometre per hour (1000 m / 3600 s).
MPS_PER_KPH: float = 1000.0 / 3600.0

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
