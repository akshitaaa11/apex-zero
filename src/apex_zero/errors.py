"""Project-specific exceptions.

Every layer raises these instead of bare ``ValueError``/``RuntimeError`` when the failure is
a domain failure the caller may want to handle (or report) specifically. Never swallow them.
"""

from __future__ import annotations

__all__ = [
    "ApexZeroError",
    "CalibrationError",
    "DataUnavailableError",
    "GhostAlignmentError",
    "SchemaValidationError",
    "TrackReconstructionError",
]


class ApexZeroError(Exception):
    """Base class for all APEX ZERO domain errors."""


class DataUnavailableError(ApexZeroError):
    """Requested session/lap/driver data is missing from the source or the cache."""


class SchemaValidationError(ApexZeroError):
    """A table violates its column contract (``apex_zero.data.schemas``)."""


class TrackReconstructionError(ApexZeroError):
    """A track could not be reconstructed, or the result failed its validation checks."""


class GhostAlignmentError(ApexZeroError):
    """A driver lap could not be aligned to the track within the required tolerances."""


class CalibrationError(ApexZeroError):
    """Car calibration did not converge to the target lap time within tolerance."""
