from __future__ import annotations

import pytest

from apex_zero import errors


@pytest.mark.parametrize(
    "exc",
    [
        errors.DataUnavailableError,
        errors.TrackReconstructionError,
        errors.GhostAlignmentError,
        errors.CalibrationError,
    ],
)
def test_domain_errors_share_base(exc: type[errors.ApexZeroError]) -> None:
    assert issubclass(exc, errors.ApexZeroError)
    assert issubclass(exc, Exception)
    with pytest.raises(errors.ApexZeroError, match="boom"):
        raise exc("boom")


def test_all_exports_exist() -> None:
    for name in errors.__all__:
        assert hasattr(errors, name)
