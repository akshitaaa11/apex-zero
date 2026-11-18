from __future__ import annotations

import json

import pytest

from apex_zero.utils.logging import configure_logging, get_logger


def test_json_output_has_event_and_fields(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="INFO", fmt="json")
    get_logger("apex_zero.test").info("track_built", length_m=5807.0)
    record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert record["event"] == "track_built"
    assert record["length_m"] == 5807.0
    assert record["logger"] == "apex_zero.test"
    assert record["level"] == "info"
    assert "timestamp" in record


def test_level_filters_lower_levels(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level="WARNING", fmt="json")
    log = get_logger("apex_zero.test")
    log.info("hidden")
    log.warning("shown")
    err = capsys.readouterr().err
    assert "hidden" not in err
    assert "shown" in err


def test_console_format_and_numeric_level(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(level=10, fmt="console")
    get_logger("apex_zero.test").debug("debug_event", key="value")
    err = capsys.readouterr().err
    assert "debug_event" in err
    assert "value" in err


def test_unknown_level_raises() -> None:
    with pytest.raises(ValueError, match="unknown log level"):
        configure_logging(level="LOUD")
