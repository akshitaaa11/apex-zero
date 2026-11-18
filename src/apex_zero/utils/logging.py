"""Structured logging through structlog.

Use ``get_logger(__name__)`` in every module and log key/value pairs, e.g.
``log.info("track_built", name=name, length_m=length_m)``. No ``print`` in ``src/``; user-facing
CLI output goes through ``rich`` in ``apex_zero.cli``.

Note: ``AGENTS.md`` §2 lists stdlib/numpy/pydantic for L0. ``structlog`` is the one extra
third-party import allowed in L0, and only in this module (P0.5 requires it here); the layering
test enforces that exception.
"""

from __future__ import annotations

import logging
import sys
from typing import Literal

import structlog

__all__ = ["LogFormat", "configure_logging", "get_logger"]

LogFormat = Literal["console", "json"]


def configure_logging(level: str | int = "INFO", fmt: LogFormat = "console") -> None:
    """Configure structlog and the stdlib root logger once, at program start.

    Args:
        level: Log level name (``"DEBUG"``, ``"INFO"``, ...) or numeric level.
        fmt: ``"console"`` for human-readable output, ``"json"`` for one JSON object per line.

    Raises:
        ValueError: If ``level`` is not a known level name.
    """
    if isinstance(level, str):
        numeric = logging.getLevelNamesMapping().get(level.upper())
        if numeric is None:
            raise ValueError(f"unknown log level: {level!r}")
    else:
        numeric = level

    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )
    logging.basicConfig(level=numeric, stream=sys.stderr, format="%(name)s: %(message)s")


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    """Return a structlog logger bound to ``logger=name``.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A bound logger; call ``.info(event, **kv)`` etc.
    """
    logger: structlog.typing.FilteringBoundLogger = structlog.get_logger().bind(logger=name)
    return logger
