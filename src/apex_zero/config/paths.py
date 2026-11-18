"""Filesystem locations, resolved from environment variables (see ``.env.example``).

Only resolves paths; it never creates directories. The cache *layout* inside these
directories is owned by ``apex_zero.data.cache`` (Phase 1).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = ["DEFAULTS", "ENV_VARS", "Paths", "resolve_paths"]

#: Environment variable name for each path.
ENV_VARS: dict[str, str] = {
    "cache_dir": "APEX_CACHE_DIR",
    "processed_dir": "APEX_PROCESSED_DIR",
    "runs_dir": "APEX_RUNS_DIR",
}

#: Defaults, relative to the working directory.
DEFAULTS: dict[str, str] = {
    "cache_dir": "data/cache",
    "processed_dir": "data/processed",
    "runs_dir": "runs",
}


@dataclass(frozen=True, slots=True)
class Paths:
    """Resolved, absolute project directories."""

    cache_dir: Path
    processed_dir: Path
    runs_dir: Path


def resolve_paths(env: Mapping[str, str] | None = None, base: Path | None = None) -> Paths:
    """Resolve project directories from environment variables, falling back to defaults.

    Args:
        env: Environment mapping; defaults to ``os.environ``. Empty values count as unset.
        base: Directory that relative paths are resolved against; defaults to the CWD.

    Returns:
        Absolute paths for the cache, processed-data and runs directories.
    """
    env = os.environ if env is None else env
    base = Path.cwd() if base is None else base

    def _one(key: str) -> Path:
        raw = env.get(ENV_VARS[key]) or DEFAULTS[key]
        path = Path(raw).expanduser()
        return (path if path.is_absolute() else base / path).resolve()

    return Paths(
        cache_dir=_one("cache_dir"),
        processed_dir=_one("processed_dir"),
        runs_dir=_one("runs_dir"),
    )
