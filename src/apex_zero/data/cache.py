"""Cache layout and keys (P1.1). Layout version 1, see ``docs/migrations.md``.

Raw source caches live under ``APEX_CACHE_DIR``::

    <cache>/fastf1/                 FastF1's own HTTP cache (``fastf1.Cache``)
    <cache>/openf1/<h[:2]>/<h>.json OpenF1 responses keyed by sha256 of the request URL

Validated, SI-unit tables live under ``APEX_PROCESSED_DIR``::

    <processed>/aliases.json                       query -> session dir (offline re-fetch)
    <processed>/<year>/<event_slug>/<session>/
        session_meta.json                          written last; marks the session complete
        laps.parquet
        <DRIVER>/pos_lap<NNN>.parquet
        <DRIVER>/car_lap<NNN>.parquet

Keys are deterministic: ``event_slug`` comes from the official event name, ``session`` is the
canonical code (``FP1``..``FP3``, ``SQ``, ``SS``, ``S``, ``Q``, ``R``), and lap files are named by
lap number.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from apex_zero.config.paths import Paths

__all__ = [
    "CACHE_LAYOUT_VERSION",
    "SESSION_NAMES",
    "FileRecord",
    "SessionKey",
    "SessionMeta",
    "alias_lookup",
    "alias_record",
    "car_path",
    "fastf1_cache_dir",
    "file_record",
    "is_complete",
    "laps_path",
    "list_sessions",
    "meta_path",
    "normalize_session",
    "openf1_cache_dir",
    "pos_path",
    "read_meta",
    "session_dir",
    "slugify",
    "write_meta",
]

#: Bump (with a ``docs/migrations.md`` entry) whenever the layout or ``SessionMeta`` changes.
CACHE_LAYOUT_VERSION: int = 1

#: Canonical session code -> session name used by FastF1 and OpenF1.
SESSION_NAMES: dict[str, str] = {
    "FP1": "Practice 1",
    "FP2": "Practice 2",
    "FP3": "Practice 3",
    "SQ": "Sprint Qualifying",
    "SS": "Sprint Shootout",
    "S": "Sprint",
    "Q": "Qualifying",
    "R": "Race",
}


def normalize_session(identifier: str) -> str:
    """Canonical session code for a code or name (``"q"``, ``"Qualifying"`` -> ``"Q"``).

    Raises:
        ValueError: If the identifier is not a known session.
    """
    ident = identifier.strip()
    if ident.upper() in SESSION_NAMES:
        return ident.upper()
    for code, name in SESSION_NAMES.items():
        if ident.casefold() == name.casefold():
            return code
    raise ValueError(f"unknown session {identifier!r}; use one of {', '.join(SESSION_NAMES)}")


def slugify(name: str) -> str:
    """Deterministic ASCII slug: ``"São Paulo Grand Prix"`` -> ``"sao_paulo_grand_prix"``.

    Raises:
        ValueError: If nothing usable remains.
    """
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")
    if not slug:
        raise ValueError(f"cannot slugify {name!r}")
    return slug


@dataclass(frozen=True, slots=True, order=True)
class SessionKey:
    """Identifies one session in the processed cache."""

    year: int
    event_slug: str
    session: str

    @property
    def relpath(self) -> Path:
        """``<year>/<event_slug>/<session>`` relative to ``APEX_PROCESSED_DIR``."""
        return Path(str(self.year), self.event_slug, self.session)

    def __str__(self) -> str:
        return self.relpath.as_posix()

    @classmethod
    def from_relpath(cls, rel: str | Path) -> SessionKey:
        """Inverse of :attr:`relpath`."""
        year, slug, session = Path(rel).parts
        return cls(int(year), slug, session)


def fastf1_cache_dir(paths: Paths) -> Path:
    """FastF1's HTTP cache directory."""
    return paths.cache_dir / "fastf1"


def openf1_cache_dir(paths: Paths) -> Path:
    """OpenF1 response cache directory."""
    return paths.cache_dir / "openf1"


def session_dir(paths: Paths, key: SessionKey) -> Path:
    """Processed directory of one session."""
    return paths.processed_dir / key.relpath


def meta_path(paths: Paths, key: SessionKey) -> Path:
    """``session_meta.json`` of one session."""
    return session_dir(paths, key) / "session_meta.json"


def laps_path(paths: Paths, key: SessionKey) -> Path:
    """Lap table of one session."""
    return session_dir(paths, key) / "laps.parquet"


def pos_path(paths: Paths, key: SessionKey, driver: str, lap: int) -> Path:
    """Position table of one driver lap."""
    return session_dir(paths, key) / driver / f"pos_lap{lap:03d}.parquet"


def car_path(paths: Paths, key: SessionKey, driver: str, lap: int) -> Path:
    """Car-data table of one driver lap."""
    return session_dir(paths, key) / driver / f"car_lap{lap:03d}.parquet"


class FileRecord(BaseModel):
    """A file written for a session, relative to the session directory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    sha256: str
    size_bytes: int


class SessionMeta(BaseModel):
    """Contents of ``session_meta.json``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    layout_version: int = CACHE_LAYOUT_VERSION
    source: Literal["fastf1", "openf1"]
    source_version: str
    apex_zero_version: str
    fetched_at_utc: str
    year: int
    event_name: str
    event_slug: str
    round_number: int
    country: str
    location: str
    session: str
    session_name: str
    drivers: list[str]
    #: Fastest valid (not deleted) lap time per driver [s], from the official timing data.
    fastest_lap_s: dict[str, float]
    n_laps: int
    files: list[FileRecord]

    @property
    def key(self) -> SessionKey:
        """Cache key of this session."""
        return SessionKey(self.year, self.event_slug, self.session)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(root: Path, path: Path) -> FileRecord:
    """Hash a written file for ``SessionMeta.files``."""
    return FileRecord(
        path=path.relative_to(root).as_posix(),
        sha256=_sha256(path),
        size_bytes=path.stat().st_size,
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_meta(paths: Paths, meta: SessionMeta) -> Path:
    """Atomically write ``session_meta.json``. Call this last, after every table is written."""
    path = meta_path(paths, meta.key)
    _atomic_write_text(path, meta.model_dump_json(indent=2) + "\n")
    return path


def read_meta(paths: Paths, key: SessionKey) -> SessionMeta | None:
    """Read a session's metadata, or ``None`` if it has not been (fully) fetched."""
    path = meta_path(paths, key)
    if not path.is_file():
        return None
    return SessionMeta.model_validate_json(path.read_text(encoding="utf-8"))


def is_complete(paths: Paths, key: SessionKey, verify_hashes: bool = True) -> bool:
    """Whether a session is fully in the processed cache and intact.

    Args:
        paths: Project paths.
        key: Session key.
        verify_hashes: Also re-hash every file (otherwise only existence and size).
    """
    meta = read_meta(paths, key)
    if meta is None or meta.layout_version != CACHE_LAYOUT_VERSION:
        return False
    root = session_dir(paths, key)
    for rec in meta.files:
        f = root / rec.path
        if not f.is_file() or f.stat().st_size != rec.size_bytes:
            return False
        if verify_hashes and _sha256(f) != rec.sha256:
            return False
    return True


def _aliases_path(paths: Paths) -> Path:
    return paths.processed_dir / "aliases.json"


def _alias(year: int, event_query: str, session: str) -> str:
    return f"{year}|{' '.join(event_query.casefold().split())}|{normalize_session(session)}"


def _read_aliases(paths: Paths) -> dict[str, str]:
    path = _aliases_path(paths)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        raise ValueError(f"{path} is not a str->str mapping")
    return data


def alias_lookup(paths: Paths, year: int, event_query: str, session: str) -> SessionKey | None:
    """Session key previously resolved for this (year, event query, session), if any.

    Lets a repeated ``fetch`` find the processed session without resolving the event schedule.
    """
    rel = _read_aliases(paths).get(_alias(year, event_query, session))
    return None if rel is None else SessionKey.from_relpath(rel)


def alias_record(paths: Paths, year: int, event_query: str, session: str, key: SessionKey) -> None:
    """Remember which session a query resolved to."""
    aliases = _read_aliases(paths)
    aliases[_alias(year, event_query, session)] = key.relpath.as_posix()
    _atomic_write_text(_aliases_path(paths), json.dumps(aliases, indent=2, sort_keys=True) + "\n")


def list_sessions(paths: Paths, year: int | None = None) -> list[SessionMeta]:
    """All sessions in the processed cache (optionally one year), sorted by key."""
    root = paths.processed_dir
    pattern = f"{year}/*/*/session_meta.json" if year is not None else "*/*/*/session_meta.json"
    metas = [
        SessionMeta.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(root.glob(pattern))
    ]
    return sorted(metas, key=lambda m: m.key)
