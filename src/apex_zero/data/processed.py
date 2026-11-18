"""Source-independent export of processed sessions (shared by the FastF1 and OpenF1 loaders).

A loader converts its source to the SI contracts in ``data.schemas`` (a lap table plus
whole-session position and car telemetry per car number) and hands them to
:func:`write_session`, which slices each lap, validates it, writes the Parquet files and writes
``session_meta.json`` last. :func:`cached_session` returns an intact earlier export without any
network access.

Each lap file holds the raw samples in ``[LapStartTime, LapEndTime]`` plus ``PAD_SAMPLES`` on
either side (for exact line-crossing interpolation later). Nothing is resampled.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from apex_zero import __version__
from apex_zero.config.paths import Paths
from apex_zero.data import cache
from apex_zero.data.schemas import CAR, LAPS, POSITION, assert_valid, validate_frame
from apex_zero.utils.logging import get_logger

__all__ = [
    "PAD_SAMPLES",
    "DriverLap",
    "FetchResult",
    "RejectedLap",
    "SessionIdentity",
    "Source",
    "cached_session",
    "fastest_laps",
    "iter_laps",
    "lap_window",
    "remember_query",
    "write_session",
]

log = get_logger(__name__)

#: Raw samples kept on each side of the lap window.
PAD_SAMPLES: int = 1

Source = Literal["fastf1", "openf1"]


class RejectedLap(BaseModel):
    """A lap whose telemetry failed validation and was not written."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    driver: str
    lap_number: int
    issues: list[str]


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Outcome of a ``fetch``: the session's metadata and whether it came from the cache."""

    meta: cache.SessionMeta
    from_cache: bool
    rejected: list[RejectedLap]


@dataclass(frozen=True, slots=True)
class SessionIdentity:
    """Who/what/where of a session, as recorded in ``session_meta.json``."""

    key: cache.SessionKey
    event_name: str
    round_number: int | None
    country: str
    location: str
    session_name: str


@dataclass(frozen=True, slots=True)
class DriverLap:
    """One driver lap: lap-table row and raw SI telemetry for its window."""

    driver: str
    driver_number: str
    lap_number: int
    lap: pd.Series
    pos: pd.DataFrame = field(repr=False)
    car: pd.DataFrame = field(repr=False)


def lap_window(telemetry: pd.DataFrame, start_s: float, end_s: float) -> pd.DataFrame:
    """Samples with ``SessionTime`` in ``[start_s, end_s]`` plus ``PAD_SAMPLES`` either side.

    Args:
        telemetry: SI telemetry sorted by ``SessionTime`` [s].
        start_s: Window start [s].
        end_s: Window end [s].

    Returns:
        The slice (index reset); empty if no sample lies inside the window.
    """
    t = telemetry["SessionTime"].to_numpy()
    inside = (t >= start_s) & (t <= end_s)
    if not inside.any():
        return telemetry.iloc[0:0].reset_index(drop=True)
    idx = inside.nonzero()[0]
    lo = max(int(idx[0]) - PAD_SAMPLES, 0)
    hi = min(int(idx[-1]) + PAD_SAMPLES + 1, len(telemetry))
    return telemetry.iloc[lo:hi].reset_index(drop=True)


def iter_laps(
    laps: pd.DataFrame,
    pos_by_number: Mapping[str, pd.DataFrame],
    car_by_number: Mapping[str, pd.DataFrame],
    driver: str | None = None,
    accurate_only: bool = False,
) -> Iterator[DriverLap]:
    """Yield driver laps with their telemetry windows, in lap-table order.

    Laps without a time window, or whose car has no telemetry, are skipped (and logged).

    Args:
        laps: ``LAPS``-contract table.
        pos_by_number: ``POSITION``-contract frames by car number (whole session).
        car_by_number: ``CAR``-contract frames by car number (whole session).
        driver: Three-letter code or car number; ``None`` for all.
        accurate_only: Only laps with ``IsAccurate``.
    """
    if driver is not None:
        laps = laps[(laps["Driver"] == driver) | (laps["DriverNumber"] == str(driver))]
    if accurate_only:
        laps = laps[laps["IsAccurate"].fillna(False).astype(bool)]
    for _, lap in laps.iterrows():
        drv, num = str(lap["Driver"]), str(lap["DriverNumber"])
        start, end = lap["LapStartTime"], lap["LapEndTime"]
        if pd.isna(lap["LapNumber"]) or pd.isna(start) or pd.isna(end):
            log.info("lap_skipped_no_window", driver=drv, lap=str(lap["LapNumber"]))
            continue
        if num not in pos_by_number or num not in car_by_number:
            log.info("lap_skipped_no_telemetry", driver=drv, lap=int(lap["LapNumber"]))
            continue
        yield DriverLap(
            driver=drv,
            driver_number=num,
            lap_number=int(lap["LapNumber"]),
            lap=lap,
            pos=lap_window(pos_by_number[num], float(start), float(end)),
            car=lap_window(car_by_number[num], float(start), float(end)),
        )


def fastest_laps(laps: pd.DataFrame) -> dict[str, float]:
    """Fastest lap time [s] per driver over timed laps not flagged ``Deleted``."""
    ok = laps[laps["LapTime"].notna() & ~laps["Deleted"].fillna(False).astype(bool)]
    return {str(d): float(t) for d, t in ok.groupby("Driver")["LapTime"].min().items()}


def write_session(
    paths: Paths,
    ident: SessionIdentity,
    laps: pd.DataFrame,
    driver_laps: Iterable[DriverLap],
    source: Source,
    source_version: str,
) -> tuple[cache.SessionMeta, list[RejectedLap]]:
    """Replace a session's processed export with validated tables and ``session_meta.json``.

    Laps whose telemetry violates its contract, or has no samples in the lap window, are not
    written; they are returned and listed in ``rejected_laps.json``.

    Raises:
        SchemaValidationError: The lap table violates ``LAPS`` (nothing is left on disk).
    """
    key = ident.key
    assert_valid(laps, LAPS, context=str(key))
    root = cache.session_dir(paths, key)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    written = [cache.laps_path(paths, key)]
    laps.to_parquet(written[0], index=False)
    rejected: list[RejectedLap] = []
    for dl in driver_laps:
        issues = [str(i) for i in (*validate_frame(dl.pos, POSITION), *validate_frame(dl.car, CAR))]
        for name, frame in (("position", dl.pos), ("car", dl.car)):
            if frame.empty:
                issues.append(f"{name}: empty: no samples in the lap window")
        if issues:
            rej = RejectedLap(driver=dl.driver, lap_number=dl.lap_number, issues=issues)
            log.warning("lap_rejected", driver=dl.driver, lap=dl.lap_number, issues=rej.issues)
            rejected.append(rej)
            continue
        for frame, path in (
            (dl.pos, cache.pos_path(paths, key, dl.driver, dl.lap_number)),
            (dl.car, cache.car_path(paths, key, dl.driver, dl.lap_number)),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            written.append(path)

    if rejected:
        rej_path = root / "rejected_laps.json"
        rej_path.write_text(
            json.dumps([r.model_dump() for r in rejected], indent=2) + "\n", encoding="utf-8"
        )
        written.append(rej_path)

    meta = cache.SessionMeta(
        source=source,
        source_version=source_version,
        apex_zero_version=__version__,
        fetched_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        year=key.year,
        event_name=ident.event_name,
        event_slug=key.event_slug,
        round_number=ident.round_number,
        country=ident.country,
        location=ident.location,
        session=key.session,
        session_name=ident.session_name,
        drivers=sorted(laps["Driver"].unique().tolist()),
        fastest_lap_s=fastest_laps(laps),
        n_laps=len(laps),
        files=[cache.file_record(root, p) for p in written],
    )
    cache.write_meta(paths, meta)
    log.info(
        "session_exported",
        key=str(key),
        source=source,
        laps=len(laps),
        files=len(written),
        rejected=len(rejected),
    )
    return meta, rejected


def remember_query(
    paths: Paths, source: Source, year: int, event: str, session: str, key: cache.SessionKey
) -> None:
    """Record which session a fetch query resolved to (for :func:`cached_session`)."""
    cache.alias_record(paths, source, year, event, session, key)


def cached_session(
    paths: Paths, source: Source, year: int, event: str, session: str
) -> FetchResult | None:
    """An intact earlier export of this query from ``source``, read without network access."""
    key = cache.alias_lookup(paths, source, year, event, session)
    if key is None or not cache.is_complete(paths, key):
        return None
    meta = cache.read_meta(paths, key)
    if meta is None or meta.source != source:
        return None
    log.info("fetch_cache_hit", key=str(key), source=source)
    return FetchResult(meta=meta, from_cache=True, rejected=_read_rejected(paths, key))


def _read_rejected(paths: Paths, key: cache.SessionKey) -> list[RejectedLap]:
    path = cache.session_dir(paths, key) / "rejected_laps.json"
    if not path.is_file():
        return []
    return [RejectedLap.model_validate(r) for r in json.loads(path.read_text(encoding="utf-8"))]
