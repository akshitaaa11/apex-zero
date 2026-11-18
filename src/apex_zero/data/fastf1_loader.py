"""FastF1 session loading and export to the processed cache (P1.2).

Written against FastF1 3.9.x (checked with ``inspect`` on the installed version):

- ``fastf1.get_event_schedule(year, include_testing=False)`` -> ``EventSchedule`` with
  ``get_event_by_round`` / ``get_event_by_name``; ``Event.get_session(name)``.
- ``Session.load(laps=, telemetry=, weather=, messages=)``; ``Session.laps`` (``Laps`` frame:
  ``Driver``, ``DriverNumber``, ``LapNumber``, ``LapStartTime``, ``Time`` (lap end), ``LapTime``,
  ``Sector1Time``..``Sector3Time``, ``PitInTime``, ``PitOutTime``, ``Compound``, ``TyreLife``,
  ``IsAccurate``, ``Deleted``, timedeltas for times).
- ``Session.pos_data`` / ``Session.car_data``: ``dict[driver_number, Telemetry]`` of raw
  samples with ``SessionTime`` (timedelta); position ``X, Y, Z`` [1/10 m] and ``Status``; car
  ``Speed`` [km/h], ``RPM``, ``nGear``, ``Throttle`` [%], ``Brake`` (bool), ``DRS`` (int code).

Since FastF1 3.9, live-timing data requires a Formula 1 account with an F1TV subscription
(``F1_USERNAME``/``F1_PASSWORD``, see FastF1 docs). Without it, ``Session.load`` gets HTTP 401 and
loads no laps; :func:`load_session` turns that into :class:`DataUnavailableError`.

Telemetry is stored **raw** (no resampling or interpolation): each lap file holds the samples in
``[LapStartTime, LapEndTime]`` plus one sample either side, so the exact line crossing can be
interpolated later (ghost extraction). Car and position data are kept separate (merging is a
ghost-extraction concern). Laps are not filtered at ingest; flags (``IsAccurate``, ``Deleted``)
are kept for downstream filtering.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import fastf1
import pandas as pd
from pydantic import BaseModel, ConfigDict

from apex_zero import __version__
from apex_zero.config.paths import Paths, resolve_paths
from apex_zero.data import cache
from apex_zero.data.schemas import CAR, LAPS, POSITION, assert_valid, validate_frame
from apex_zero.errors import DataUnavailableError
from apex_zero.utils.logging import get_logger
from apex_zero.utils.units import FASTF1_POS_TO_M, MPS_PER_KPH

__all__ = [
    "DriverLap",
    "FastF1Session",
    "FetchResult",
    "RejectedLap",
    "SessionHandle",
    "convert_car",
    "convert_laps",
    "convert_pos",
    "export_processed",
    "fetch_session",
    "iter_driver_laps",
    "load_session",
]

log = get_logger(__name__)

#: Raw samples kept on each side of the lap window.
PAD_SAMPLES: int = 1


class FastF1Session(Protocol):
    """The subset of ``fastf1.core.Session`` this module uses (also implemented by test fakes)."""

    @property
    def name(self) -> str: ...
    @property
    def event(self) -> pd.Series: ...
    @property
    def laps(self) -> pd.DataFrame: ...
    @property
    def pos_data(self) -> Mapping[str, pd.DataFrame]: ...
    @property
    def car_data(self) -> Mapping[str, pd.DataFrame]: ...


@dataclass(frozen=True, slots=True)
class SessionHandle:
    """A loaded session plus its cache key and identity."""

    key: cache.SessionKey
    session: FastF1Session
    event_name: str
    round_number: int
    country: str
    location: str
    session_name: str


@dataclass(frozen=True, slots=True)
class DriverLap:
    """One driver lap: lap-table row and raw telemetry, already converted to SI."""

    driver: str
    driver_number: str
    lap_number: int
    lap: pd.Series
    pos: pd.DataFrame = field(repr=False)
    car: pd.DataFrame = field(repr=False)


class RejectedLap(BaseModel):
    """A lap whose telemetry failed validation and was not written."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    driver: str
    lap_number: int
    issues: list[str]


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Outcome of :func:`fetch_session`."""

    meta: cache.SessionMeta
    from_cache: bool
    rejected: list[RejectedLap]


# --------------------------------------------------------------------------------------------
# Conversion to the processed (SI) contracts
# --------------------------------------------------------------------------------------------


def _seconds(col: pd.Series) -> pd.Series:
    """Timedelta (or numeric seconds) -> float64 seconds, NaT -> NaN."""
    if pd.api.types.is_timedelta64_dtype(col.dtype):
        out: pd.Series = col.dt.total_seconds().astype("float64")
        return out
    return pd.to_numeric(col, errors="coerce").astype("float64")


def _str_or_na(col: pd.Series) -> pd.Series:
    return col.map(lambda v: v if isinstance(v, str) and v != "" else None).astype("object")


def convert_pos(raw: pd.DataFrame) -> pd.DataFrame:
    """FastF1 position telemetry -> ``POSITION`` contract (SI).

    Args:
        raw: Frame with ``SessionTime`` (timedelta), ``X, Y, Z`` [1/10 m], ``Status``.

    Returns:
        ``SessionTime`` [s], ``X, Y, Z`` [m], ``Status`` (str), index reset.
    """
    out = pd.DataFrame(
        {
            "SessionTime": _seconds(raw["SessionTime"]),
            "X": pd.to_numeric(raw["X"], errors="coerce").astype("float64") * FASTF1_POS_TO_M,
            "Y": pd.to_numeric(raw["Y"], errors="coerce").astype("float64") * FASTF1_POS_TO_M,
            "Z": pd.to_numeric(raw["Z"], errors="coerce").astype("float64") * FASTF1_POS_TO_M,
            "Status": _str_or_na(raw["Status"]),
        }
    )
    return out.reset_index(drop=True)


def convert_car(raw: pd.DataFrame) -> pd.DataFrame:
    """FastF1 car telemetry -> ``CAR`` contract (SI).

    Args:
        raw: Frame with ``SessionTime`` (timedelta), ``Speed`` [km/h], ``RPM``, ``nGear``,
            ``Throttle`` [%], ``Brake`` (bool), ``DRS`` (int code).

    Returns:
        ``SessionTime`` [s], ``Speed`` [m/s], ``RPM``, ``nGear`` (Int64), ``Throttle``
        [fraction], ``Brake`` (boolean), ``DRS`` (Int64), index reset.
    """
    speed_kph = pd.to_numeric(raw["Speed"], errors="coerce").astype("float64")
    out = pd.DataFrame(
        {
            "SessionTime": _seconds(raw["SessionTime"]),
            "Speed": speed_kph * MPS_PER_KPH,
            "RPM": pd.to_numeric(raw["RPM"], errors="coerce").astype("float64"),
            "nGear": pd.to_numeric(raw["nGear"], errors="coerce").astype("Int64"),
            "Throttle": pd.to_numeric(raw["Throttle"], errors="coerce").astype("float64") / 100.0,
            "Brake": raw["Brake"].astype("boolean"),
            "DRS": pd.to_numeric(raw["DRS"], errors="coerce").astype("Int64"),
        }
    )
    return out.reset_index(drop=True)


def convert_laps(raw: pd.DataFrame) -> pd.DataFrame:
    """FastF1 ``Laps`` -> ``LAPS`` contract (SI), sorted by driver and lap number.

    ``Deleted`` stays missing where FastF1 could not determine it (no race-control messages).
    """
    seconds = {
        "LapStartTime": "LapStartTime",
        "LapEndTime": "Time",
        "LapTime": "LapTime",
        "Sector1Time": "Sector1Time",
        "Sector2Time": "Sector2Time",
        "Sector3Time": "Sector3Time",
        "PitInTime": "PitInTime",
        "PitOutTime": "PitOutTime",
    }
    out = pd.DataFrame(
        {
            "Driver": raw["Driver"].astype(str).astype("object"),
            "DriverNumber": raw["DriverNumber"].astype(str).astype("object"),
            "LapNumber": pd.to_numeric(raw["LapNumber"], errors="coerce").astype("Int64"),
            **{dst: _seconds(raw[src]) for dst, src in seconds.items()},
            "Compound": _str_or_na(raw["Compound"]),
            "TyreLife": pd.to_numeric(raw["TyreLife"], errors="coerce").astype("float64"),
            "IsAccurate": raw["IsAccurate"].astype("boolean"),
            "Deleted": raw["Deleted"].astype("boolean"),
        }
    )
    return out.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)


def _window(telemetry: pd.DataFrame, start_s: float, end_s: float) -> pd.DataFrame:
    """Raw samples in ``[start_s, end_s]`` plus ``PAD_SAMPLES`` either side."""
    t = _seconds(telemetry["SessionTime"]).to_numpy()
    inside = (t >= start_s) & (t <= end_s)
    if not inside.any():
        return telemetry.iloc[0:0]
    idx = inside.nonzero()[0]
    lo = max(int(idx[0]) - PAD_SAMPLES, 0)
    hi = min(int(idx[-1]) + PAD_SAMPLES + 1, len(telemetry))
    return telemetry.iloc[lo:hi]


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------


def load_session(
    year: int, event: str | int, session: str, *, paths: Paths | None = None
) -> SessionHandle:
    """Resolve an event via the FastF1 schedule, enable the cache and load the session.

    Args:
        year: Championship year (FastF1 telemetry exists from 2018).
        event: Round number, or a name FastF1 can match (``"Japan"``, ``"Suzuka"``).
        session: Session code or name (``"Q"``, ``"Qualifying"``).
        paths: Project paths; defaults to the environment.

    Raises:
        DataUnavailableError: Unknown event/session, or the session loaded without laps or
            telemetry (e.g. live-timing authentication failed).
    """
    paths = resolve_paths() if paths is None else paths
    code = cache.normalize_session(session)
    ff1_dir = cache.fastf1_cache_dir(paths)
    ff1_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(ff1_dir))

    schedule = fastf1.get_event_schedule(year, include_testing=False)
    try:
        if isinstance(event, int) or str(event).strip().isdigit():
            ev = schedule.get_event_by_round(int(event))
        else:
            ev = schedule.get_event_by_name(str(event))
    except (ValueError, KeyError) as exc:
        raise DataUnavailableError(f"no {year} event matching {event!r}: {exc}") from exc
    if ev is None or len(ev) == 0:
        raise DataUnavailableError(f"no {year} event matching {event!r}")
    try:
        sess = ev.get_session(cache.SESSION_NAMES[code])
    except ValueError as exc:
        raise DataUnavailableError(f"{year} {ev['EventName']}: no session {code}") from exc

    log.info("fastf1_load", year=year, event_name=str(ev["EventName"]), session=code)
    sess.load(laps=True, telemetry=True, weather=False, messages=True)
    handle = _handle_from_session(year, code, sess)
    _require_data(handle)
    return handle


def _handle_from_session(year: int, code: str, sess: FastF1Session) -> SessionHandle:
    ev = sess.event
    name = str(ev["EventName"])
    return SessionHandle(
        key=cache.SessionKey(year, cache.slugify(name), code),
        session=sess,
        event_name=name,
        round_number=int(ev["RoundNumber"]),
        country=str(ev["Country"]),
        location=str(ev["Location"]),
        session_name=str(sess.name),
    )


def _require_data(handle: SessionHandle) -> None:
    hint = (
        "FastF1 >= 3.9 needs an F1 account with F1TV (F1_USERNAME/F1_PASSWORD) for live-timing "
        "data; check the FastF1 log above for HTTP 401."
    )
    try:
        laps = handle.session.laps
        pos = handle.session.pos_data
    except Exception as exc:  # FastF1 raises DataNotLoadedError when nothing loaded
        raise DataUnavailableError(f"{handle.key}: session data not loaded. {hint}") from exc
    if len(laps) == 0 or len(pos) == 0:
        raise DataUnavailableError(f"{handle.key}: no laps or position data. {hint}")


def iter_driver_laps(
    handle: SessionHandle, driver: str | None = None, accurate_only: bool = False
) -> Iterator[DriverLap]:
    """Yield driver laps with SI telemetry, sorted by driver and lap number.

    Laps without a defined time window (no start or end time) are skipped, with a log entry.

    Args:
        handle: Loaded session.
        driver: Three-letter code or car number; ``None`` for all drivers.
        accurate_only: Only laps FastF1 flags ``IsAccurate``.
    """
    laps = convert_laps(handle.session.laps)
    if driver is not None:
        laps = laps[(laps["Driver"] == driver) | (laps["DriverNumber"] == str(driver))]
    if accurate_only:
        laps = laps[laps["IsAccurate"].fillna(False).astype(bool)]
    pos_all = handle.session.pos_data
    car_all = handle.session.car_data
    for _, lap in laps.iterrows():
        drv, num = str(lap["Driver"]), str(lap["DriverNumber"])
        start, end = lap["LapStartTime"], lap["LapEndTime"]
        if pd.isna(lap["LapNumber"]) or pd.isna(start) or pd.isna(end):
            log.info("lap_skipped_no_window", driver=drv, lap=str(lap["LapNumber"]))
            continue
        if num not in pos_all or num not in car_all:
            log.info("lap_skipped_no_telemetry", driver=drv, lap=int(lap["LapNumber"]))
            continue
        yield DriverLap(
            driver=drv,
            driver_number=num,
            lap_number=int(lap["LapNumber"]),
            lap=lap,
            pos=convert_pos(_window(pos_all[num], float(start), float(end))),
            car=convert_car(_window(car_all[num], float(start), float(end))),
        )


# --------------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------------


def _fastest(laps: pd.DataFrame) -> dict[str, float]:
    ok = laps[laps["LapTime"].notna() & ~laps["Deleted"].fillna(False).astype(bool)]
    return {str(d): float(t) for d, t in ok.groupby("Driver")["LapTime"].min().items()}


def export_processed(
    handle: SessionHandle, paths: Paths | None = None
) -> tuple[cache.SessionMeta, list[RejectedLap]]:
    """Write validated Parquet tables and ``session_meta.json`` for every driver lap.

    Any previous export of the session is replaced. Laps whose telemetry violates its contract
    are not written; they are returned (and logged). The lap table must be valid.

    Raises:
        SchemaValidationError: The lap table is invalid.
    """
    paths = resolve_paths() if paths is None else paths
    root = cache.session_dir(paths, handle.key)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    laps = convert_laps(handle.session.laps)
    assert_valid(laps, LAPS, context=str(handle.key))
    written: list[Path] = [cache.laps_path(paths, handle.key)]
    laps.to_parquet(written[0], index=False)

    rejected: list[RejectedLap] = []
    for dl in iter_driver_laps(handle):
        issues = [*validate_frame(dl.pos, POSITION), *validate_frame(dl.car, CAR)]
        if issues:
            rej = RejectedLap(
                driver=dl.driver, lap_number=dl.lap_number, issues=[str(i) for i in issues]
            )
            log.warning("lap_rejected", driver=dl.driver, lap=dl.lap_number, issues=rej.issues)
            rejected.append(rej)
            continue
        for frame, path in (
            (dl.pos, cache.pos_path(paths, handle.key, dl.driver, dl.lap_number)),
            (dl.car, cache.car_path(paths, handle.key, dl.driver, dl.lap_number)),
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
        source="fastf1",
        source_version=str(fastf1.__version__),
        apex_zero_version=__version__,
        fetched_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        year=handle.key.year,
        event_name=handle.event_name,
        event_slug=handle.key.event_slug,
        round_number=handle.round_number,
        country=handle.country,
        location=handle.location,
        session=handle.key.session,
        session_name=handle.session_name,
        drivers=sorted(laps["Driver"].unique().tolist()),
        fastest_lap_s=_fastest(laps),
        n_laps=len(laps),
        files=[cache.file_record(root, p) for p in written],
    )
    cache.write_meta(paths, meta)
    log.info(
        "session_exported", key=str(handle.key), laps=len(laps), files=len(written),
        rejected=len(rejected),
    )
    return meta, rejected


def fetch_session(
    year: int,
    event: str | int,
    session: str,
    *,
    paths: Paths | None = None,
    force: bool = False,
) -> FetchResult:
    """Load and export a session, or return it from the processed cache without network calls.

    A session that was fetched before with the same (year, event, session) query and is intact
    (hashes match) is returned straight from disk; no schedule lookup, no FastF1 call.
    """
    paths = resolve_paths() if paths is None else paths
    if not force:
        key = cache.alias_lookup(paths, year, str(event), session)
        if key is not None and cache.is_complete(paths, key):
            meta = cache.read_meta(paths, key)
            assert meta is not None
            log.info("fetch_cache_hit", key=str(key))
            return FetchResult(meta=meta, from_cache=True, rejected=_read_rejected(paths, key))
    handle = load_session(year, event, session, paths=paths)
    meta, rejected = export_processed(handle, paths)
    cache.alias_record(paths, year, str(event), session, handle.key)
    return FetchResult(meta=meta, from_cache=False, rejected=rejected)


def _read_rejected(paths: Paths, key: cache.SessionKey) -> list[RejectedLap]:
    path = cache.session_dir(paths, key) / "rejected_laps.json"
    if not path.is_file():
        return []
    return [RejectedLap.model_validate(r) for r in json.loads(path.read_text(encoding="utf-8"))]
