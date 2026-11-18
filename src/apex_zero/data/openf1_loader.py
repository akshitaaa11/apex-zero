"""OpenF1 session loading and export to the processed cache.

OpenF1 is the primary data source (ADR 0002): it is free and needs no account, but covers 2023
onwards only. The export has the same layout and column contracts as the FastF1 path
(``data.schemas``), with these differences, all documented in ``docs/data.md``:

- ``SessionTime`` is seconds since the session's scheduled ``date_start`` (OpenF1 has no session
  clock of its own). Position and car samples keep their own timestamps.
- ``LapEndTime`` is ``LapStartTime + lap_duration`` for timed laps that are not pit-out laps.
  For pit-out laps and untimed laps it is the start of the driver's next lap (the same line
  crossing): OpenF1's ``lap_duration`` of a pit-out lap counts from the previous crossing, so it
  includes the time in the garage. ``LapTime`` is OpenF1's ``lap_duration`` as published
  (garage time included for pit-out laps, which are never ``IsAccurate``), missing if untimed.
- ``date_start`` is approximate: it can differ from the previous lap's start + duration by
  ~0.2 s. Lap windows are padded by one sample either side, which covers that.
- ``IsAccurate`` is derived: the lap is timed, is not a pit-out lap, and the next lap is not a
  pit-out lap (so it is not an in-lap). A driver's last lap counts as not accurate, because
  without the next lap an in-lap cannot be ruled out. FastF1's flag also runs timing-consistency
  checks that this does not.
- ``PitInTime``/``PitOutTime`` are missing (OpenF1 only flags pit-out laps); ``Deleted`` is
  missing (deletions are only in race-control messages, not parsed); ``Status`` is missing.
- ``Compound`` and ``TyreLife`` come from ``stints``; ``TyreLife`` counts the current lap,
  ``tyre_age_at_start + (lap_number - lap_start) + 1``, as FastF1's does.
- ``Brake`` is ``brake > 0`` (OpenF1 reports 0 or 100: still on/off only).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from apex_zero.config.paths import Paths, resolve_paths
from apex_zero.data import cache, processed
from apex_zero.data.openf1_client import JSONValue, OpenF1Client, OpenF1Session, Record
from apex_zero.data.processed import FetchResult, RejectedLap, SessionIdentity
from apex_zero.errors import DataUnavailableError
from apex_zero.utils.logging import get_logger
from apex_zero.utils.units import MPS_PER_KPH, OPENF1_POS_TO_M

__all__ = [
    "SOURCE_VERSION",
    "OpenF1SessionData",
    "convert_car_data",
    "convert_laps",
    "convert_location",
    "export_openf1",
    "fetch_openf1_session",
    "load_openf1_session",
]

log = get_logger(__name__)

#: Recorded as ``source_version`` (OpenF1 publishes no data version; this is the API path).
SOURCE_VERSION: str = "api-v1"


@dataclass(frozen=True)
class OpenF1SessionData:
    """A session from OpenF1, converted to the processed (SI) contracts."""

    session: OpenF1Session
    laps: pd.DataFrame
    pos: dict[str, pd.DataFrame] = field(repr=False)
    car: dict[str, pd.DataFrame] = field(repr=False)

    @property
    def identity(self) -> SessionIdentity:
        """Identity recorded in ``session_meta.json`` (OpenF1 has no round number)."""
        s = self.session
        return SessionIdentity(
            key=s.cache_key,
            event_name=s.meeting_name,
            round_number=None,
            country=s.country_name,
            location=s.location,
            session_name=s.session_name,
        )


def _t0(session: OpenF1Session) -> pd.Timestamp:
    t0 = pd.Timestamp(session.date_start)
    if t0.tzinfo is None:
        raise DataUnavailableError(f"OpenF1 session {session.session_key}: date_start has no TZ")
    return t0


def _seconds_since(dates: Sequence[JSONValue], t0: pd.Timestamp) -> np.ndarray:
    t = pd.to_datetime(pd.Series([str(d) if d is not None else None for d in dates]),
                       format="ISO8601", utc=True)
    return ((t - t0).dt.total_seconds()).to_numpy(dtype=np.float64)


def _num(records: Iterable[Record], key: str) -> np.ndarray:
    return pd.to_numeric(pd.Series([r.get(key) for r in records], dtype="object"),
                         errors="coerce").to_numpy(dtype=np.float64)


def convert_location(records: list[Record], t0: pd.Timestamp) -> pd.DataFrame:
    """OpenF1 ``location`` records -> ``POSITION`` contract, sorted by time.

    Args:
        records: Records with ``date`` and ``x, y, z`` [1/10 m].
        t0: Session start (``SessionTime`` origin).
    """
    out = pd.DataFrame(
        {
            "SessionTime": _seconds_since([r.get("date") for r in records], t0),
            "X": _num(records, "x") * OPENF1_POS_TO_M,
            "Y": _num(records, "y") * OPENF1_POS_TO_M,
            "Z": _num(records, "z") * OPENF1_POS_TO_M,
            "Status": pd.Series([None] * len(records), dtype="object"),
        }
    )
    return out.sort_values("SessionTime", kind="stable").reset_index(drop=True)


def convert_car_data(records: list[Record], t0: pd.Timestamp) -> pd.DataFrame:
    """OpenF1 ``car_data`` records -> ``CAR`` contract, sorted by time.

    Args:
        records: Records with ``date``, ``speed`` [km/h], ``rpm``, ``n_gear``, ``throttle`` [%],
            ``brake`` (0/100), ``drs`` (code).
        t0: Session start (``SessionTime`` origin).
    """
    brake = _num(records, "brake")
    out = pd.DataFrame(
        {
            "SessionTime": _seconds_since([r.get("date") for r in records], t0),
            "Speed": _num(records, "speed") * MPS_PER_KPH,
            "RPM": _num(records, "rpm"),
            "nGear": pd.array(_num(records, "n_gear"), dtype="Float64").astype("Int64"),
            "Throttle": _num(records, "throttle") / 100.0,
            "Brake": pd.array([None if np.isnan(b) else bool(b > 0) for b in brake],
                              dtype="boolean"),
            "DRS": pd.array(_num(records, "drs"), dtype="Float64").astype("Int64"),
        }
    )
    return out.sort_values("SessionTime", kind="stable").reset_index(drop=True)


def _opt_float(v: JSONValue) -> float:
    return float(v) if isinstance(v, int | float) and not isinstance(v, bool) else np.nan


def convert_laps(
    laps: list[Record], drivers: list[Record], stints: list[Record], t0: pd.Timestamp
) -> pd.DataFrame:
    """OpenF1 ``laps`` + ``drivers`` + ``stints`` -> ``LAPS`` contract (see module docstring).

    Raises:
        DataUnavailableError: A lap's driver number has no ``drivers`` record.
    """
    codes = {int(str(d["driver_number"])): str(d["name_acronym"]) for d in drivers
             if d.get("driver_number") is not None and d.get("name_acronym")}
    rows: list[dict[str, object]] = []
    by_driver: dict[int, list[Record]] = {}
    for lap in laps:
        by_driver.setdefault(int(str(lap["driver_number"])), []).append(lap)

    for num, dl in sorted(by_driver.items()):
        if num not in codes:
            raise DataUnavailableError(f"OpenF1: no driver record for car #{num}")
        dl = sorted(dl, key=lambda r: int(str(r["lap_number"])))
        starts = _seconds_since([r.get("date_start") for r in dl], t0)
        my_stints = [s for s in stints if s.get("driver_number") == num]
        for i, lap in enumerate(dl):
            n = int(str(lap["lap_number"]))
            duration = _opt_float(lap.get("lap_duration"))
            nxt = dl[i + 1] if i + 1 < len(dl) else None
            out_lap = bool(lap.get("is_pit_out_lap"))
            next_start = (
                starts[i + 1]
                if nxt is not None and int(str(nxt["lap_number"])) == n + 1
                else np.nan
            )
            # Timed flying laps: start + official duration (date_start is only approximate).
            # Out-laps: OpenF1's lap_duration includes garage time, so use the next crossing.
            if not np.isnan(duration) and not out_lap:
                end = starts[i] + duration
            elif not np.isnan(next_start):
                end = next_start
            else:
                end = starts[i] + duration
            in_lap = nxt is None or bool(nxt.get("is_pit_out_lap"))
            compound, tyre_life = _stint_info(my_stints, n)
            rows.append(
                {
                    "Driver": codes[num],
                    "DriverNumber": str(num),
                    "LapNumber": n,
                    "LapStartTime": starts[i],
                    "LapEndTime": end,
                    "LapTime": duration,
                    "Sector1Time": _opt_float(lap.get("duration_sector_1")),
                    "Sector2Time": _opt_float(lap.get("duration_sector_2")),
                    "Sector3Time": _opt_float(lap.get("duration_sector_3")),
                    "PitInTime": np.nan,
                    "PitOutTime": np.nan,
                    "Compound": compound,
                    "TyreLife": tyre_life,
                    "IsAccurate": not np.isnan(duration) and not out_lap and not in_lap,
                    "Deleted": pd.NA,
                }
            )
    df = pd.DataFrame(rows, columns=[
        "Driver", "DriverNumber", "LapNumber", "LapStartTime", "LapEndTime", "LapTime",
        "Sector1Time", "Sector2Time", "Sector3Time", "PitInTime", "PitOutTime", "Compound",
        "TyreLife", "IsAccurate", "Deleted",
    ])
    floats = ["LapStartTime", "LapEndTime", "LapTime", "Sector1Time", "Sector2Time",
              "Sector3Time", "PitInTime", "PitOutTime", "TyreLife"]
    df[floats] = df[floats].astype("float64")
    df["Driver"] = df["Driver"].astype("object")
    df["DriverNumber"] = df["DriverNumber"].astype("object")
    df["Compound"] = df["Compound"].astype("object")
    df["LapNumber"] = df["LapNumber"].astype("Int64")
    df["IsAccurate"] = df["IsAccurate"].astype("boolean")
    df["Deleted"] = df["Deleted"].astype("boolean")
    return df.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)


def _stint_info(stints: list[Record], lap_number: int) -> tuple[str | None, float]:
    for s in stints:
        start, end = s.get("lap_start"), s.get("lap_end")
        if isinstance(start, int) and isinstance(end, int) and start <= lap_number <= end:
            compound = s.get("compound")
            age = s.get("tyre_age_at_start")
            life = float(age) + (lap_number - start) + 1 if isinstance(age, int) else np.nan
            return (compound if isinstance(compound, str) and compound else None), life
    return None, np.nan


def load_openf1_session(
    client: OpenF1Client, year: int, event: str, session: str
) -> OpenF1SessionData:
    """Resolve a session on OpenF1 and download laps, drivers, stints and all telemetry.

    Telemetry is requested once per car for the whole session (about 17k samples per channel
    for a one-hour session) and sliced per lap at export.

    Raises:
        DataUnavailableError: Unknown event/session or a session without laps.
    """
    sess = client.find_session(year, event, session)
    t0 = _t0(sess)
    laps_raw = client.laps(sess.session_key)
    if not laps_raw:
        raise DataUnavailableError(f"OpenF1: {sess.cache_key} has no laps")
    laps = convert_laps(laps_raw, client.drivers(sess.session_key),
                        client.stints(sess.session_key), t0)
    pos: dict[str, pd.DataFrame] = {}
    car: dict[str, pd.DataFrame] = {}
    for num in sorted(laps["DriverNumber"].unique(), key=int):
        log.info("openf1_telemetry", session_key=sess.session_key, driver_number=num)
        loc = client.location(sess.session_key, int(num))
        cd = client.car_data(sess.session_key, int(num))
        if loc and cd:
            pos[num] = convert_location(loc, t0)
            car[num] = convert_car_data(cd, t0)
    return OpenF1SessionData(session=sess, laps=laps, pos=pos, car=car)


def export_openf1(
    data: OpenF1SessionData, paths: Paths | None = None
) -> tuple[cache.SessionMeta, list[RejectedLap]]:
    """Write a loaded OpenF1 session to the processed cache (replacing any earlier export)."""
    paths = resolve_paths() if paths is None else paths
    return processed.write_session(
        paths,
        data.identity,
        data.laps,
        processed.iter_laps(data.laps, data.pos, data.car),
        source="openf1",
        source_version=SOURCE_VERSION,
    )


def fetch_openf1_session(
    year: int,
    event: str,
    session: str,
    *,
    paths: Paths | None = None,
    force: bool = False,
    client: OpenF1Client | None = None,
) -> FetchResult:
    """Fetch and export a session from OpenF1, or return an intact earlier export offline.

    A cache hit makes no HTTP request and does not even create a client. On a miss, OpenF1
    responses still come from the client's disk cache where possible.
    """
    paths = resolve_paths() if paths is None else paths
    if not force:
        hit = processed.cached_session(paths, "openf1", year, event, session)
        if hit is not None:
            return hit
    own = client is None
    c = OpenF1Client(cache.openf1_cache_dir(paths)) if client is None else client
    try:
        data = load_openf1_session(c, year, event, session)
    finally:
        if own:
            c.close()
    meta, rejected = export_openf1(data, paths)
    processed.remember_query(paths, "openf1", year, event, session, data.session.cache_key)
    return FetchResult(meta=meta, from_cache=False, rejected=rejected)
