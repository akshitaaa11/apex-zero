from __future__ import annotations

import copy
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest

from apex_zero.config.paths import Paths
from apex_zero.data import cache, openf1_loader
from apex_zero.data.openf1_client import OpenF1Client
from apex_zero.data.openf1_loader import (
    convert_car_data,
    convert_laps,
    convert_location,
    fetch_openf1_session,
)
from apex_zero.data.schemas import CAR, LAPS, POSITION, validate_frame
from apex_zero.errors import DataUnavailableError

T0 = pd.Timestamp("2024-04-06T06:00:00+00:00")

#: VER's real lap records from 2024 Suzuka Q (session_key 9492), sectors trimmed.
VER_LAPS: list[dict[str, Any]] = [
    {"lap_number": 1, "date_start": "2024-04-06T06:04:13.393000+00:00", "lap_duration": None,
     "is_pit_out_lap": True},
    {"lap_number": 2, "date_start": "2024-04-06T06:06:13.988000+00:00", "lap_duration": 88.866,
     "is_pit_out_lap": False},
    {"lap_number": 3, "date_start": "2024-04-06T06:07:42.876000+00:00",
     "lap_duration": 121.685, "is_pit_out_lap": False},
    {"lap_number": 4, "date_start": "2024-04-06T06:25:21.407000+00:00",
     "lap_duration": 1054.112, "is_pit_out_lap": True},
    {"lap_number": 5, "date_start": "2024-04-06T06:27:18.568000+00:00", "lap_duration": 88.74,
     "is_pit_out_lap": False},
    {"lap_number": 10, "date_start": "2024-04-06T06:56:15.555000+00:00",
     "lap_duration": 276.133, "is_pit_out_lap": True},
    {"lap_number": 11, "date_start": "2024-04-06T06:58:14.889000+00:00", "lap_duration": 88.197,
     "duration_sector_1": 30.777, "duration_sector_2": 39.85, "duration_sector_3": 17.57,
     "is_pit_out_lap": False},
    {"lap_number": 12, "date_start": "2024-04-06T06:59:42.925000+00:00", "lap_duration": 119.61,
     "is_pit_out_lap": False},
]
DRIVERS = [{"driver_number": 1, "name_acronym": "VER"}, {"driver_number": 11,
                                                        "name_acronym": "PER"}]
STINTS = [
    {"driver_number": 1, "stint_number": 1, "lap_start": 1, "lap_end": 3, "compound": "SOFT",
     "tyre_age_at_start": 0},
    {"driver_number": 1, "stint_number": 2, "lap_start": 4, "lap_end": 9, "compound": "SOFT",
     "tyre_age_at_start": 3},
    {"driver_number": 1, "stint_number": 3, "lap_start": 10, "lap_end": 12, "compound": "SOFT",
     "tyre_age_at_start": 0},
]


def records() -> list[dict[str, Any]]:
    return [{**r, "driver_number": 1, "session_key": 9492} for r in copy.deepcopy(VER_LAPS)]


def t(ts: str) -> float:
    return (pd.Timestamp(ts) - T0).total_seconds()


# --- conversions ------------------------------------------------------------------------------


def test_convert_location_real_lap(real_lap: dict[str, Any]) -> None:
    loc = list(reversed(real_lap["location"]))  # order must not matter
    df = convert_location(loc, T0)
    assert validate_frame(df, POSITION) == []
    assert df["SessionTime"].is_monotonic_increasing
    first = real_lap["location"][0]
    assert df["SessionTime"].iloc[0] == pytest.approx(t(first["date"]))
    assert df["X"].iloc[0] == pytest.approx(first["x"] * 0.1)
    assert df["Status"].isna().all()


def test_convert_car_data_real_lap(real_lap: dict[str, Any]) -> None:
    df = convert_car_data(real_lap["car_data"], T0)
    assert validate_frame(df, CAR) == []
    assert 80 < df["Speed"].max() < 95  # VER's pole lap peaks at ~324 km/h
    assert set(df["Brake"].unique()) == {True, False}
    raw_brake = {r["brake"] for r in real_lap["car_data"]}
    assert raw_brake <= {0, 100}
    assert df["Throttle"].max() <= 1.05


def test_convert_car_data_missing_values() -> None:
    recs = [{"date": "2024-04-06T06:00:01+00:00", "speed": None, "rpm": 1, "n_gear": None,
             "throttle": 50, "brake": None, "drs": 8}]
    df = convert_car_data(recs, T0)
    assert df["nGear"].isna().all() and df["Brake"].isna().all() and df["Speed"].isna().all()


def test_convert_laps_rules() -> None:
    laps = convert_laps(records(), DRIVERS, STINTS, T0)
    assert validate_frame(laps, LAPS) == []
    by = laps.set_index("LapNumber")
    starts = {r["lap_number"]: t(r["date_start"]) for r in VER_LAPS}
    # untimed out-lap and pit-out laps (whose duration includes garage time): next crossing
    assert by.loc[1, "LapEndTime"] == pytest.approx(starts[2])
    assert by.loc[4, "LapEndTime"] == pytest.approx(starts[5])
    assert by.loc[10, "LapEndTime"] == pytest.approx(starts[11])
    assert by.loc[4, "LapTime"] == pytest.approx(1054.112)  # published value kept
    # timed laps: start + official duration (not the approximate next date_start)
    assert by.loc[11, "LapEndTime"] == pytest.approx(starts[11] + 88.197)
    # accuracy: timed, not out-lap, not followed by an out-lap, not the last lap
    accurate = by.index[by["IsAccurate"].astype(bool)].tolist()
    assert accurate == [2, 11]
    assert pd.isna(by.loc[1, "LapTime"])
    # stints
    assert by.loc[5, "TyreLife"] == 5.0  # age 3 at lap 4, +1 lap, counting the current one
    assert (by["Compound"] == "SOFT").all()
    assert by["Deleted"].isna().all() and by["PitInTime"].isna().all()
    assert by.loc[11, "Sector2Time"] == pytest.approx(39.85)
    assert laps["Driver"].unique().tolist() == ["VER"]


def test_convert_laps_without_stint_and_unknown_driver() -> None:
    laps = convert_laps(records()[:2], DRIVERS, [], T0)
    assert laps["Compound"].isna().all() and laps["TyreLife"].isna().all()
    bad = records()[:1]
    bad[0]["driver_number"] = 99
    with pytest.raises(DataUnavailableError, match="#99"):
        convert_laps(bad, DRIVERS, STINTS, T0)


# --- end to end against a fake OpenF1 ---------------------------------------------------------


class FakeOpenF1:
    """Serves a two-driver session: VER's real lap 11 telemetry; PER has laps but no telemetry."""

    def __init__(self, real_lap: dict[str, Any], laps: list[dict[str, Any]] | None = None) -> None:
        per = [{"driver_number": 11, "session_key": 9492, "lap_number": 1,
                "date_start": "2024-04-06T06:10:00+00:00", "lap_duration": 90.0,
                "is_pit_out_lap": False}]
        self.data: dict[str, list[dict[str, Any]]] = {
            "meetings": [{"meeting_key": 1232, "meeting_name": "Japanese Grand Prix",
                          "meeting_official_name": "X", "country_name": "Japan",
                          "location": "Suzuka", "circuit_short_name": "Suzuka", "year": 2024}],
            "sessions": [{"session_key": 9492, "session_name": "Qualifying", "meeting_key": 1232,
                          "year": 2024, "circuit_short_name": "Suzuka", "country_name": "Japan",
                          "location": "Suzuka", "date_start": "2024-04-06T06:00:00+00:00"}],
            "drivers": [{**d, "session_key": 9492} for d in DRIVERS],
            "laps": records() + per if laps is None else laps,
            "stints": [{**s, "session_key": 9492} for s in STINTS],
            "location": real_lap["location"],
            "car_data": real_lap["car_data"],
        }
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        rows = self.data[request.url.path.rsplit("/", 1)[-1]]
        for k, v in request.url.params.items():
            rows = [r for r in rows if str(r.get(k)) == v]
        if not rows:
            return httpx.Response(404, json={"detail": "No results found."})
        return httpx.Response(200, json=rows)


def make_client(paths: Paths, api: FakeOpenF1) -> OpenF1Client:
    return OpenF1Client(cache.openf1_cache_dir(paths), transport=httpx.MockTransport(api),
                        sleep=lambda s: None)


def test_fetch_exports_validated_session(
    paths: Paths, real_lap: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    api = FakeOpenF1(real_lap)
    with make_client(paths, api) as client:
        result = fetch_openf1_session(2024, "Japan", "Q", paths=paths, client=client)
    meta = result.meta
    assert not result.from_cache
    assert (meta.source, meta.source_version, meta.round_number) == ("openf1", "api-v1", None)
    assert meta.key == cache.SessionKey(2024, "japanese_grand_prix", "Q")
    assert meta.drivers == ["PER", "VER"] and meta.fastest_lap_s["VER"] == pytest.approx(88.197)
    assert cache.is_complete(paths, meta.key)

    pos = pd.read_parquet(cache.pos_path(paths, meta.key, "VER", 11))
    car = pd.read_parquet(cache.car_path(paths, meta.key, "VER", 11))
    assert validate_frame(pos, POSITION) == [] and validate_frame(car, CAR) == []
    xy = np.vstack([pos[["X", "Y"]].to_numpy(), pos[["X", "Y"]].to_numpy()[:1]])
    assert 5700 < np.linalg.norm(np.diff(xy, axis=0), axis=1).sum() < 5900  # metres

    # VER's other laps fall outside the fixture's telemetry; PER has none at all
    rejected = {(r.driver, r.lap_number) for r in result.rejected}
    assert ("VER", 2) in rejected and ("VER", 11) not in rejected
    assert all("empty" in r.issues[0] for r in result.rejected if r.driver == "VER")
    assert not cache.pos_path(paths, meta.key, "PER", 1).exists()

    # Gate: the second fetch makes zero network calls (no client is even created)
    calls = api.calls

    def no_client(*args: object, **kwargs: object) -> None:
        raise AssertionError("must not create an OpenF1 client for a cached session")

    monkeypatch.setattr(openf1_loader, "OpenF1Client", no_client)
    again = fetch_openf1_session(2024, "japan", "qualifying", paths=paths)
    assert again.from_cache and again.meta == meta and again.rejected == result.rejected
    assert api.calls == calls


def test_forced_refetch_reuses_response_cache(paths: Paths, real_lap: dict[str, Any]) -> None:
    api = FakeOpenF1(real_lap)
    with make_client(paths, api) as c:
        fetch_openf1_session(2024, "Japan", "Q", paths=paths, client=c)
    calls = api.calls
    with make_client(paths, api) as c:
        again = fetch_openf1_session(2024, "Japan", "Q", paths=paths, client=c, force=True)
    assert not again.from_cache
    assert api.calls == calls  # every OpenF1 response came from the disk cache


def test_session_without_laps(paths: Paths, real_lap: dict[str, Any]) -> None:
    with make_client(paths, FakeOpenF1(real_lap, laps=[])) as c, pytest.raises(
        DataUnavailableError, match="no laps"
    ):
        fetch_openf1_session(2024, "Japan", "Q", paths=paths, client=c)


def test_fetch_creates_and_closes_own_client(
    paths: Paths, real_lap: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    api = FakeOpenF1(real_lap)
    made: list[OpenF1Client] = []

    def factory(cache_dir: Any) -> OpenF1Client:
        made.append(OpenF1Client(cache_dir, transport=httpx.MockTransport(api),
                                 sleep=lambda s: None))
        return made[-1]

    monkeypatch.setattr(openf1_loader, "OpenF1Client", factory)
    assert not fetch_openf1_session(2024, "Japan", "Q", paths=paths).from_cache
    assert len(made) == 1 and made[0]._http.is_closed
