from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from apex_zero.config.paths import Paths
from apex_zero.data import cache, fastf1_loader
from apex_zero.data.fastf1_loader import (
    SessionHandle,
    convert_car,
    convert_laps,
    convert_pos,
    export_processed,
    fetch_session,
    iter_driver_laps,
    load_session,
)
from apex_zero.data.schemas import CAR, LAPS, POSITION, validate_frame
from apex_zero.errors import DataUnavailableError, SchemaValidationError
from tests.fixtures.synthetic import FakeSession


class FakeEvent:
    def __init__(self, session: FakeSession | None) -> None:
        self._session = session
        self._data = dict(session.event) if session is not None else {"EventName": "Empty GP"}

    def __getitem__(self, key: str) -> object:
        return self._data[key]

    def __len__(self) -> int:
        return len(self._data)

    def get_session(self, name: str) -> FakeSession:
        if self._session is None or name != self._session.name:
            raise ValueError(f"Session type '{name}' does not exist for this event")
        return self._session


class FakeSchedule:
    def __init__(self, session: FakeSession | None) -> None:
        self.session = session
        self.calls: list[tuple[str, object]] = []

    def get_event_by_name(self, name: str) -> FakeEvent:
        self.calls.append(("name", name))
        if name == "nowhere":
            raise ValueError("no match")
        return FakeEvent(self.session)

    def get_event_by_round(self, rnd: int) -> FakeEvent:
        self.calls.append(("round", rnd))
        return FakeEvent(self.session)


@pytest.fixture
def patch_fastf1(monkeypatch: pytest.MonkeyPatch) -> Callable[[FakeSession | None], FakeSchedule]:
    def install(session: FakeSession | None) -> FakeSchedule:
        schedule = FakeSchedule(session)
        monkeypatch.setattr(fastf1_loader.fastf1, "get_event_schedule", lambda *a, **k: schedule)
        monkeypatch.setattr(fastf1_loader.fastf1.Cache, "enable_cache", lambda *a, **k: None)
        return schedule

    return install


def handle_for(session: FakeSession) -> SessionHandle:
    return fastf1_loader._handle_from_session(2024, "Q", session)


# --- conversions ------------------------------------------------------------------------------


def test_convert_pos_units() -> None:
    raw = pd.DataFrame(
        {
            "SessionTime": pd.to_timedelta([1.0, 1.25], unit="s"),
            "X": [12345.0, -10.0],
            "Y": [0.0, 5.0],
            "Z": [735.0, 736.0],
            "Status": ["OnTrack", ""],
        }
    )
    out = convert_pos(raw)
    np.testing.assert_allclose(out["SessionTime"], [1.0, 1.25])
    np.testing.assert_allclose(out["X"], [1234.5, -1.0])  # 1/10 m -> m
    np.testing.assert_allclose(out["Z"], [73.5, 73.6])
    assert out["Status"].tolist() == ["OnTrack", None]
    # 1 of 2 statuses missing is over the NaN budget; everything else conforms
    assert {i.check for i in validate_frame(out, POSITION)} == {"nan_budget"}


def test_convert_car_units() -> None:
    raw = pd.DataFrame(
        {
            "SessionTime": pd.to_timedelta([0.0, 0.3], unit="s"),
            "Speed": [360.0, 36.0],
            "RPM": [11000, 9000],
            "nGear": [8, 2],
            "Throttle": [100, 50],
            "Brake": [False, True],
            "DRS": [12, 0],
        }
    )
    out = convert_car(raw)
    np.testing.assert_allclose(out["Speed"], [100.0, 10.0])  # km/h -> m/s
    np.testing.assert_allclose(out["Throttle"], [1.0, 0.5])  # % -> fraction
    assert out["Brake"].tolist() == [False, True]
    assert validate_frame(out, CAR) == []


def test_convert_laps(fake_session: FakeSession) -> None:
    laps = convert_laps(fake_session.laps)
    assert validate_frame(laps, LAPS) == []
    assert laps["LapEndTime"].iloc[0] == pytest.approx(
        fake_session.laps["Time"].iloc[0].total_seconds()
    )
    assert laps[["Driver", "LapNumber"]].equals(
        laps[["Driver", "LapNumber"]].sort_values(["Driver", "LapNumber"])
    )
    raw = fake_session.laps.copy()
    raw["Deleted"] = None  # FastF1 without race-control messages
    assert convert_laps(raw)["Deleted"].isna().all()


# --- lap iteration ----------------------------------------------------------------------------


def test_iter_driver_laps_windows_and_padding(fake_session: FakeSession) -> None:
    laps = list(iter_driver_laps(handle_for(fake_session)))
    assert len(laps) == len(fake_session.laps)
    for dl in laps:
        start, end = float(dl.lap["LapStartTime"]), float(dl.lap["LapEndTime"])
        t = dl.pos["SessionTime"].to_numpy()
        inside = (t >= start) & (t <= end)
        assert inside.sum() >= len(t) - 2  # exactly one padding sample each side
        assert t[0] < start <= t[1] and t[-2] <= end < t[-1]
        assert validate_frame(dl.pos, POSITION) == []
        assert validate_frame(dl.car, CAR) == []


def test_iter_driver_laps_filters(fake_session: FakeSession) -> None:
    h = handle_for(fake_session)
    assert {dl.driver for dl in iter_driver_laps(h, driver="BBB")} == {"BBB"}
    assert {dl.driver for dl in iter_driver_laps(h, driver="3")} == {"CCC"}
    accurate = list(iter_driver_laps(h, accurate_only=True))
    assert accurate and all(dl.lap_number > 1 for dl in accurate)


def test_iter_driver_laps_skips_laps_without_window_or_telemetry(fake_session: FakeSession) -> None:
    fake_session.laps.loc[0, "LapStartTime"] = pd.NaT
    del fake_session.pos_data["2"]
    laps = list(iter_driver_laps(handle_for(fake_session)))
    keys = {(dl.driver, dl.lap_number) for dl in laps}
    assert ("AAA", 1) not in keys
    assert not any(d == "BBB" for d, _ in keys)


def test_window_with_no_samples_is_empty(fake_session: FakeSession) -> None:
    tel = fake_session.pos_data["1"]
    assert fastf1_loader._window(tel, 1e9, 1e9 + 1).empty


# --- export -----------------------------------------------------------------------------------


def test_export_processed_round_trip(paths: Paths, fig8_session: FakeSession) -> None:
    h = handle_for(fig8_session)
    meta, rejected = export_processed(h, paths)
    assert rejected == []
    assert meta.source == "fastf1"
    assert meta.drivers == ["AAA", "BBB", "CCC"]
    assert meta.n_laps == 6
    assert cache.is_complete(paths, h.key)
    assert h.key == cache.SessionKey(2024, "synthetic_figure8_grand_prix", "Q")
    for drv, times in fig8_session.lap_times_s.items():
        assert meta.fastest_lap_s[drv] == pytest.approx(min(times), abs=1e-6)

    laps = pd.read_parquet(cache.laps_path(paths, h.key))
    assert validate_frame(laps, LAPS) == []
    pos = pd.read_parquet(cache.pos_path(paths, h.key, "AAA", 2))
    car = pd.read_parquet(cache.car_path(paths, h.key, "AAA", 2))
    assert validate_frame(pos, POSITION) == []
    assert validate_frame(car, CAR) == []
    # positions are metres: the figure-eight spans about +-500 m
    assert 450 < pos["X"].abs().max() < 520


def test_export_rejects_corrupt_lap_and_replaces_old_export(
    paths: Paths, fake_session: FakeSession
) -> None:
    h = handle_for(fake_session)
    stale = cache.session_dir(paths, h.key) / "STALE" / "pos_lap001.parquet"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    pos = fake_session.pos_data["1"]
    pos.loc[pos.index[40:60], "X"] = np.nan  # NaN burst inside lap 2 of AAA... or lap 1
    meta, rejected = export_processed(h, paths)
    assert not stale.exists()
    assert rejected and rejected[0].driver == "AAA"
    assert any("nan_budget" in i for i in rejected[0].issues)
    lap = rejected[0].lap_number
    assert not cache.pos_path(paths, h.key, "AAA", lap).exists()
    rec = json.loads((cache.session_dir(paths, h.key) / "rejected_laps.json").read_text())
    assert rec[0]["lap_number"] == lap
    assert cache.is_complete(paths, h.key)


def test_export_refuses_invalid_lap_table(paths: Paths, fake_session: FakeSession) -> None:
    fake_session.laps.loc[1, "LapNumber"] = fake_session.laps.loc[0, "LapNumber"]
    with pytest.raises(SchemaValidationError, match="unique"):
        export_processed(handle_for(fake_session), paths)


# --- load_session / fetch_session -------------------------------------------------------------


def test_load_session_resolves_event(
    paths: Paths, fake_session: FakeSession, patch_fastf1: Callable[[FakeSession | None], FakeSchedule]
) -> None:
    schedule = patch_fastf1(fake_session)
    h = load_session(2024, "Ellipse", "qualifying", paths=paths)
    assert h.key == cache.SessionKey(2024, "synthetic_ellipse_grand_prix", "Q")
    assert h.round_number == 99 and h.session_name == "Qualifying"
    load_session(2024, 99, "Q", paths=paths)
    load_session(2024, "99", "Q", paths=paths)
    assert schedule.calls == [("name", "Ellipse"), ("round", 99), ("round", 99)]
    assert cache.fastf1_cache_dir(paths).is_dir()


def test_load_session_errors(
    paths: Paths, fake_session: FakeSession, patch_fastf1: Callable[[FakeSession | None], FakeSchedule]
) -> None:
    patch_fastf1(fake_session)
    with pytest.raises(DataUnavailableError, match="no 2024 event"):
        load_session(2024, "nowhere", "Q", paths=paths)
    with pytest.raises(DataUnavailableError, match="no session R"):
        load_session(2024, "Ellipse", "R", paths=paths)
    with pytest.raises(ValueError, match="unknown session"):
        load_session(2024, "Ellipse", "warmup", paths=paths)


def test_load_session_without_data_explains_auth(
    paths: Paths, fake_session: FakeSession, patch_fastf1: Callable[[FakeSession | None], FakeSchedule]
) -> None:
    fake_session.laps = fake_session.laps.iloc[0:0]
    patch_fastf1(fake_session)
    with pytest.raises(DataUnavailableError, match="F1_USERNAME"):
        load_session(2024, "Ellipse", "Q", paths=paths)


def test_require_data_wraps_not_loaded(fake_session: FakeSession) -> None:
    class NotLoaded:
        name = fake_session.name
        event = fake_session.event
        pos_data = fake_session.pos_data
        car_data = fake_session.car_data

        @property
        def laps(self) -> pd.DataFrame:
            raise RuntimeError("DataNotLoadedError: The data you are trying to access ...")

    handle = SessionHandle(
        key=cache.SessionKey(2024, "x", "Q"), session=NotLoaded(), event_name="x",
        round_number=1, country="c", location="l", session_name="Qualifying",
    )
    with pytest.raises(DataUnavailableError, match="not loaded"):
        fastf1_loader._require_data(handle)


def test_second_fetch_makes_no_network_calls(
    paths: Paths,
    fake_session: FakeSession,
    patch_fastf1: Callable[[FakeSession | None], FakeSchedule],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gate: fetching the same session twice makes zero network calls the second time."""
    schedule = patch_fastf1(fake_session)
    first = fetch_session(2024, "Ellipse", "Q", paths=paths)
    assert not first.from_cache
    assert len(schedule.calls) == 1

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("FastF1 must not be called for a cached session")

    monkeypatch.setattr(fastf1_loader.fastf1, "get_event_schedule", boom)
    monkeypatch.setattr(fastf1_loader, "load_session", boom)
    second = fetch_session(2024, "ellipse", "qualifying", paths=paths)
    assert second.from_cache
    assert second.meta == first.meta
    assert second.rejected == []


def test_fetch_refetches_when_cache_damaged_or_forced(
    paths: Paths, fake_session: FakeSession, patch_fastf1: Callable[[FakeSession | None], FakeSchedule]
) -> None:
    schedule = patch_fastf1(fake_session)
    fake_session.pos_data["1"].loc[fake_session.pos_data["1"].index[40:60], "X"] = np.nan
    first = fetch_session(2024, "Ellipse", "Q", paths=paths)
    assert first.rejected
    again = fetch_session(2024, "Ellipse", "Q", paths=paths)
    assert again.from_cache and again.rejected == first.rejected
    cache.laps_path(paths, first.meta.key).write_bytes(b"corrupt")
    assert not fetch_session(2024, "Ellipse", "Q", paths=paths).from_cache
    assert not fetch_session(2024, "Ellipse", "Q", paths=paths, force=True).from_cache
    assert len(schedule.calls) == 3
