"""P1.4: pin position units and axes against official circuit lengths (live data).

For each circuit, the fastest lap's raw position samples inside the lap window, joined into a
closed polyline and multiplied by the unit factor, must be within 2 % of the official length.
At ~4 Hz the polyline cuts corners, so it reads slightly short; 2 % covers that.

Run with ``uv run pytest -m network tests/network/test_units.py``. FastF1 tests need an F1
account with F1TV (``F1_USERNAME``/``F1_PASSWORD``) since FastF1 3.9; they are skipped, with that
reason, when the variables are not set.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from apex_zero.config.paths import resolve_paths
from apex_zero.data.cache import openf1_cache_dir
from apex_zero.data.fastf1_loader import convert_laps, convert_pos, load_session
from apex_zero.data.openf1_client import OpenF1Client, Record
from apex_zero.utils.units import FASTF1_POS_TO_M, OPENF1_POS_TO_M

pytestmark = pytest.mark.network

#: (id, year, event query, official lap length [m] as published by F1/FIA).
CIRCUITS = [
    pytest.param(2024, "Suzuka", 5807.0, id="suzuka"),
    pytest.param(2024, "Monza", 5793.0, id="monza"),
]
TOLERANCE = 0.02

needs_f1_account = pytest.mark.skipif(
    not os.environ.get("F1_USERNAME"),
    reason="FastF1 >= 3.9 live timing needs a paid F1TV account (F1_USERNAME/F1_PASSWORD)",
)


def closed_length(xy: np.ndarray) -> float:
    closed = np.vstack([xy, xy[:1]])
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())


def openf1_fastest_lap(client: OpenF1Client, year: int, event: str) -> tuple[Record, list[Record]]:
    sess = client.find_session(year, event, "Q")
    laps = [r for r in client.laps(sess.session_key) if isinstance(r["lap_duration"], float)]
    lap = min(laps, key=lambda r: float(r["lap_duration"]))  # type: ignore[arg-type]
    start = datetime.fromisoformat(str(lap["date_start"]))
    end = start + timedelta(seconds=float(lap["lap_duration"]))  # type: ignore[arg-type]
    loc = client.location(
        sess.session_key,
        int(lap["driver_number"]),  # type: ignore[arg-type]
        start.isoformat(),
        end.isoformat(),
    )
    return lap, loc


@pytest.mark.parametrize(("year", "event", "official_m"), CIRCUITS)
def test_openf1_position_units(year: int, event: str, official_m: float) -> None:
    with OpenF1Client(openf1_cache_dir(resolve_paths())) as client:
        lap, loc = openf1_fastest_lap(client, year, event)
    xy = np.array([[p["x"], p["y"]] for p in loc], dtype=np.float64) * OPENF1_POS_TO_M
    length = closed_length(xy)
    print(
        f"OpenF1 {event} {year}: driver {lap['driver_number']} {lap['lap_duration']} s, "
        f"{len(loc)} samples, polyline {length:.1f} m vs official {official_m:.0f} m "
        f"({length / official_m - 1:+.2%})"
    )
    assert abs(length / official_m - 1) < TOLERANCE


@needs_f1_account
@pytest.mark.parametrize(("year", "event", "official_m"), CIRCUITS)
def test_fastf1_position_units(year: int, event: str, official_m: float) -> None:
    handle = load_session(year, event, "Q")
    laps = convert_laps(handle.session.laps)
    ok = laps[laps["LapTime"].notna() & ~laps["Deleted"].fillna(False).astype(bool)]
    lap = ok.loc[ok["LapTime"].idxmin()]
    raw = handle.session.pos_data[str(lap["DriverNumber"])]
    pos = convert_pos(raw)
    inside = pos[
        (pos["SessionTime"] >= lap["LapStartTime"]) & (pos["SessionTime"] <= lap["LapEndTime"])
    ]
    length = closed_length(inside[["X", "Y"]].to_numpy())
    print(
        f"FastF1 {event} {year}: {lap['Driver']} {lap['LapTime']:.3f} s, {len(inside)} samples, "
        f"polyline {length:.1f} m vs official {official_m:.0f} m "
        f"({length / official_m - 1:+.2%}), factor {FASTF1_POS_TO_M}"
    )
    assert abs(length / official_m - 1) < TOLERANCE


@needs_f1_account
@pytest.mark.parametrize(("year", "event", "official_m"), CIRCUITS)
def test_openf1_axes_match_fastf1(year: int, event: str, official_m: float) -> None:
    """Same lap from both sources: matched by UTC timestamp, positions must coincide."""
    with OpenF1Client(openf1_cache_dir(resolve_paths())) as client:
        lap, loc = openf1_fastest_lap(client, year, event)
    o_t = pd.to_datetime([p["date"] for p in loc], format="ISO8601").tz_convert(None)
    o_xy = np.array([[p["x"], p["y"]] for p in loc], dtype=np.float64) * OPENF1_POS_TO_M

    handle = load_session(year, event, "Q")
    raw = handle.session.pos_data[str(lap["driver_number"])]
    f_t = pd.to_datetime(raw["Date"])
    f_xy = raw[["X", "Y"]].to_numpy(dtype=np.float64) * FASTF1_POS_TO_M
    idx = np.clip(np.searchsorted(f_t.to_numpy(), o_t.to_numpy()), 0, len(f_t) - 1)
    d = np.linalg.norm(f_xy[idx] - o_xy, axis=1)
    print(f"{event}: median |FastF1 - OpenF1| = {np.median(d):.2f} m")
    assert math.isfinite(float(np.median(d)))
    assert np.median(d) < 2.0, "axes differ: record the transform in utils/units.py"
