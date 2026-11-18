"""Live end-to-end OpenF1 fetch (Phase 1 deliverable, OpenF1 is the primary source: ADR 0002).

``uv run pytest -m network tests/network/test_openf1_live.py``. Uses the real OpenF1 response
cache (``APEX_CACHE_DIR``) so repeated runs do not re-download, but a temporary processed dir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apex_zero.config.paths import Paths, resolve_paths
from apex_zero.data import cache
from apex_zero.data.openf1_loader import fetch_openf1_session
from apex_zero.data.schemas import CAR, LAPS, POSITION, validate_frame

pytestmark = pytest.mark.network

#: Official 2024 qualifying results (pole and 2nd), lap times in seconds.
EXPECTED = [
    pytest.param(2024, "Japan", {"VER": 88.197, "PER": 88.263}, 5807.0, id="suzuka"),
    pytest.param(2024, "Monza", {"NOR": 79.327, "PIA": 79.436}, 5793.0, id="monza"),
]


@pytest.mark.parametrize(("year", "event", "top2", "official_m"), EXPECTED)
def test_fetch_real_session(
    tmp_path: object, year: int, event: str, top2: dict[str, float], official_m: float
) -> None:
    real = resolve_paths()
    paths = Paths(
        cache_dir=real.cache_dir,
        processed_dir=tmp_path / "processed",  # type: ignore[operator]
        runs_dir=real.runs_dir,
    )
    result = fetch_openf1_session(year, event, "Q", paths=paths)
    meta = result.meta
    print(
        f"{meta.event_name}: {len(meta.drivers)} drivers, {meta.n_laps} laps, "
        f"{len(result.rejected)} rejected"
    )
    assert len(meta.drivers) == 20
    for drv, t in top2.items():
        assert meta.fastest_lap_s[drv] == pytest.approx(t, abs=1e-3)
    assert min(meta.fastest_lap_s.values()) == pytest.approx(min(top2.values()), abs=1e-3)

    laps = pd.read_parquet(cache.laps_path(paths, meta.key))
    assert validate_frame(laps, LAPS) == []
    pole_driver = min(top2, key=lambda d: top2[d])
    pole = laps[(laps["Driver"] == pole_driver) & laps["LapTime"].notna()]
    pole = pole.loc[pole["LapTime"].idxmin()]
    pos = pd.read_parquet(cache.pos_path(paths, meta.key, pole_driver, int(pole["LapNumber"])))
    car = pd.read_parquet(cache.car_path(paths, meta.key, pole_driver, int(pole["LapNumber"])))
    assert validate_frame(pos, POSITION) == [] and validate_frame(car, CAR) == []
    inside = pos[
        (pos["SessionTime"] >= pole["LapStartTime"]) & (pos["SessionTime"] <= pole["LapEndTime"])
    ]
    xy = inside[["X", "Y"]].to_numpy()
    length = float(np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1).sum())
    print(f"pole lap polyline {length:.1f} m vs {official_m:.0f} m")
    assert abs(length / official_m - 1) < 0.02

    again = fetch_openf1_session(year, event, "Q", paths=paths)
    assert again.from_cache and again.meta == meta
