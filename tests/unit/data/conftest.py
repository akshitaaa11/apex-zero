from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from apex_zero.config.paths import Paths
from apex_zero.utils.units import OPENF1_POS_TO_M
from tests.fixtures.synthetic import FakeSession, ellipse_track, figure_eight_track, make_session

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
REAL_LAP = FIXTURES / "openf1_suzuka_2024_q_ver_fastest.json"


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    return Paths(
        cache_dir=tmp_path / "cache",
        processed_dir=tmp_path / "processed",
        runs_dir=tmp_path / "runs",
    )


@pytest.fixture
def fake_session() -> FakeSession:
    return make_session(ellipse_track(), np.random.default_rng(7), n_laps=2)


@pytest.fixture
def fig8_session() -> FakeSession:
    return make_session(figure_eight_track(), np.random.default_rng(11), n_laps=2)


@pytest.fixture(scope="session")
def real_lap() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(REAL_LAP.read_text(encoding="utf-8"))
    return data


def openf1_position_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Real OpenF1 location samples -> POSITION-contract frame (time relative to first sample)."""
    t = pd.to_datetime([r["date"] for r in records], format="ISO8601")
    return pd.DataFrame(
        {
            "SessionTime": (t - t[0]).total_seconds().astype("float64"),
            "X": np.array([r["x"] for r in records], dtype=np.float64) * OPENF1_POS_TO_M,
            "Y": np.array([r["y"] for r in records], dtype=np.float64) * OPENF1_POS_TO_M,
            "Z": np.array([r["z"] for r in records], dtype=np.float64) * OPENF1_POS_TO_M,
            "Status": pd.Series(["OnTrack"] * len(records), dtype="object"),
        }
    )
