from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from apex_zero.config.track import TrackConfig
from apex_zero.data.fastf1_loader import convert_laps, convert_pos
from apex_zero.data.processed import iter_laps
from apex_zero.track.centerline import LapPoints
from tests.fixtures.synthetic import (
    FakeSession,
    SyntheticTrack,
    ellipse_track,
    figure_eight_track,
    make_session,
)

#: A realistic session size: 20 cars, 3 laps each (lap 1 is an out-lap -> 40 accurate laps).
DRIVERS = tuple(f"D{i:02d}" for i in range(20))


def synthetic_config(track: SyntheticTrack, **overrides: Any) -> TrackConfig:
    """Config for the synthetic tracks.

    The synthetic feed is deliberately noisier than the real one (0.3 m position noise per
    axis, drivers spread up to ±4.8 m across a 12 m road), so it uses longer smoothing scales
    than configs/tracks/suzuka.yaml, whose feed is map-matched (spread ~0.2 m).
    """
    data: dict[str, Any] = {
        "name": "synthetic",
        "official_length_m": track.length_m,
        "width_prior_m": {"min": 8.0, "max": 20.0},
        "width_margin_m": 1.0,
        "smoothing": {"seed": 25.0, "xy": 15.0, "z": 60.0, "width": 40.0, "kappa": 20.0},
        "sessions": {"reference": {"year": 2024, "event": "Synthetic", "session": "Q"}},
    }
    data.update(overrides)
    return TrackConfig.model_validate(data)


def laps_of(session: FakeSession) -> list[LapPoints]:
    laps = convert_laps(session.laps)
    pos = {k: convert_pos(v) for k, v in session.pos_data.items()}
    return [
        LapPoints(
            lap_id=f"{dl.driver}/{dl.lap_number}",
            t=dl.pos["SessionTime"].to_numpy(),
            xyz=dl.pos[["X", "Y", "Z"]].to_numpy(),
            t_start=float(dl.lap["LapStartTime"]),
            t_end=float(dl.lap["LapEndTime"]),
        )
        for dl in iter_laps(laps, pos, pos, accurate_only=True)
    ]


@pytest.fixture(scope="session")
def ellipse() -> SyntheticTrack:
    return ellipse_track()


@pytest.fixture(scope="session")
def figure8() -> SyntheticTrack:
    return figure_eight_track()


@pytest.fixture(scope="session")
def ellipse_laps(ellipse: SyntheticTrack) -> list[LapPoints]:
    return laps_of(make_session(ellipse, np.random.default_rng(1), drivers=DRIVERS, n_laps=3))


@pytest.fixture(scope="session")
def figure8_laps(figure8: SyntheticTrack) -> list[LapPoints]:
    return laps_of(make_session(figure8, np.random.default_rng(1), drivers=DRIVERS, n_laps=3))
