"""Regenerate ``tests/fixtures/openf1_suzuka_2024_q_ver_fastest.json`` from the live OpenF1 API.

The fixture is one real lap, unmodified apart from trimming to the lap window: the lap record
from ``laps`` plus the raw ``location`` and ``car_data`` samples between the lap start and
start + lap duration (one extra second either side). Run:

    uv run python scripts/make_openf1_lap_fixture.py
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from apex_zero.config.paths import resolve_paths
from apex_zero.data.cache import openf1_cache_dir
from apex_zero.data.openf1_client import OpenF1Client

YEAR, EVENT, SESSION, DRIVER_NUMBER = 2024, "Suzuka", "Q", 1
PAD = timedelta(seconds=1)
OUT = Path(__file__).resolve().parents[1] / "tests/fixtures/openf1_suzuka_2024_q_ver_fastest.json"


def main() -> None:
    with OpenF1Client(openf1_cache_dir(resolve_paths())) as client:
        sess = client.find_session(YEAR, EVENT, SESSION)
        laps = [lap for lap in client.laps(sess.session_key, DRIVER_NUMBER) if lap["lap_duration"]]
        lap = min(laps, key=lambda r: float(r["lap_duration"]))  # type: ignore[arg-type]
        start = datetime.fromisoformat(str(lap["date_start"]))
        end = start + timedelta(seconds=float(lap["lap_duration"]))  # type: ignore[arg-type]
        lo, hi = (start - PAD).isoformat(), (end + PAD).isoformat()
        fixture = {
            "source": "OpenF1 https://api.openf1.org/v1 (live timing relay)",
            "query": {
                "year": YEAR,
                "event": EVENT,
                "session": SESSION,
                "driver_number": DRIVER_NUMBER,
                "window_pad_s": PAD.total_seconds(),
            },
            "session": {
                "session_key": sess.session_key,
                "meeting_name": sess.meeting_name,
                "session_name": sess.session_name,
                "circuit_short_name": sess.circuit_short_name,
            },
            "lap": lap,
            "location": client.location(sess.session_key, DRIVER_NUMBER, lo, hi),
            "car_data": client.car_data(sess.session_key, DRIVER_NUMBER, lo, hi),
        }
    OUT.write_text(json.dumps(fixture, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
