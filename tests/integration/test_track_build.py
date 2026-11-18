"""Real Suzuka build (P2.11, slow + network): the Phase 2 real-data checks.

``uv run pytest -m "slow and network" tests/integration/test_track_build.py -s``. Fetches 2024
Japanese GP qualifying from OpenF1 (cached after the first run) into the configured cache, builds
into a temporary processed dir, and checks the asset against the official circuit length and
the fastest laps against the boundaries.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from apex_zero.config.paths import Paths, resolve_paths
from apex_zero.data.openf1_loader import fetch_openf1_session
from apex_zero.track.mesh import export_glb
from apex_zero.track.reconstruct import build_track, load_track_config, resolve_sessions

pytestmark = [pytest.mark.slow, pytest.mark.network]

REPO = Path(__file__).resolve().parents[2]


def test_suzuka_real_data(tmp_path: Path) -> None:
    cfg = load_track_config(REPO / "configs/tracks/suzuka.yaml")
    real = resolve_paths()
    paths = Paths(
        cache_dir=real.cache_dir, processed_dir=tmp_path / "processed", runs_dir=tmp_path / "runs"
    )
    ref = cfg.sessions.reference
    fetch_openf1_session(ref.year, ref.event, ref.session, paths=paths)

    t0 = time.perf_counter()
    asset = build_track(cfg, resolve_sessions(paths, cfg.sessions.all), paths)
    export_glb(asset, tmp_path / "track.glb")
    seconds = time.perf_counter() - t0
    r = asset.report
    print(
        f"suzuka: {asset.length:.1f} m vs {cfg.official_length_m} m "
        f"({r['length_error_pct']:+.3f} %), inside {r['fastest_laps_inside_frac']:.4f}, "
        f"width {r['width_source']}, build+glb {seconds:.1f} s"
    )
    assert seconds < 60.0
    assert float(r["fastest_laps_inside_frac"]) >= 0.99  # type: ignore[arg-type]
    assert abs(float(r["length_error_pct"])) <= 0.5  # type: ignore[arg-type]
