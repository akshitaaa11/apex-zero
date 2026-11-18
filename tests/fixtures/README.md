# Test fixtures

Small, committed data used by unit and integration tests (no network in unit tests).

Rules (`AGENTS.md` §6.8):

- Keep each fixture under 1 MB. Prefer synthetic data with known ground truth.
- For trimmed real data, document below: source (FastF1/OpenF1), library version, session
  (year, event, session type), drivers/laps kept, and how it was trimmed.

| Fixture | Source | Notes |
|---|---|---|
| `openf1_suzuka_2024_q_ver_fastest.json` (101 KiB) | OpenF1 API, fetched 2026-11-18 by `scripts/make_openf1_lap_fixture.py` | 2024 Japanese GP qualifying (session_key 9492), car #1 VER, lap 11 (88.197 s, pole). The `laps` record plus raw `location` and `car_data` samples from 1 s before the lap start to 1 s after its end. Unmodified values; only trimmed by time. |
| `synthetic.py` | generated | Ellipse and figure-eight (self-crossing at different heights) tracks with elevation; multi-driver, ~4 Hz noisy sessions shaped like a loaded FastF1 session, with ground truth. Deterministic given an `np.random.Generator`. |

A FastF1-sourced trimmed lap is not included: FastF1 3.9 needs an F1 account for live-timing
data (`docs/data.md`). When one is available, add it here with the same provenance details.
