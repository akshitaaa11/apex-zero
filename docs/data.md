# Data sources, units and limits

Phase 1 (data ingestion). Code: `src/apex_zero/data/`. Formats: `docs/migrations.md`.

## Sources

| Source | Years | Access | Used for |
|---|---|---|---|
| FastF1 (`fastf1` 3.9.2) | 2018+ | **F1 account with F1TV** for live-timing data (see below) | `apex data fetch`, processed Parquet |
| OpenF1 (`https://api.openf1.org/v1`) | 2023+ | free, no key for historical data | `OpenF1Client`, unit pinning, real-lap fixture |

### FastF1 needs an F1 account (since 3.9)

FastF1 3.9 sends an F1TV subscription token with every request to `livetiming.formula1.com`. It
reads the account from `F1_USERNAME` / `F1_PASSWORD` (or prompts in an interactive shell). Without
it, F1's CDN answers **HTTP 401**: the event schedule and driver list still load (they come from
GitHub and Ergast/jolpi.ca), but laps, telemetry, position data and race-control messages do not.
`fastf1_loader.load_session` detects the empty session and raises `DataUnavailableError` with
that explanation instead of exporting empty tables.

Checked on 2026-11-18 from the development container: `apex data fetch --year 2024 --event Japan
--session Q` logs `Failed to authenticate with Formula 1 account ... No F1 account credentials
available` and exits 1. **No FastF1 session has been fetched from live data yet.**

## Processed tables (SI)

See the docstring of `data/schemas.py` for the column contracts. Conversions at ingest:
positions 1/10 m → m, speed km/h → m/s, throttle % → fraction, times timedelta → s. Nothing is
resampled or smoothed; each lap file holds the raw samples in the lap window plus one sample
either side.

What the telemetry does **not** contain: steering, brake pressure (brake is on/off only),
tyre or suspension data. Position and car data arrive at about 4 Hz each, on separate clocks.

## Units and axes (P1.4)

`tests/network/test_units.py`: fastest qualifying lap, raw samples inside the lap window, closed
polyline, times the unit factor, against the official circuit length. Tolerance 2 % (the ~4 Hz
polyline cuts corners and reads short).

| Source | Circuit | Lap | Samples | Polyline | Official | Error | Status |
|---|---|---|---|---|---|---|---|
| OpenF1 | Suzuka 2024 Q | #1 VER 88.197 s | 341 | 5771.2 m | 5807 m | −0.62 % | **pass** |
| OpenF1 | Monza 2024 Q | #4 NOR 79.327 s | 320 | 5760.6 m | 5793 m | −0.56 % | **pass** |
| FastF1 | Suzuka 2024 Q | | | | 5807 m | | not run: needs F1 account |
| FastF1 | Monza 2024 Q | | | | 5793 m | | not run: needs F1 account |
| FastF1 vs OpenF1 axes | both | | | | | | not run: needs F1 account |

Results (OpenF1 rows) from `uv run pytest -m network tests/network/test_units.py -s` on
2026-11-18.

- `OPENF1_POS_TO_M = 0.1`: **verified** at two circuits.
- `FASTF1_POS_TO_M = 0.1`: from FastF1's documentation (`X position [1/10 m]`) and consistent
  with OpenF1 (both relay the same live-timing feed), but **not verified by our test yet**.
- Axis transform between the two sources: **unknown**. Do not mix them in one track build until
  `test_openf1_axes_match_fastf1` has passed.

## OpenF1 API facts the client relies on

Checked against live responses on 2026-11-18:

- Each endpoint returns a JSON list of flat objects. Equality filters are query parameters;
  comparison filters carry the operator (`date>=...`), raw or percent-encoded.
- No results → HTTP 404 `{"detail": "No results found."}` (the client returns `[]`, cached).
- `meetings`: `meeting_key, meeting_name, meeting_official_name, country_name, location,
  circuit_short_name, circuit_key, year, date_start, ...`
- `sessions`: `session_key, session_name, session_type, meeting_key, year, circuit_short_name,
  country_name, location, date_start, date_end, ...`. `session_name` uses the same names as
  FastF1 (`Qualifying`, `Practice 1`, `Sprint`, ...), so it maps onto the same cache key.
- `laps`: `driver_number, lap_number, date_start, lap_duration, duration_sector_1..3,
  is_pit_out_lap, i1_speed, i2_speed, st_speed, segments_sector_*`.
- `location`: `date, driver_number, x, y, z` (integers, 1/10 m), ~3.7–4 Hz.
- `car_data`: `date, driver_number, speed (km/h), rpm, n_gear, throttle (%), brake (0/100),
  drs`, ~3.7–4 Hz.

## Data etiquette

Everything is cached: FastF1 under `APEX_CACHE_DIR/fastf1`, OpenF1 responses under
`APEX_CACHE_DIR/openf1` keyed by request-URL hash. The OpenF1 client throttles to about 3 req/s
with exponential backoff on 429/5xx. Raw data is not redistributed; the only committed real data
is one trimmed lap in `tests/fixtures/` (see its README).
