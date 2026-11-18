# Data sources, units and limits

Phase 1 (data ingestion). Code: `src/apex_zero/data/`. Formats: `docs/migrations.md`.

## Sources

OpenF1 is the primary source ([ADR 0002](decisions/0002-openf1-primary-source.md)).

| Source | Years | Access | Used for |
|---|---|---|---|
| OpenF1 (`https://api.openf1.org/v1`) | 2023+ | free, no key for historical data | `apex data fetch` (default), unit pinning, real-lap fixture |
| FastF1 (`fastf1` 3.9.2) | 2018+ | **paid F1TV subscription** for live-timing data (see below) | `apex data fetch --source fastf1` |

### Live sessions fetched so far (OpenF1, 2026-11-18)

| Session | Drivers | Laps | Rejected | Pole (matches official) | Cold fetch |
|---|---|---|---|---|---|
| 2024 Japanese GP Q | 20 | 223 | 0 | VER 88.197 s, 2nd PER 88.263 s | ~2 min |
| 2024 Italian GP Q | 20 | 276 | 0 | NOR 79.327 s, 2nd PIA 79.436 s | ~1.5 min |

Re-fetching either takes under a second from the processed cache, with no HTTP requests. Each
driver's final untimed in-lap is skipped (no end crossing), and so is a first lap without a
start time; both are logged.

### How OpenF1 maps onto the processed tables

See the module docstring of `data/openf1_loader.py`. In short: `SessionTime` counts from the
session's scheduled start; a lap ends at start + official duration, except pit-out laps (whose
`lap_duration` includes garage time) and untimed laps, which end at the next lap's start;
`IsAccurate` means timed, not an out-lap, not followed by an out-lap, and not the driver's last
lap; pit times, deleted-lap flags and `Status` are missing; compound and tyre age come from
`stints`; brake is on/off (0/100).

### FastF1 needs a paid F1TV account (since 3.9)

FastF1 3.9 sends an F1TV subscription token with every request to `livetiming.formula1.com`. It
reads the account from `F1_USERNAME` / `F1_PASSWORD` (or prompts in an interactive shell). Without
it, F1's CDN answers **HTTP 401**. A free F1 account is not enough: FastF1 needs the
subscription token from the login ("An active F1TV subscription is required"). Without one: the event schedule and driver list still load (they come from
GitHub and Ergast/jolpi.ca), but laps, telemetry, position data and race-control messages do not.
`fastf1_loader.load_session` detects the empty session and raises `DataUnavailableError` with
that explanation instead of exporting empty tables.

Checked on 2026-11-18 from the development container: `apex data fetch --year 2024 --event Japan
--session Q` logs `Failed to authenticate with Formula 1 account ... No F1 account credentials
available` and exits 1. **No FastF1 session has been fetched from live data yet.** The project
owner has no F1TV subscription, so OpenF1 became the primary source.

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

## Positions are map-matched (found in Phase 2)

Across 74 accurate laps of 20 drivers at Suzuka 2024 Q, the lateral spread of all samples around
the reconstructed line is ~0.2 m (1st to 99th percentile per 10 m bin, median over the lap).
Different cars' racing lines differ by metres, so the feed's positions are snapped onto one
reference line. Consequences ([ADR 0003](decisions/0003-track-width-from-map-matched-data.md)):

- **Track width and banking cannot be measured from this data.** The Suzuka asset uses an
  assumed 12 m width, recorded as `width_source: config_nominal`.
- The reconstructed centerline is F1's reference line, recovered to within ~0.2 m.
- The line comes out **~0.5 % shorter than the official lap length** at both circuits tried:

| Circuit | Reconstructed | Official | Error | Dense multi-lap medians (independent check) |
|---|---|---|---|---|
| Suzuka 2024 Q | 5777.9 m | 5807 m | −0.502 % | 5776–5782 m |
| Monza 2024 Q | 5761.5 m | 5793 m | −0.544 % | not computed |

The Suzuka value does not move with smoothing once smoothing is small enough not to cut
corners (5777.8–5778.2 m over seed 5–15 m, offsets 3–5 m), so it is the line's length, not a
fitting artefact. Possible causes, not resolved: the feed's reference line is not the line the
official length is measured on (e.g. closer to a racing line), or the position unit is not
exactly 0.1 m. Resolving it needs independent geometry (survey, imagery, or FastF1 data).

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
  is_pit_out_lap, i1_speed, i2_speed, st_speed, segments_sector_*`. `date_start` is approximate
  (up to ~0.2 s off the previous lap's start + duration); `lap_duration` of a pit-out lap counts
  from the previous crossing, garage time included; the first out-lap has no duration.
- `drivers`: `driver_number, name_acronym, full_name, team_name, ...` per session.
- `stints`: `driver_number, stint_number, lap_start, lap_end, compound, tyre_age_at_start`.
- `location`: `date, driver_number, x, y, z` (integers, 1/10 m), ~3.7–4 Hz.
- `car_data`: `date, driver_number, speed (km/h), rpm, n_gear, throttle (%), brake (0/100),
  drs`, ~3.7–4 Hz.

## Data etiquette

Everything is cached: FastF1 under `APEX_CACHE_DIR/fastf1`, OpenF1 responses under
`APEX_CACHE_DIR/openf1` keyed by request-URL hash. OpenF1 limits clients to 3 req/s and
30 req/min (429 "Max 30 requests/minute" observed); the client spaces requests evenly at
29/min and backs off (honouring `Retry-After`) on 429/5xx. A full session is ~45 requests. Raw data is not redistributed; the only committed real data
is one trimmed lap in `tests/fixtures/` (see its README).
