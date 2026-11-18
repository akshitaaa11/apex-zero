# ADR 0002: OpenF1 is the primary data source; FastF1 is optional

- **Status:** accepted
- **Date:** 2026-11-18
- **Phase:** 1

## Context

The plan assumed FastF1 (2018+) as the main source. Since FastF1 3.9, reading F1 live-timing
data requires signing in to an F1 account **with an active F1TV subscription**: FastF1 extracts a
`subscriptionToken` from the login and fails with "An active F1TV subscription is required"
otherwise. Without it, `livetiming.formula1.com` answers HTTP 401 (also to plain `curl`, so
pinning an older FastF1 does not help). The project owner has no paid F1TV account.

OpenF1 (`api.openf1.org`) is free, needs no account for historical data, relays the same
live-timing feed, and covers 2023 onwards.

## Decision

- OpenF1 is the default source: `apex data fetch` uses `--source openf1` unless told otherwise,
  via `data/openf1_loader.py`, writing the same processed layout and column contracts as the
  FastF1 path (`data/processed.py` is shared). Differences are documented in
  `data/openf1_loader.py` and `docs/data.md`.
- The FastF1 path stays, tested against fakes, for anyone with F1TV (`--source fastf1`).
- P1.4 is closed on OpenF1: units verified at Suzuka and Monza. The FastF1 unit and axis tests
  stay in `tests/network/test_units.py` and skip without credentials.

## Consequences

- No seasons before 2023 without FastF1. Cross-era races (Phase 11, e.g. 2019 vs 2024) need a
  FastF1 user, or are limited to 2023+.
- OpenF1 lacks some FastF1 fields: pit in/out times, deleted-lap flags, on/off-track status,
  round numbers. `IsAccurate` is derived by a simpler rule. Later phases must treat these as
  missing (they are nullable in the contracts), not as "false".
- `FASTF1_POS_TO_M` remains documented but unverified; positions from the two sources must not
  be mixed in one track build until the axis test has passed.
- OpenF1 rate limits (3 req/s, 30 req/min) make a full session ~45 requests, about 1.5 min on
  a cold cache; everything is cached afterwards.
