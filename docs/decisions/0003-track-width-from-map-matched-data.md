# ADR 0003: Track width is not observable from F1 position data; use an explicit nominal width

- **Status:** accepted (provisional: replace with a measured width when one is available)
- **Date:** 2026-11-18
- **Phase:** 2

## Context

Phase 2 planned to estimate track boundaries from the lateral spread of every car's samples
(`SKILLS.md` -> *reconstruct-track*, step 6). On the real 2024 Suzuka qualifying data (OpenF1,
74 accurate laps, 20 drivers, 25,685 samples) the per-10 m-bin spread between the 1st and 99th
percentile of the lateral offset is **0.20 m in the median** (`docs/media/tracks/
suzuka_debug_lateral.png`). Real cars on different lines differ by metres. The live-timing
position feed is evidently map-matched onto one reference line, so it carries no width (and no
banking) information. Monza 2024 behaves the same way.

## Decision

- The reconstruction measures the spread and, below `min_observed_spread_m` (2 m), **refuses**
  to invent a width unless the track config sets `nominal_width_m`. With it, both sides get
  half the nominal width and the asset report records `width_source: config_nominal` and
  `bank_source: not_observable`. `apex track build` and `track inspect` print this.
- `configs/tracks/suzuka.yaml` sets `nominal_width_m: 12.0`, the FIA minimum width for new
  Grade 1 circuits. This is an **assumption**, not Suzuka's measured width.
- Width estimation from the spread stays implemented and tested (synthetic data with real
  lateral spread), for any future source with genuine lateral information.

## Consequences

- Track limits for the RL environment (Phase 7) rest on an assumed constant width. Comparisons
  that depend on line choice or track limits (corner-by-corner "line offset") are not credible
  until a measured width replaces it, and reports must say so.
- The "≥ 99 % of fastest-lap points inside bounds" check passes trivially (all samples lie
  within ~0.2 m of the line). It is still run, but it no longer validates the width.
- The centerline is F1's reference line, recovered very precisely (samples within ~0.2 m).
