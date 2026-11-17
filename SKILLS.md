# SKILLS.md

Reusable workflows for AI agents (and humans) working on APEX ZERO. Each skill gives **when to use it**, **prerequisites**, **step-by-step instructions**, **tools and commands**, **required expertise**, **pitfalls** and a **worked example**.

Conventions and constraints come from `AGENTS.md`. Phases and acceptance criteria come from `IMPLEMENTATION.md`.

---

## Skill index

| # | Skill | Use when | Phase |
|---|---|---|---|
| 1 | [ingest-session](#1-ingest-session) | Fetching or caching a new F1 session | 1 |
| 2 | [reconstruct-track](#2-reconstruct-track) | Building or fixing a track asset | 2 |
| 3 | [add-new-track](#3-add-new-track) | Supporting a new circuit end to end | 2–3 |
| 4 | [extract-ghost](#4-extract-ghost) | Turning a driver's lap into a ghost | 3 |
| 5 | [modify-vehicle-physics](#5-modify-vehicle-physics) | Any change to the car model | 4, 6 |
| 6 | [calibrate-car](#6-calibrate-car) | Matching the car to a real era or lap time | 5 |
| 7 | [train-agent](#7-train-agent) | Launching or resuming an RL training run | 8 |
| 8 | [debug-rl-agent](#8-debug-rl-agent) | Training stalls, diverges, or the agent "cheats" | 7–9 |
| 9 | [compare-laps](#9-compare-laps) | Corner-by-corner AI vs ghost analysis | 9 |
| 10 | [export-replay](#10-export-replay) | Producing a bundle for the viewer | 9–10 |
| 11 | [viewer-feature](#11-viewer-feature) | Adding UI, cameras or effects to the 3D viewer | 10 |
| 12 | [cross-era-race](#12-cross-era-race) | Racing ghosts from different seasons | 11 |
| 13 | [make-clip](#13-make-clip) | Rendering a shareable head-to-head video | 12 |
| 14 | [performance-profiling](#14-performance-profiling) | Sim throughput or viewer fps regressions | 6, 10 |

---

## 1. ingest-session

**When:** a task needs data for a session (year, event, session type) that isn't in the cache yet, or the data schema needs checking.

**Prerequisites:** network access, `APEX_CACHE_DIR` set (default `./data/cache`).

**Expertise:** pandas, FastF1 API, REST APIs, time-series alignment.

**Steps**
1. Check the cache first: `uv run apex data ls --year 2024`.
2. Fetch:
   ```bash
   uv run apex data fetch --year 2024 --event "Japan" --session Q
   uv run apex data fetch --year 2024 --event "Japan" --session Q --source openf1   # 2023+ only
   ```
3. Internally, `data/fastf1_loader.py` should:
   - call `fastf1.Cache.enable_cache(cache_dir)`
   - `session = fastf1.get_session(year, event, session_type)`, then `session.load(laps=True, telemetry=True, weather=False, messages=True)`
   - for each driver, store per-lap position data and car data as Parquet under `processed/<year>/<event>/<session>/<driver>/`
   - write `session_meta.json` with the FastF1 version, fetch time, event name, circuit key and official lap times.
4. Validate with `data/schemas.py`: required columns, monotonic `SessionTime` per lap, no duplicated timestamps, and position units converted to metres (the conversion factor is pinned by a test).
5. For OpenF1, `openf1_client.py` calls `GET https://api.openf1.org/v1/location?session_key=…&driver_number=…` and the matching `car_data`/`laps`, throttled and cached by URL hash.

**Pitfalls**
- Session names differ by weekend format (sprint weekends). Resolve them with `fastf1.get_event_schedule`, never hard-code.
- Some laps have missing position data or pit in/out laps. Keep `is_accurate`-style flags and filter downstream, not at ingest.
- Don't merge car and position data at ingest. Keep both raw. Merging is a ghost-extraction concern.

**Example:** "Add 2023 Suzuka qualifying" → `apex data fetch --year 2023 --event "Japan" --session Q`, then run `pytest -m network tests/network/test_fastf1_schema.py` once to confirm the schema hasn't drifted.

---

## 2. reconstruct-track

**When:** building a `TrackAsset` from cached data, or a track looks wrong (kinks, wrong width, a stepped elevation).

**Expertise:** computational geometry, splines, robust statistics, Frenet frames.

**Steps**
1. Collect points: all accurate laps from all drivers in the session (and optionally FP sessions) → `(M, 3)` world points with lap IDs.
2. Build an initial centerline: take the cleanest fast lap as a seed polyline and resample it every 2 m.
3. Project every point onto the seed to get `(s, n)`. Bin by `s` (1 m bins) and take the **median** world position per bin as the refined centerline. Iterate 2–3 times.
4. Smooth: fit a **periodic smoothing spline** `x(s), y(s), z(s)` (`scipy.interpolate.make_smoothing_spline`, or `splprep` with `per=1`). Pick the smoothing factor so that curvature has no spikes over 1/15 m⁻¹ that aren't in the raw data.
5. Derive `psi(s)` (heading), `kappa(s)` (signed curvature), `grade(s) = dz/ds`, and `bank(s)` (optional, estimated from systematic lateral z-slope across `n`).
6. Width: per `s` bin, take the 1st and 99th percentile of `n` over all cars plus a margin (`configs/tracks/<name>.yaml: width_margin_m`), clip to `[w_min, w_max]` from config, and smooth.
7. Close the loop: enforce `x(0) = x(L)` with matching first and second derivatives. Align `s = 0` to the timing line (the start/finish position from the fastest lap's start).
8. Save the `.npz` and `meta.json` (with its hash). Export `track.glb` through `track/mesh.py`.
9. Inspect: `uv run apex track inspect <name>`. Check the curvature plot, the width plot, the elevation profile, and a 2D overlay of raw points on the centerline.

**Acceptance checks:** synthetic tests pass (< 0.5 m RMS), the real track length is within 0.5% of the official circuit length in `configs/tracks/<name>.yaml`, and ≥ 99% of the fastest laps' points fall inside the boundaries.

**Pitfalls**
- Pit lane points pollute the width. Drop points with `|n| > 30 m` or flagged as pit laps.
- A figure-of-eight layout (Suzuka!) crosses itself. Never use nearest-point projection without an `s` continuity window. Project with a search window around the previous `s`.
- Z noise is large. Smooth elevation more strongly than x/y, but keep total climb within 10% of the raw median profile.

---

## 3. add-new-track

**When:** supporting a circuit that isn't in `configs/tracks/` yet.

**Steps**
1. Create `configs/tracks/<name>.yaml`:
   ```yaml
   name: monaco
   official_length_m: 3337
   width_prior_m: {min: 7.0, max: 16.0}
   width_margin_m: 0.5
   smoothing: {xy: 2.0, z: 25.0}
   self_intersecting: false
   corners:   # filled after first build, by s-range
     - {name: "Sainte Dévote", s_start: 180, s_end: 300}
   sessions:
     reference: {year: 2024, event: "Monaco", session: Q}
   ```
2. Run *ingest-session*, then *reconstruct-track*.
3. Run `apex analysis corners --track <name>` to auto-detect corners, then name them in the YAML.
4. Add a smoke config `configs/training/<name>_smoke.yaml`.
5. Add an integration test case in `tests/integration/test_tracks.py` (parametrised over names, marked `slow` and `network`).

**Pitfalls:** the official length includes the timing-line definition. Small differences are fine, but more than 0.5% means a reconstruction problem.

---

## 4. extract-ghost

**When:** you need a real driver's lap as a ghost.

**Expertise:** time-series resampling, interpolation, telemetry semantics.

**Steps**
1. Pick the lap: `fastest`, a lap number, or `--lap-time 1:28.197` (closest match).
2. Merge position and car data on `SessionTime` with linear interpolation for continuous channels, and nearest-neighbour for `nGear`, `Brake`, `DRS`.
3. Project positions onto the track Frenet frame (windowed, see skill 2) → `s(t), n(t)`.
4. Make `s(t)` monotonic. Fix small reversals by isotonic regression on `s`.
5. Resample to 50 Hz in time with a monotone cubic (PCHIP) for `s`, and linear for the rest.
6. Validate:
   - `t_end − t_start` equals the official lap time within 0.05 s
   - ≥ 99% of samples are inside the boundaries
   - speed derived from `ds/dt` agrees with telemetry `Speed` within 5% RMS
7. Save `ghosts/<track>/<year>_<session>_<DRIVER>_<lap>.npz` with metadata (driver, team, year, lap time, source versions).

**Pitfalls**
- Brake is boolean. Compute `brake_point` from `dv/ds < −threshold` and use the flag only for confirmation.
- Lap start and finish don't line up with the 50 Hz grid. Interpolate the exact crossing time of `s = 0`.

**Example:** `apex ghost extract --year 2024 --event Japan --session Q --driver LEC --lap fastest --track suzuka`

---

## 5. modify-vehicle-physics

**When:** adding or changing any term in the car model (tyres, aero, load transfer, powertrain, slope, kerbs).

**Expertise:** vehicle dynamics, numerical integration, PyTorch vectorisation.

**Steps**
1. Write the equation and its source in the module docstring.
2. Add an **analytic test** to `tests/unit/sim/` (see `AGENTS.md` §6.2).
3. Implement it in `sim/reference.py` (via the component modules) first.
4. Mirror it in `sim/batched/` as tensor ops only: no Python branches on tensor values; use `torch.where`.
5. Run parity: `uv run pytest tests/gpu/test_parity.py`.
6. Re-run QSS calibration on the reference track. If the calibrated lap time moved by more than 0.5%, note it and regenerate the calibration report.
7. Benchmark: `uv run python scripts/bench_sim.py --num-envs 4096`. No more than a 10% throughput loss without justification.

**Model reference (default)**
- Dynamic bicycle model in curvilinear coordinates:
  - `ṡ = (vx·cos(Δψ) − vy·sin(Δψ)) / (1 − n·κ(s))`
  - `ṅ = vx·sin(Δψ) + vy·cos(Δψ)`
  - `Δψ̇ = r − κ(s)·ṡ`
- Tyre lateral force: simplified Pacejka `F = D·sin(C·atan(B·α − E·(B·α − atan(B·α))))`, with `D = μ·Fz` and load sensitivity `μ(Fz) = μ0·(1 − k·(Fz − Fz0)/Fz0)`.
- Combined slip: friction ellipse scaling.
- Aero: `F_down = ½ρ·ClA·v²` split by `aero_balance`, and `F_drag = ½ρ·CdA·v²`.
- Powertrain: `F_drive = min(P_max / max(v, v_min), μ·Fz_rear)`. Brakes: `F_brake = pedal·F_brake_max`, split by `brake_bias`.
- Longitudinal load transfer: `ΔFz = m·ax·h_cg / L`.
- Slope: gravity component `−m·g·grade` along `s`. The normal load uses `g·cos(θ)`, plus a vertical-curvature term `m·v²·κ_vertical`.
- Integrator: semi-implicit Euler at `dt = 0.01 s` (RK4 available for validation).

**Pitfalls:** divide-by-zero at low speed in slip angles. Use the `v_min` blending already in `tyre.py`. Never copy the reference with Python `for` loops over envs.

---

## 6. calibrate-car

**When:** before training on a new track or era, or after a physics change.

**Steps**
1. Choose a target: usually the pole lap (or a top-3 average) for that session.
2. Run `apex car calibrate --track suzuka --era 2024 --target-ghost VER`.
3. `qss/calibrate.py`:
   - builds a racing line (the target ghost's own `n(s)` line by default, or min-curvature)
   - runs the QSS solver: a forward pass for acceleration limits (power, traction, drag, grade), a backward pass for braking limits, both bounded by the lateral grip `v ≤ sqrt(μ_eff(v)·g_eff / |κ|)` with aero-dependent `μ_eff`
   - optimises 2–3 scalars (`mu_scale`, `cl_a_scale`, `p_max_scale`) with `scipy.optimize.minimize` (Nelder–Mead, bounded) to match the target lap time **and** the speed-trace shape (an L2 loss on `v(s)` against the ghost, weighted)
4. Accept if: lap-time error ≤ 1.5%, speed-trace RMS ≤ 8 km/h, and all scales stay inside the physically plausible bounds in `configs/cars/f1_<era>.yaml`.
5. Save `configs/cars/calibrated/<track>_<era>.yaml` and `calibration_report.json`.

**Pitfalls:** don't calibrate against the AI's result. Fitting only the lap time can give an unphysical mix, such as huge grip with tiny power. The speed-trace term prevents that.

---

## 7. train-agent

**When:** starting, resuming or sweeping training.

**Expertise:** PPO/SAC, GPU training, experiment hygiene.

**Steps**
1. Smoke first: `apex train --config configs/training/<track>_ppo.yaml --smoke` (about 2 min). Confirm reward rises and there are no NaNs.
2. Full run: `apex train --config configs/training/<track>_ppo.yaml --name <track>_ppo_v1`.
   - Logs go to `runs/<name>/` (TensorBoard), with checkpoints every N updates and `best.pt` chosen by eval lap time.
3. Monitor: `uv run tensorboard --logdir runs/`. Key curves: `eval/lap_time`, `eval/completion_rate`, `train/episode_progress_m`, `train/offtrack_rate`, `ppo/approx_kl`, `ppo/entropy`, `ppo/clip_frac`, `sps`.
4. Curriculum (`rl/curriculum.py`):
   - Stage A: random start positions along the track at random low speeds, with a 15 s episode cap.
   - Stage B: full-lap episodes from a flying start.
   - Stage C: tighten `offtrack_tolerance_m` from 2.0 to 1.0, and fine-tune with lower entropy and a lower LR.
5. Resume: `apex train --resume runs/<name>/latest`.
6. Record the final checkpoint, config hash, git SHA and calibration hash in `runs/<name>/summary.json`.

**Default PPO setup:** 4096 envs, 32-step rollouts, an MLP of 3×256 with SiLU for both actor and critic, a tanh-squashed Gaussian, observation normalisation (running mean/var), reward scaling, GAE λ = 0.95, γ = 0.995 (an effective horizon of about 10 s at 20 Hz).

---

## 8. debug-rl-agent

**When:** reward plateaus, training diverges, or the agent is **suspiciously fast**.

**Checklist (in order)**
1. **Sanity of the env:** run `scripts/env_check.py`. A scripted controller (pure-pursuit on the centerline plus the QSS speed profile) should complete a lap within about 5% of the QSS time. If it can't, the bug is in the env or sim, not in the RL.
2. **Observation normalisation:** look at per-dimension running stats. Any std ≈ 0 or values over 10 after normalisation is a problem.
3. **Reward scale:** the per-step reward should be O(0.1–1). Check the off-track penalty against the progress reward.
4. **Termination rate:** if over 50% of episodes end in under 3 s, the curriculum start states are too hard.
5. **Action saturation:** if more than 80% of actions are at ±1, check the tanh squashing, the log-prob correction and the steering limit.
6. **PPO health:** `approx_kl` within 0.005–0.03, `clip_frac` < 0.3, entropy decaying smoothly.
7. **Suspiciously fast (exploit hunt):**
   - plot `n(s)` against the boundaries. Is it cutting through bounds the tolerance allows?
   - check for physics blow-ups (spikes in `vy`, `r`, slip angles) and integrator instability at high speed
   - compare the AI's speed trace with the QSS bound. **The AI must never beat the QSS bound on the same line by more than numerical noise.** If it does, the bug is real.
   - check that lap timing triggers only on a valid `s` wrap (no reverse-crossing exploits)
8. Fix the root cause and add a regression test. Don't paper over it with reward shaping.

---

## 9. compare-laps

**When:** producing the corner-by-corner story.

**Steps**
1. `apex compare --a agent:runs/suzuka_ppo_v1/best.pt --b ghost:2024_Q_VER_fastest --track suzuka`
2. `analysis/corners.py` detects corners where `|κ(s)| > κ_thr` (default 1/400 m⁻¹), merges regions closer than 30 m apart, and labels them from the track config.
3. For each corner, `analysis/metrics.py` computes:
   - **brake point** `s` where decel first exceeds 8 m/s² before the corner (`dv/ds`-based)
   - **entry speed** at `s_start`, **minimum speed** and its `s` (apex), and **exit speed** at `s_end + 50 m`
   - **line**: mean `n` at entry, apex and exit, and the lateral offset between laps
   - **time delta** gained or lost across the corner window, from `delta.py`
4. `delta.py`: Δt(s) = t_a(s) − t_b(s) on a common 1 m `s` grid.
5. `report.py` outputs `analysis.json` (for the viewer), `report.md`, and figures: speed vs s, Δt vs s, line overlay per corner.
6. Generate a headline automatically: the corner with the biggest gain, plus its main cause ("brakes 14 m later", "+9 km/h minimum speed", "wider entry").

**Pitfalls:** don't compare at matched times. Always compare at matched distance `s`. With low-rate ghost data, minimum speed can be under-sampled, so report it with a `±` band based on sample spacing.

---

## 10. export-replay

**When:** building the bundle the viewer loads.

**Bundle layout (`REPLAY_FORMAT_VERSION = 1`)**
```
replays/<name>/
├── manifest.json        # version, track hash, calibration report, cars[], fps, duration
├── track.glb            # road mesh + boundaries + centerline curve
├── cars/<id>.bin        # float32 interleaved [t, x, y, z, yaw, pitch, roll, speed, throttle, brake, gear]
└── analysis.json        # corners, metrics, delta curves (decimated)
```

**Steps**
1. `apex export replay --track suzuka --agent runs/suzuka_ppo_v1/best.pt --ghosts 2024:VER,2024:LEC --name suzuka_ai_vs_ver`
2. The exporter runs a deterministic evaluation lap, converts `(s, n)` to world coordinates with orientation from the track frame and the car's yaw relative to it, and samples at 50 Hz.
3. It validates against the schema and refuses to export if the calibration report or the track hash is missing (an `AGENTS.md` invariant).
4. Open it in the viewer and confirm that the cars start together, ghost lap times match the official times, and nothing clips through the road.

---

## 11. viewer-feature

**When:** adding cameras, HUD elements, effects or controls.

**Expertise:** TypeScript, Three.js, rendering performance.

**Steps**
1. Read `viewer/src/replay/schema.ts`. Features may only use data from the bundle.
2. Put scene objects in `scene/`, cameras in `cameras/`, and DOM/HUD in `ui/`. One class per feature, with `init()`, `update(t)` and `dispose()`.
3. No per-frame allocations. Preallocate `Vector3`/`Quaternion`.
4. Add a Vitest test for any data logic (interpolation, delta lookup).
5. Check fps in the dev overlay (`?debug=1`) with 1 AI car and 5 ghosts at 1080p. It must hold 60.
6. Add a screenshot to the PR.

**Ideas backlog:** a brake-point marker on the road, a speed-coloured trail, a split-screen two-car chase cam, a picture-in-picture minimap, slow-motion corner replay.

---

## 12. cross-era-race

**When:** racing ghosts from different seasons on one track.

**Steps**
1. Use one **reference track build** (usually the most recent season).
2. Extract ghosts from the other seasons against the reference asset: `apex ghost extract ... --track suzuka --reference-build <hash>`.
3. Layout check: the ghost's points must sit inside the reference bounds at ≥ 98%, and the length match must be within 0.5%. If the check fails, the layout changed, so refuse and explain.
4. Optional `--normalise` mode: scale each ghost's time so its lap equals the field-median ratio for its era (labelled clearly as "era-normalised" in the viewer).
5. Export with an `era` tag per car, so the viewer shows the year badges.

**Pitfalls:** the same circuit name doesn't mean the same layout. Chicane changes and resurfacing all matter. Always run the layout check.

---

## 13. make-clip

**When:** rendering a shareable video.

**Steps**
1. Pick the story: `apex compare ...` → headline corner.
2. `apex clip --replay replays/suzuka_ai_vs_ver --preset corner --corner "130R" --camera chase --duration 12 --out clips/130r.mp4`
   - Under the hood this runs the viewer headless (Playwright + Chromium), steps time deterministically per frame, captures frames, and encodes with ffmpeg (`-c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p`).
3. Presets: `full-lap`, `corner`, `split-screen`, `topdown-lines`, `vertical-9x16` (for shorts).
4. Overlays: the title card ("AI vs VER · Suzuka 2024 Q · simulated car calibrated to ±1.5%"), the delta bar and the corner name.
5. Every clip includes the honesty label. Don't remove it.

---

## 14. performance-profiling

**Sim (GPU)**
```bash
uv run python scripts/bench_sim.py --num-envs 1024 2048 4096 8192
uv run python -m torch.utils.bottleneck scripts/bench_sim.py --num-envs 4096
```
- Use `torch.compile` on the step function. Keep all state in one contiguous `(N, S)` tensor and avoid host-device syncs (`.item()`, `.cpu()`) inside the loop.
- Track lookups are a gather from a precomputed 1 m grid with linear interpolation. Don't use searchsorted per step.

**Viewer**
- Use the Chrome Performance panel and the `?debug=1` stats overlay.
- Use instanced meshes for kerbs and markers, a single merged geometry for the road, and update trail buffers with `setDrawRange`.
