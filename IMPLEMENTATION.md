# IMPLEMENTATION.md

The complete step-by-step development roadmap for **APEX ZERO**, written for Claude Code (and humans) to follow phase by phase.

**How to use this file**
- Work on phases **in order**. A phase starts only when its dependencies' acceptance criteria are met.
- Each task has an ID (`P2.3`). Reference it in commits: `feat(track): P2.3 periodic smoothing spline`.
- Tick `[x]` when a task meets the definition of done in `AGENTS.md` §9.
- Each phase ends with a **gate**: every acceptance criterion must pass before you move on.
- Workflows for each kind of task are in `SKILLS.md`. Conventions are in `AGENTS.md`.

---

## Overview

| Phase | Name | Depends on | Main deliverable | Est. effort |
|---|---|---|---|---|
| 0 | Scaffolding and tooling | — | Repo skeleton, CI, CLI stub | 1–2 days |
| 1 | Data ingestion | 0 | Cached, validated sessions | 3–4 days |
| 2 | Track reconstruction | 1 | `TrackAsset` + glTF for Suzuka | 5–7 days |
| 3 | Ghost extraction | 2 | Validated driver ghosts | 2–3 days |
| 4 | Reference vehicle sim | 2 | NumPy dynamic bicycle model | 5–7 days |
| 5 | QSS solver and calibration | 3, 4 | Calibrated 2024 car | 3–5 days |
| 6 | Batched GPU sim | 4 | Torch sim with parity, ≥ 2M steps/s | 4–6 days |
| 7 | RL environment | 5, 6 | Gymnasium + vectorised env | 3–4 days |
| 8 | RL training | 7 | Agent that completes clean Suzuka laps | 7–14 days |
| 9 | Evaluation and analysis | 3, 8 | Corner-by-corner reports | 3–4 days |
| 10 | Replay export and 3D viewer | 9 | Three.js head-to-head viewer | 7–10 days |
| 11 | Multi-track and cross-era | 10 | 3+ tracks, era races | 4–6 days |
| 12 | Clips, polish and release | 11 | Clip renderer, docs, v1.0 | 4–6 days |

Critical path: 0 → 1 → 2 → 4 → 6 → 7 → 8 → 9 → 10. Phases 3 and 5 can run in parallel with 4 and 6.

---

## Phase 0: Scaffolding and tooling

**Goal:** a clean, reproducible repo where every later phase can drop in code and tests.

**Tasks**
- [x] **P0.1** Initialise the repo: `pyproject.toml` (uv, Python 3.12, `src/` layout, package `apex_zero`), the `.gitignore` (`data/`, `runs/`, `replays/`, `clips/`, `.env`, `viewer/node_modules`, `viewer/dist`), `.env.example`, MIT `LICENSE`.
- [x] **P0.2** Dependencies:
  - runtime: `numpy`, `scipy`, `pandas`, `pyarrow`, `pydantic>=2`, `pyyaml`, `typer`, `rich`, `structlog`, `fastf1`, `httpx`, `trimesh`, `pygltflib`, `matplotlib`, `gymnasium`, `torch`, `tensorboard`
  - extras: `gpu` (CUDA torch index), `dev` (`pytest`, `pytest-cov`, `hypothesis`, `ruff`, `mypy`, `pandas-stubs`, `types-PyYAML`), `wandb` (optional)
- [x] **P0.3** Create the folder structure exactly as in `AGENTS.md` §3, with an `__init__.py` and a module docstring per package.
- [x] **P0.4** Tooling config: ruff (line length 100, rules `E,F,I,B,UP,N,SIM,NPY`), mypy strict for `src/`, and pytest markers `slow`, `gpu`, `network`.
- [x] **P0.5** `apex_zero/errors.py`, `utils/logging.py` (structlog), `utils/seeding.py`, `utils/units.py` (kph↔mps, deg↔rad, and the telemetry unit constants, filled in during P1).
- [x] **P0.6** CLI skeleton (`cli.py`, Typer) with subcommand groups `data`, `track`, `ghost`, `car`, `qss`, `train`, `eval`, `compare`, `export`, `clip`, `doctor`. `apex doctor` checks the Python version, torch, CUDA availability, cache dir permissions and the Node version.
- [x] **P0.7** `tests/unit/test_layering.py`: parse imports with `ast` and enforce the layer table in `AGENTS.md` §2.
- [x] **P0.8** Viewer scaffold: `viewer/` with Vite + TS + Three.js, ESLint, Prettier, Vitest, and a spinning-cube placeholder scene.
- [x] **P0.9** GitHub Actions CI: jobs `python` (ruff, mypy, pytest fast suite on ubuntu-latest, CPU torch) and `viewer` (lint, typecheck, test, build). Cache uv and pnpm.
- [x] **P0.10** Add the five docs files (`README.md`, `AGENTS.md`, `CLAUDE.md`, `SKILLS.md`, `IMPLEMENTATION.md`) and a `docs/decisions/0001-architecture.md` ADR.

**Deliverables:** a repo that installs with `uv sync --extra dev`, `apex --help` and `apex doctor` both work, and CI is green.

**Gate (acceptance criteria)**
- `uv run apex doctor` exits 0 on a clean machine (CUDA reported as optional).
- The quality gate in `AGENTS.md` §4 passes, both locally and in CI.
- The layering test fails when a forbidden import is added (check manually once).

---

## Phase 1: Data ingestion

**Goal:** reliable, cached, schema-validated access to positions and telemetry for any session from 2018 on (FastF1) and from 2023 on (OpenF1).

**Dependencies:** Phase 0.

**Tasks**
- [x] **P1.1** `data/cache.py`: cache layout under `APEX_CACHE_DIR` (FastF1's own cache) and `APEX_PROCESSED_DIR/<year>/<event_slug>/<session>/`. Keys are deterministic. Metadata goes in `session_meta.json`.
- [x] **P1.2** `data/fastf1_loader.py`:
  - `load_session(year, event, session) -> SessionHandle` (resolves the event via the event schedule, enables the cache)
  - `iter_driver_laps(handle, driver=None, accurate_only=False)`
  - `export_processed(handle)`: per-driver, per-lap Parquet files for position (`SessionTime, X, Y, Z, Status`) and car data (`SessionTime, Speed, RPM, nGear, Throttle, Brake, DRS`), plus a lap table (`LapNumber, LapTime, Sector1-3, Compound, TyreLife, IsAccurate, PitIn/Out`).
- [x] **P1.3** `data/schemas.py`: column contracts and validators (dtypes, monotonic time, NaN budget ≤ 2% per channel per lap).
- [ ] **P1.4** **Pin units and axes.** Write `tests/network/test_units.py`, which loads one known session and checks that the fastest lap's polyline length, after the unit conversion, is within 2% of the official circuit length. Record the conversion factor in `utils/units.py` as `FASTF1_POS_TO_M`. Do the same for OpenF1 (`OPENF1_POS_TO_M`), and check whether its axes match FastF1's, recording any transform.
  - _Status (2026-11-18): OpenF1 half done and passing at Suzuka and Monza (docs/data.md). FastF1 half and the axis comparison are written but not run: FastF1 3.9 needs an F1 account (`F1_USERNAME`/`F1_PASSWORD`). `FASTF1_POS_TO_M = 0.1` is from FastF1's docs, not yet verified._
- [x] **P1.5** `data/openf1_client.py`: an httpx client with base URL `https://api.openf1.org/v1`, endpoints `sessions`, `meetings`, `laps`, `location`, `car_data`, a token-bucket limit (~3 req/s), exponential backoff, and an on-disk cache keyed by URL hash. Map `session_key` to the FastF1 session.
- [x] **P1.6** CLI: `apex data fetch`, `apex data ls`, `apex data info <year> <event> <session>`.
- [x] **P1.7** Fixtures: a trimmed real lap (one driver, one lap, about 300 KB) and a synthetic session generator `tests/fixtures/synthetic.py` (ellipse and figure-eight tracks with noise and elevation, multiple "drivers").
- [x] **P1.8** Unit tests with fixtures (no network) and network tests marked.

**Deliverables:** `apex data fetch --year 2024 --event Japan --session Q` produces validated Parquet for all drivers.

**Gate**
- Fetching the same session twice makes zero network calls the second time.
- The unit-conversion test passes for at least 2 circuits (Suzuka plus one other) and both sources where they're available.
- Schema validation flags a deliberately corrupted fixture.
- Fast-suite coverage of `data/` ≥ 80% (network paths excluded).

---

## Phase 2: Track reconstruction

**Goal:** an accurate 3D `TrackAsset` (centerline, width, elevation, curvature, banking) and a glTF mesh, built only from data.

**Dependencies:** Phase 1. Workflow: `SKILLS.md` → *reconstruct-track*.

**Tasks**
- [ ] **P2.1** `track/asset.py`: a `TrackAsset` frozen dataclass with arrays on a 1 m `s` grid: `s, x, y, z, psi, kappa, grade, bank, w_left, w_right`, plus `length`, `name`, `source_sessions`, `build_hash`, `params`. `.save(path)` and `.load(path)` (`.npz` + `meta.json`).
- [ ] **P2.2** `track/frenet.py`: vectorised `world_to_frenet(points, s_hint=None, window_m=50)` (windowed, handles self-intersection) and `frenet_to_world(s, n)`, plus interpolation helpers for every asset channel at arbitrary `s` (periodic).
- [ ] **P2.3** `track/centerline.py`: seed from the fastest accurate lap, run iterative median refinement (bins over `s`, 2–3 iterations), and fit a periodic smoothing spline, with smoothing params from `configs/tracks/<name>.yaml`.
- [ ] **P2.4** Start/finish alignment: `s = 0` at the timing line (the interpolated position at lap start across fastest laps).
- [ ] **P2.5** `track/width.py`: percentile-based left/right boundaries with priors, margin and smoothing. Exclude pit and out laps.
- [ ] **P2.6** `track/elevation.py`: heavy z smoothing, `grade(s)`, vertical curvature, and an optional banking estimate (a regression of z on n per s window, clipped to ±10°).
- [ ] **P2.7** `track/mesh.py`: generate a road ribbon (left/right boundaries at 1 m), kerb strips at corners (from curvature), a run-off plane, and centerline and boundary polylines as glTF nodes. Normals and UVs for a tiling asphalt texture. Export `track.glb`.
- [ ] **P2.8** `track/reconstruct.py`: the orchestrator `build_track(config, sessions) -> TrackAsset`.
- [ ] **P2.9** CLI: `apex track build`, `apex track inspect <name> [--debug]`. Inspect saves `docs/media/tracks/<name>_{xy,curvature,width,elevation}.png`.
- [ ] **P2.10** `configs/tracks/suzuka.yaml`, with `self_intersecting: true` and the official length.
- [ ] **P2.11** Tests:
  - synthetic ellipse: centerline RMS < 0.5 m, length error < 0.2%, curvature error < 5% relative
  - synthetic figure-eight with elevation: correct `s` continuity through the crossing, z RMS < 0.5 m
  - property test (hypothesis): `frenet_to_world(world_to_frenet(p)) ≈ p` for points within 20 m of the line
  - real Suzuka (marked `slow` and `network`): length within 0.5% of official, ≥ 99% of fastest-lap points inside bounds

**Deliverables:** `data/processed/tracks/suzuka/{asset.npz, meta.json, track.glb}`, plus inspection plots.

**Gate**
- All synthetic tests pass. Suzuka passes the real-data checks.
- The track build takes < 60 s on a laptop CPU.
- `track.glb` opens correctly in a glTF viewer: no flipped normals, and the loop is closed.

---

## Phase 3: Ghost extraction

**Goal:** turn any driver's lap into a validated ghost aligned to the track asset.

**Dependencies:** Phase 2. Workflow: `SKILLS.md` → *extract-ghost*.

**Tasks**
- [ ] **P3.1** `ghosts/ghost.py`: a `Ghost` dataclass with `t, s, n, x, y, z, speed, throttle, brake, gear, drs` at 50 Hz, plus meta (driver, team, year, session, lap number, official lap time, source versions, track build hash).
- [ ] **P3.2** `ghosts/extract.py`: lap selection (`fastest`, number, nearest lap time) and the merge of position and car data (interpolation rules from `SKILLS.md`).
- [ ] **P3.3** `ghosts/align.py`: windowed Frenet projection, monotone `s` (isotonic), exact start/finish crossing interpolation, and 50 Hz PCHIP resampling.
- [ ] **P3.4** Validation: lap time within 0.05 s, ≥ 99% inside bounds, and `ds/dt` against `Speed` RMS within 5%. Raise `GhostAlignmentError` with diagnostics if any check fails.
- [ ] **P3.5** CLI: `apex ghost extract`, `apex ghost ls`, `apex ghost plot <id>` (speed vs s, n vs s).
- [ ] **P3.6** Tests on the synthetic session (known ground-truth lap) and the trimmed real fixture.

**Deliverables:** ghosts for the 2024 Suzuka Q top 5.

**Gate**
- The top-5 ghosts pass validation.
- A synthetic ghost reproduces its ground-truth lap time within 0.01 s.

---

## Phase 4: Reference vehicle simulation (NumPy)

**Goal:** a correct, readable, deterministic single-car simulator that's the ground truth for everything else.

**Dependencies:** Phase 2 (track asset, read-only). Workflow: `SKILLS.md` → *modify-vehicle-physics*.

**Tasks**
- [ ] **P4.1** `sim/params.py`: `CarParams` (SI units): mass, wheelbase, CG position `lf/lr`, `h_cg`, yaw inertia `Iz`, `track_width`, tyre coefficients (`B, C, E, mu0, load_sensitivity, Fz0`), `ClA`, `CdA`, `aero_balance`, `rho`, `P_max`, `F_brake_max`, `brake_bias`, `delta_max`, `delta_rate_max`, `v_min`. Load from `configs/cars/f1_<era>.yaml`.
- [ ] **P4.2** `configs/cars/f1_2018.yaml` … `f1_2026.yaml`: plausible public-domain-level estimates with min/max bounds per parameter, and comments on each assumption. (2026 regulations: lighter car, more electric power share, active aero. Model active aero as two `ClA/CdA` modes switched by a straight/corner flag from curvature.)
- [ ] **P4.3** `sim/tyre.py`: simplified Pacejka lateral, longitudinal force from the demanded force capped by `μ·Fz`, friction-ellipse combined slip, load sensitivity, and low-speed blending.
- [ ] **P4.4** `sim/aero.py`, `sim/powertrain.py` (power-limited drive, rear-wheel drive, brake split).
- [ ] **P4.5** `sim/vehicle.py`: state `[s, n, dpsi, vx, vy, r]` (wheel speeds optional, behind a flag), the curvilinear dynamic bicycle model (equations in `SKILLS.md` §5), longitudinal load transfer, grade and vertical-curvature effects on normal load, and banking on the lateral force balance.
- [ ] **P4.6** `sim/integrator.py`: semi-implicit Euler (default, `dt = 0.01`) and RK4 (validation). Steering rate limiting and actuator first-order lag.
- [ ] **P4.7** `sim/reference.py`: `ReferenceSim(track, params, dt)` with `reset(state)`, `step(action) -> state, info` (info: `ax, ay, slip_f, slip_r, Fz, offtrack`).
- [ ] **P4.8** `scripts/env_check.py`: a scripted driver (pure pursuit with lookahead plus a PI speed controller tracking a target speed profile) to drive full laps.
- [ ] **P4.9** Analytic tests (`AGENTS.md` §6.2): steady-state cornering, top speed, braking distance, grade, and a Euler vs RK4 agreement test at `dt = 0.005`.
- [ ] **P4.10** Determinism test: bit-identical trajectories across runs.

**Deliverables:** the scripted driver completes a Suzuka lap in the reference sim, with a plot of speed vs s.

**Gate**
- Every analytic test is within tolerance.
- The scripted driver completes 10/10 laps without going off track at 90% of the QSS speed (the QSS speed can be stubbed with a constant-μ estimate until Phase 5).
- No NaN or inf in 10⁶ random-action steps (with resets on termination).

---

## Phase 5: QSS lap-time solver and calibration

**Goal:** a fast physics bound on lap time, used to calibrate the car to real eras and to catch an AI that's impossibly fast.

**Dependencies:** Phases 3 and 4. Workflow: `SKILLS.md` → *calibrate-car*.

**Tasks**
- [ ] **P5.1** `qss/solver.py`: given a line `n(s)`, compute the path curvature, then the velocity profile from (a) lateral-limit apex speeds with aero-dependent grip, (b) a forward pass under power, traction, drag and grade, and (c) a backward pass under braking, drag and grade. Use the g-g-v envelope from the same tyre, aero and powertrain functions as the sim. Integrate `t = ∫ ds / v`.
- [ ] **P5.2** `qss/line_opt.py`: a minimum-curvature line within bounds (quadratic programme via `scipy.optimize` or `osqp` if approved), as a baseline line independent of the ghosts.
- [ ] **P5.3** `qss/calibrate.py`: fit `mu_scale`, `cl_a_scale` and `p_max_scale` to the target ghost (lap time plus the speed-trace loss), within the bounds in the car config. Output `configs/cars/calibrated/<track>_<era>.yaml` and `calibration_report.json` (target, achieved, error %, speed RMS, params, hashes).
- [ ] **P5.4** CLI: `apex qss solve`, `apex car calibrate`.
- [ ] **P5.5** Cross-check: the scripted driver following the QSS line and profile in the reference sim must lap within 3% of the QSS time. This validates that QSS and the sim agree.
- [ ] **P5.6** Tests: a constant-radius circle (an analytic lap time), a straight-plus-hairpin with known braking distance, and calibration recovering known params from a synthetic ghost generated by the sim itself.

**Deliverables:** a calibrated 2024 car for Suzuka, and a report.

**Gate**
- Calibration error ≤ 1.5% on lap time, and speed-trace RMS ≤ 8 km/h against the target ghost.
- Synthetic param recovery within 2% per parameter.
- The QSS-vs-sim cross-check is within 3%.

---

## Phase 6: Batched GPU simulation (PyTorch)

**Goal:** the same physics as the reference sim, vectorised over thousands of cars on the GPU.

**Dependencies:** Phase 4. Workflow: *modify-vehicle-physics*, *performance-profiling*.

**Tasks**
- [ ] **P6.1** `sim/batched/track_tensors.py`: upload the track asset channels as `float32` tensors, with periodic linear interpolation by gather (`idx = floor(s)`, `frac`).
- [ ] **P6.2** `sim/batched/vehicle.py`: tensorised dynamics with no data-dependent Python control flow (`torch.where`), and the state as a `(N, S)` tensor.
- [ ] **P6.3** `sim/batched/sim.py`: `BatchedSim(track, params, num_envs, device, dt)` with `reset(mask, states)` and `step(actions) -> states, info`. Partial resets by mask. A params tensor per env (for domain randomisation later).
- [ ] **P6.4** `torch.compile` the step function, with a fallback flag. Benchmark script `scripts/bench_sim.py`.
- [ ] **P6.5** `tests/gpu/test_parity.py`: the same initial states and action sequences (random and scripted) for 2,000 steps, with relative error < 1e-3 on every state component. The CPU-device variant of the batched sim runs in CI (non-GPU) at small N.
- [ ] **P6.6** `scripts/parity_bisect.py`: find the first divergent step and component.

**Deliverables:** a batched sim with a benchmark table in `docs/perf.md`.

**Gate**
- Parity passes on both CPU and CUDA.
- ≥ 2M env-steps/s at N = 4096 on an RTX 4090-class GPU (record the actual hardware in `docs/perf.md`; on smaller GPUs, ≥ 0.5M steps/s).
- No host-device sync inside `step` (checked with `torch.cuda.set_sync_debug_mode("error")` in a test).

---

## Phase 7: RL environment

**Goal:** Gymnasium-compatible single and vectorised environments with a well-designed observation, action and reward.

**Dependencies:** Phases 5 and 6.

**Tasks**
- [ ] **P7.1** `env/observation.py`:
  - ego: `vx, vy, r, n / half_width, dpsi, last_action (2), slip_f, slip_r`
  - lookahead at configurable distances (default 11 points, 10–300 m): `kappa`, `w_left`, `w_right`, `grade`, plus optional `bank`
  - normalisation constants from config, with running mean/variance handled in the RL layer
- [ ] **P7.2** Action: `[steer, pedal] ∈ [-1,1]²`, with steering rate limiting in the sim. Action repeat of 5 (20 Hz policy over a 100 Hz sim).
- [ ] **P7.3** `env/reward.py`: `r = w_p·Δs/Δs_ref + w_off·offtrack + w_sr·|Δsteer|² + w_time·(−dt)`. Δs is unwrapped across the finish line. Rewards are returned as components in `info` for logging.
- [ ] **P7.4** `env/termination.py`: off-track beyond tolerance, going backwards for more than 2 s, speed < 2 m/s for more than 3 s, physics NaN guard, and a time limit. Lap-completion detection only on a forward `s` wrap after passing ≥ 95% of `L` (anti-exploit).
- [ ] **P7.5** `env/racing_env.py`: `RacingEnv(gymnasium.Env)` on the reference sim, for debugging, tests and eval.
- [ ] **P7.6** `env/vec_env.py`: `TorchVecEnv` on the batched sim, auto-reset, and start-state sampling (random `s`, random `n` within ±30% of the width, speed from 0.5–1.0 × the QSS speed at `s`, small heading noise).
- [ ] **P7.7** Tests: Gymnasium `check_env` on `RacingEnv`; obs/reward parity between `RacingEnv` and `TorchVecEnv` for identical seeds; lap-completion logic (including a reverse-crossing exploit attempt, which must not count); reward unwrap at `s` wrap.

**Deliverables:** envs ready for training, and `scripts/env_check.py` updated to drive through the env API.

**Gate**
- `check_env` passes. Parity between the single and vec envs holds.
- The scripted driver through the env gets a lap time within 5% of QSS.
- The exploit tests pass.

---

## Phase 8: RL training

**Goal:** an agent that reliably completes clean, fast laps of Suzuka in the calibrated car.

**Dependencies:** Phase 7. Workflows: *train-agent*, *debug-rl-agent*.

**Tasks**
- [ ] **P8.1** `rl/networks.py`: separate actor and critic MLPs (3×256, SiLU, orthogonal init), a tanh-squashed Gaussian with state-independent log-std, and a correct log-prob correction.
- [ ] **P8.2** `rl/ppo.py`: a GPU-native PPO (CleanRL-style) with GAE, clipped objective, value clipping (optional), advantage normalisation, grad-norm clip, KL early stop, and running observation normalisation plus reward scaling.
- [ ] **P8.3** `rl/train.py`: the loop, TensorBoard logging (and W&B optional), checkpoints (`latest.pt`, `best.pt` by eval lap time, a ring buffer of the last 5), resume, and `summary.json` (git SHA, config hash, calibration hash, track hash, hardware).
- [ ] **P8.4** `rl/curriculum.py`: stages A → B → C (`SKILLS.md` §7), advanced on metrics (completion rate ≥ 90% → next stage).
- [ ] **P8.5** `rl/evaluate.py`: deterministic policy (mean action), flying-lap protocol (one out-lap, then timed laps), and track-limit validation of every timed lap.
- [ ] **P8.6** `rl/sac.py`: a SAC baseline on the same vec env, for comparison (lower priority).
- [ ] **P8.7** Optional domain randomisation (±3% grip, ±2% power) for robustness, off by default for headline runs.
- [ ] **P8.8** Configs: `configs/training/smoke.yaml` (CPU, 64 envs, 2 min), `suzuka_ppo.yaml` (full).
- [ ] **P8.9** Tests: PPO learns `Pendulum`-like toy dynamics (a fast CPU sanity check), a smoke training run improves progress by > 50% in 2 minutes on CPU (marked `slow`), and checkpoint save/load round-trip.

**Deliverables:** `runs/suzuka_ppo_v1/` with `best.pt`, curves and summary.

**Gate**
- Eval completion rate is 100% over 20 laps, with all laps inside track limits.
- Best AI lap time is within 3% of the calibrated QSS bound **and** not faster than the QSS bound on its own line by more than 0.3% (anything faster is an exploit, so go to *debug-rl-agent*).
- Training is reproducible: two seeds land within 1% of each other's lap time.

---

## Phase 9: Evaluation and analysis

**Goal:** turn laps into a corner-by-corner story that holds up.

**Dependencies:** Phases 3 and 8. Workflow: *compare-laps*.

**Tasks**
- [ ] **P9.1** `analysis/delta.py`: time-at-distance curves and Δt(s) between any two laps (agent or ghost).
- [ ] **P9.2** `analysis/corners.py`: curvature-threshold detection, merging, and naming from the track config. Add Suzuka corner names (Turn 1–2, the S Curves, Dunlop, Degner 1 and 2, Hairpin, Spoon, 130R, Casio Triangle).
- [ ] **P9.3** `analysis/metrics.py`: brake point, entry/min/exit speed, apex `s`, line `n` at entry/apex/exit, time gained per corner, and uncertainty bands from the ghost sample spacing.
- [ ] **P9.4** `analysis/report.py`: `analysis.json`, `report.md`, figures (speed vs s, Δt vs s, per-corner line overlay), and an automatic headline sentence for the biggest gain and its cause.
- [ ] **P9.5** CLI: `apex compare`, and `apex eval --ghosts ... --export` to chain eval → analysis → export.
- [ ] **P9.6** Tests: synthetic laps with known differences (a later brake point by 10 m, +5 km/h minimum speed) are detected correctly, and the delta sign convention is documented and tested.

**Deliverables:** `reports/suzuka_ai_vs_VER/` with JSON, markdown and figures.

**Gate**
- Synthetic difference detection is exact within the sample resolution.
- The report for AI vs VER includes the calibration report and its uncertainty notes.

---

## Phase 10: Replay export and 3D viewer

**Goal:** a smooth, honest, good-looking 3D head-to-head.

**Dependencies:** Phase 9. Workflows: *export-replay*, *viewer-feature*.

**Tasks: export (Python)**
- [ ] **P10.1** `export/replay_format.py`: a Pydantic schema for `manifest.json`, the binary layout, and `REPLAY_FORMAT_VERSION = 1`.
- [ ] **P10.2** `export/exporter.py`: deterministic agent lap → world poses (yaw from the track `psi + dpsi`; pitch from grade; roll from bank plus a small lateral-g roll term for visuals only, flagged as such), ghosts → poses, 50 Hz float32 binaries, glTF track copy, `analysis.json` (decimated). Enforces the credibility invariants.
- [ ] **P10.3** Tests: schema round-trip, refusal without a calibration report, pose continuity (no yaw jumps across the 2π wrap).

**Tasks: viewer (TypeScript)**
- [ ] **P10.4** `replay/schema.ts` (mirrors Python), `loader.ts` (fetch, or drag-and-drop a folder or zip), `interpolate.ts` (binary search on t plus slerp for orientation).
- [ ] **P10.5** `scene/track.ts`: load `track.glb`, asphalt and kerb materials, a sky, a directional light with shadows, and a ground plane.
- [ ] **P10.6** `scene/car.ts`: a generic low-poly open-wheel car model (original geometry, no liveries), colour per car, driver-code label sprite, and an era badge.
- [ ] **P10.7** `scene/trails.ts`: line trails coloured by speed or by car, with a toggle.
- [ ] **P10.8** Cameras: chase (spring-damped follow), TV (auto-switching fixed positions along the track by `s`), and top-down (orthographic).
- [ ] **P10.9** UI: timeline scrubber with corner markers, a play/pause/speed control (0.25×–4×), a live delta bar (AI vs the selected ghost), a corner table (from `analysis.json`), and a permanent honesty label.
- [ ] **P10.10** `record/`: MediaRecorder capture of the canvas to WebM (in-browser clip capture).
- [ ] **P10.11** Keyboard controls (README table), and `?debug=1` for the stats overlay.
- [ ] **P10.12** Vitest: schema parse, interpolation accuracy, delta lookup.

**Deliverables:** `pnpm --dir viewer dev` plays `replays/suzuka_ai_vs_ver/` with the AI and 3+ ghosts.

**Gate**
- 60 fps at 1080p with 1 AI car and 5 ghosts on a mid-range laptop GPU.
- Ghost lap times shown in the UI match the official times to 0.01 s.
- Cars sit on the road surface everywhere: max vertical error < 5 cm, visually verified at the 3 highest-grade points.
- Screenshots saved in `docs/media/` and linked from the README.

---

## Phase 11: Multi-track and cross-era racing

**Goal:** generalise beyond Suzuka and enable 2018-vs-2026 style matchups.

**Dependencies:** Phase 10. Workflows: *add-new-track*, *cross-era-race*.

**Tasks**
- [ ] **P11.1** Add at least 3 more tracks with different character: Monaco (tight, low speed), Monza (power), Silverstone (fast sweeps). Run the full pipeline per track.
- [ ] **P11.2** A reference-build mechanism: ghosts from any season are aligned to a chosen track build, with the layout check (≥ 98% inside bounds, length within 0.5%) and a clear refusal message when it fails.
- [ ] **P11.3** Era car configs, calibrated per track and era where reference data exists. Store the calibration matrix in `docs/calibration.md`.
- [ ] **P11.4** The optional era-normalisation mode, clearly labelled in the viewer and in clips.
- [ ] **P11.5** Multi-track training: one policy across tracks (track ID not in the observation, so it relies on lookahead features), compared with per-track policies. Report the generalisation gap.
- [ ] **P11.6** The viewer shows the year badge per car and has an era filter.
- [ ] **P11.7** Tests: the layout-change refusal (a synthetic track with a moved chicane), and normalisation math.

**Deliverables:** "2019 HAM vs 2024 VER vs AI at Suzuka", plus 3 other tracks with replays.

**Gate**
- Every new track passes the Phase 2 real-data checks and the Phase 8 completion gate.
- Cross-era replays pass the layout check or refuse with a clear message.

---

## Phase 12: Clips, polish and release

**Goal:** make it easy to share and easy for others to run and contribute to.

**Dependencies:** Phase 11. Workflow: *make-clip*.

**Tasks**
- [ ] **P12.1** `apex clip`: a headless renderer (Playwright + Chromium driving the viewer in deterministic frame-step mode via `?render=1`), frame capture, and an ffmpeg encode (`libx264 -preset veryfast -crf 18 -pix_fmt yuv420p`).
- [ ] **P12.2** Presets: `full-lap`, `corner`, `split-screen`, `topdown-lines`, `vertical-9x16`. A title card and an end card with the honesty label and data credits.
- [ ] **P12.3** An auto-story: `apex story --track suzuka --vs VER` picks the top 3 corners by |Δt| and renders a 30–45 s montage.
- [ ] **P12.4** Optionally deploy a static viewer build (GitHub Pages) with 2–3 bundled demo replays (small, decimated).
- [ ] **P12.5** Docs: finish the README with real screenshots and GIFs, `docs/physics.md` (model and assumptions), `docs/data.md` (sources, limits, units), `docs/fairness.md` (calibration methodology), and `CONTRIBUTING.md`.
- [ ] **P12.6** Release hygiene: pinned versions in `uv.lock` and `pnpm-lock.yaml`, `CHANGELOG.md`, a v1.0.0 tag, and pretrained checkpoints published as release assets (if within data terms).
- [ ] **P12.7** An end-to-end test (`slow`): synthetic track → ghost → calibrate → smoke train → eval → export → headless render of 2 s.

**Deliverables:** v1.0.0, 3+ shareable clips, and a public demo viewer.

**Gate**
- A fresh clone gets to the first replay by following the README only (tested on a clean machine or container).
- The end-to-end test passes in under 10 minutes on CPU.
- Every published clip carries the honesty label and data credits.

---

## Cross-cutting requirements

| Area | Requirement |
|---|---|
| Reproducibility | Every artefact (track, ghost, calibration, run, replay) records its input hashes, git SHA and tool versions. |
| Determinism | Fixed seeds give identical CPU results. GPU results are reproducible within float tolerance. |
| Observability | structlog JSON logs, TensorBoard for training, and timing logs for every pipeline stage. |
| Performance | Budgets in `AGENTS.md` §7, tracked in `docs/perf.md`. |
| Credibility | Invariants in `AGENTS.md` §7 are enforced in code, not just in docs. |
| Legal | No F1 or team IP assets. Data credits on every output. |

## Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Position data too noisy or sparse for a clean rebuild | Bad track, bad comparisons | Multi-lap, multi-driver fusion; median refinement; synthetic ground-truth tests; FP sessions as extra input |
| Units or axes misread | Everything wrong in scale | P1.4 pinning tests against official lengths |
| Self-intersecting tracks (Suzuka) | Projection jumps | Windowed Frenet projection with `s` continuity |
| Car model too simple or miscalibrated | "AI beat VER" is meaningless | QSS calibration plus speed-trace fit; the QSS bound as an exploit detector; published calibration report |
| RL exploits (cutting, physics glitches) | Fake results | Strict termination, lap-validity checks, the QSS-bound check, regression tests |
| GPU throughput too low | Slow iteration | Batched sim, `torch.compile`, no host syncs, smaller-N fallback |
| Data access or terms change | Pipeline breaks | Aggressive caching, pinned versions, schema tests, no redistribution of raw data |
| Cross-era layout changes | Invalid comparisons | Layout check with refusal |

## Milestone demos

1. **M1 (end of Phase 2):** a rotating 3D Suzuka rebuilt from data, with an elevation profile plot.
2. **M2 (end of Phase 5):** "Physics-optimal lap vs VER", a QSS speed trace overlaid on real telemetry.
3. **M3 (end of Phase 8):** the AI completing a clean Suzuka lap, from training-curve GIFs to a first lap.
4. **M4 (end of Phase 10):** **the hook clip**, "I trained an AI to drive Suzuka, then raced it against Verstappen's real pole lap."
5. **M5 (end of Phase 11):** the cross-era showdown, 2019 vs 2024 vs AI.
