# AGENTS.md

Instructions for AI coding agents (Claude Code, Codex, Cursor, Copilot, Aider and similar) working in the **APEX ZERO** repository. Human contributors follow the same rules.

Read this file before your first change. Then read the section of `IMPLEMENTATION.md` for the phase you're working on, and the matching workflow in `SKILLS.md`.

---

## 1. What the project is

APEX ZERO rebuilds real F1 circuits from car position telemetry (FastF1 2018+, OpenF1 2023+), simulates a racing car with a dynamic bicycle model, trains an RL agent (PPO on a batched GPU simulator) to drive it, and renders the agent racing ghost replays of real drivers' laps in a Three.js viewer. It also produces corner-by-corner comparisons. **The core value is credibility.** A comparison is only worth something if the track, the car calibration and the ghost alignment are all correct and verifiable.

## 2. Architecture

```
data  ──▶ track ──▶ ghosts ─────────────────────────┐
  │         │                                       ▼
  │         └──▶ sim (ref NumPy) ◀─parity─▶ sim (batched Torch)
  │                   │                             │
  │                   ▼                             ▼
  │                  qss (calibration)          env ──▶ rl
  │                                                 │
  └──────────────────────────────▶ analysis ◀───────┘
                                       │
                                     export ──▶ viewer (TS/Three.js)
```

**Dependency direction is strict. Lower layers never import higher ones.**

| Layer | Package | May import |
|---|---|---|
| L0 | `apex_zero.config`, `apex_zero.utils` | stdlib, numpy, pydantic |
| L1 | `apex_zero.data` | L0 |
| L2 | `apex_zero.track`, `apex_zero.ghosts` | L0–L1 |
| L3 | `apex_zero.sim`, `apex_zero.qss` | L0–L2 (track only, read-only) |
| L4 | `apex_zero.env` | L0–L3 |
| L5 | `apex_zero.rl` | L0–L4 |
| L6 | `apex_zero.analysis`, `apex_zero.export` | L0–L5 |
| L7 | `apex_zero.cli` | everything |

- `sim/` must not import `torch` except in `sim/batched/`. The reference sim is pure NumPy.
- `fastf1` and `httpx` may only be imported inside `data/`. Nothing else touches the network.
- The viewer consumes **only** the replay bundle format (`export/replay_format.py`, mirrored in `viewer/src/replay/schema.ts`). It never reads raw data.

`tests/unit/test_layering.py` enforces these import rules. Do not weaken it.

## 3. Folder structure

```
apex-zero/
├── configs/
│   ├── tracks/<name>.yaml        # reconstruction params, corner names, known width prior
│   ├── cars/f1_<era>.yaml        # vehicle params per era
│   └── training/<name>.yaml      # full training config
├── src/apex_zero/
│   ├── config/                   # Pydantic models: TrackConfig, CarParams, TrainConfig...
│   ├── utils/                    # units.py, geometry.py, logging.py, seeding.py, io.py
│   ├── data/
│   │   ├── fastf1_loader.py      # session load, lap selection, telemetry merge
│   │   ├── openf1_client.py      # thin httpx client with retries + on-disk cache
│   │   ├── cache.py              # cache layout + keys
│   │   └── schemas.py            # canonical DataFrame column contracts (pandera-style checks)
│   ├── track/
│   │   ├── reconstruct.py        # fuse multi-lap positions → raw centerline
│   │   ├── centerline.py         # periodic smoothing spline over arc length s
│   │   ├── frenet.py             # s,n ⇄ x,y,z; curvature, heading, grade, banking
│   │   ├── width.py              # left/right boundary estimation
│   │   ├── elevation.py
│   │   ├── asset.py              # TrackAsset dataclass + .npz save/load
│   │   └── mesh.py               # glTF export (road, kerbs, run-off)
│   ├── ghosts/
│   │   ├── extract.py            # driver lap → resampled trace
│   │   ├── align.py              # project onto track Frenet frame, start/finish sync
│   │   └── ghost.py              # Ghost dataclass + validation
│   ├── sim/
│   │   ├── params.py             # CarParams (dataclass, SI units)
│   │   ├── tyre.py               # Pacejka MF (simplified), combined slip (friction ellipse)
│   │   ├── aero.py
│   │   ├── powertrain.py         # power-limited drive force, brake force + bias
│   │   ├── vehicle.py            # dynamic bicycle model, load transfer, slope terms
│   │   ├── integrator.py         # semi-implicit Euler / RK4
│   │   ├── reference.py          # single-car NumPy sim (ground truth)
│   │   └── batched/              # Torch implementation, (N,) tensors on device
│   ├── qss/
│   │   ├── solver.py             # forward/backward velocity profile on a given line
│   │   ├── line_opt.py           # min-curvature racing line (baseline)
│   │   └── calibrate.py          # fit CarParams scalars to a target lap time
│   ├── env/
│   │   ├── racing_env.py         # Gymnasium env wrapping reference sim
│   │   ├── vec_env.py            # batched env wrapping batched sim
│   │   ├── observation.py
│   │   ├── reward.py
│   │   └── termination.py
│   ├── rl/
│   │   ├── networks.py
│   │   ├── ppo.py
│   │   ├── sac.py
│   │   ├── buffers.py
│   │   ├── train.py              # loop, checkpointing, logging
│   │   ├── curriculum.py
│   │   └── evaluate.py           # deterministic hot laps
│   ├── analysis/
│   │   ├── corners.py            # corner detection from curvature
│   │   ├── metrics.py            # brake point, apex speed, exit speed, line offset
│   │   ├── delta.py              # time delta vs distance
│   │   └── report.py             # markdown/HTML/JSON reports + matplotlib figures
│   ├── export/
│   │   ├── replay_format.py      # schema + versioning
│   │   └── exporter.py
│   └── cli.py                    # Typer app: `apex ...`
├── viewer/
│   ├── src/
│   │   ├── main.ts
│   │   ├── replay/               # schema.ts, loader.ts, interpolate.ts
│   │   ├── scene/                # track.ts, car.ts, trails.ts, lights.ts
│   │   ├── cameras/              # chase.ts, tv.ts, topdown.ts
│   │   ├── ui/                   # deltaBar.ts, cornerTable.ts, timeline.ts
│   │   └── record/               # clip recorder (MediaRecorder)
│   ├── public/
│   └── tests/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── gpu/                      # marked @pytest.mark.gpu
│   └── fixtures/                 # small, committed, synthetic or trimmed real data
├── scripts/
├── docs/
├── data/                         # gitignored
└── runs/                         # gitignored
```

New modules go in the layer that owns the concept. If you aren't sure which layer that is, ask rather than create a new top-level package.

## 4. Commands

### Setup

```bash
uv sync --extra dev            # CPU dev environment
uv sync --extra dev --extra gpu
pnpm --dir viewer install
```

### Quality gate (run before saying a task is done)

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -m "not gpu and not network and not slow"   # fast suite, < 60 s
pnpm --dir viewer lint
pnpm --dir viewer typecheck
pnpm --dir viewer test
```

### Extended tests

```bash
uv run pytest -m slow                 # integration: full track build on fixture, QSS lap
uv run pytest -m gpu                  # batched sim parity, training smoke (requires CUDA)
uv run pytest -m network              # live FastF1/OpenF1 (never in CI)
uv run pytest --cov=apex_zero --cov-report=term-missing
```

### Pipeline

```bash
uv run apex data fetch --year 2024 --event "Japan" --session Q
uv run apex track build --year 2024 --event "Japan" --session Q --name suzuka
uv run apex track inspect suzuka
uv run apex ghost extract --year 2024 --event "Japan" --session Q --driver VER --lap fastest --track suzuka
uv run apex car calibrate --track suzuka --era 2024 --target-ghost VER
uv run apex train --config configs/training/suzuka_ppo.yaml
uv run apex train --config configs/training/suzuka_ppo.yaml --smoke    # 2 min, any device
uv run apex eval --run runs/suzuka_ppo/latest --ghosts VER,LEC --export
pnpm --dir viewer dev
```

## 5. Coding conventions

### Python

- **Python 3.12**, full type hints, `mypy --strict` on `src/`. No `Any` in public signatures.
- Format and lint with **ruff** (line length 100). Imports sorted by ruff.
- **SI units everywhere internally**: metres, seconds, kilograms, radians, newtons. Convert at the edges, in `data/` and `export/`, using `utils/units.py`. If a variable isn't SI, its name carries a suffix: `speed_kph`, `angle_deg`.
- **Name the frame.** `x, y, z` are world (track asset frame, metres). `s, n` are Frenet (arc length, lateral offset, positive to the left). `vx, vy` are body frame. Mixing frames without converting is a bug.
- Dataclasses (`@dataclass(frozen=True, slots=True)`) for value objects. Pydantic only for configs and anything read from disk.
- Pure functions in `track/`, `qss/`, `analysis/`. State lives in env and trainer classes only.
- NumPy arrays: document shape and dtype in the docstring (`(N, 3) float64`). Use `float64` for reconstruction and analysis, and `float32` on the GPU path.
- **No hidden randomness.** Every stochastic function takes a `np.random.Generator` or a `torch.Generator`. Seeds go through `utils/seeding.py`.
- Logging through `utils/logging.py` (structlog). No `print` in `src/`. CLI output goes through `rich`.
- Docstrings in Google style. Physics functions cite their equation source (for example `Pacejka 2012, eq. 4.E1`).
- Errors: raise specific exceptions from `apex_zero.errors` (`DataUnavailableError`, `TrackReconstructionError`, `GhostAlignmentError`, `CalibrationError`). Never swallow them.

### TypeScript (viewer)

- Strict TS, ES modules, no default exports except Vite entry points.
- Three.js objects are created and disposed in one owner class. Every `geometry`/`material`/`texture` gets disposed.
- The replay schema in `viewer/src/replay/schema.ts` must match `export/replay_format.py`. Bump `REPLAY_FORMAT_VERSION` in both places together.
- No frameworks beyond Three.js and lil-gui unless approved.
- The render loop never allocates per frame. Reuse vectors and matrices.

### Git

- Conventional Commits with a scope from the package name: `feat(track): estimate banking from z gradient`.
- One logical change per commit. Physics changes and refactors are never mixed.
- Never commit `data/`, `runs/`, `.env`, model checkpoints, or anything over 1 MB outside `tests/fixtures/`.

## 6. Testing rules

1. **Every new function in `src/` gets a unit test.** Bug fixes start with a failing test.
2. **Physics needs analytic checks.** Each sim component has at least one test against a closed-form case:
   - Steady-state cornering: lateral acceleration = v²/R within 1% at low slip.
   - Straight-line top speed: drag equals drive force at v_max.
   - Braking on a flat surface: stopping distance matches μ-limited deceleration including aero.
   - Grade: on a 5% uphill slope, acceleration drops by g·sin(θ).
3. **Parity between reference and batched sims.** `tests/gpu/test_parity.py` runs identical actions through both for 2,000 steps. Max state error must be < 1e-3 relative (float32). Any change to `sim/` must keep this passing. If you change one sim, change the other in the same PR.
4. **Determinism.** The same seed and config give bit-identical rollouts on CPU. A test checks this.
5. **Track reconstruction is tested on synthetic tracks** (an ellipse, a figure-eight with elevation) with known ground truth: centerline error < 0.5 m RMS, length error < 0.2%.
6. **Ghost alignment**: a resampled ghost's lap time must match the official lap time within 0.05 s, and at least 99% of its samples must sit inside the track boundaries.
7. **No network in unit tests.** Use `tests/fixtures/`. Live calls are marked `@pytest.mark.network`.
8. **Fixtures stay small.** Trim real data to under 1 MB and document where it came from in `tests/fixtures/README.md`.
9. **Viewer**: Vitest for replay parsing and interpolation. Rendering is checked by hand with a screenshot in the PR description.
10. Coverage target: 85% for `track/`, `sim/`, `qss/`, `analysis/`, and 70% overall.

## 7. Constraints and invariants

- **Credibility invariants** (never break these, never paper over them):
  - A replay bundle **must** include the calibration report (target lap time, QSS lap time, error %) and the track build hash. The exporter refuses to export without them.
  - The agent's lap counts only if it stays within track limits: `|n| ≤ half_width + offtrack_tolerance_m` for the whole lap.
  - Ghost data is never smoothed so much that its lap time changes by more than 0.05 s.
  - No hand-edited telemetry. Every correction is code, logged, and reproducible.
- **Performance budgets**:
  - Batched sim: at least 2M env-steps/s at `num_envs=4096` on a 24 GB consumer GPU (RTX 4090 class). Profile before and after any sim change.
  - Track build: under 60 s per session on a laptop CPU.
  - Viewer: a steady 60 fps at 1080p with 1 AI car and 5 ghosts.
- **Data etiquette**: cache everything. Never loop over live API calls without the cache. Respect OpenF1 rate limits (the client enforces about 3 req/s with backoff).
- **Legal**: no F1, FIA or team logos, liveries or driver photos in the repo or the viewer. Driver three-letter codes and names in text are fine.
- **Dependencies**: adding a runtime dependency needs a one-line justification in the PR. Prefer what's already in `pyproject.toml`.

## 8. Do's and don'ts

**Do**
- Read the phase's acceptance criteria in `IMPLEMENTATION.md` before coding.
- Keep changes small and finish one phase task end to end, with tests.
- Add a `track inspect`-style plot when you change reconstruction or analysis, and attach it to the PR.
- Update `configs/` schemas, docs and examples when you change a config field.
- Prefer vectorised NumPy and Torch over Python loops in hot paths.
- Write down every physics assumption in the module docstring.
- Run the quality gate and report its actual output.

**Don't**
- Don't tune reward weights or car params to "make the AI win". Calibration targets the real lap time. It doesn't target the outcome.
- Don't change `REPLAY_FORMAT_VERSION`, the cache layout, or the `TrackAsset` schema without a migration note in `docs/migrations.md`.
- Don't add an `if track == "suzuka"` special case. Track-specific data lives in `configs/tracks/`.
- Don't disable, skip or loosen tests to get green. If a tolerance is wrong, argue it in the PR.
- Don't import `torch` in the reference sim, or `fastf1` outside `data/`.
- Don't commit generated data, checkpoints or large binaries.
- Don't call a task done while any quality-gate command fails.
- Don't invent API fields for FastF1 or OpenF1. Check the installed library or the docs, and pin what you rely on in `data/schemas.py`.

## 9. Definition of done

A task is done when:
1. The code is implemented in the right layer, with types and docstrings.
2. Tests cover the new behaviour and pass, including the relevant `slow`/`gpu` markers if you touched those paths.
3. The quality gate is green.
4. Docs are updated: README usage if user-facing, `IMPLEMENTATION.md` checkbox ticked, `docs/` notes for design decisions.
5. For visual or analysis changes, a plot or screenshot is attached or saved to `docs/media/`.
