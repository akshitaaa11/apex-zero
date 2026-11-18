# ADR 0001: Layered architecture, three physics sources of truth, replay bundle as the only interface

- **Status:** accepted
- **Date:** 2026-11-18
- **Phase:** 0 (P0.10)

## Context

APEX ZERO's value is credibility: "the agent beat a real lap" only means something if the
track reconstruction, the car calibration and the ghost alignment can each be checked
independently. The project also spans very different technologies (telemetry ingestion,
geometry, NumPy physics, batched Torch physics, RL, a TypeScript/Three.js viewer), and is
developed largely by AI coding agents working phase by phase.

## Decision

1. **Strict layering** (`AGENTS.md` §2). Packages are ordered L0 (`config`, `utils`,
   `errors`) to L7 (`cli`); a module imports only its own layer or lower ones. L3 (`sim`, `qss`)
   reads `track` only. Network libraries (`fastf1`, `httpx`) are confined to `data`; `torch` is
   kept out of the reference sim (`sim.batched` only). `tests/unit/test_layering.py` enforces
   this with `ast` on every test run.
2. **Three sources of truth for physics.** `sim/reference.py` (NumPy, float64) is ground truth;
   `sim/batched/` (Torch, float32) must match it in a parity test; `qss/solver.py` gives the
   quasi-steady-state lap-time bound used to calibrate the car to a real lap. Any car-model
   change touches all three or explains why not.
3. **The replay bundle is the only Python↔viewer interface.** `export/replay_format.py` and
   `viewer/src/replay/schema.ts` share one versioned schema; the viewer never reads raw data.
4. **SI units and explicit frames internally**, conversions only at the edges (`data`,
   `export`) through `utils/units.py`; explicit RNGs (`utils/seeding.py`), no global seeds.
5. **Tooling:** uv + Python 3.12, ruff, `mypy --strict`, pytest markers `slow`/`gpu`/`network`;
   pnpm + Vite + strict TypeScript + ESLint/Prettier/Vitest for the viewer; GitHub Actions runs
   the fast suites on CPU.

## Consequences

- Lower layers can be tested in isolation with synthetic fixtures; nothing below `data` needs the
  network, and nothing below `sim.batched` needs a GPU.
- Some convenience imports are forbidden (e.g. `sim` reading a `Ghost`); the CLI composes layers
  instead.
- Changing a shared format (`TrackAsset`, `CarParams`, replay bundle, cache layout) requires a
  version bump and a note in `docs/migrations.md`.
- **torch packaging:** uv cannot make "CPU wheels unless `--extra gpu`" the default. The default
  is the PyPI wheel (CUDA 13 build on Linux, CPU on macOS; works without a GPU); `--extra cpu`
  selects CPU-only wheels (used by CI) and `--extra gpu` the CUDA 13.0 index. After syncing with
  either extra, use `uv run --no-sync` / `UV_NO_SYNC=1`, because plain `uv run` re-syncs the
  default resolution.
- **`structlog` in L0:** `AGENTS.md` §2 lists stdlib/numpy/pydantic for L0, but P0.5 places
  structured logging in `utils/logging.py`. The layering test allows `structlog` in that one
  module only.
