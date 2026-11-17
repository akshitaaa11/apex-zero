# CLAUDE.md

Instructions for **Claude** (Claude Code) working on APEX ZERO. This file adds Claude-specific workflow on top of `AGENTS.md`. **`AGENTS.md` is the source of truth for architecture, conventions and constraints.** If the two ever disagree, follow `AGENTS.md` and point out the conflict.

---

## 1. Read order at the start of every session

1. `CLAUDE.md` (this file)
2. `AGENTS.md`: layering, conventions, testing rules, invariants
3. `IMPLEMENTATION.md`: find the **current phase** (the first phase with unchecked boxes) and read its acceptance criteria
4. `SKILLS.md`: the workflow that matches the task
5. `docs/decisions/`: architecture decision records (ADRs), if the task touches that area

Don't read `data/` or `runs/`. They're large generated artefacts. Use `uv run apex data ls`, `uv run apex track inspect`, or small `python -c` snippets to query them.

## 2. Project context Claude must keep in mind

- **Credibility is the whole product.** "AI beats Verstappen" is only interesting if the track, the car calibration and the ghost alignment can be checked. When in doubt, choose correctness and transparency over a more impressive result.
- **Three sources of truth for physics:**
  1. `sim/reference.py`, the NumPy ground truth
  2. `sim/batched/`, which must match the reference (parity test)
  3. `qss/solver.py`, the quasi-steady-state lap-time bound used for calibration
  A change to the car model touches all three, or explains why it doesn't.
- **Telemetry is coarse.** Positions arrive at about 4 Hz, brake is on/off, and there's no steering data. Never write code or docs that suggest otherwise.
- **The two halves meet at the replay bundle.** The Python side (`export/`) and the TypeScript side (`viewer/src/replay/`) share a versioned schema. Change both together.

## 3. How Claude should work here

### 3.1 Plan first for anything non-trivial

For any task that touches more than one module, changes physics, or changes a file format:
1. Write a short plan: files to change, tests to add, risks, and which acceptance criteria it satisfies.
2. If the task is ambiguous, or changes a public schema (`TrackAsset`, `CarParams`, replay format, cache layout), **stop and ask** before coding.
3. Otherwise, go ahead and keep the plan updated in the task list.

### 3.2 Test-first loop

```
write/adjust failing test → implement → uv run pytest <targeted path> -q → quality gate → report
```

Use targeted runs while iterating (`uv run pytest tests/unit/track -q`). Run the full fast suite before you finish.

### 3.3 Verify, don't assume

- **External APIs:** before using any FastF1 or OpenF1 field, check it against the installed version:
  ```bash
  uv run python -c "import fastf1, inspect; print(fastf1.__version__)"
  uv run python -c "from fastf1.core import Telemetry; help(Telemetry)" | head -80
  ```
  For OpenF1, check `src/apex_zero/data/schemas.py` first. If a field isn't there, fetch one sample with `-m network` tests and record its schema.
- **Units and axes:** never assume telemetry units (for example, the scale of position X/Y/Z). The Phase 1 tests pin them. Read those tests.
- **Physics:** when you add a formula, add the analytic test from `AGENTS.md` §6.2 in the same change.

### 3.4 Report honestly

When you finish, report:
- what changed, file by file in one line each
- the quality-gate output: pass/fail counts, and any skipped markers with the reason (for example "gpu tests skipped: no CUDA in this environment")
- anything not done, or done differently from the plan
- any number that moved (lap time, parity error, steps/s), with before and after

Never claim a GPU test, a training run or a benchmark passed unless you ran it and saw the output.

## 4. Important commands (quick reference)

```bash
# environment
uv sync --extra dev                    # add --extra gpu on CUDA machines
uv run apex doctor

# quality gate
uv run ruff check . && uv run ruff format --check . && uv run mypy src
uv run pytest -m "not gpu and not network and not slow" -q
pnpm --dir viewer lint && pnpm --dir viewer typecheck && pnpm --dir viewer test

# auto-fix formatting/lint
uv run ruff format . && uv run ruff check . --fix

# targeted
uv run pytest tests/unit/sim -q -x
uv run pytest tests/gpu/test_parity.py -q        # only if CUDA available
uv run pytest -m slow tests/integration/test_track_build.py

# pipeline smoke (CPU, offline, uses fixtures)
uv run apex track build --fixture synthetic_figure8 --name f8
uv run apex train --config configs/training/smoke.yaml --smoke
uv run apex eval --run runs/smoke/latest --export
```

## 5. Development rules specific to Claude

1. **Stay in scope.** Do the task you were asked to do. If you spot an unrelated bug, note it in your report or in `docs/todo.md`. Don't fix it in the same change.
2. **Respect the layering.** Before adding an import, check the table in `AGENTS.md` §2. `tests/unit/test_layering.py` fails otherwise.
3. **No silent schema drift.** Any change to `TrackAsset`, `Ghost`, `CarParams`, replay format or cache keys needs:
   - a version bump
   - a migration note in `docs/migrations.md`
   - updated fixtures (`uv run python scripts/regen_fixtures.py`)
4. **Don't game the result.** Never change reward weights, track limits, calibration tolerances or ghost smoothing to improve the AI-vs-ghost outcome. If the AI is implausibly fast, treat it as a bug: an exploit, a calibration error, or track limits that are too loose. Investigate with `SKILLS.md` → *debug-rl-agent*.
5. **Long-running jobs.** Full training runs take hours. Don't start one unless asked. Use `--smoke` configs to validate training code. If you're asked to train, run it in the background, log to `runs/<name>/`, and report the TensorBoard path.
6. **Don't touch generated or heavy dirs.** Never edit files under `data/`, `runs/`, `viewer/dist/` or `viewer/node_modules/`.
7. **Small, reviewable diffs.** Aim for under about 400 changed lines per change, excluding tests and fixtures. Split larger work into phase sub-tasks.
8. **Commits** (only when asked): Conventional Commits, scoped by package, and the body explains *why*.
9. **Ask before you:** add a runtime dependency, change a public schema, delete a test, change a tolerance in `AGENTS.md` §6 or §7, or restructure folders.

## 6. Coding standards summary

The full rules are in `AGENTS.md` §5. The essentials:
- SI units internally, suffixes for anything else (`_kph`, `_deg`)
- Frame names on every coordinate: world `x,y,z`, Frenet `s,n`, body `vx,vy`
- `mypy --strict`, ruff at line length 100, Google docstrings with array shapes
- Explicit RNGs, never global seeds
- `float64` for reconstruction and analysis, `float32` on the GPU path
- Viewer: strict TS, dispose every Three.js resource, zero per-frame allocations

## 7. Domain cheat-sheet

| Concept | Where | Notes |
|---|---|---|
| Frenet frame | `track/frenet.py` | `s` ∈ [0, L) periodic, `n` > 0 to the left. Curvature `κ(s)` > 0 for left turns. |
| Track asset | `track/asset.py` | `.npz` with `s, x, y, z, psi, kappa, grade, bank, w_left, w_right`, resampled every 1 m, plus a `meta.json` hash. |
| Car state | `sim/vehicle.py` | `[s, n, dpsi, vx, vy, r, omega_f, omega_r]`. The sim integrates in curvilinear coordinates. |
| Actions | `env/racing_env.py` | `steer ∈ [-1,1]` → `δ = steer·δ_max`, `pedal ∈ [-1,1]` with > 0 throttle and < 0 brake. |
| Observation | `env/observation.py` | Normalised; includes curvature, width and grade lookahead at fixed distances. |
| Reward | `env/reward.py` | Δs progress. Off-track penalty plus termination. Small steer-rate penalty. |
| Calibration | `qss/calibrate.py` | Fits `mu_scale`, `cl_a`, `p_max` so that QSS lap time is within ±1.5% of the target. |
| Corner | `analysis/corners.py` | A region where `|κ|` > threshold, merged by gap. Names come from `configs/tracks/<name>.yaml`. |
| Replay bundle | `export/replay_format.py` | `manifest.json` + `track.glb` + `cars/*.bin` (float32, 50 Hz) + `analysis.json`. |

## 8. When stuck

- **Reconstruction looks wrong:** run `uv run apex track inspect <name> --debug` and look at the raw-point scatter against the fitted centerline, and at the width histogram.
- **Parity fails:** bisect with `scripts/parity_bisect.py`, which reports the first step and state component that diverges.
- **Training doesn't learn:** follow `SKILLS.md` → *debug-rl-agent*. Check reward scale, observation normalisation, termination rate and action saturation before touching hyperparameters.
- **Unsure about the physics:** write the assumption in the docstring, add a test that pins the behaviour, and flag it in your report instead of guessing silently.
