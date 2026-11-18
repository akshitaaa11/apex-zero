# apex-zero

An RL agent that learns to drive real F1 circuits, then races ghosts of real drivers' laps.

The tracks are rebuilt from car position data (x, y, z, so elevation comes along for free). The car is a physics sim. The agent is trained with PPO on a batched GPU version of that sim. Ghosts are replays of actual laps from FastF1/OpenF1 telemetry, so you can watch the agent go round Suzuka next to Verstappen's pole lap and see, corner by corner, where it brakes later, carries more speed or takes a different line.

You can also put drivers from different seasons on the same track, e.g. Hamilton's 2019 Suzuka lap against Verstappen's 2024 one.

**Status:** early. See [IMPLEMENTATION.md](IMPLEMENTATION.md) for what's done and what's next. Commands below describe the intended CLI; not all of them exist yet.

## How it works

```
telemetry -> track rebuild -> ghosts
                  |
                  v
           car sim (numpy reference + batched torch)
                  |
          calibrate car to real lap time (QSS solver)
                  |
             train agent (PPO)
                  |
      compare vs ghosts -> export replay -> three.js viewer
```

1. Pull positions and telemetry for a session and cache them.
2. Fuse every car's position samples into a smooth centerline, parameterised by distance along the track. Width comes from how far cars spread across the track; elevation comes from z.
3. Turn a driver's lap into a ghost aligned to that centerline.
4. Calibrate the car. A quasi-steady-state lap-time solver tunes grip, downforce and power until the best possible lap lands within about 1.5% of the real pole time. Without this step "the AI was faster" doesn't mean anything.
5. Train the agent on thousands of cars in parallel. Reward is progress along the track; leaving the track ends the episode.
6. Detect corners, compute per-corner deltas, export a replay bundle and open it in the viewer.

## Stack

Python 3.12 managed with uv. FastF1 and the OpenF1 API for data; NumPy/SciPy/pandas for track work; PyTorch for the batched sim and PPO; Typer for the CLI; trimesh + pygltflib for the track mesh. The viewer is TypeScript + Vite + Three.js.

## Requirements

- Linux or macOS (WSL2 on Windows)
- Python 3.12 and [uv](https://github.com/astral-sh/uv)
- Node 20+ and pnpm, for the viewer
- An NVIDIA GPU with 8 GB+ for real training. CPU is fine for everything else and for smoke runs.
- A few GB of disk per season of cached data
- Data: OpenF1 (2023+) is the default source and needs no account. FastF1 (2018+) is optional and needs a paid F1TV subscription (`F1_USERNAME`/`F1_PASSWORD`, FastF1 3.9+). See [docs/data.md](docs/data.md).

## Install

```bash
git clone https://github.com/<you>/apex-zero.git
cd apex-zero

uv sync --extra dev            # add --extra gpu on a CUDA machine
uv run apex doctor             # checks python, torch, cuda, cache dir, node

pnpm --dir viewer install
```

Which torch build you get: the default is the PyPI wheel (a CUDA 13 build on Linux that also runs
without a GPU; CPU on macOS). `--extra gpu` pins the CUDA 13.0 wheel from the PyTorch index, and
`--extra cpu` installs the much smaller CPU-only wheel (what CI uses). The two extras are mutually
exclusive. After `uv sync --extra cpu` or `--extra gpu`, run commands with `uv run --no-sync ...`
(or `export UV_NO_SYNC=1`), because plain `uv run` re-syncs to the default torch.

Paths can be overridden in `.env` (see `.env.example`): `APEX_CACHE_DIR`, `APEX_PROCESSED_DIR`, `APEX_RUNS_DIR`.

## Quick start (Suzuka)

To use another circuit, change `--event` (and `--name`/`--track`) in the same commands, for example `--event Monaco --name monaco`.

```bash
# data + track
uv run apex data fetch --year 2024 --event Japan --session Q
uv run apex track build --year 2024 --event Japan --session Q --name suzuka

# ghost
uv run apex ghost extract --year 2024 --event Japan --session Q --driver VER --lap fastest --track suzuka

# calibrate and train
uv run apex car calibrate --track suzuka --era 2024 --target-ghost VER
uv run apex train --config configs/training/suzuka_ppo.yaml

# compare and export
uv run apex eval --run runs/suzuka_ppo/latest --ghosts VER,LEC,NOR --export

# watch
pnpm --dir viewer dev
```

Then open http://localhost:5173 and load the replay folder.

To check the training code without a GPU, use the smoke config. It finishes in a couple of minutes:

```bash
uv run apex train --config configs/training/smoke.yaml --smoke
```

## CLI

| Command | What it does |
|---|---|
| `apex doctor` | Environment checks |
| `apex data fetch / ls` | Download and cache a session, list cached sessions |
| `apex track build / inspect` | Rebuild a track; plot centerline, width, elevation, curvature |
| `apex ghost extract / ls` | Turn a driver's lap into a ghost |
| `apex car calibrate` | Fit car parameters to a target lap |
| `apex qss solve` | Run the lap-time solver on its own |
| `apex train` | Train from a YAML config (`--resume`, `--smoke`) |
| `apex eval` | Evaluate a checkpoint, compare against ghosts, optionally export |
| `apex compare` | Corner table between any two laps |
| `apex export replay` | Build a replay bundle for the viewer |
| `apex clip` | Render a video from a replay (later phase) |

`--help` on any of them lists the flags.

### Mixing seasons

```bash
uv run apex ghost extract --year 2019 --event Japan --session Q --driver HAM --lap fastest --track suzuka
uv run apex export replay --track suzuka --ghosts 2019:HAM,2024:VER --agent runs/suzuka_ppo/latest
```

Older ghosts are projected onto the reference track build. If the layout changed between seasons (a new chicane, say), the projection check fails and nothing is exported.

### Viewer keys

Space play/pause, arrow keys scrub (Shift for fine steps), 1/2/3 chase/TV/top-down camera, C cycle car, [ and ] previous/next corner, L toggle racing lines, R record a clip. Add `?debug=1` to the URL for an fps overlay.

## Config

Training runs are a single YAML file. A trimmed example:

```yaml
track: suzuka
car: configs/cars/f1_2024.yaml
sim:
  dt: 0.01
  action_repeat: 5        # policy runs at 20 Hz
env:
  num_envs: 4096
  lookahead_m: [10, 20, 40, 60, 80, 100, 130, 160, 200, 250, 300]
  offtrack_tolerance_m: 1.0
algo:
  name: ppo
  total_steps: 2_000_000_000
  lr: 3.0e-4
  gamma: 0.995
seed: 42
```

Schemas live in `src/apex_zero/config/`.

## Layout

```
configs/          tracks, cars, training configs
src/apex_zero/
  data/           FastF1 + OpenF1 loading and caching
  track/          reconstruction, Frenet frame, mesh export
  ghosts/         lap extraction and alignment
  sim/            vehicle model (numpy reference, batched torch)
  qss/            lap-time solver and calibration
  env/            gymnasium env, observations, reward
  rl/             PPO, SAC, training loop
  analysis/       corners, deltas, reports
  export/         replay bundle format
viewer/           three.js app
tests/
```

`data/`, `runs/` and `replays/` are gitignored.

## About the data

FastF1 covers 2018 onwards: lap timing, car data (speed, RPM, gear, throttle, brake, DRS) and position. OpenF1 adds a REST API with 3D location from 2023. Both are free.

The data has some limits:

- Sample rates are low, around 4 Hz for position. The track rebuild leans on fusing many laps from many drivers.
- Brake is on/off, not pressure, so brake points come from deceleration.
- There's no steering, tyre temperature or setup data. A ghost is a path and a speed trace. The agent drives a model of a car, not the real thing.

## Is the comparison fair?

Fair enough to be interesting, as long as it's labelled honestly. The car is calibrated against the real lap before training. Ghosts and agent run on the same rebuilt surface. The agent's laps only count if it stays inside track limits. If it ever beats the solver's theoretical best on its own line, that's treated as a bug, not a result. Every replay carries its calibration report, and the viewer always says it's a simulated car against real telemetry.

## Contributing

Issues and PRs are welcome. For anything touching the physics or the replay format, open an issue first.

Before pushing:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src
uv run pytest -m "not gpu and not network and not slow"
pnpm --dir viewer lint && pnpm --dir viewer test
```

Commit messages follow Conventional Commits with the package as scope, e.g. `feat(sim): tyre load sensitivity`. Physics changes need a test against a case with a known answer. [AGENTS.md](AGENTS.md) has the full conventions. It's written for coding agents but applies to everyone.

## Disclaimer

Unofficial, non-commercial project. Not affiliated with Formula 1, the FIA or any team or driver. F1 and FORMULA 1 are trademarks of Formula One Licensing B.V. Data comes via FastF1 and OpenF1 and stays subject to their terms. The repo doesn't include logos, liveries or driver images, and the viewer uses a generic car model.

## Credits

[FastF1](https://github.com/theOehrly/Fast-F1), [OpenF1](https://openf1.org), and [CleanRL](https://github.com/vwxyzjn/cleanrl), whose PPO implementation this one is modelled on.

MIT licensed.
