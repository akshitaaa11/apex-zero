"""``apex`` command-line interface (Typer).

Phase 0 provides the command tree and ``apex doctor``. Every other command is a placeholder
that says which phase implements it and exits with status 1; nothing is faked.

User-facing output goes through ``rich``; diagnostics go through ``utils.logging``.
"""

from __future__ import annotations

import importlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

import typer
from rich.console import Console
from rich.table import Table

from apex_zero import __version__
from apex_zero.config.paths import resolve_paths

__all__ = [
    "CheckResult",
    "app",
    "check_cache_dir",
    "check_cuda",
    "check_node",
    "check_python",
    "check_torch",
    "main",
    "run_doctor",
]

REQUIRED_PYTHON: tuple[int, int] = (3, 12)
MIN_NODE_MAJOR: int = 20

console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    name="apex",
    help="APEX ZERO: rebuild F1 tracks from telemetry, train an RL driver, race real ghosts.",
    no_args_is_help=True,
    add_completion=False,
)


# --------------------------------------------------------------------------------------------
# Command tree. Groups get their sub-commands in the phase named in their help text.
# --------------------------------------------------------------------------------------------

_GROUPS: dict[str, str] = {
    "track": "Build and inspect track assets. [Phase 2]",
    "ghost": "Extract and validate driver ghosts. [Phase 3]",
    "car": "Car parameters and calibration. [Phase 5]",
    "qss": "Quasi-steady-state lap-time solver. [Phase 5]",
    "export": "Export replay bundles for the viewer. [Phase 10]",
}

for _name, _help in _GROUPS.items():
    app.add_typer(typer.Typer(help=_help, no_args_is_help=True), name=_name)

data_app = typer.Typer(
    help="Fetch, cache and list sessions (OpenF1, FastF1).", no_args_is_help=True
)
app.add_typer(data_app, name="data")


class DataSource(StrEnum):
    """Where ``apex data fetch`` gets a session from."""

    openf1 = "openf1"
    fastf1 = "fastf1"


_SOURCE_OPTION = typer.Option(
    DataSource.openf1, help="openf1 (free, 2023+) or fastf1 (2018+, needs a paid F1TV account)."
)


@data_app.command("fetch")
def data_fetch(
    year: int = typer.Option(..., help="Championship year (OpenF1: 2023+, FastF1: 2018+)."),
    event: str = typer.Option(..., help="Event name, location or country, e.g. Japan."),
    session: str = typer.Option(..., help="Session code or name: FP1-3, SQ, SS, S, Q, R."),
    source: DataSource = _SOURCE_OPTION,
    force: bool = typer.Option(False, help="Re-download and re-export even if cached."),
) -> None:
    """Fetch a session and write validated Parquet for every driver lap."""
    from apex_zero.data.processed import FetchResult
    from apex_zero.errors import DataUnavailableError, SchemaValidationError

    try:
        result: FetchResult
        if source is DataSource.openf1:
            from apex_zero.data.openf1_loader import fetch_openf1_session

            result = fetch_openf1_session(year, event, session, force=force)
        else:
            from apex_zero.data.fastf1_loader import fetch_session

            result = fetch_session(year, event, session, force=force)
    except (DataUnavailableError, SchemaValidationError, ValueError) as exc:
        err_console.print(f"[red]fetch failed:[/] {exc}")
        raise typer.Exit(code=1) from exc
    m = result.meta
    origin = (
        "processed cache (no network)" if result.from_cache else f"{m.source} {m.source_version}"
    )
    console.print(
        f"{m.year} {m.event_name} {m.session}: {len(m.drivers)} drivers, {m.n_laps} laps, "
        f"{len(m.files)} files from {origin}"
    )
    if result.rejected:
        err_console.print(
            f"[yellow]{len(result.rejected)} lap(s) failed validation and were not written "
            "(see rejected_laps.json)"
        )


@data_app.command("ls")
def data_ls(
    year: int | None = typer.Option(None, help="Only this year."),
) -> None:
    """List sessions in the processed cache (offline)."""
    from apex_zero.data.cache import list_sessions

    metas = list_sessions(resolve_paths(), year)
    if not metas:
        console.print("no processed sessions")
        return
    table = Table(title="processed sessions")
    for col in ("year", "event", "session", "source", "drivers", "laps", "fetched (UTC)"):
        table.add_column(col)
    for m in metas:
        table.add_row(
            str(m.year),
            m.event_name,
            m.session,
            f"{m.source} {m.source_version}",
            str(len(m.drivers)),
            str(m.n_laps),
            m.fetched_at_utc,
        )
    console.print(table)


@data_app.command("info")
def data_info(
    year: int = typer.Argument(..., help="Championship year."),
    event: str = typer.Argument(..., help="Event query used for fetch, or the event name."),
    session: str = typer.Argument(..., help="Session code or name."),
) -> None:
    """Show metadata of a processed session (offline)."""
    from apex_zero.data import cache

    paths = resolve_paths()
    try:
        code = cache.normalize_session(session)
    except ValueError as exc:
        err_console.print(f"[red]{exc}")
        raise typer.Exit(code=1) from exc
    key = (
        cache.alias_lookup(paths, "openf1", year, event, code)
        or cache.alias_lookup(paths, "fastf1", year, event, code)
        or cache.SessionKey(year, cache.slugify(event), code)
    )
    meta = cache.read_meta(paths, key)
    if meta is None:
        err_console.print(
            f"[red]{year} {event} {code} is not in the processed cache; run `apex data fetch`."
        )
        raise typer.Exit(code=1)
    intact = cache.is_complete(paths, key)
    console.print(f"[bold]{meta.year} {meta.event_name} - {meta.session_name}[/] ({key.relpath})")
    console.print(
        f"round {meta.round_number or '?'}, {meta.location}, {meta.country}; source {meta.source} "
        f"{meta.source_version}; fetched {meta.fetched_at_utc}; "
        f"{'intact' if intact else '[red]INCOMPLETE OR MODIFIED[/]'}"
    )
    table = Table(title=f"{len(meta.drivers)} drivers, {meta.n_laps} laps")
    table.add_column("driver")
    table.add_column("fastest lap [s]", justify="right")
    for drv in sorted(meta.drivers, key=lambda d: meta.fastest_lap_s.get(d, float("inf"))):
        t = meta.fastest_lap_s.get(drv)
        table.add_row(drv, "-" if t is None else f"{t:.3f}")
    console.print(table)


def _not_implemented(command: str, phase: int) -> Callable[[], None]:
    def _cmd() -> None:
        err_console.print(
            f"[yellow]apex {command}[/] is not implemented yet (planned for Phase {phase}, "
            "see IMPLEMENTATION.md)."
        )
        raise typer.Exit(code=1)

    _cmd.__doc__ = f"[Phase {phase}] Not implemented yet."
    return _cmd


for _cmd_name, _phase in {"train": 8, "eval": 9, "compare": 9, "clip": 12}.items():
    app.command(name=_cmd_name)(_not_implemented(_cmd_name, _phase))


def _print_version(value: bool) -> None:
    if value:
        console.print(__version__)
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        help="Print the version and exit.",
        is_eager=True,
        callback=_print_version,
    ),
) -> None:
    """APEX ZERO command-line interface."""


# --------------------------------------------------------------------------------------------
# apex doctor
# --------------------------------------------------------------------------------------------

Status = Literal["ok", "warn", "fail"]


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of one ``apex doctor`` check.

    Attributes:
        name: Short check name.
        status: ``"ok"``, ``"warn"`` (optional component missing) or ``"fail"`` (required).
        detail: Human-readable explanation.
    """

    name: str
    status: Status
    detail: str


def check_python(version: Sequence[int] | None = None) -> CheckResult:
    """Check that the interpreter is Python 3.12.

    Args:
        version: ``(major, minor, ...)``; defaults to ``sys.version_info``.
    """
    v = tuple(sys.version_info[:3] if version is None else version)
    text = ".".join(str(p) for p in v)
    if v[:2] == REQUIRED_PYTHON:
        return CheckResult("python", "ok", text)
    want = ".".join(str(p) for p in REQUIRED_PYTHON)
    return CheckResult("python", "fail", f"{text} (need {want}.x)")


def check_torch(import_module: Callable[[str], object] = importlib.import_module) -> CheckResult:
    """Check that PyTorch imports.

    Args:
        import_module: Module importer (injectable for tests).
    """
    try:
        torch = import_module("torch")
    except ImportError as exc:
        return CheckResult("torch", "fail", f"not importable: {exc}")
    return CheckResult("torch", "ok", str(getattr(torch, "__version__", "unknown version")))


def check_cuda(import_module: Callable[[str], object] = importlib.import_module) -> CheckResult:
    """Report CUDA availability. CUDA is optional: absence is a warning, never a failure.

    Args:
        import_module: Module importer (injectable for tests).
    """
    try:
        torch = import_module("torch")
    except ImportError:
        return CheckResult("cuda", "warn", "skipped: torch not importable (optional)")
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and bool(cuda.is_available()):
        name = str(cuda.get_device_name(0))
        return CheckResult("cuda", "ok", f"{int(cuda.device_count())} device(s); {name}")
    return CheckResult("cuda", "warn", "not available (optional; GPU training and parity off)")


def check_cache_dir(path: Path) -> CheckResult:
    """Check that the cache directory is (or can be created) readable and writable.

    Does not create the directory. If it exists, a temporary file is written and removed.

    Args:
        path: Cache directory (``APEX_CACHE_DIR``).
    """
    if path.exists():
        if not path.is_dir():
            return CheckResult("cache dir", "fail", f"{path} exists but is not a directory")
        try:
            with tempfile.NamedTemporaryFile(dir=path, prefix=".apex-doctor-"):
                pass
        except OSError as exc:
            return CheckResult("cache dir", "fail", f"{path} is not writable: {exc}")
        return CheckResult("cache dir", "ok", f"{path} (writable)")

    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if parent.is_dir() and os.access(parent, os.W_OK | os.X_OK):
        return CheckResult("cache dir", "ok", f"{path} (does not exist yet; creatable)")
    return CheckResult("cache dir", "fail", f"{path} cannot be created: {parent} not writable")


def check_node(
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[[list[str]], str] | None = None,
) -> CheckResult:
    """Check the Node.js version used by the viewer (>= 20). Missing Node is a warning.

    Args:
        which: Executable lookup (injectable for tests).
        run: Runs a command and returns its stdout (injectable for tests).
    """
    exe = which("node")
    if exe is None:
        return CheckResult(
            "node", "warn", f"not found (needed only for the viewer; >= {MIN_NODE_MAJOR})"
        )
    runner = run if run is not None else _run_stdout
    try:
        out = runner([exe, "--version"]).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return CheckResult("node", "warn", f"{exe} --version failed: {exc}")
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", out)
    if match is None:
        return CheckResult("node", "warn", f"unrecognised version string {out!r}")
    if int(match.group(1)) < MIN_NODE_MAJOR:
        return CheckResult("node", "warn", f"{out} (viewer needs >= {MIN_NODE_MAJOR})")
    return CheckResult("node", "ok", out)


def _run_stdout(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10).stdout


def run_doctor(cache_dir: Path | None = None) -> list[CheckResult]:
    """Run every environment check.

    Args:
        cache_dir: Cache directory to check; defaults to the resolved ``APEX_CACHE_DIR``.

    Returns:
        One result per check, in display order.
    """
    cache = resolve_paths().cache_dir if cache_dir is None else cache_dir
    return [check_python(), check_torch(), check_cuda(), check_cache_dir(cache), check_node()]


_STYLE: dict[Status, str] = {
    "ok": "[green]ok[/]",
    "warn": "[yellow]warn[/]",
    "fail": "[red]FAIL[/]",
}


@app.command()
def doctor() -> None:
    """Check the environment: Python, torch, CUDA (optional), cache dir, Node."""
    results = run_doctor()
    table = Table(title=f"apex doctor (apex-zero {__version__})")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    for r in results:
        table.add_row(r.name, _STYLE[r.status], r.detail)
    console.print(table)
    failed = [r.name for r in results if r.status == "fail"]
    if failed:
        err_console.print(
            f"[red]doctor: {len(failed)} required check(s) failed: {', '.join(failed)}"
        )
        raise typer.Exit(code=1)


def main() -> None:
    """Console-script entry point (``apex``)."""
    app()
