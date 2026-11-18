from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from apex_zero import __version__, cli

runner = CliRunner()


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rich wraps at the terminal width; make it wide so assertions see whole lines."""
    monkeypatch.setattr(cli.console, "width", 200)
    monkeypatch.setattr(cli.err_console, "width", 200)


GROUPS = ["ghost", "car", "qss", "export"]
STUBS = ["train", "eval", "compare", "clip"]


def test_help_lists_every_command() -> None:
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for name in [*GROUPS, *STUBS, "data", "track", "doctor"]:
        assert name in result.output


def test_version() -> None:
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


@pytest.mark.parametrize("group", GROUPS)
def test_groups_show_help(group: str) -> None:
    result = runner.invoke(cli.app, [group, "--help"])
    assert result.exit_code == 0
    assert "Phase" in result.output


@pytest.mark.parametrize("command", STUBS)
def test_stub_commands_fail_loudly(command: str) -> None:
    result = runner.invoke(cli.app, [command])
    assert result.exit_code == 1
    assert "not implemented" in result.output


# --- individual doctor checks -----------------------------------------------------------------


def test_check_python() -> None:
    assert cli.check_python((3, 12, 7)).status == "ok"
    assert cli.check_python((3, 11, 9)).status == "fail"
    assert cli.check_python((3, 13, 0)).status == "fail"
    assert cli.check_python().status == "ok"  # the test env itself is 3.12


def _importer(module: object | None) -> Callable[[str], object]:
    def _import(name: str) -> object:
        if module is None:
            raise ImportError(f"No module named {name!r}")
        return module

    return _import


def test_check_torch() -> None:
    assert cli.check_torch(_importer(SimpleNamespace(__version__="9.9"))).detail == "9.9"
    assert cli.check_torch(_importer(None)).status == "fail"
    assert cli.check_torch().status == "ok"  # torch is a runtime dependency


def test_check_cuda_is_optional() -> None:
    no_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    assert cli.check_cuda(_importer(no_cuda)).status == "warn"
    assert cli.check_cuda(_importer(None)).status == "warn"
    gpu = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True, device_count=lambda: 1, get_device_name=lambda i: "FakeGPU"
        )
    )
    result = cli.check_cuda(_importer(gpu))
    assert result.status == "ok"
    assert "FakeGPU" in result.detail
    assert cli.check_cuda().status in {"ok", "warn"}


def test_check_cache_dir(tmp_path: Path) -> None:
    assert cli.check_cache_dir(tmp_path).status == "ok"
    missing = tmp_path / "a" / "b"
    result = cli.check_cache_dir(missing)
    assert result.status == "ok"
    assert "does not exist yet" in result.detail
    assert not missing.exists()
    a_file = tmp_path / "file"
    a_file.write_text("x")
    assert cli.check_cache_dir(a_file).status == "fail"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_check_cache_dir_read_only(tmp_path: Path) -> None:
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        if cli.check_cache_dir(ro).status == "ok":
            pytest.skip("running as a user that ignores permissions (root)")
        assert cli.check_cache_dir(ro).status == "fail"
        assert cli.check_cache_dir(ro / "sub").status == "fail"
    finally:
        ro.chmod(0o700)


def test_check_node() -> None:
    found = lambda _: "/usr/bin/node"  # noqa: E731
    assert cli.check_node(found, lambda _: "v22.3.0\n").status == "ok"
    assert cli.check_node(found, lambda _: "v18.20.0").status == "warn"
    assert cli.check_node(found, lambda _: "garbage").status == "warn"
    assert cli.check_node(lambda _: None).status == "warn"

    def boom(_: list[str]) -> str:
        raise subprocess.CalledProcessError(1, "node")

    assert cli.check_node(found, boom).status == "warn"


# --- the doctor command ----------------------------------------------------------------------


def test_doctor_exits_zero_with_optional_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APEX_CACHE_DIR", str(tmp_path / "cache"))
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 0, result.output
    for name in ["python", "torch", "cuda", "cache dir", "node"]:
        assert name in result.output


def test_doctor_exits_nonzero_on_required_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    not_a_dir = tmp_path / "file"
    not_a_dir.write_text("x")
    monkeypatch.setenv("APEX_CACHE_DIR", str(not_a_dir))
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 1
    assert "cache dir" in result.output


def test_run_doctor_order(tmp_path: Path) -> None:
    names = [r.name for r in cli.run_doctor(cache_dir=tmp_path)]
    assert names == ["python", "torch", "cuda", "cache dir", "node"]


# --- apex data ---------------------------------------------------------------------------------


def _meta(year: int = 2024) -> object:
    from apex_zero.data.cache import SessionMeta

    return SessionMeta(
        source="fastf1",
        source_version="3.9.2",
        apex_zero_version=__version__,
        fetched_at_utc="2026-01-01T00:00:00+00:00",
        year=year,
        event_name="Japanese Grand Prix",
        event_slug="japanese_grand_prix",
        round_number=4,
        country="Japan",
        location="Suzuka",
        session="Q",
        session_name="Qualifying",
        drivers=["LEC", "VER"],
        fastest_lap_s={"VER": 88.197},
        n_laps=30,
        files=[],
    )


@pytest.mark.parametrize(
    ("source", "module", "func"),
    [
        (None, "openf1_loader", "fetch_openf1_session"),
        ("openf1", "openf1_loader", "fetch_openf1_session"),
        ("fastf1", "fastf1_loader", "fetch_session"),
    ],
)
def test_data_fetch_dispatches_by_source(
    monkeypatch: pytest.MonkeyPatch, source: str | None, module: str, func: str
) -> None:
    import importlib

    from apex_zero.data.processed import FetchResult, RejectedLap

    calls: list[tuple[object, ...]] = []

    def fake_fetch(*args: object, **kwargs: object) -> FetchResult:
        calls.append((*args, kwargs.get("force")))
        rej = [RejectedLap(driver="VER", lap_number=3, issues=["x"])]
        return FetchResult(meta=_meta(), from_cache=bool(calls[1:]), rejected=rej)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.import_module(f"apex_zero.data.{module}"), func, fake_fetch)
    args = ["data", "fetch", "--year", "2024", "--event", "Japan", "--session", "Q"]
    if source is not None:
        args += ["--source", source]
    first = runner.invoke(cli.app, args)
    assert first.exit_code == 0, first.output
    assert "2 drivers, 30 laps" in first.output and "fastf1 3.9.2" in first.output
    assert "1 lap(s) failed validation" in first.output
    second = runner.invoke(cli.app, [*args, "--force"])
    assert "processed cache (no network)" in second.output
    assert calls == [(2024, "Japan", "Q", False), (2024, "Japan", "Q", True)]


def test_data_fetch_rejects_unknown_source() -> None:
    args = ["data", "fetch", "--year", "2024", "--event", "Japan", "--session", "Q"]
    assert runner.invoke(cli.app, [*args, "--source", "ergast"]).exit_code == 2


def test_data_fetch_failure_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    from apex_zero.data import openf1_loader
    from apex_zero.errors import DataUnavailableError

    def fail(*args: object, **kwargs: object) -> None:
        raise DataUnavailableError("OpenF1: no 2024 meeting matching 'Atlantis'")

    monkeypatch.setattr(openf1_loader, "fetch_openf1_session", fail)
    result = runner.invoke(
        cli.app, ["data", "fetch", "--year", "2024", "--event", "Atlantis", "--session", "Q"]
    )
    assert result.exit_code == 1
    assert "Atlantis" in result.output


def test_data_ls_and_info(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from apex_zero.config.paths import resolve_paths
    from apex_zero.data import cache

    monkeypatch.setenv("APEX_PROCESSED_DIR", str(tmp_path / "processed"))
    assert "no processed sessions" in runner.invoke(cli.app, ["data", "ls"]).output
    missing = runner.invoke(cli.app, ["data", "info", "2024", "Japan", "Q"])
    assert missing.exit_code == 1 and "not in the processed cache" in missing.output
    assert runner.invoke(cli.app, ["data", "info", "2024", "Japan", "warmup"]).exit_code == 1

    paths = resolve_paths()
    meta = _meta()
    cache.write_meta(paths, meta)  # type: ignore[arg-type]
    cache.alias_record(paths, "openf1", 2024, "Japan", "Q", meta.key)  # type: ignore[attr-defined]

    listing = runner.invoke(cli.app, ["data", "ls", "--year", "2024"])
    assert listing.exit_code == 0 and "Japanese Grand Prix" in listing.output
    for query in (["2024", "Japan", "Q"], ["2024", "Japanese Grand Prix", "qualifying"]):
        info = runner.invoke(cli.app, ["data", "info", *query])
        assert info.exit_code == 0, info.output
        assert "88.197" in info.output and "LEC" in info.output and "intact" in info.output


# --- apex track ---------------------------------------------------------------------------------


@pytest.fixture
def synthetic_track_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A processed synthetic session plus configs/tracks/synthetic.yaml in a temp CWD."""
    import numpy as np
    import yaml

    from apex_zero.config.paths import resolve_paths
    from apex_zero.data import cache
    from apex_zero.data.fastf1_loader import _handle_from_session, export_processed
    from tests.fixtures.synthetic import ellipse_track, make_session
    from tests.unit.track.conftest import DRIVERS, synthetic_config

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("APEX_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("APEX_PROCESSED_DIR", str(tmp_path / "processed"))
    paths = resolve_paths()
    track = ellipse_track()
    handle = _handle_from_session(
        2024, "Q", make_session(track, np.random.default_rng(1), drivers=DRIVERS, n_laps=3)
    )
    export_processed(handle, paths)
    cache.alias_record(paths, "fastf1", 2024, "Synthetic", "Q", handle.key)
    cfg = synthetic_config(
        track,
        sessions={
            "reference": {"year": 2024, "event": "Synthetic", "session": "Q", "source": "fastf1"}
        },
    )
    (tmp_path / "configs/tracks").mkdir(parents=True)
    (tmp_path / "configs/tracks/synthetic.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json", by_alias=True))
    )
    return tmp_path


def test_track_build_and_inspect(synthetic_track_env: Path) -> None:
    built = runner.invoke(cli.app, ["track", "build", "--name", "synthetic", "--no-fetch"])
    assert built.exit_code == 0, built.output
    assert "length vs official" in built.output and "pass" in built.output
    out = synthetic_track_env / "processed/tracks/synthetic"
    assert {p.name for p in out.iterdir()} == {"asset.npz", "meta.json", "track.glb"}

    inspected = runner.invoke(
        cli.app,
        ["track", "inspect", "synthetic", "--debug", "--out", str(synthetic_track_env / "plots")],
    )
    assert inspected.exit_code == 0, inspected.output
    names = {p.name for p in (synthetic_track_env / "plots").iterdir()}
    assert names == {
        f"synthetic_{k}.png"
        for k in ("xy", "curvature", "width", "elevation", "debug_raw", "debug_lateral")
    }


def test_track_build_reports_failed_checks(synthetic_track_env: Path) -> None:
    import yaml

    cfg_path = synthetic_track_env / "configs/tracks/synthetic.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["official_length_m"] = cfg["official_length_m"] * 1.02  # 2 % off: must fail
    cfg_path.write_text(yaml.safe_dump(cfg))
    built = runner.invoke(cli.app, ["track", "build", "--name", "synthetic", "--no-fetch"])
    assert built.exit_code == 1
    assert "FAIL" in built.output and "acceptance checks failed" in built.output
    assert (synthetic_track_env / "processed/tracks/synthetic/asset.npz").is_file()


def test_track_build_errors(synthetic_track_env: Path) -> None:
    wrong = runner.invoke(
        cli.app,
        [
            "track",
            "build",
            "--name",
            "other",
            "--config",
            "configs/tracks/synthetic.yaml",
            "--no-fetch",
        ],
    )
    assert wrong.exit_code == 1 and "does not match" in wrong.output
    missing = runner.invoke(
        cli.app, ["track", "build", "--name", "synthetic", "--no-fetch", "--year", "2023"]
    )
    assert missing.exit_code == 1 and "apex data fetch" in missing.output
    assert runner.invoke(cli.app, ["track", "inspect", "nothing"]).exit_code == 1
