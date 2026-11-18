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

GROUPS = ["data", "track", "ghost", "car", "qss", "export"]
STUBS = ["train", "eval", "compare", "clip"]


def test_help_lists_every_command() -> None:
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for name in [*GROUPS, *STUBS, "doctor"]:
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
