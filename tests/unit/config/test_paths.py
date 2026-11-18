from __future__ import annotations

from pathlib import Path

import pytest

from apex_zero.config.paths import DEFAULTS, ENV_VARS, resolve_paths


def test_defaults_relative_to_base(tmp_path: Path) -> None:
    paths = resolve_paths(env={}, base=tmp_path)
    assert paths.cache_dir == (tmp_path / DEFAULTS["cache_dir"]).resolve()
    assert paths.processed_dir == (tmp_path / "data" / "processed").resolve()
    assert paths.runs_dir == (tmp_path / "runs").resolve()
    assert not paths.cache_dir.exists()  # resolving never creates directories


def test_env_overrides(tmp_path: Path) -> None:
    env = {
        ENV_VARS["cache_dir"]: str(tmp_path / "abs_cache"),
        ENV_VARS["runs_dir"]: "rel/runs",
        ENV_VARS["processed_dir"]: "",  # empty -> default
    }
    paths = resolve_paths(env=env, base=tmp_path)
    assert paths.cache_dir == (tmp_path / "abs_cache").resolve()
    assert paths.runs_dir == (tmp_path / "rel" / "runs").resolve()
    assert paths.processed_dir == (tmp_path / "data" / "processed").resolve()


def test_defaults_to_os_environ_and_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("APEX_CACHE_DIR", "c")
    assert resolve_paths().cache_dir == (tmp_path / "c").resolve()
