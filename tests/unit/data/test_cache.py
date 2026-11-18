from __future__ import annotations

import json
from pathlib import Path

import pytest

from apex_zero.config.paths import Paths
from apex_zero.data import cache


def make_meta(paths: Paths, key: cache.SessionKey, content: bytes = b"abc") -> cache.SessionMeta:
    root = cache.session_dir(paths, key)
    f = cache.laps_path(paths, key)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(content)
    meta = cache.SessionMeta(
        source="fastf1",
        source_version="3.9.2",
        apex_zero_version="0.0.1",
        fetched_at_utc="2026-01-01T00:00:00+00:00",
        year=key.year,
        event_name="Japanese Grand Prix",
        event_slug=key.event_slug,
        round_number=4,
        country="Japan",
        location="Suzuka",
        session=key.session,
        session_name="Qualifying",
        drivers=["VER"],
        fastest_lap_s={"VER": 88.197},
        n_laps=1,
        files=[cache.file_record(root, f)],
    )
    cache.write_meta(paths, meta)
    return meta


@pytest.mark.parametrize(
    ("ident", "code"),
    [("q", "Q"), ("Qualifying", "Q"), (" race ", "R"), ("FP2", "FP2"), ("practice 3", "FP3"),
     ("Sprint Shootout", "SS"), ("sq", "SQ"), ("Sprint", "S")],
)
def test_normalize_session(ident: str, code: str) -> None:
    assert cache.normalize_session(ident) == code


def test_normalize_session_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown session"):
        cache.normalize_session("warmup")


def test_slugify() -> None:
    assert cache.slugify("Japanese Grand Prix") == "japanese_grand_prix"
    assert cache.slugify("São Paulo  Grand-Prix!") == "sao_paulo_grand_prix"
    with pytest.raises(ValueError):
        cache.slugify("???")


def test_layout_is_deterministic(paths: Paths) -> None:
    key = cache.SessionKey(2024, "japanese_grand_prix", "Q")
    assert key.relpath == Path("2024/japanese_grand_prix/Q")
    assert str(key) == "2024/japanese_grand_prix/Q"
    assert cache.SessionKey.from_relpath("2024/japanese_grand_prix/Q") == key
    root = paths.processed_dir / "2024/japanese_grand_prix/Q"
    assert cache.session_dir(paths, key) == root
    assert cache.meta_path(paths, key) == root / "session_meta.json"
    assert cache.pos_path(paths, key, "VER", 7) == root / "VER/pos_lap007.parquet"
    assert cache.car_path(paths, key, "VER", 12) == root / "VER/car_lap012.parquet"
    assert cache.fastf1_cache_dir(paths) == paths.cache_dir / "fastf1"
    assert cache.openf1_cache_dir(paths) == paths.cache_dir / "openf1"


def test_meta_round_trip_and_integrity(paths: Paths) -> None:
    key = cache.SessionKey(2024, "japanese_grand_prix", "Q")
    assert cache.read_meta(paths, key) is None
    assert not cache.is_complete(paths, key)
    meta = make_meta(paths, key)
    assert cache.read_meta(paths, key) == meta
    assert meta.key == key
    assert cache.is_complete(paths, key)

    # same size, different content -> only the hash check catches it
    cache.laps_path(paths, key).write_bytes(b"xyz")
    assert cache.is_complete(paths, key, verify_hashes=False)
    assert not cache.is_complete(paths, key)
    cache.laps_path(paths, key).unlink()
    assert not cache.is_complete(paths, key, verify_hashes=False)


def test_old_layout_version_is_incomplete(paths: Paths) -> None:
    key = cache.SessionKey(2024, "japanese_grand_prix", "Q")
    make_meta(paths, key)
    path = cache.meta_path(paths, key)
    data = json.loads(path.read_text())
    data["layout_version"] = 0
    path.write_text(json.dumps(data))
    assert not cache.is_complete(paths, key)


def test_aliases(paths: Paths) -> None:
    key = cache.SessionKey(2024, "japanese_grand_prix", "Q")
    assert cache.alias_lookup(paths, 2024, "Japan", "Q") is None
    cache.alias_record(paths, 2024, "Japan", "Qualifying", key)
    assert cache.alias_lookup(paths, 2024, " japan ", "q") == key
    assert cache.alias_lookup(paths, 2023, "Japan", "Q") is None
    (paths.processed_dir / "aliases.json").write_text("[1, 2]")
    with pytest.raises(ValueError, match="mapping"):
        cache.alias_lookup(paths, 2024, "Japan", "Q")


def test_list_sessions(paths: Paths) -> None:
    assert cache.list_sessions(paths) == []
    k1 = cache.SessionKey(2024, "japanese_grand_prix", "Q")
    k2 = cache.SessionKey(2023, "japanese_grand_prix", "R")
    make_meta(paths, k1)
    make_meta(paths, k2)
    assert [m.key for m in cache.list_sessions(paths)] == [k2, k1]
    assert [m.key for m in cache.list_sessions(paths, 2024)] == [k1]
