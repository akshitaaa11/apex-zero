from __future__ import annotations

import numpy as np
import pytest

from apex_zero.utils.seeding import make_rng, spawn_rngs, spawn_seeds


def test_same_seed_same_stream() -> None:
    a = make_rng(123).standard_normal(100)
    b = make_rng(123).standard_normal(100)
    np.testing.assert_array_equal(a, b)


def test_different_seed_different_stream() -> None:
    assert not np.array_equal(make_rng(1).random(10), make_rng(2).random(10))


def test_make_rng_does_not_touch_global_state() -> None:
    # Deliberately uses the legacy global API: that is the state being checked.
    np.random.seed(0)  # noqa: NPY002
    expected = np.random.random()  # noqa: NPY002
    np.random.seed(0)  # noqa: NPY002
    make_rng(42).random(1000)
    assert np.random.random() == expected  # noqa: NPY002


def test_spawn_rngs_deterministic_and_independent() -> None:
    first = [g.random(5) for g in spawn_rngs(7, 3)]
    second = [g.random(5) for g in spawn_rngs(7, 3)]
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(first[0], first[1])
    # Spawning more children leaves the earlier children's streams unchanged.
    np.testing.assert_array_equal(spawn_rngs(7, 5)[0].random(5), first[0])


def test_spawn_seeds() -> None:
    seeds = spawn_seeds(7, 4)
    assert seeds == spawn_seeds(7, 4)
    assert len(set(seeds)) == 4
    assert all(isinstance(s, int) and 0 <= s < 2**63 for s in seeds)
    assert spawn_seeds(7, 0) == []


@pytest.mark.parametrize("fn", [make_rng, lambda s: spawn_rngs(s, 1), lambda s: spawn_seeds(s, 1)])
def test_rejects_bad_seeds(fn: object) -> None:
    assert callable(fn)
    with pytest.raises(ValueError):
        fn(-1)
    with pytest.raises(TypeError):
        fn(1.5)
    with pytest.raises(TypeError):
        fn(True)


def test_rejects_negative_count() -> None:
    with pytest.raises(ValueError):
        spawn_rngs(0, -1)
    with pytest.raises(ValueError):
        spawn_seeds(0, -1)
