"""Explicit, reproducible random number generation.

There is no global seed anywhere in APEX ZERO. Every stochastic function takes a
``np.random.Generator`` (or, on the Torch path, a ``torch.Generator`` built in the layer that
owns Torch). Generators are created here from an integer seed, and independent child streams
are derived with ``np.random.SeedSequence.spawn`` so that adding a consumer never changes the
numbers another consumer sees.
"""

from __future__ import annotations

import numpy as np

__all__ = ["make_rng", "spawn_rngs", "spawn_seeds"]


def _check_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")


def make_rng(seed: int) -> np.random.Generator:
    """Create a NumPy generator (PCG64) from a non-negative integer seed.

    Args:
        seed: Non-negative integer seed.

    Returns:
        A new, independent ``np.random.Generator``.

    Raises:
        TypeError: If ``seed`` is not an ``int``.
        ValueError: If ``seed`` is negative.
    """
    _check_seed(seed)
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))


def spawn_seeds(seed: int, n: int) -> list[int]:
    """Derive ``n`` independent 63-bit integer seeds from one root seed.

    Useful for handing seeds to components that need a plain ``int`` (e.g. a Torch generator
    or a Gymnasium ``reset(seed=...)``).

    Args:
        seed: Non-negative root seed.
        n: Number of child seeds, ``n >= 0``.

    Returns:
        ``n`` deterministic child seeds in ``[0, 2**63)``.
    """
    _check_seed(seed)
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    children = np.random.SeedSequence(seed).spawn(n)
    return [int(child.generate_state(1, dtype=np.uint64)[0] >> np.uint64(1)) for child in children]


def spawn_rngs(seed: int, n: int) -> list[np.random.Generator]:
    """Derive ``n`` statistically independent generators from one root seed.

    Args:
        seed: Non-negative root seed.
        n: Number of generators, ``n >= 0``.

    Returns:
        ``n`` deterministic, independent ``np.random.Generator`` objects.
    """
    _check_seed(seed)
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    return [np.random.Generator(np.random.PCG64(s)) for s in np.random.SeedSequence(seed).spawn(n)]
