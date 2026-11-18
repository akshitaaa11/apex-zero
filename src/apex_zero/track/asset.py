"""``TrackAsset``: the reconstructed track on a 1 m ``s`` grid (P2.1). Format version 1.

On disk (``<APEX_PROCESSED_DIR>/tracks/<name>/``)::

    asset.npz   float64 arrays: s, x, y, z, psi, kappa, grade, bank, w_left, w_right
    meta.json   name, length, source_sessions, build_hash, params, report, versions,
                sha256 of asset.npz

Channels (all ``(N,)`` float64, SI, ``s`` uniform in ``[0, length)``):
``x, y, z`` world [m]; ``psi`` heading [rad] (from +x, counter-clockwise); ``kappa`` signed
curvature [1/m] (> 0 turning left); ``grade`` ``dz/ds``; ``bank`` [rad] (> 0: left edge
higher); ``w_left``, ``w_right`` boundary distances [m] (both positive).

``s = 0`` is the timing line. Change the format only with a version bump and a
``docs/migrations.md`` note.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from apex_zero import __version__
from apex_zero.errors import TrackReconstructionError
from apex_zero.track.frenet import RefLine, interp_angle, interp_periodic

__all__ = ["ASSET_VERSION", "CHANNELS", "JSONScalar", "TrackAsset"]

ASSET_VERSION: int = 1
CHANNELS: tuple[str, ...] = (
    "s",
    "x",
    "y",
    "z",
    "psi",
    "kappa",
    "grade",
    "bank",
    "w_left",
    "w_right",
)

FloatArray = NDArray[np.float64]
JSONScalar = str | int | float | bool | None


@dataclass(frozen=True)
class TrackAsset:
    """Immutable track asset (arrays are made read-only)."""

    name: str
    length: float
    s: FloatArray
    x: FloatArray
    y: FloatArray
    z: FloatArray
    psi: FloatArray
    kappa: FloatArray
    grade: FloatArray
    bank: FloatArray
    w_left: FloatArray
    w_right: FloatArray
    source_sessions: tuple[str, ...]
    build_hash: str
    params: Mapping[str, JSONScalar | list[JSONScalar]] = field(default_factory=dict)
    #: Build statistics and validation results (length error, inside fraction, timings...).
    report: Mapping[str, JSONScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.s)
        for name in CHANNELS:
            arr = np.ascontiguousarray(getattr(self, name), dtype=np.float64)
            if arr.shape != (n,):
                raise ValueError(f"channel {name} has shape {arr.shape}, expected ({n},)")
            if not np.isfinite(arr).all():
                raise ValueError(f"channel {name} has non-finite values")
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)
        if not np.allclose(self.s, np.arange(n) * self.length / n, atol=1e-6 * self.length):
            raise ValueError("s must be the uniform grid k * length / N")
        if (self.w_left <= 0).any() or (self.w_right <= 0).any():
            raise ValueError("widths must be positive")

    @property
    def ds(self) -> float:
        """Grid spacing [m]."""
        return self.length / len(self.s)

    @property
    def xyz(self) -> FloatArray:
        """``(N, 3)`` centerline [m]."""
        return np.column_stack([self.x, self.y, self.z])

    @property
    def line(self) -> RefLine:
        """The centerline as a :class:`RefLine` (for Frenet conversions)."""
        return RefLine(s=np.array(self.s), xyz=self.xyz, length=self.length)

    def at(self, channel: str, s: FloatArray) -> FloatArray:
        """Periodic interpolation of a channel at arbitrary ``s`` (``psi`` without wrap jumps)."""
        if channel not in CHANNELS or channel == "s":
            raise KeyError(f"unknown channel {channel!r}")
        values = getattr(self, channel)
        if channel == "psi":
            return interp_angle(self.s, values, s, self.length)
        return interp_periodic(self.s, values, s, self.length)

    # -- persistence ------------------------------------------------------------------------

    def save(self, directory: Path) -> Path:
        """Write ``asset.npz`` and ``meta.json`` into ``directory`` (created if needed)."""
        directory.mkdir(parents=True, exist_ok=True)
        npz = directory / "asset.npz"
        np.savez(npz, **{c: getattr(self, c) for c in CHANNELS})
        meta = {
            "asset_version": ASSET_VERSION,
            "apex_zero_version": __version__,
            "name": self.name,
            "length": self.length,
            "n_points": len(self.s),
            "source_sessions": list(self.source_sessions),
            "build_hash": self.build_hash,
            "params": dict(self.params),
            "report": dict(self.report),
            "asset_sha256": _sha256(npz),
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: Path) -> TrackAsset:
        """Read an asset written by :meth:`save`, verifying version and checksum.

        Raises:
            TrackReconstructionError: Missing files, wrong version or a checksum mismatch.
        """
        npz, meta_path = directory / "asset.npz", directory / "meta.json"
        if not npz.is_file() or not meta_path.is_file():
            raise TrackReconstructionError(f"no track asset in {directory}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("asset_version") != ASSET_VERSION:
            raise TrackReconstructionError(
                f"{directory}: asset version {meta.get('asset_version')} != {ASSET_VERSION}; "
                "rebuild with `apex track build`"
            )
        if meta.get("asset_sha256") != _sha256(npz):
            raise TrackReconstructionError(f"{npz}: checksum mismatch (file modified?)")
        with np.load(npz) as data:
            arrays = {c: np.array(data[c]) for c in CHANNELS}
        return cls(
            name=str(meta["name"]),
            length=float(meta["length"]),
            source_sessions=tuple(meta["source_sessions"]),
            build_hash=str(meta["build_hash"]),
            params=meta.get("params", {}),
            report=meta.get("report", {}),
            **arrays,
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
