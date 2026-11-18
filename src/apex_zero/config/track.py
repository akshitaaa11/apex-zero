"""``configs/tracks/<name>.yaml``: reconstruction parameters and known facts about a circuit.

Loaded with ``apex_zero.track.reconstruct.load_track_config`` (YAML parsing lives outside L0).
Every smoothing value is a **length scale in metres** (see ``track/centerline.py``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["CornerName", "SessionRef", "Smoothing", "TrackConfig", "TrackSessions", "WidthPrior"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class WidthPrior(_Frozen):
    """Plausible total track width range [m]; each side is clipped to half of it."""

    w_min: float = Field(alias="min", gt=0)
    w_max: float = Field(alias="max", gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> WidthPrior:
        if self.w_min >= self.w_max:
            raise ValueError("width_prior_m.min must be < max")
        return self


class Smoothing(_Frozen):
    """Smoothing length scales [m]."""

    seed: float = Field(
        default=25.0, ge=0, description="seed lap polyline (removes 4 Hz chord noise)"
    )
    xy: float = Field(default=15.0, ge=0, description="lateral offsets of the centerline")
    z: float = Field(default=60.0, ge=0, description="elevation (z noise is larger than x/y noise)")
    width: float = Field(default=40.0, ge=0, description="boundary widths")
    kappa: float = Field(default=10.0, ge=0, description="extra smoothing of curvature (0: none)")
    bank: float = Field(default=50.0, ge=0, description="bank angle")


class SessionRef(_Frozen):
    """A session in the processed cache, by the query used to fetch it."""

    year: int
    event: str
    session: str
    source: Literal["openf1", "fastf1"] = "openf1"


class TrackSessions(_Frozen):
    """Sessions used to build the track."""

    reference: SessionRef
    extra: list[SessionRef] = Field(default_factory=list)

    @property
    def all(self) -> list[SessionRef]:
        """Reference session first, then extras."""
        return [self.reference, *self.extra]


class CornerName(_Frozen):
    """A named corner by ``s`` range (filled in after the first build)."""

    name: str
    s_start: float
    s_end: float


class TrackConfig(_Frozen):
    """Reconstruction config of one circuit."""

    name: str = Field(pattern=r"^[a-z0-9_]+$")
    official_length_m: float = Field(gt=0)
    width_prior_m: WidthPrior
    width_margin_m: float = Field(
        default=1.0,
        ge=0,
        description="added outside the observed car-centre band: about half a car's width",
    )
    smoothing: Smoothing = Smoothing()
    self_intersecting: bool = False
    sessions: TrackSessions
    corners: list[CornerName] = Field(default_factory=list)
    bin_m: float = Field(default=5.0, gt=0, description="refinement bin size along s")
    width_bin_m: float = Field(default=10.0, gt=0, description="width percentile bin size along s")
    refine_iterations: int = Field(default=3, ge=1, le=10)
    window_m: float = Field(default=50.0, gt=0, description="s-continuity projection window")
    estimate_bank: bool = True
    recentre: bool = True
    min_observed_spread_m: float = Field(
        default=2.0,
        ge=0,
        description="median per-bin lateral spread below which widths are not observable",
    )
    nominal_width_m: float | None = Field(
        None,
        gt=0,
        description=(
            "total width used when the data has no lateral information (map-matched "
            "positions); an assumption, recorded as width_source=config_nominal"
        ),
    )
