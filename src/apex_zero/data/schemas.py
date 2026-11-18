"""Column contracts for the processed session tables, and their validators (P1.3).

Processed tables are written by ``data.openf1_loader`` or ``data.fastf1_loader`` and read by
every later phase. They are **SI**: the conversion from source units happens once, at ingest.
``SessionTime`` is seconds on the source's session clock: FastF1's ``SessionTime``, or for
OpenF1 seconds since the session's scheduled ``date_start``. Times are only comparable within
one session export.

``position`` (one Parquet file per driver lap, raw samples, ~4 Hz):
    ``SessionTime`` [s], ``X``/``Y``/``Z`` [m, world frame of the source], ``Status`` (str,
    ``"OnTrack"``/``"OffTrack"``; missing throughout for OpenF1, which has no such channel).

``car`` (one Parquet file per driver lap, raw samples, ~4 Hz):
    ``SessionTime`` [s], ``Speed`` [m/s], ``RPM`` [rev/min, name says it], ``nGear`` (int),
    ``Throttle`` [fraction, 0..1, nominally; the feed occasionally reports slightly above 100 %],
    ``Brake`` (bool: **on/off only**, no pressure), ``DRS`` (int, raw feed code).

``laps`` (one Parquet file per session):
    ``Driver`` (3-letter code), ``DriverNumber`` (str), ``LapNumber`` (int), ``LapStartTime`` and
    ``LapEndTime`` [s, session time], ``LapTime`` and ``Sector1Time``..``Sector3Time`` [s],
    ``PitInTime``/``PitOutTime`` [s, session time], ``Compound`` (str), ``TyreLife`` [laps],
    ``IsAccurate`` (bool, FastF1's lap-timing accuracy flag), ``Deleted`` (bool, lap time deleted
    by race control, e.g. track limits; missing when the source could not determine it).

Checks: required columns, dtypes, nulls only where allowed, a strictly increasing time column
(no duplicate timestamps), and a NaN budget of at most 2 % per channel per lap.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from apex_zero.errors import SchemaValidationError

__all__ = [
    "CAR",
    "CONTRACTS",
    "LAPS",
    "NAN_BUDGET",
    "POSITION",
    "ColumnSpec",
    "TableContract",
    "ValidationIssue",
    "assert_valid",
    "validate_frame",
]

#: Maximum fraction of missing values per channel per lap (``position`` and ``car`` tables).
NAN_BUDGET: float = 0.02

Kind = Literal["float", "int", "bool", "str"]


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """One column of a table contract.

    Attributes:
        name: Column name.
        kind: Expected dtype family.
        unit: Unit (documentation; ``""`` if dimensionless or categorical).
        nullable: Whether missing values are allowed at all. For tables with a NaN budget,
            nullable channels may be missing in at most ``NAN_BUDGET`` of rows.
        budgeted: Whether the table's NaN budget applies. ``False`` for channels a source may
            not provide at all (``Status`` from OpenF1).
    """

    name: str
    kind: Kind
    unit: str = ""
    nullable: bool = False
    budgeted: bool = True


@dataclass(frozen=True, slots=True)
class TableContract:
    """Columns plus table-level rules.

    Attributes:
        name: Table name, used in messages and file names.
        columns: Required columns, in canonical order.
        time_column: Column that must be strictly increasing, or ``None``.
        nan_budget: Max missing fraction per nullable column, or ``None`` for no budget.
        unique: Columns whose combination must be unique, or empty.
    """

    name: str
    columns: tuple[ColumnSpec, ...]
    time_column: str | None = None
    nan_budget: float | None = None
    unique: tuple[str, ...] = ()

    @property
    def names(self) -> list[str]:
        """Column names in canonical order."""
        return [c.name for c in self.columns]


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A single contract violation."""

    table: str
    column: str | None
    check: str
    message: str

    def __str__(self) -> str:
        where = f"{self.table}.{self.column}" if self.column else self.table
        return f"{where}: {self.check}: {self.message}"


POSITION = TableContract(
    name="position",
    columns=(
        ColumnSpec("SessionTime", "float", "s"),
        ColumnSpec("X", "float", "m", nullable=True),
        ColumnSpec("Y", "float", "m", nullable=True),
        ColumnSpec("Z", "float", "m", nullable=True),
        ColumnSpec("Status", "str", nullable=True, budgeted=False),
    ),
    time_column="SessionTime",
    nan_budget=NAN_BUDGET,
)

CAR = TableContract(
    name="car",
    columns=(
        ColumnSpec("SessionTime", "float", "s"),
        ColumnSpec("Speed", "float", "m/s", nullable=True),
        ColumnSpec("RPM", "float", "rev/min", nullable=True),
        ColumnSpec("nGear", "int"),
        ColumnSpec("Throttle", "float", "fraction", nullable=True),
        ColumnSpec("Brake", "bool"),
        ColumnSpec("DRS", "int"),
    ),
    time_column="SessionTime",
    nan_budget=NAN_BUDGET,
)

LAPS = TableContract(
    name="laps",
    columns=(
        ColumnSpec("Driver", "str"),
        ColumnSpec("DriverNumber", "str"),
        ColumnSpec("LapNumber", "int"),
        ColumnSpec("LapStartTime", "float", "s", nullable=True),
        ColumnSpec("LapEndTime", "float", "s", nullable=True),
        ColumnSpec("LapTime", "float", "s", nullable=True),
        ColumnSpec("Sector1Time", "float", "s", nullable=True),
        ColumnSpec("Sector2Time", "float", "s", nullable=True),
        ColumnSpec("Sector3Time", "float", "s", nullable=True),
        ColumnSpec("PitInTime", "float", "s", nullable=True),
        ColumnSpec("PitOutTime", "float", "s", nullable=True),
        ColumnSpec("Compound", "str", nullable=True),
        ColumnSpec("TyreLife", "float", "laps", nullable=True),
        ColumnSpec("IsAccurate", "bool"),
        ColumnSpec("Deleted", "bool", nullable=True),
    ),
    unique=("Driver", "LapNumber"),
)

#: Contracts by table name.
CONTRACTS: Mapping[str, TableContract] = {c.name: c for c in (POSITION, CAR, LAPS)}


def _kind_ok(series: pd.Series, kind: Kind) -> bool:
    dtype = series.dtype
    if kind == "float":
        return pd.api.types.is_float_dtype(dtype)
    if kind == "int":
        return pd.api.types.is_integer_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)
    if kind == "bool":
        return pd.api.types.is_bool_dtype(dtype)
    # str: a string dtype, or object holding only str (missing values allowed).
    if pd.api.types.is_string_dtype(dtype):
        return bool(series.dropna().map(lambda v: isinstance(v, str)).all())
    return False


def validate_frame(df: pd.DataFrame, contract: TableContract) -> list[ValidationIssue]:
    """Check a table against its contract.

    Args:
        df: Table to check. Extra columns are allowed and ignored.
        contract: Contract to check against.

    Returns:
        All violations found (empty if the table is valid).
    """
    issues: list[ValidationIssue] = []

    def issue(column: str | None, check: str, message: str) -> None:
        issues.append(ValidationIssue(contract.name, column, check, message))

    missing = [n for n in contract.names if n not in df.columns]
    for name in missing:
        issue(name, "required", "column missing")

    n_rows = len(df)
    for spec in contract.columns:
        if spec.name in missing:
            continue
        col = df[spec.name]
        if not _kind_ok(col, spec.kind):
            issue(spec.name, "dtype", f"expected {spec.kind}, got {col.dtype}")
        n_null = int(col.isna().sum())
        if n_null == 0:
            continue
        if not spec.nullable:
            issue(spec.name, "nulls", f"{n_null} missing value(s) in a non-nullable column")
        elif spec.budgeted and contract.nan_budget is not None and n_rows > 0:
            frac = n_null / n_rows
            if frac > contract.nan_budget:
                issue(
                    spec.name,
                    "nan_budget",
                    f"{frac:.1%} missing exceeds budget {contract.nan_budget:.0%}",
                )

    t = contract.time_column
    if t is not None and t not in missing and n_rows > 1:
        diffs = df[t].diff().iloc[1:]
        n_dup = int((diffs == 0).sum())
        n_back = int((diffs < 0).sum())
        if n_dup:
            issue(t, "duplicates", f"{n_dup} duplicated timestamp(s)")
        if n_back:
            issue(t, "monotonic", f"time goes backwards {n_back} time(s)")

    if contract.unique and not any(c in missing for c in contract.unique):
        n_dup = int(df.duplicated(subset=list(contract.unique)).sum())
        if n_dup:
            issue(None, "unique", f"{n_dup} duplicated {'/'.join(contract.unique)} row(s)")
    return issues


def assert_valid(df: pd.DataFrame, contract: TableContract, context: str = "") -> None:
    """Raise if ``df`` violates ``contract``.

    Args:
        df: Table to check.
        contract: Contract to check against.
        context: Prefix for the error message (e.g. ``"VER lap 11"``).

    Raises:
        SchemaValidationError: With every violation listed.
    """
    issues = validate_frame(df, contract)
    if issues:
        prefix = f"{context}: " if context else ""
        lines = "\n  ".join(str(i) for i in issues)
        raise SchemaValidationError(f"{prefix}{len(issues)} schema violation(s):\n  {lines}")
