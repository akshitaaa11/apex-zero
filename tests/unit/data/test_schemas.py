from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from apex_zero.data.schemas import (
    CAR,
    CONTRACTS,
    LAPS,
    NAN_BUDGET,
    POSITION,
    ColumnSpec,
    TableContract,
    assert_valid,
    validate_frame,
)
from apex_zero.errors import SchemaValidationError
from tests.unit.data.conftest import openf1_position_frame


def checks(df: pd.DataFrame, contract: TableContract) -> set[tuple[str | None, str]]:
    return {(i.column, i.check) for i in validate_frame(df, contract)}


def car_frame(n: int = 50) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SessionTime": np.arange(n, dtype=np.float64) * 0.25,
            "Speed": np.full(n, 70.0),
            "RPM": np.full(n, 11000.0),
            "nGear": pd.array([7] * n, dtype="Int64"),
            "Throttle": np.ones(n),
            "Brake": pd.array([False] * n, dtype="boolean"),
            "DRS": pd.array([8] * n, dtype="Int64"),
        }
    )


def test_real_lap_fixture_is_valid(real_lap: dict[str, Any]) -> None:
    df = openf1_position_frame(real_lap["location"])
    assert validate_frame(df, POSITION) == []
    assert_valid(df, POSITION)  # does not raise


def test_corrupted_real_fixture_is_flagged(real_lap: dict[str, Any]) -> None:
    """Gate: schema validation flags a deliberately corrupted fixture."""
    df = openf1_position_frame(real_lap["location"])
    bad = df.copy()
    bad.loc[10, "SessionTime"] = bad.loc[9, "SessionTime"]  # duplicated timestamp
    bad.loc[20, "SessionTime"] = bad.loc[18, "SessionTime"] - 1.0  # time goes backwards
    n_nan = int(len(bad) * NAN_BUDGET) + 2
    bad.loc[100 : 100 + n_nan - 1, "X"] = np.nan  # NaN burst over budget
    bad["Y"] = bad["Y"].astype(str)  # wrong dtype
    bad = bad.drop(columns=["Z"])  # missing channel
    found = checks(bad, POSITION)
    assert {
        ("SessionTime", "duplicates"),
        ("SessionTime", "monotonic"),
        ("X", "nan_budget"),
        ("Y", "dtype"),
        ("Z", "required"),
    } <= found
    with pytest.raises(SchemaValidationError, match="schema violation"):
        assert_valid(bad, POSITION, context="VER lap 11")


def test_nan_budget_boundary() -> None:
    df = car_frame(100)
    df.loc[:1, "Speed"] = np.nan  # exactly 2 %
    assert validate_frame(df, CAR) == []
    df.loc[:2, "Speed"] = np.nan  # 3 %
    assert ("Speed", "nan_budget") in checks(df, CAR)


def test_non_nullable_and_int_bool_dtypes() -> None:
    df = car_frame()
    df["nGear"] = df["nGear"].astype("float64")
    df.loc[0, "SessionTime"] = np.nan
    df["Brake"] = df["Brake"].astype("Int64")
    df.loc[3, "DRS"] = pd.NA
    found = checks(df, CAR)
    assert ("nGear", "dtype") in found
    assert ("SessionTime", "nulls") in found
    assert ("Brake", "dtype") in found
    assert ("DRS", "nulls") in found


def test_lap_table_unique_and_str_columns() -> None:
    laps = pd.DataFrame(
        {c.name: pd.Series([None, None], dtype="float64") for c in LAPS.columns}
    )
    laps["Driver"] = ["VER", "VER"]
    laps["DriverNumber"] = ["1", "1"]
    laps["LapNumber"] = pd.array([3, 3], dtype="Int64")
    laps["Compound"] = pd.Series(["SOFT", None], dtype="object")
    laps["IsAccurate"] = pd.array([True, False], dtype="boolean")
    laps["Deleted"] = pd.array([pd.NA, False], dtype="boolean")
    assert checks(laps, LAPS) == {(None, "unique")}
    laps.loc[1, "LapNumber"] = 4
    assert validate_frame(laps, LAPS) == []
    laps["DriverNumber"] = [1, 1]
    assert ("DriverNumber", "dtype") in checks(laps, LAPS)


def test_str_kind_rejects_mixed_objects() -> None:
    contract = TableContract("t", (ColumnSpec("a", "str"),))
    assert validate_frame(pd.DataFrame({"a": ["x", "y"]}), contract) == []
    assert ("a", "dtype") in checks(pd.DataFrame({"a": ["x", 3]}), contract)
    assert ("a", "dtype") in checks(pd.DataFrame({"a": [1.0, 2.0]}), contract)


def test_contract_registry_and_issue_str() -> None:
    assert set(CONTRACTS) == {"position", "car", "laps"}
    assert POSITION.names[0] == "SessionTime"
    issue = validate_frame(pd.DataFrame({"SessionTime": [0.0]}), POSITION)[0]
    assert str(issue).startswith("position.X: required")
    assert validate_frame(car_frame(0), CAR) == []
