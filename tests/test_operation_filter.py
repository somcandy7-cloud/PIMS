# tests/test_operation_filter.py
import pandas as pd
import numpy as np
import pytest
from src.utils.operation_discovery import OperationCondition
from src.utils.operation_filter import OperationFilter


def _make_df() -> pd.DataFrame:
    t = pd.date_range("2026-03-12", periods=5, freq="2s", tz="UTC")
    return pd.DataFrame({
        "ctrl":  [0.0, 1.0, 1.0, 1.0, 0.0],
        "speed": [0.0, 50_000.0, 150_000.0, 200_000.0, 0.0],
        "temp":  [20.0, 180.0, 200.0, 195.0, 20.0],
    }, index=t)


def _conds_two() -> list[OperationCondition]:
    return [
        OperationCondition("ctrl", "==", 1.0, 0.9, "binary"),
        OperationCondition("speed", ">", 100_000.0, 0.8, "bimodal"),
    ]


def test_and_filter_keeps_correct_rows():
    df = _make_df()
    result = OperationFilter(_conds_two()).filter(df)
    assert len(result) == 2
    assert result["temp"].tolist() == [200.0, 195.0]


def test_three_conditions_and():
    df = _make_df()
    conds = _conds_two() + [
        OperationCondition("temp", ">", 190.0, 0.7, "bimodal"),
    ]
    result = OperationFilter(conds).filter(df)
    assert len(result) == 1


def test_all_stopped_returns_empty():
    t = pd.date_range("2026-03-12", periods=3, freq="2s", tz="UTC")
    df = pd.DataFrame({
        "ctrl":  [0.0, 0.0, 0.0],
        "speed": [0.0, 0.0, 0.0],
    }, index=t)
    result = OperationFilter(_conds_two()).filter(df)
    assert result.empty


def test_missing_column_raises():
    t = pd.date_range("2026-03-12", periods=2, freq="2s", tz="UTC")
    df = pd.DataFrame({"other": [1.0, 2.0]}, index=t)
    with pytest.raises(KeyError):
        OperationFilter(_conds_two()).filter(df)


def test_supported_operators():
    t = pd.date_range("2026-03-12", periods=3, freq="2s", tz="UTC")
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=t)
    ops_expected = [
        (">",  2.0, [3.0]),
        (">=", 2.0, [2.0, 3.0]),
        ("<",  2.0, [1.0]),
        ("<=", 2.0, [1.0, 2.0]),
        ("==", 2.0, [2.0]),
        ("!=", 2.0, [1.0, 3.0]),
    ]
    for op, val, expected in ops_expected:
        cond = OperationCondition("v", op, val, 1.0, "binary")
        result = OperationFilter([cond]).filter(df)
        assert result["v"].tolist() == expected, f"op={op} failed"


def test_empty_conditions_returns_all_rows():
    df = _make_df()
    result = OperationFilter([]).filter(df)
    assert len(result) == len(df)
