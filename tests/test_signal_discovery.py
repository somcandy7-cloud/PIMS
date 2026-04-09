import pandas as pd
import numpy as np
import pytest
from src.utils.signal_discovery import classify_numeric_columns, filter_by_patterns, SignalGroup


@pytest.fixture
def mixed_df():
    n = 200
    rng = np.random.default_rng(1)
    t = pd.date_range("2026-03-20", periods=n, freq="10ms", tz="UTC")
    return pd.DataFrame(
        {
            "MD 100": rng.normal(50, 10, n),           # analog (많은 unique값)
            "DB420.DBW 2": rng.normal(120, 5, n),      # analog
            "QB 0": rng.integers(0, 2, n).astype(float),  # digital (0/1)
            "MB 10": rng.integers(0, 2, n).astype(float), # digital
            "MW 50": rng.normal(0, 1, n),              # analog
        },
        index=t,
    )


def test_filter_by_patterns_md(mixed_df):
    result = filter_by_patterns(mixed_df, ["MD "])
    assert result == ["MD 100"]


def test_filter_by_patterns_multiple(mixed_df):
    result = filter_by_patterns(mixed_df, ["DB420.DBW", "MW "])
    assert set(result) == {"DB420.DBW 2", "MW 50"}


def test_filter_by_patterns_empty(mixed_df):
    result = filter_by_patterns(mixed_df, ["NONEXISTENT"])
    assert result == []


def test_classify_returns_signal_groups(mixed_df):
    groups = classify_numeric_columns(mixed_df)
    assert len(groups) == 5
    assert all(isinstance(g, SignalGroup) for g in groups)


def test_classify_binary_detection(mixed_df):
    groups = {g.name: g for g in classify_numeric_columns(mixed_df)}
    assert groups["QB 0"].is_binary is True
    assert groups["MB 10"].is_binary is True
    assert groups["MD 100"].is_binary is False
    assert groups["DB420.DBW 2"].is_binary is False


def test_classify_stores_stats(mixed_df):
    groups = {g.name: g for g in classify_numeric_columns(mixed_df)}
    g = groups["MD 100"]
    assert abs(g.mean - mixed_df["MD 100"].mean()) < 1e-6
    assert g.unique_count > 2
