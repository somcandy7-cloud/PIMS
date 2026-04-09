import pandas as pd
import numpy as np
import pytest
from src.utils.preprocessor import Preprocessor


@pytest.fixture
def raw_df():
    n = 100
    t = pd.date_range("2026-03-20 15:49:00", periods=n, freq="10ms", tz="UTC")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "MD 100": np.linspace(0, 100, n),
            "DB420.DBW 2": rng.normal(50, 5, n),
            "QB 0": rng.integers(0, 2, n).astype(float),
            "bad_col": ["str"] * n,  # 비수치형 컬럼
        },
        index=t,
    )
    # NaN 삽입
    df.iloc[10:15, 0] = np.nan
    return df


def test_drop_non_numeric(raw_df):
    proc = Preprocessor()
    result = proc.process(raw_df)
    assert "bad_col" not in result.columns


def test_fill_missing_values(raw_df):
    proc = Preprocessor()
    result = proc.process(raw_df)
    assert result["MD 100"].isna().sum() == 0


def test_numeric_columns_are_float(raw_df):
    proc = Preprocessor()
    result = proc.process(raw_df)
    for col in result.columns:
        assert result[col].dtype in (np.float64, np.float32, float)


def test_index_is_datetimeindex(raw_df):
    proc = Preprocessor()
    result = proc.process(raw_df)
    assert isinstance(result.index, pd.DatetimeIndex)


def test_empty_dataframe_handled():
    proc = Preprocessor()
    empty = pd.DataFrame()
    result = proc.process(empty)
    assert result.empty
