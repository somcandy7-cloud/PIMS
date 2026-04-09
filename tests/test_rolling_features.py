# tests/test_rolling_features.py
import pandas as pd
import numpy as np
from src.utils.rolling_features import RollingFeatureExtractor


def _make_df(n: int = 50) -> pd.DataFrame:
    t = pd.date_range("2026-03-12", periods=n, freq="2s", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "sig_a": np.full(n, 100.0) + rng.normal(0, 1, n),
        "sig_b": np.linspace(0, 50, n),
    }, index=t)


def test_output_has_same_index():
    result = RollingFeatureExtractor(window=5).transform(_make_df())
    assert list(result.index) == list(_make_df().index)


def test_column_names_contain_suffix():
    cols = RollingFeatureExtractor(window=5).transform(_make_df()).columns.tolist()
    assert any("_rmean" in c for c in cols)
    assert any("_rstd" in c for c in cols)
    assert any("_roc" in c for c in cols)


def test_no_nan_after_transform():
    result = RollingFeatureExtractor(window=5).transform(_make_df(n=30))
    assert result.isna().sum().sum() == 0


def test_constant_signal_zero_rstd():
    t = pd.date_range("2026-03-12", periods=20, freq="2s", tz="UTC")
    df = pd.DataFrame({"flat": np.ones(20)}, index=t)
    result = RollingFeatureExtractor(window=5).transform(df)
    assert (result["flat_rstd"] == 0.0).all()
