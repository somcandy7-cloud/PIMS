import numpy as np
import pandas as pd
import pytest
from src.utils.operation_discovery import AutoOperationDiscovery, OperationCondition


def _make_df_with_known_signals(n: int = 200) -> pd.DataFrame:
    t = pd.date_range("2026-03-12", periods=n, freq="2s", tz="UTC")
    rng = np.random.default_rng(42)
    ctrl = np.array([1.0 if i < n // 2 else 0.0 for i in range(n)])
    speed = np.where(ctrl == 1.0,
                     rng.normal(300, 5, n),
                     rng.normal(0, 1, n))
    return pd.DataFrame({
        "ctrl_on":   ctrl,
        "speed":     speed,
        "noise":     rng.normal(50, 5, n),
        "always_0":  np.zeros(n),
    }, index=t)


def test_discovers_binary_candidate():
    df = _make_df_with_known_signals()
    conditions = AutoOperationDiscovery().discover(df)
    binary_cols = [c.column for c in conditions if c.op == "=="]
    assert "ctrl_on" in binary_cols


def test_discovers_bimodal_candidate():
    df = _make_df_with_known_signals()
    conditions = AutoOperationDiscovery().discover(df)
    bimodal_cols = [c.column for c in conditions if c.op == ">"]
    assert "speed" in bimodal_cols


def test_noise_signal_not_selected():
    df = _make_df_with_known_signals()
    conditions = AutoOperationDiscovery().discover(df)
    all_cols = [c.column for c in conditions]
    assert "noise" not in all_cols


def test_always_zero_not_selected():
    df = _make_df_with_known_signals()
    conditions = AutoOperationDiscovery().discover(df)
    all_cols = [c.column for c in conditions]
    assert "always_0" not in all_cols


def test_returns_operation_condition_objects():
    df = _make_df_with_known_signals()
    conditions = AutoOperationDiscovery().discover(df)
    assert len(conditions) >= 1
    for c in conditions:
        assert isinstance(c, OperationCondition)
        assert c.op in ("==", "!=", ">", ">=", "<", "<=")
        assert isinstance(c.value, float)
        assert 0.0 <= c.confidence <= 1.0


def test_bimodal_threshold_between_clusters():
    df = _make_df_with_known_signals()
    conditions = AutoOperationDiscovery().discover(df)
    speed_cond = next((c for c in conditions if c.column == "speed"), None)
    assert speed_cond is not None
    assert 10.0 < speed_cond.value < 290.0


def test_empty_df_returns_empty():
    df = pd.DataFrame(columns=["a", "b"])
    conditions = AutoOperationDiscovery().discover(df)
    assert conditions == []
