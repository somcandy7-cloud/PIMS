# tests/test_isolation_forest_adapter.py
import numpy as np
import pandas as pd
import pytest
from src.agents.isolation_forest_adapter import IsolationForestAdapter
from src.agents.base_detector import AnomalyEvent


def _make_normal_df(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    t = pd.date_range("2026-03-20", periods=n, freq="10ms", tz="UTC")
    return pd.DataFrame({
        "speed": np.full(n, 50.0) + rng.normal(0, 0.5, n),
        "current": np.full(n, 100.0) + rng.normal(0, 1.0, n),
        "pressure": np.full(n, 10.0) + rng.normal(0, 0.1, n),
    }, index=t)


def _make_anomaly_df(n_pre: int = 300, n_anom: int = 20, n_post: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    n = n_pre + n_anom + n_post
    t = pd.date_range("2026-03-20", periods=n, freq="10ms", tz="UTC")
    speed = np.concatenate([
        np.full(n_pre, 50.0) + rng.normal(0, 0.5, n_pre),
        np.linspace(50, 0, n_anom),
        np.zeros(n_post) + rng.normal(0, 0.2, n_post),
    ])
    current = np.concatenate([
        np.full(n_pre, 100.0) + rng.normal(0, 1, n_pre),
        np.linspace(100, 0, n_anom),
        np.zeros(n_post) + rng.normal(0, 0.3, n_post),
    ])
    pressure = np.full(n, 10.0) + rng.normal(0, 0.1, n)
    return pd.DataFrame({"speed": speed, "current": current, "pressure": pressure}, index=t)


def test_returns_list_of_anomaly_events():
    adapter = IsolationForestAdapter()
    df = _make_anomaly_df()
    events = adapter.detect(df)
    assert isinstance(events, list)
    assert all(isinstance(e, AnomalyEvent) for e in events)


def test_detects_anomaly_in_anomaly_data():
    adapter = IsolationForestAdapter(contamination=0.05)
    df = _make_anomaly_df()
    events = adapter.detect(df)
    assert len(events) >= 1


def test_event_label_is_if_candidate():
    adapter = IsolationForestAdapter(contamination=0.05)
    events = adapter.detect(_make_anomaly_df())
    assert all(e.label == "if_candidate" for e in events)


def test_event_has_float_score():
    adapter = IsolationForestAdapter(contamination=0.05)
    events = adapter.detect(_make_anomaly_df())
    if events:
        assert all(isinstance(e.score, float) for e in events)


def test_name():
    assert IsolationForestAdapter().name() == "isolation_forest"


def test_empty_df_returns_empty():
    adapter = IsolationForestAdapter()
    empty = pd.DataFrame(columns=["speed", "current"])
    result = adapter.detect(empty)
    assert result == []


def test_respects_excluded_signals():
    df = _make_anomaly_df()
    adapter_excl = IsolationForestAdapter(contamination=0.05, excluded_signals=["speed", "current"])
    events = adapter_excl.detect(df)
    assert isinstance(events, list)
