# tests/test_zscore_adapter.py
import numpy as np
import pandas as pd
from src.agents.zscore_adapter import ZScoreAdapter
from src.agents.base_detector import AnomalyEvent


def _make_trip_df() -> pd.DataFrame:
    n_pre, n_trip, n_post = 300, 10, 100
    rng = np.random.default_rng(99)
    n = n_pre + n_trip + n_post
    t = pd.date_range("2026-03-20 15:49:00", periods=n, freq="10ms", tz="UTC")
    speed = np.concatenate([
        np.full(n_pre, 50.0) + rng.normal(0, 0.5, n_pre),
        np.linspace(50, 0, n_trip),
        np.zeros(n_post) + rng.normal(0, 0.1, n_post),
    ])
    current = np.concatenate([
        np.full(n_pre, 100.0) + rng.normal(0, 1, n_pre),
        np.linspace(100, 0, n_trip),
        np.zeros(n_post) + rng.normal(0, 0.2, n_post),
    ])
    return pd.DataFrame({"speed": speed, "current": current}, index=t)


def test_returns_anomaly_events():
    adapter = ZScoreAdapter()
    df = _make_trip_df()
    events = adapter.detect(df)
    assert len(events) >= 1
    assert all(isinstance(e, AnomalyEvent) for e in events)


def test_event_has_score_and_signals():
    adapter = ZScoreAdapter()
    events = adapter.detect(_make_trip_df())
    evt = events[0]
    assert evt.score > 0
    assert len(evt.top_signals) >= 1


def test_name():
    assert ZScoreAdapter().name() == "zscore"


def test_no_event_in_flat_data():
    rng = np.random.default_rng(0)
    t = pd.date_range("2026-03-20", periods=500, freq="10ms", tz="UTC")
    df = pd.DataFrame({
        "speed": np.full(500, 50.0) + rng.normal(0, 0.3, 500),
        "current": np.full(500, 100.0) + rng.normal(0, 0.5, 500),
    }, index=t)
    events = ZScoreAdapter().detect(df)
    assert len(events) == 0
