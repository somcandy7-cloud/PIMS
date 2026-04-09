# tests/test_context_formatter.py
import json
import pandas as pd
import numpy as np
import pytest
from src.agents.base_detector import AnomalyEvent
from src.utils.context_formatter import AnomalyContextFormatter


def _make_event() -> AnomalyEvent:
    return AnomalyEvent(
        timestamp=pd.Timestamp("2026-03-12 06:54:00+00:00"),
        score=-0.35,
        top_signals=[("sig_a", 0.8), ("sig_b", 0.4)],
        label="if_candidate",
        metadata={"window_size": 30},
    )


def _make_df(ts: pd.Timestamp, n: int = 300) -> pd.DataFrame:
    start = ts - pd.Timedelta(minutes=5)
    t = pd.date_range(start, periods=n, freq="2s", tz="UTC")
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "sig_a":       rng.normal(100, 5, n),
        "sig_b":       rng.normal(50, 2, n),
        "MD    348_1": rng.normal(150_000, 10_000, n),
    }, index=t)


def test_returns_required_keys():
    event = _make_event()
    result = AnomalyContextFormatter().format(event, _make_df(event.timestamp))
    assert "json_payload" in result
    assert "text_summary" in result


def test_json_payload_valid():
    event = _make_event()
    payload = json.loads(
        AnomalyContextFormatter().format(event, _make_df(event.timestamp))["json_payload"]
    )
    for key in ("timestamp", "score_level", "top_signals", "context_stats"):
        assert key in payload


def test_cross_signal_in_summary():
    event = _make_event()
    result = AnomalyContextFormatter().format(event, _make_df(event.timestamp))
    assert "MD    348_1" in result["text_summary"]


def test_missing_cross_signal_no_error():
    event = _make_event()
    t = pd.date_range("2026-03-12 06:49:00", periods=50, freq="2s", tz="UTC")
    df = pd.DataFrame({"sig_a": np.ones(50)}, index=t)
    result = AnomalyContextFormatter().format(event, df)
    assert "json_payload" in result
