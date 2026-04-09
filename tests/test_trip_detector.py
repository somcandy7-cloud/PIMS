import pandas as pd
import numpy as np
from src.agents.trip_detector import TripDetector, TripEvent


def _make_trip_df(n_pre=300, n_trip=10, n_post=100) -> pd.DataFrame:
    """정상 → 급정지(10샘플, 100ms) → 정지 패턴 (실제 PLC Trip 특성)"""
    rng = np.random.default_rng(99)
    n = n_pre + n_trip + n_post
    t = pd.date_range("2026-03-20 15:49:00", periods=n, freq="10ms", tz="UTC")

    speed = np.concatenate([
        np.full(n_pre, 50.0) + rng.normal(0, 0.5, n_pre),   # 정상 운전
        np.linspace(50, 0, n_trip),                           # 급정지 (Trip)
        np.zeros(n_post) + rng.normal(0, 0.1, n_post),       # 정지
    ])
    current = np.concatenate([
        np.full(n_pre, 100.0) + rng.normal(0, 1, n_pre),
        np.linspace(100, 0, n_trip),
        np.zeros(n_post) + rng.normal(0, 0.2, n_post),
    ])
    return pd.DataFrame({"speed": speed, "current": current}, index=t)


def _make_flat_df(n=500) -> pd.DataFrame:
    """완전 정상 (이상없음)"""
    rng = np.random.default_rng(0)
    t = pd.date_range("2026-03-20", periods=n, freq="10ms", tz="UTC")
    return pd.DataFrame({
        "speed": np.full(n, 50.0) + rng.normal(0, 0.3, n),
        "current": np.full(n, 100.0) + rng.normal(0, 0.5, n),
    }, index=t)


def test_trip_detected_in_trip_data():
    df = _make_trip_df()
    detector = TripDetector(window=50, roc_zscore_threshold=3.0)
    events = detector.detect(df)
    assert len(events) >= 1


def test_no_trip_in_flat_data():
    df = _make_flat_df()
    detector = TripDetector(window=50, roc_zscore_threshold=3.0)
    events = detector.detect(df)
    assert len(events) == 0


def test_trip_event_fields():
    df = _make_trip_df()
    detector = TripDetector(window=50, roc_zscore_threshold=3.0)
    events = detector.detect(df)
    e = events[0]
    assert isinstance(e, TripEvent)
    assert isinstance(e.timestamp, pd.Timestamp)
    assert isinstance(e.top_changed_signals, list)
    assert len(e.top_changed_signals) >= 1
    assert e.trigger in ("multi_signal_drop", "single_signal_drop", "spike")


def test_trip_timestamp_near_drop_start():
    df = _make_trip_df(n_pre=300)
    detector = TripDetector(window=50, roc_zscore_threshold=3.0)
    events = detector.detect(df)
    # 이벤트 중 하나가 실제 급정지 구간(인덱스 295~315) 안에 있으면 통과
    # min_duration=3이므로 첫 rising edge는 trip 시작+2 이후
    idxs = [df.index.get_loc(e.timestamp) for e in events]
    assert any(295 <= i <= 315 for i in idxs), f"trip 근처 이벤트 없음: {idxs}"


def test_top_changed_signals_sorted_by_magnitude():
    df = _make_trip_df()
    detector = TripDetector(window=50, roc_zscore_threshold=3.0, top_n=5)
    events = detector.detect(df)
    signals = events[0].top_changed_signals
    magnitudes = [abs(mag) for _, mag in signals]
    assert magnitudes == sorted(magnitudes, reverse=True)
