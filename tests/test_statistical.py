import pandas as pd
import numpy as np
import pytest
from src.agents.statistical import ZScoreDetector, AnomalyResult


@pytest.fixture
def normal_df():
    n = 500
    rng = np.random.default_rng(42)
    t = pd.date_range("2026-03-20", periods=n, freq="10ms", tz="UTC")
    return pd.DataFrame(
        {
            "MD 100": rng.normal(50, 5, n),
            "DB420.DBW 2": rng.normal(120, 3, n),
        },
        index=t,
    )


@pytest.fixture
def spike_df():
    n = 500
    rng = np.random.default_rng(7)
    t = pd.date_range("2026-03-20", periods=n, freq="10ms", tz="UTC")
    data = rng.normal(50, 2, n)
    data[250] = 200.0   # 명확한 스파이크
    data[251] = 195.0
    return pd.DataFrame({"MD 100": data}, index=t)


def test_no_anomaly_in_normal_signal(normal_df):
    # 정규분포 500샘플에서 3σ 자연 이탈이 생길 수 있으므로 5σ로 확인
    detector = ZScoreDetector(threshold=5.0)
    results = detector.detect(normal_df)
    assert len(results) == 0


def test_spike_detected(spike_df):
    detector = ZScoreDetector(threshold=3.0)
    results = detector.detect(spike_df)
    assert len(results) >= 1
    cols = [r.column for r in results]
    assert "MD 100" in cols


def test_anomaly_result_fields(spike_df):
    detector = ZScoreDetector(threshold=3.0)
    results = detector.detect(spike_df)
    r = results[0]
    assert isinstance(r, AnomalyResult)
    assert r.zscore > 3.0
    assert isinstance(r.timestamp, pd.Timestamp)
    assert r.value == pytest.approx(200.0, abs=1.0)


def test_higher_threshold_fewer_results(spike_df):
    low = ZScoreDetector(threshold=2.0).detect(spike_df)
    high = ZScoreDetector(threshold=5.0).detect(spike_df)
    assert len(low) >= len(high)
