# tests/test_base_detector.py
import pandas as pd
import numpy as np
import pytest
from src.agents.base_detector import AnomalyDetector, AnomalyEvent


def _make_df() -> pd.DataFrame:
    t = pd.date_range("2026-01-01", periods=100, freq="10ms")
    return pd.DataFrame({"val": np.random.randn(100)}, index=t)


def test_anomaly_event_fields():
    evt = AnomalyEvent(
        timestamp=pd.Timestamp("2026-01-01"),
        score=3.5,
        top_signals=[("val", 3.5)],
        label="anomaly",
    )
    assert evt.score == 3.5
    assert evt.label == "anomaly"


def test_detector_is_abstract():
    with pytest.raises(TypeError):
        AnomalyDetector()  # 직접 인스턴스화 불가


class ConcreteDetector(AnomalyDetector):
    def detect(self, df: pd.DataFrame) -> list[AnomalyEvent]:
        return []

    def name(self) -> str:
        return "concrete"


def test_concrete_detector_works():
    d = ConcreteDetector()
    df = _make_df()
    result = d.detect(df)
    assert isinstance(result, list)
