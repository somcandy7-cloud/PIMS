# tests/test_feedback_store.py
import pandas as pd
import pytest
from src.agents.base_detector import AnomalyEvent
from src.services.feedback_store import FeedbackStore


def _make_event(idx: int = 0) -> AnomalyEvent:
    return AnomalyEvent(
        timestamp=pd.Timestamp(f"2026-03-12 06:5{idx}:00+00:00"),
        score=-0.3 - idx * 0.05,
        top_signals=[("sig_a", 0.5)],
        label="if_candidate",
    )


@pytest.fixture
def store(tmp_path):
    return FeedbackStore(db_path=str(tmp_path / "feedback.db"))


def test_save_and_count(store):
    store.save(_make_event(0), label="O", reason="전류 급등")
    store.save(_make_event(1), label="X", reason="정상 기동")
    assert len(store.load_labeled()) == 2


def test_label_preserved(store):
    store.save(_make_event(0), label="O", reason="이상 확정")
    df = store.load_labeled()
    assert df.iloc[0]["admin_label"] == "O"
    assert df.iloc[0]["reason"] == "이상 확정"


def test_invalid_label_raises(store):
    with pytest.raises(ValueError):
        store.save(_make_event(0), label="Y", reason="잘못된 라벨")


def test_retrain_scaffold_structure(store):
    store.save(_make_event(0), label="O", reason="이상")
    store.save(_make_event(1), label="X", reason="정상")
    result = store.retrain_scaffold()
    assert {"normal_count", "anomaly_count", "ready", "message"}.issubset(result)
