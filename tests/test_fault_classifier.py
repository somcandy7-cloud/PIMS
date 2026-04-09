import pandas as pd
import pytest
from src.agents.trip_detector import TripEvent
from src.agents.fault_classifier import FaultClassifier, FaultReport


def _make_event(trigger, signals):
    return TripEvent(
        timestamp=pd.Timestamp("2026-03-20 15:49:03", tz="UTC"),
        trigger=trigger,
        top_changed_signals=signals,
        description="test",
        window_size=50,
    )


def test_multi_signal_drop_classified():
    event = _make_event("multi_signal_drop", [("speed", 5.0), ("current", 4.0)])
    report = FaultClassifier().classify(event)
    assert isinstance(report, FaultReport)
    assert report.fault_type != ""
    assert report.severity in ("높음", "중간", "낮음")


def test_single_signal_drop_classified():
    event = _make_event("single_signal_drop", [("speed", 3.0)])
    report = FaultClassifier().classify(event)
    assert report.fault_type != ""


def test_spike_classified():
    event = _make_event("spike", [("current", 8.0)])
    report = FaultClassifier().classify(event)
    assert report.fault_type != ""


def test_report_has_inspection_items():
    event = _make_event("multi_signal_drop", [("speed", 5.0), ("current", 4.0)])
    report = FaultClassifier().classify(event)
    assert isinstance(report.inspection_items, list)
    assert len(report.inspection_items) >= 1


def test_multi_drop_is_high_severity():
    event = _make_event("multi_signal_drop", [("s1", 10.0), ("s2", 9.0)])
    report = FaultClassifier().classify(event)
    assert report.severity == "높음"


def test_report_korean_text():
    event = _make_event("multi_signal_drop", [("speed", 5.0)])
    report = FaultClassifier().classify(event)
    # 점검항목이 한국어로 작성되어야 함
    assert any("\uAC00" <= c <= "\uD7A3" for item in report.inspection_items for c in item)
