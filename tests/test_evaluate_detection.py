# tests/test_evaluate_detection.py
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest


def _write_eval_csv(path: Path, has_trip: bool = True):
    """테스트용 평가 CSV를 생성한다 (표준 CSV, iba 포맷 아님)."""
    n = 500
    rng = np.random.default_rng(42)
    t = pd.date_range("2026-03-20 15:49:00", periods=n, freq="10ms", tz="UTC")
    if has_trip:
        speed = np.concatenate([
            np.full(300, 50.0) + rng.normal(0, 0.3, 300),
            np.linspace(50, 0, 10),
            np.zeros(190) + rng.normal(0, 0.1, 190),
        ])
        current = np.concatenate([
            np.full(300, 100.0) + rng.normal(0, 0.5, 300),
            np.linspace(100, 0, 10),
            np.zeros(190) + rng.normal(0, 0.2, 190),
        ])
    else:
        speed = np.full(n, 50.0) + rng.normal(0, 0.3, n)
        current = np.full(n, 100.0) + rng.normal(0, 0.5, n)

    df = pd.DataFrame({"speed": speed, "current": current}, index=t)
    df.index.name = "timestamp"
    df.to_csv(path)


class SimpleCsvLoader:
    def load(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, format="ISO8601", utc=True)
        return df


def test_evaluate_returns_expected_keys(tmp_path, monkeypatch):
    """evaluate()가 올바른 키를 포함한 dict를 반환한다."""
    eval_dir = tmp_path / "eval_data"
    eval_dir.mkdir()
    csv_path = eval_dir / "trip_sample.csv"
    _write_eval_csv(csv_path, has_trip=True)

    import evaluate_detection as ed
    monkeypatch.setattr(ed, "EVAL_CSV_DIR", eval_dir)
    monkeypatch.setattr(ed, "KNOWN_EVENTS", {
        "trip_sample.csv": ["2026-03-20 15:49:03.000000+00:00"],
    })

    result = ed.evaluate(loader=SimpleCsvLoader())
    for key in ("f1", "hit_rate", "fp_rate", "alarm_count", "hit_count", "miss_count", "false_positive"):
        assert key in result, f"Missing key: {key}"


def test_empty_known_events_returns_zeros(tmp_path, monkeypatch):
    """KNOWN_EVENTS가 비어있으면 f1=0, alarm_count>=0를 반환한다."""
    eval_dir = tmp_path / "eval_data"
    eval_dir.mkdir()

    import evaluate_detection as ed
    monkeypatch.setattr(ed, "EVAL_CSV_DIR", eval_dir)
    monkeypatch.setattr(ed, "KNOWN_EVENTS", {})

    result = ed.evaluate()
    assert result["f1"] == 0.0
    assert result["alarm_count"] == 0


def test_validate_signals_raises_on_out_of_range():
    """signal_overrides의 임계값이 범위 밖이면 ValueError를 발생시킨다."""
    import evaluate_detection as ed
    signals = {"signal_overrides": {"DB420.DBW 2": 10.0}}
    with pytest.raises(ValueError, match="허용 범위"):
        ed._validate_signals(signals)


def test_validate_signals_passes_on_valid_range():
    """signal_overrides의 임계값이 범위 내면 예외가 발생하지 않는다."""
    import evaluate_detection as ed
    signals = {"signal_overrides": {"DB420.DBW 2": 3.0}}
    ed._validate_signals(signals)  # Should not raise


# ── 새로운 테스트 ─────────────────────────────────────────────────────────────

def test_evaluate_returns_candidate_count(tmp_path, monkeypatch):
    """evaluate()의 반환 dict에 candidate_count 키가 포함된다."""
    eval_dir = tmp_path / "eval_data"
    eval_dir.mkdir()

    import evaluate_detection as ed
    monkeypatch.setattr(ed, "EVAL_CSV_DIR", eval_dir)
    monkeypatch.setattr(ed, "KNOWN_EVENTS", {})

    result = ed.evaluate()
    assert "candidate_count" in result, "candidate_count 키가 반환 dict에 없습니다."
    assert isinstance(result["candidate_count"], int)


def test_evaluate_use_llm_false_does_not_call_llm_filter(tmp_path, monkeypatch):
    """use_llm=False(기본값)이면 LLMFilter.filter가 호출되지 않는다."""
    eval_dir = tmp_path / "eval_data"
    eval_dir.mkdir()

    import evaluate_detection as ed
    monkeypatch.setattr(ed, "EVAL_CSV_DIR", eval_dir)
    monkeypatch.setattr(ed, "KNOWN_EVENTS", {})

    mock_llm = MagicMock()
    with patch.object(ed, "build_llm_filter", return_value=mock_llm) as mock_build:
        result = ed.evaluate(use_llm=False)
        # build_llm_filter가 호출되지 않아야 한다
        mock_build.assert_not_called()
        mock_llm.filter.assert_not_called()

    assert result["alarm_count"] == 0


def test_validate_signals_contamination_out_of_range():
    """model_params.isolation_forest.contamination이 0.5 초과면 ValueError."""
    import evaluate_detection as ed
    signals = {
        "model_params": {
            "isolation_forest": {"contamination": 0.99, "window_size": 5}
        }
    }
    with pytest.raises(ValueError, match="contamination"):
        ed._validate_signals(signals)


def test_validate_signals_window_size_out_of_range():
    """model_params.isolation_forest.window_size가 50 초과면 ValueError."""
    import evaluate_detection as ed
    signals = {
        "model_params": {
            "isolation_forest": {"contamination": 0.03, "window_size": 100}
        }
    }
    with pytest.raises(ValueError, match="window_size"):
        ed._validate_signals(signals)


def test_build_detector_reads_model_params():
    """build_detector가 signals dict의 isolation_forest 파라미터를 올바르게 읽는다."""
    import evaluate_detection as ed
    from src.agents.isolation_forest_adapter import IsolationForestAdapter

    signals = {
        "model_params": {
            "isolation_forest": {
                "window_size": 10,
                "contamination": 0.05,
                "n_estimators": 200,
            }
        },
        "excluded_signals": ["noise_signal"],
        "signal_overrides": {},
    }
    settings = {"analysis": {"top_n_signals": 3}}

    detector = ed.build_detector(signals, settings)

    assert isinstance(detector, IsolationForestAdapter)
    assert detector.window_size == 10
    assert detector.contamination == 0.05
    assert detector.n_estimators == 200
    assert "noise_signal" in detector.excluded_signals
    assert detector.top_n == 3


def test_build_llm_filter_reads_settings():
    """build_llm_filter가 settings dict의 llm 설정을 올바르게 읽는다."""
    import evaluate_detection as ed
    from src.agents.llm_filter import LLMFilter
    from src.agents.llm_backends import OllamaBackend

    settings = {
        "llm": {
            "backend": "ollama",
            "ollama": {
                "model": "llama3:8b",
                "host": "http://myhost:11434",
                "timeout": 60,
            },
        }
    }

    llm_filter = ed.build_llm_filter(settings)

    assert isinstance(llm_filter, LLMFilter)
    assert isinstance(llm_filter.backend, OllamaBackend)
    assert llm_filter.backend.model == "llama3:8b"
    assert "myhost" in llm_filter.backend.host
    assert llm_filter.backend.timeout == 60
