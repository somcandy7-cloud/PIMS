# tests/test_llm_filter.py
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock
from src.agents.llm_filter import LLMFilter
from src.agents.llm_backends import LLMBackend
from src.agents.base_detector import AnomalyEvent


def _make_event(ts: str = "2026-03-20 15:49:03+00:00", score: float = -0.2) -> AnomalyEvent:
    return AnomalyEvent(
        timestamp=pd.Timestamp(ts),
        score=score,
        top_signals=[("speed", 5.0), ("current", 3.0)],
        label="if_candidate",
    )


def _make_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    t = pd.date_range("2026-03-20 15:49:00", periods=n, freq="10ms", tz="UTC")
    return pd.DataFrame({
        "speed": np.full(n, 50.0) + rng.normal(0, 0.5, n),
        "current": np.full(n, 100.0) + rng.normal(0, 1, n),
    }, index=t)


def _make_backend(response: str = "KEEP: 이상 확인됨") -> LLMBackend:
    """generate()가 고정 응답을 반환하는 mock 백엔드."""
    backend = MagicMock(spec=LLMBackend)
    backend.generate.return_value = response
    backend.name.return_value = "mock/test"
    return backend


def test_disabled_filter_passes_all():
    """backend=None이면 모든 후보를 그대로 반환한다."""
    f = LLMFilter(backend=None)
    events = [_make_event(), _make_event("2026-03-20 15:49:10+00:00")]
    result = f.filter(events, _make_df())
    assert len(result) == 2


def test_empty_candidates_returns_empty():
    f = LLMFilter(backend=None)
    result = f.filter([], _make_df())
    assert result == []


def test_keep_response_passes_event():
    """KEEP 응답이 오면 이벤트가 결과에 포함된다."""
    backend = _make_backend("KEEP: 복수 신호 동시 급변 감지됨")
    f = LLMFilter(backend=backend)
    result = f.filter([_make_event()], _make_df())
    assert len(result) == 1
    assert result[0].metadata.get("llm_verdict") == "KEEP"


def test_reject_response_filters_event():
    """REJECT 응답이 오면 이벤트가 결과에서 제외된다."""
    backend = _make_backend("REJECT: 단일 신호 노이즈로 판단")
    f = LLMFilter(backend=backend)
    result = f.filter([_make_event()], _make_df())
    assert len(result) == 0


def test_llm_error_falls_back_to_keep():
    """백엔드 호출 오류 시 해당 이벤트를 통과시킨다 (안전 fallback)."""
    backend = MagicMock(spec=LLMBackend)
    backend.generate.side_effect = Exception("Connection refused")
    backend.name.return_value = "mock/test"
    f = LLMFilter(backend=backend)
    result = f.filter([_make_event()], _make_df())
    assert len(result) == 1
    assert result[0].metadata.get("llm_verdict") == "ERROR_KEEP"


def test_reasoning_stored_in_metadata():
    """LLM 판단 이유가 metadata['llm_reason']에 저장된다."""
    backend = _make_backend("KEEP: 속도 신호 0으로 급감")
    f = LLMFilter(backend=backend)
    result = f.filter([_make_event()], _make_df())
    assert "속도 신호 0으로 급감" in result[0].metadata.get("llm_reason", "")


def test_backend_name_stored_in_metadata():
    """사용된 백엔드 이름이 metadata['llm_backend']에 저장된다."""
    backend = _make_backend("KEEP: 이상 확인")
    f = LLMFilter(backend=backend)
    result = f.filter([_make_event()], _make_df())
    assert result[0].metadata.get("llm_backend") == "mock/test"


def test_unparseable_response_is_error_keep():
    """KEEP:/REJECT: 형식이 아닌 응답은 ERROR_KEEP으로 처리되어 이벤트가 통과된다."""
    backend = _make_backend("이상치입니다. 확인이 필요합니다.")
    f = LLMFilter(backend=backend)
    result = f.filter([_make_event()], _make_df())
    assert len(result) == 1
    assert result[0].metadata.get("llm_verdict") == "ERROR_KEEP"
    assert "파싱 불가" in result[0].metadata.get("llm_reason", "")


def test_disabled_filter_does_not_add_llm_metadata():
    """backend=None이면 llm_verdict 등 LLM 메타데이터가 추가되지 않는다."""
    f = LLMFilter(backend=None)
    event = _make_event()
    result = f.filter([event], _make_df())
    assert len(result) == 1
    assert "llm_verdict" not in result[0].metadata
    assert "llm_backend" not in result[0].metadata
