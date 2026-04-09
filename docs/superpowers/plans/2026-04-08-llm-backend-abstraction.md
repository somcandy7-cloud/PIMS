# LLM Backend Abstraction & Dashboard Model Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** LLM 백엔드를 교체 가능한 구조로 추상화하고, 대시보드에서 백엔드/모델을 선택할 수 있는 UI를 추가한다.

**Architecture:** `LLMBackend` ABC 하나에 `OllamaBackend`(로컬), `OpenAIBackend`(사내망/GPT API) 두 구현체를 제공한다. `LLMFilter`는 `LLMBackend` 인스턴스를 주입받아 사용한다. 대시보드 사이드바에서 백엔드·모델을 선택하면 `settings.yaml`에 저장되고 다음 분석 실행 시 반영된다.

**Tech Stack:** Python 3.14, `openai` SDK (pip), `requests`, `streamlit`, `pyyaml`

---

## 파일 구조

| 파일 | 역할 |
|------|------|
| **Create** `src/analysis/llm_backends.py` | `LLMBackend` ABC + `OllamaBackend` + `OpenAIBackend` + `build_llm_backend()` factory |
| **Modify** `src/analysis/llm_filter.py` | `LLMFilter(backend)` — 백엔드 인스턴스 주입받도록 시그니처 변경 |
| **Modify** `config/settings.yaml` | llm 섹션 구조 변경 (`backend`, `anthropic.*`, `ollama.*`) |
| **Modify** `dashboard.py` | 사이드바 백엔드/모델 선택 UI, `build_llm_filter()` 업데이트 |
| **Modify** `evaluate_detection.py` | `build_llm_filter()` 팩토리 업데이트 |
| **Create** `tests/test_llm_backends.py` | 각 백엔드 단위 테스트 |
| **Modify** `tests/test_llm_filter.py` | 백엔드 mock 방식으로 업데이트 |

---

## Task 1: LLMBackend ABC + OllamaBackend

**Files:**
- Create: `src/analysis/llm_backends.py`
- Create: `tests/test_llm_backends.py`

- [ ] **Step 1-1: 실패하는 테스트 작성**

```python
# tests/test_llm_backends.py
from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock
from src.analysis.llm_backends import LLMBackend, OllamaBackend


def test_llm_backend_is_abstract():
    """LLMBackend는 직접 인스턴스화할 수 없다."""
    with pytest.raises(TypeError):
        LLMBackend()


def test_ollama_backend_name():
    b = OllamaBackend(model="qwen3:4b", host="http://localhost:11434", timeout=5)
    assert b.name() == "ollama/qwen3:4b"


def test_ollama_backend_generate_success():
    """Ollama API 정상 응답 시 response 텍스트를 반환한다."""
    b = OllamaBackend(model="qwen3:4b", host="http://localhost:11434", timeout=5)
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "KEEP: 이상 확인됨"}
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp) as mock_post:
        result = b.generate("테스트 프롬프트")

    assert result == "KEEP: 이상 확인됨"
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert "qwen3:4b" in str(call_kwargs)


def test_ollama_backend_generate_raises_on_error():
    """Ollama 호출 실패 시 예외를 그대로 전파한다."""
    b = OllamaBackend(model="qwen3:4b", host="http://localhost:11434", timeout=5)
    with patch("requests.post", side_effect=ConnectionError("refused")):
        with pytest.raises(ConnectionError):
            b.generate("프롬프트")
```

- [ ] **Step 1-2: 테스트 실패 확인**

```
pytest tests/test_llm_backends.py -v
```
Expected: `ImportError: cannot import name 'LLMBackend'`

- [ ] **Step 1-3: LLMBackend ABC + OllamaBackend 구현**

```python
# src/analysis/llm_backends.py
"""LLM 백엔드 추상화 — 교체 가능한 LLM 연결 인터페이스."""
from __future__ import annotations
from abc import ABC, abstractmethod

import requests


class LLMBackend(ABC):
    """LLM 호출 추상 인터페이스. 모든 백엔드가 구현해야 한다."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """프롬프트를 보내고 텍스트 응답을 반환한다. 오류 시 예외를 발생시킨다."""

    @abstractmethod
    def name(self) -> str:
        """'backend/model' 형식의 식별자를 반환한다."""


class OllamaBackend(LLMBackend):
    """로컬 Ollama HTTP API 백엔드."""

    def __init__(
        self,
        model: str = "qwen3:4b",
        host: str = "http://localhost:11434",
        timeout: int = 30,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def name(self) -> str:
        return f"ollama/{self.model}"
```

- [ ] **Step 1-4: 테스트 통과 확인**

```
pytest tests/test_llm_backends.py -v
```
Expected: 4 passed

- [ ] **Step 1-5: 커밋**

```bash
git add src/analysis/llm_backends.py tests/test_llm_backends.py
git commit -m "feat: add LLMBackend ABC and OllamaBackend"
```

---

## Task 2: OpenAIBackend

**Files:**
- Modify: `src/analysis/llm_backends.py`
- Modify: `tests/test_llm_backends.py`

**사전 조건:** `openai` 패키지 설치

```bash
pip install openai
```

- [ ] **Step 2-1: 실패하는 테스트 추가**

아래 테스트를 `tests/test_llm_backends.py` 하단에 추가한다.  
`pytest.importorskip("openai")`을 사용해 패키지가 없으면 해당 테스트만 스킵되도록 한다.

```python
openai = pytest.importorskip("openai")  # 패키지 없으면 이 아래 테스트 전체 스킵
from src.analysis.llm_backends import OpenAIBackend


def test_openai_backend_name():
    b = OpenAIBackend(model="gpt-4o", api_key="sk-test")
    assert b.name() == "openai/gpt-4o"


def test_openai_backend_generate_success():
    """OpenAI API 정상 응답 시 텍스트를 반환한다."""
    b = OpenAIBackend(model="gpt-4o", api_key="sk-test")

    mock_message = MagicMock()
    mock_message.content = "KEEP: 복수 신호 동시 급변"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_cls.return_value = mock_client

        result = b.generate("테스트 프롬프트")

    assert result == "KEEP: 복수 신호 동시 급변"


def test_openai_backend_generate_raises_on_error():
    """OpenAI API 호출 실패 시 예외를 그대로 전파한다."""
    b = OpenAIBackend(model="gpt-4o", api_key="sk-test")
    with patch("openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        mock_cls.return_value = mock_client

        with pytest.raises(Exception, match="API error"):
            b.generate("프롬프트")
```

- [ ] **Step 2-2: 테스트 실패 확인**

```
pytest tests/test_llm_backends.py::test_openai_backend_name -v
```
Expected: `ImportError: cannot import name 'OpenAIBackend'`

- [ ] **Step 2-3: OpenAIBackend 구현**

`src/analysis/llm_backends.py`의 `OllamaBackend` 클래스 바로 아래에 추가한다.

```python
class OpenAIBackend(LLMBackend):
    """OpenAI GPT API 백엔드 (사내망/클라우드 사용)."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        timeout: int = 60,
        max_tokens: int = 256,
    ):
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        import openai
        client = openai.OpenAI(api_key=self.api_key, timeout=self.timeout)
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def name(self) -> str:
        return f"openai/{self.model}"
```

`OllamaBackend` 다음 줄에 삽입. `build_llm_backend()` factory는 Task 4에서 추가한다.

- [ ] **Step 2-4: 테스트 통과 확인**

```
pytest tests/test_llm_backends.py -v
```
Expected: 7 passed

- [ ] **Step 2-5: 커밋**

```bash
git add src/analysis/llm_backends.py tests/test_llm_backends.py
git commit -m "feat: add OpenAIBackend (GPT API)"
```

---

## Task 3: LLMFilter 리팩토링

**Files:**
- Modify: `src/analysis/llm_filter.py`
- Modify: `tests/test_llm_filter.py`

**핵심 변경:** `LLMFilter(backend: LLMBackend | None)` — 백엔드를 외부에서 주입받는다.  
`backend=None`이면 disabled (기존 `enabled=False`와 동일).

- [ ] **Step 3-1: 기존 테스트가 새 시그니처로 업데이트되면 실패하는지 확인 (확인용)**

기존 `tests/test_llm_filter.py`를 읽어서 `requests.post` mock 방식임을 확인. 리팩토링 후 이 mock이 깨진다.

- [ ] **Step 3-2: tests/test_llm_filter.py를 새 시그니처로 전면 교체**

아래 내용으로 `tests/test_llm_filter.py`를 완전히 덮어쓴다.

```python
# tests/test_llm_filter.py
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from src.analysis.llm_filter import LLMFilter
from src.analysis.llm_backends import LLMBackend
from src.analysis.base_detector import AnomalyEvent


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
```

- [ ] **Step 3-3: 테스트 실패 확인**

```
pytest tests/test_llm_filter.py -v
```
Expected: 여러 테스트가 `TypeError` 또는 `ImportError`로 실패

- [ ] **Step 3-4: llm_filter.py 리팩토링**

`src/analysis/llm_filter.py` 전체를 아래 내용으로 교체한다.

```python
# src/analysis/llm_filter.py
from __future__ import annotations
import re
from pathlib import Path

import pandas as pd

from src.analysis.base_detector import AnomalyEvent
from src.analysis.llm_backends import LLMBackend

_PROMPT_PATH = Path(__file__).parent.parent.parent / "config" / "llm_filter_prompt.md"
_CONTEXT_SEC = 10


def _load_prompt_template() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    return (
        "이상 후보를 분석하세요.\n"
        "timestamp: {timestamp}, score: {score:.4f}, signals: {top_signals}\n"
        "context: {context_stats}\n"
        "첫 줄에 KEEP: 이유 또는 REJECT: 이유 형식으로 응답하세요."
    )


def _build_context_stats(event: AnomalyEvent, df: pd.DataFrame) -> str:
    ts = event.timestamp
    t_start = ts - pd.Timedelta(seconds=_CONTEXT_SEC)
    t_end = ts + pd.Timedelta(seconds=_CONTEXT_SEC)

    sig_names = [s[0] for s in event.top_signals]
    available = [s for s in sig_names if s in df.columns]
    if not available:
        return "(신호 데이터 없음)"

    try:
        window = df.loc[t_start:t_end, available]
    except Exception:
        window = df[available]

    if window.empty:
        return "(윈도우 데이터 없음)"

    lines = []
    for col in available:
        s = window[col].dropna()
        if len(s) == 0:
            continue
        lines.append(
            f"  {col}: 평균={s.mean():.2f}, 최솟값={s.min():.2f}, "
            f"최댓값={s.max():.2f}, 표준편차={s.std():.2f}"
        )
    return "\n".join(lines) if lines else "(통계 계산 불가)"


def _parse_verdict(response_text: str) -> tuple[str, str]:
    first_line = response_text.strip().splitlines()[0] if response_text.strip() else ""
    m = re.match(r"^(KEEP|REJECT):\s*(.+)", first_line, re.IGNORECASE)
    if m:
        return m.group(1).upper(), m.group(2).strip()
    return "KEEP", f"파싱 불가 응답: {first_line[:80]}"


class LLMFilter:
    """LLM 백엔드를 사용해 이상치 후보 목록에서 유효 이상치만 선별한다.

    backend=None이면 비활성화 — 모든 후보를 통과시킨다 (안전 fallback).
    프롬프트 템플릿: config/llm_filter_prompt.md
    """

    def __init__(self, backend: LLMBackend | None):
        self.backend = backend

    def filter(
        self,
        candidates: list[AnomalyEvent],
        df: pd.DataFrame,
    ) -> list[AnomalyEvent]:
        """후보 목록을 LLM으로 검증하고 KEEP 판정된 이벤트만 반환한다."""
        if not candidates:
            return []
        if self.backend is None:
            return candidates

        template = _load_prompt_template()
        validated = []

        for event in candidates:
            verdict, reason = self._call_backend(event, df, template)
            updated_meta = {
                **event.metadata,
                "llm_verdict": verdict,
                "llm_reason": reason,
                "llm_backend": self.backend.name(),
            }
            updated_event = AnomalyEvent(
                timestamp=event.timestamp,
                score=event.score,
                top_signals=event.top_signals,
                label=event.label,
                metadata=updated_meta,
            )
            if verdict in ("KEEP", "ERROR_KEEP"):
                validated.append(updated_event)

        return validated

    def _call_backend(
        self,
        event: AnomalyEvent,
        df: pd.DataFrame,
        template: str,
    ) -> tuple[str, str]:
        top_signals_text = "\n".join(
            f"  - {name}: 변동성={val:.3f}" for name, val in event.top_signals
        )
        context_stats = _build_context_stats(event, df)

        try:
            prompt = template.format(
                timestamp=str(event.timestamp),
                score=event.score,
                top_signals=top_signals_text,
                context_sec=_CONTEXT_SEC,
                context_stats=context_stats,
            )
        except KeyError:
            prompt = (
                f"이상 후보: timestamp={event.timestamp}, score={event.score:.4f}\n"
                f"신호: {top_signals_text}\n"
                "첫 줄에 KEEP: 이유 또는 REJECT: 이유 형식으로 응답하세요."
            )

        try:
            response_text = self.backend.generate(prompt)
            return _parse_verdict(response_text)
        except Exception as exc:
            return "ERROR_KEEP", f"LLM 오류: {exc}"
```

- [ ] **Step 3-5: 테스트 통과 확인**

```
pytest tests/test_llm_filter.py tests/test_llm_backends.py -v
```
Expected: 15 passed (7 backends + 8 filter)

**주의:** 이 단계에서 `pytest --tb=short -q` 전체를 실행하면 `test_evaluate_detection.py:test_build_llm_filter_reads_settings`가 실패한다. Step 3-6에서 해당 테스트를 업데이트할 것이므로 지금은 무시한다.

- [ ] **Step 3-6: test_evaluate_detection.py의 build_llm_filter 테스트 업데이트**

`tests/test_evaluate_detection.py`의 `test_build_llm_filter_reads_settings` 함수(175~195번 줄)를 아래로 교체한다.

```python
def test_build_llm_filter_reads_settings():
    """build_llm_filter가 settings dict를 읽어 LLMFilter(backend=...)를 반환한다."""
    import evaluate_detection as ed
    from src.analysis.llm_filter import LLMFilter
    from src.analysis.llm_backends import OllamaBackend

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
```

- [ ] **Step 3-7: 전체 테스트 통과 확인**

```
pytest --tb=short -q
```
Expected: 전체 통과 (기존 테스트 깨진 것 없어야 함)

- [ ] **Step 3-8: 커밋**

```bash
git add src/analysis/llm_filter.py tests/test_llm_filter.py tests/test_evaluate_detection.py
git commit -m "refactor: LLMFilter accepts LLMBackend injection, removes direct Ollama coupling"
```

---

## Task 4: Config 구조 변경 + build_llm_backend() factory + evaluate_detection.py 업데이트

**Files:**
- Modify: `config/settings.yaml`
- Modify: `src/analysis/llm_backends.py`
- Modify: `evaluate_detection.py`
- Modify: `tests/test_llm_backends.py`

**Config 새 구조:**
```yaml
llm:
  backend: "openai"   # "openai" | "ollama" | "disabled"
  openai:
    model: "gpt-4o"
    api_key_env: "OPENAI_API_KEY"   # 환경변수 이름
    timeout: 60
    max_tokens: 256
  ollama:
    model: "qwen3:4b"
    host: "http://localhost:11434"
    timeout: 30
```

- [ ] **Step 4-1: factory 테스트 추가**

`tests/test_llm_backends.py` **파일 상단** (기존 `import pytest` 바로 다음)에 `import os`를 추가한다. 이미 있다면 생략.

그다음 파일 하단에 아래 테스트를 추가한다.

```python
from src.analysis.llm_backends import build_llm_backend


def test_build_llm_backend_disabled():
    cfg = {"backend": "disabled"}
    assert build_llm_backend(cfg) is None


def test_build_llm_backend_ollama():
    cfg = {
        "backend": "ollama",
        "ollama": {"model": "qwen3:4b", "host": "http://localhost:11434", "timeout": 10},
    }
    b = build_llm_backend(cfg)
    assert isinstance(b, OllamaBackend)
    assert b.name() == "ollama/qwen3:4b"


def test_build_llm_backend_openai():
    cfg = {
        "backend": "openai",
        "openai": {"model": "gpt-4o", "api_key_env": "OPENAI_API_KEY", "timeout": 60},
    }
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
        b = build_llm_backend(cfg)
    assert isinstance(b, OpenAIBackend)
    assert b.name() == "openai/gpt-4o"
    assert b.api_key == "sk-test-key"


def test_build_llm_backend_unknown_falls_back_to_none():
    cfg = {"backend": "unknown_backend"}
    assert build_llm_backend(cfg) is None
```

- [ ] **Step 4-2: 테스트 실패 확인**

```
pytest tests/test_llm_backends.py::test_build_llm_backend_disabled -v
```
Expected: `ImportError: cannot import name 'build_llm_backend'`

- [ ] **Step 4-3: build_llm_backend() factory 구현**

`src/analysis/llm_backends.py` 파일 **상단** (`from abc import ABC, abstractmethod` 바로 다음 줄)에 `import os`를 추가한다.

그다음 파일 맨 끝(OpenAIBackend 클래스 이후)에 factory 함수를 추가한다.

```python


def build_llm_backend(llm_cfg: dict) -> "LLMBackend | None":
    """settings.yaml llm 섹션 dict로 적합한 LLMBackend를 생성한다.

    반환값이 None이면 LLMFilter가 비활성화된다.
    """
    backend_type = (llm_cfg.get("backend") or "disabled").lower()

    if backend_type == "ollama":
        cfg = llm_cfg.get("ollama") or {}
        return OllamaBackend(
            model=cfg.get("model", "qwen3:4b"),
            host=cfg.get("host", "http://localhost:11434"),
            timeout=int(cfg.get("timeout", 30)),
        )

    if backend_type == "openai":
        cfg = llm_cfg.get("openai") or {}
        api_key_env = cfg.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(api_key_env)
        return OpenAIBackend(
            model=cfg.get("model", "gpt-4o"),
            api_key=api_key,
            timeout=int(cfg.get("timeout", 60)),
            max_tokens=int(cfg.get("max_tokens", 256)),
        )

    return None  # "disabled" 및 알 수 없는 값
```

주의: `import os`는 파일 상단 `from abc import ABC, abstractmethod` 바로 아래에 추가한다.

- [ ] **Step 4-4: config/settings.yaml llm 섹션 교체**

`config/settings.yaml`의 `llm:` 블록을 아래로 교체한다.

```yaml
llm:
  backend: "openai"           # "openai" | "ollama" | "disabled"
  openai:
    model: "gpt-4o"
    api_key_env: "OPENAI_API_KEY"
    timeout: 60
    max_tokens: 256
  ollama:
    model: "qwen3:4b"
    host: "http://localhost:11434"
    timeout: 30
```

- [ ] **Step 4-5: evaluate_detection.py build_llm_filter() 업데이트**

`evaluate_detection.py`의 import 섹션에 추가:
```python
from src.analysis.llm_backends import build_llm_backend
```

기존 `build_llm_filter()` 함수(79~87번 줄)를 아래로 교체한다:
```python
def build_llm_filter(settings: dict) -> "LLMFilter":
    """settings.yaml llm 섹션으로 LLMFilter를 생성한다."""
    llm_cfg = settings.get("llm", {})
    backend = build_llm_backend(llm_cfg)
    return LLMFilter(backend=backend)
```

- [ ] **Step 4-6: 테스트 통과 확인**

```
pytest tests/test_llm_backends.py -v
```
Expected: 11 passed

```
pytest --tb=short -q
```
Expected: 전체 통과

- [ ] **Step 4-7: 커밋**

```bash
git add src/analysis/llm_backends.py config/settings.yaml evaluate_detection.py tests/test_llm_backends.py
git commit -m "feat: add build_llm_backend factory, update settings.yaml and evaluate_detection.py"
```

---

## Task 5: 대시보드 백엔드/모델 선택 UI

**Files:**
- Modify: `dashboard.py`

**UI 설계:**
- 사이드바에 "LLM 설정" 섹션 추가 (기존 "탐지 설정" 아래)
- 백엔드 선택: `st.radio` — `["OpenAI (GPT)", "Ollama (로컬)", "비활성화"]`
- OpenAI 선택 시: 모델 `st.selectbox` (gpt-4o, gpt-4o-mini, o3-mini)
- Ollama 선택 시: 모델명 `st.text_input`, 호스트 `st.text_input`
- "설정 저장" 버튼 → `settings.yaml` 업데이트
- 연결 상태 표시 (현재 백엔드 이름)

- [ ] **Step 5-1: dashboard.py import 섹션 업데이트**

`dashboard.py` 27번 줄 (`from src.data.signal_label_mapper import SignalLabelMapper`) 다음 줄에 추가:

```python
from src.analysis.llm_backends import build_llm_backend
```

**주의:** 기존 24번 줄의 `from src.analysis.llm_filter import LLMFilter` import는 **반드시 유지**한다. 삭제하지 말 것.

- [ ] **Step 5-2: dashboard.py build_llm_filter() 함수 교체**

기존 `build_llm_filter()` 함수(96~104번 줄)를 아래로 교체한다:

```python
def build_llm_filter(settings: dict) -> LLMFilter:
    """settings.yaml llm 섹션으로 LLMFilter를 생성한다."""
    llm_cfg = settings.get("llm", {})
    backend = build_llm_backend(llm_cfg)
    return LLMFilter(backend=backend)
```

- [ ] **Step 5-3: save_llm_config() 헬퍼 추가**

`save_signal_config()` 함수(68~75번 줄) 바로 다음에 추가한다:

```python
def save_llm_config(llm_cfg: dict) -> None:
    """settings.yaml의 llm 섹션을 업데이트한다."""
    with open(SETTINGS_YAML, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["llm"] = llm_cfg
    with open(SETTINGS_YAML, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
```

- [ ] **Step 5-4: 사이드바 LLM 설정 UI 추가**

사이드바의 `run_btn = st.button(...)` 줄(124번 줄) **위에**, `st.divider()` 다음 위치에 아래 블록을 삽입한다.
(정확히: `top_n = st.slider(...)` 줄 바로 아래 `st.divider()` 이후 위치)

```python
    st.divider()
    st.subheader("LLM 설정")

    # 현재 settings.yaml에서 llm 섹션 로드
    with open(SETTINGS_YAML, encoding="utf-8") as _f:
        _settings_now = yaml.safe_load(_f)
    _llm_now = _settings_now.get("llm", {})

    _BACKEND_OPTIONS = ["OpenAI (GPT)", "Ollama (로컬)", "비활성화"]
    _backend_map = {"openai": 0, "ollama": 1, "disabled": 2}
    _current_backend_key = (_llm_now.get("backend") or "disabled").lower()
    _current_idx = _backend_map.get(_current_backend_key, 2)

    _selected_backend = st.radio(
        "백엔드", _BACKEND_OPTIONS, index=_current_idx, key="llm_backend_radio"
    )

    _new_llm_cfg = dict(_llm_now)  # 기존 설정 복사 (openai/ollama 섹션 보존)

    if _selected_backend == "OpenAI (GPT)":
        _new_llm_cfg["backend"] = "openai"
        _oai = _llm_now.get("openai") or {}
        _oai_models = ["gpt-4o", "gpt-4o-mini", "o3-mini"]
        _cur_model = _oai.get("model", "gpt-4o")
        _sel_model_idx = _oai_models.index(_cur_model) if _cur_model in _oai_models else 0
        _sel_model = st.selectbox("모델", _oai_models, index=_sel_model_idx, key="oai_model")
        _new_llm_cfg["openai"] = {**_oai, "model": _sel_model}
        st.caption("API 키: `OPENAI_API_KEY` 환경변수")

    elif _selected_backend == "Ollama (로컬)":
        _new_llm_cfg["backend"] = "ollama"
        _olla = _llm_now.get("ollama") or {}
        _olla_model = st.text_input("모델명", value=_olla.get("model", "qwen3:4b"), key="olla_model")
        _olla_host  = st.text_input("호스트", value=_olla.get("host", "http://localhost:11434"), key="olla_host")
        _new_llm_cfg["ollama"] = {**_olla, "model": _olla_model, "host": _olla_host}

    else:
        _new_llm_cfg["backend"] = "disabled"
        st.caption("LLM 필터 비활성화 — IF 후보 전체를 알람으로 처리")

    st.caption("백엔드 변경 후 반드시 '저장' 버튼을 누른 뒤 '분석 실행'을 클릭하세요.")

    if st.button("LLM 설정 저장", key="save_llm_btn"):
        save_llm_config(_new_llm_cfg)
        detect_anomalies.clear()
        st.success("저장됨")
        st.rerun()

    # 현재 적용 중인 백엔드 표시 (settings.yaml 기준 — 저장된 값)
    _active_backend = _llm_now.get("backend", "disabled")
    if _active_backend == "openai":
        _active_model = (_llm_now.get("openai") or {}).get("model", "?")
        st.info(f"현재: OpenAI / {_active_model}")
    elif _active_backend == "ollama":
        _active_model = (_llm_now.get("ollama") or {}).get("model", "?")
        st.info(f"현재: Ollama / {_active_model}")
    else:
        st.warning("현재: LLM 비활성화")
```

- [ ] **Step 5-5: 대시보드 수동 확인**

```bash
streamlit run dashboard.py
```

확인 사항:
1. 사이드바에 "LLM 설정" 섹션이 표시됨
2. 백엔드 라디오 버튼으로 OpenAI/Ollama/비활성화 전환 가능
3. OpenAI 선택 시 모델 selectbox (gpt-4o, gpt-4o-mini, o3-mini), Ollama 선택 시 텍스트 입력창 표시
4. "LLM 설정 저장" 버튼 클릭 시 `config/settings.yaml`이 업데이트됨
5. "분석 실행" 클릭 시 새 백엔드 설정으로 LLMFilter가 동작함

- [ ] **Step 5-6: 커밋**

```bash
git add dashboard.py
git commit -m "feat: add LLM backend/model selector UI in dashboard sidebar"
```

---

## 최종 검증

- [ ] **전체 테스트 통과**

```
pytest --tb=short -q
```
Expected: 전체 통과 (기존 81개 + 신규 ~11개 = 약 92개)

- [ ] **OpenAI 백엔드 실제 동작 확인**

`OPENAI_API_KEY` 환경변수 설정 후 대시보드에서 분석 실행.  
이벤트 카드의 `llm_backend` 메타데이터가 `openai/gpt-4o`로 표시되면 성공.

---

## 설계 메모

| 항목 | 결정 |
|------|------|
| `LLMFilter.enabled` 파라미터 | 제거 — `backend=None`으로 대체 |
| 기존 Ollama config (`enabled`, `host`, `model`, `filter_timeout`) | settings.yaml에서 새 구조로 교체 |
| API 키 관리 | 환경변수(`OPENAI_API_KEY`)만 사용 — yaml에 키 저장 안 함 |
| 모델 교체 | 대시보드에서 selectbox 선택 → 저장 → 다음 실행에 반영 |
| 백엔드 추가 방법 | `LLMBackend` 상속 + `build_llm_backend()` factory에 분기 추가 |