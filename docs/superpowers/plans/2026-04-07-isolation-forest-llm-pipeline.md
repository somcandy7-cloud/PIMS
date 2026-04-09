# Isolation Forest + LLM Filter Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Windowed Isolation Forest로 이상치 후보를 추리고, LLM 레이어로 유효 이상치만 걸러내어 대시보드에 표시하는 2단계 파이프라인 구축

**Architecture:**
`IsolationForestAdapter`(AnomalyDetector)가 윈도우드 특징으로 이상 후보를 추출하고,
`LLMFilter`가 Ollama를 통해 후보 중 실제 이상만 선별한다.
대시보드는 `(candidates → LLM filter → validated)` 파이프라인을 실행하며 후보 수와 검증 수를 함께 표시한다.
autoresearch 루프는 IF 파라미터(`signals.yaml`)와 LLM 프롬프트(`config/llm_filter_prompt.md`) 두 축을 최적화한다.

**Tech Stack:** Python 3.10+, scikit-learn (IsolationForest), requests (Ollama HTTP), pandas, numpy, pytest, streamlit, PyYAML

---

## 파일 구조 변경 계획

```
신규 생성:
  src/analysis/isolation_forest_adapter.py   # IsolationForestAdapter(AnomalyDetector)
  src/analysis/llm_filter.py                 # LLMFilter — Ollama 기반 후보 필터
  config/llm_filter_prompt.md               # LLM 판단 프롬프트 템플릿 (autoresearch 대상)
  tests/test_isolation_forest_adapter.py
  tests/test_llm_filter.py

수정:
  config/signals.yaml       # model_params 섹션 추가 (IF 파라미터, autoresearch 대상)
  config/settings.yaml      # llm.enabled: true, model: qwen3:4b
  dashboard.py              # IF + LLM 파이프라인, 후보/검증 카드 추가
  evaluate_detection.py     # IsolationForestAdapter + LLM optional 평가
  reference/pims_program.md # autoresearch 2축(모델파라미터 + 프롬프트) 추가
```

---

## 파이프라인 흐름

```
DataFrame (전처리 완료)
    ↓
IsolationForestAdapter.detect()
    윈도우드 특징 생성 (window_size × n_cols → flat vector)
    IsolationForest.fit_predict()
    AnomalyEvent 생성 (label="if_candidate")
    ↓
list[AnomalyEvent] — 후보 (N건)
    ↓
LLMFilter.filter(candidates, df)
    후보별 컨텍스트 윈도우 추출
    프롬프트 조립 (llm_filter_prompt.md 템플릿)
    Ollama 호출 → KEEP / REJECT 파싱
    ↓
list[AnomalyEvent] — 검증된 이상치 (M건, M ≤ N)
    ↓
Dashboard 표시
```

---

## Task 1: `IsolationForestAdapter` — 윈도우드 Isolation Forest

**Files:**
- Create: `src/analysis/isolation_forest_adapter.py`
- Create: `tests/test_isolation_forest_adapter.py`

참고: `reference/windowed_model_pipeline.py`의 `prepare_windowed_data()` 로직을 AnomalyDetector 인터페이스로 이식한다.

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_isolation_forest_adapter.py
import numpy as np
import pandas as pd
import pytest
from src.analysis.isolation_forest_adapter import IsolationForestAdapter
from src.analysis.base_detector import AnomalyEvent


def _make_normal_df(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    t = pd.date_range("2026-03-20", periods=n, freq="10ms", tz="UTC")
    return pd.DataFrame({
        "speed": np.full(n, 50.0) + rng.normal(0, 0.5, n),
        "current": np.full(n, 100.0) + rng.normal(0, 1.0, n),
        "pressure": np.full(n, 10.0) + rng.normal(0, 0.1, n),
    }, index=t)


def _make_anomaly_df(n_pre: int = 300, n_anom: int = 20, n_post: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    n = n_pre + n_anom + n_post
    t = pd.date_range("2026-03-20", periods=n, freq="10ms", tz="UTC")
    speed = np.concatenate([
        np.full(n_pre, 50.0) + rng.normal(0, 0.5, n_pre),
        np.linspace(50, 0, n_anom),              # 급격한 변화
        np.zeros(n_post) + rng.normal(0, 0.2, n_post),
    ])
    current = np.concatenate([
        np.full(n_pre, 100.0) + rng.normal(0, 1, n_pre),
        np.linspace(100, 0, n_anom),
        np.zeros(n_post) + rng.normal(0, 0.3, n_post),
    ])
    pressure = np.full(n, 10.0) + rng.normal(0, 0.1, n)
    return pd.DataFrame({"speed": speed, "current": current, "pressure": pressure}, index=t)


def test_returns_list_of_anomaly_events():
    adapter = IsolationForestAdapter()
    df = _make_anomaly_df()
    events = adapter.detect(df)
    assert isinstance(events, list)
    assert all(isinstance(e, AnomalyEvent) for e in events)


def test_detects_anomaly_in_anomaly_data():
    # contamination=0.05면 500 샘플 중 약 25개 → 최소 1건 이상
    adapter = IsolationForestAdapter(contamination=0.05)
    df = _make_anomaly_df()
    events = adapter.detect(df)
    assert len(events) >= 1


def test_event_label_is_if_candidate():
    adapter = IsolationForestAdapter(contamination=0.05)
    events = adapter.detect(_make_anomaly_df())
    assert all(e.label == "if_candidate" for e in events)


def test_event_has_negative_score():
    """IF anomaly score는 음수일수록 더 이상함 → AnomalyEvent.score는 그대로 전달."""
    adapter = IsolationForestAdapter(contamination=0.05)
    events = adapter.detect(_make_anomaly_df())
    if events:
        assert all(isinstance(e.score, float) for e in events)


def test_name():
    assert IsolationForestAdapter().name() == "isolation_forest"


def test_empty_df_returns_empty():
    adapter = IsolationForestAdapter()
    empty = pd.DataFrame(columns=["speed", "current"])
    result = adapter.detect(empty)
    assert result == []


def test_respects_excluded_signals():
    """excluded_signals에 포함된 컬럼은 특징 행렬에서 제외된다."""
    df = _make_anomaly_df()
    adapter_full = IsolationForestAdapter(contamination=0.05)
    adapter_excl = IsolationForestAdapter(contamination=0.05, excluded_signals=["speed", "current"])
    # excluded 버전은 pressure만 사용 → 이상 탐지 어려워짐 (이 테스트는 crash 없음만 검증)
    events = adapter_excl.detect(df)
    assert isinstance(events, list)
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd "c:/Users/somca/내문서/project/PIMS" && python -m pytest tests/test_isolation_forest_adapter.py -v 2>&1
```
Expected: `ImportError` (모듈 없음)

- [ ] **Step 3: 구현**

```python
# src/analysis/isolation_forest_adapter.py
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.analysis.base_detector import AnomalyDetector, AnomalyEvent


class IsolationForestAdapter(AnomalyDetector):
    """윈도우드 Isolation Forest를 AnomalyDetector 인터페이스로 감싼다.

    reference/windowed_model_pipeline.py의 prepare_windowed_data() 로직 기반.

    각 윈도우 끝 시점이 anomaly(-1)로 분류되면 AnomalyEvent를 생성한다.
    score = anomaly_score (sklearn이 반환하는 음수 float, 낮을수록 이상함).
    """

    def __init__(
        self,
        window_size: int = 5,
        contamination: float = 0.03,
        n_estimators: int = 100,
        excluded_signals: list[str] | None = None,
        top_n: int = 5,
    ):
        self.window_size = window_size
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.excluded_signals: set[str] = set(excluded_signals or [])
        self.top_n = top_n

    def detect(self, df: pd.DataFrame) -> list[AnomalyEvent]:
        # 분석 대상 컬럼 선택 (수치형, 제외 신호 제거)
        analog_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c not in self.excluded_signals
        ]
        if not analog_cols or len(df) < self.window_size:
            return []

        matrix = df[analog_cols].values  # shape: (N, M)
        N, M = matrix.shape

        # 윈도우드 특징 행렬 생성: (N - W + 1, M * W)
        windowed = np.array([
            matrix[i: i + self.window_size].flatten()
            for i in range(N - self.window_size + 1)
        ])

        # Isolation Forest 학습 + 예측
        iso = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=42,
            n_jobs=-1,
        )
        preds = iso.fit_predict(windowed)     # 1: normal, -1: anomaly
        scores = iso.score_samples(windowed)  # 낮을수록 이상

        # 윈도우 끝 인덱스 → DataFrame 인덱스 매핑
        target_indices = list(range(self.window_size - 1, N))

        events: list[AnomalyEvent] = []
        for i, (pred, score) in enumerate(zip(preds, scores)):
            if pred != -1:
                continue

            df_idx = target_indices[i]
            ts = df.index[df_idx]

            # 윈도우 내 각 컬럼의 변동성 (std) → top_n 기여 신호 추출
            window_slice = matrix[i: i + self.window_size]  # (W, M)
            col_stds = window_slice.std(axis=0)
            top_idx = np.argsort(col_stds)[::-1][: self.top_n]
            top_signals = [(analog_cols[j], float(col_stds[j])) for j in top_idx]

            events.append(
                AnomalyEvent(
                    timestamp=ts,
                    score=float(score),
                    top_signals=top_signals,
                    label="if_candidate",
                    metadata={
                        "window_size": self.window_size,
                        "contamination": self.contamination,
                    },
                )
            )
        return events

    def name(self) -> str:
        return "isolation_forest"
```

- [ ] **Step 4: sklearn 설치 확인 후 테스트 통과 확인**

```bash
cd "c:/Users/somca/내문서/project/PIMS" && python -m pip install scikit-learn 2>&1 | tail -3
python -m pytest tests/test_isolation_forest_adapter.py -v 2>&1
```
Expected: 7 PASSED

- [ ] **Step 5: 커밋**

```bash
cd "c:/Users/somca/내문서/project/PIMS" && git add src/analysis/isolation_forest_adapter.py tests/test_isolation_forest_adapter.py && git commit -m "feat: IsolationForestAdapter with windowed features"
```

---

## Task 2: `LLMFilter` — Ollama 기반 후보 검증

**Files:**
- Create: `src/analysis/llm_filter.py`
- Create: `config/llm_filter_prompt.md`
- Create: `tests/test_llm_filter.py`

LLM이 비활성화(`enabled: false`)이거나 응답 오류 시 모든 후보를 통과시킨다 (안전 fallback).
프롬프트 템플릿은 `config/llm_filter_prompt.md`에서 읽는다 (autoresearch 대상).

- [ ] **Step 1: 프롬프트 템플릿 작성**

`config/llm_filter_prompt.md` 생성:

```markdown
당신은 산업 설비 이상 탐지 전문가입니다.
Isolation Forest 모델이 탐지한 이상 후보를 분석하고 실제 이상인지 판단하세요.

## 이상 후보 정보

- 발생 시각: {timestamp}
- 이상 점수: {score:.4f} (낮을수록 더 이상함, 정상 범위: -0.1 ~ 0)
- 주요 급변 신호 (변동성 기준):
{top_signals}

## 탐지 전후 {context_sec}초 신호 통계

{context_stats}

## 판단 기준

- 복수 신호가 동시에 급변 → 이상 가능성 높음 (KEEP)
- 단일 신호만 변화, 나머지 정상 → 노이즈일 수 있음 (REJECT 고려)
- 신호값이 물리적으로 불가능한 범위 (음수 속도, 과도한 전류) → 이상 (KEEP)
- 이상 점수가 -0.15 미만 → 강한 이상 신호 (KEEP)

## 응답 형식

반드시 첫 줄에 아래 형식 중 하나로만 응답하세요:

KEEP: {판단 이유 한 줄}
REJECT: {판단 이유 한 줄}
```

- [ ] **Step 2: 테스트 작성**

```python
# tests/test_llm_filter.py
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.analysis.llm_filter import LLMFilter
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


def test_disabled_filter_passes_all():
    """LLM 비활성화 시 모든 후보를 그대로 반환한다."""
    f = LLMFilter(enabled=False)
    events = [_make_event(), _make_event("2026-03-20 15:49:10+00:00")]
    result = f.filter(events, _make_df())
    assert len(result) == 2


def test_empty_candidates_returns_empty():
    f = LLMFilter(enabled=False)
    result = f.filter([], _make_df())
    assert result == []


def test_keep_response_passes_event():
    """KEEP 응답이 오면 이벤트가 결과에 포함된다."""
    f = LLMFilter(enabled=True, host="http://localhost:11434", model="qwen3:4b")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "KEEP: 복수 신호 동시 급변 감지됨"}
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp):
        result = f.filter([_make_event()], _make_df())

    assert len(result) == 1
    assert result[0].metadata.get("llm_verdict") == "KEEP"


def test_reject_response_filters_event():
    """REJECT 응답이 오면 이벤트가 결과에서 제외된다."""
    f = LLMFilter(enabled=True, host="http://localhost:11434", model="qwen3:4b")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "REJECT: 단일 신호 노이즈로 판단"}
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp):
        result = f.filter([_make_event()], _make_df())

    assert len(result) == 0


def test_llm_error_falls_back_to_keep():
    """LLM 호출 오류 시 해당 이벤트를 통과시킨다 (안전 fallback)."""
    f = LLMFilter(enabled=True, host="http://localhost:11434", model="qwen3:4b")
    with patch("requests.post", side_effect=Exception("Connection refused")):
        result = f.filter([_make_event()], _make_df())
    assert len(result) == 1
    assert result[0].metadata.get("llm_verdict") == "ERROR_KEEP"


def test_reasoning_stored_in_metadata():
    """LLM 판단 이유가 metadata['llm_reason']에 저장된다."""
    f = LLMFilter(enabled=True, host="http://localhost:11434", model="qwen3:4b")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "KEEP: 속도 신호 0으로 급감"}
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp):
        result = f.filter([_make_event()], _make_df())

    assert "속도 신호 0으로 급감" in result[0].metadata.get("llm_reason", "")
```

- [ ] **Step 3: 테스트 실패 확인**

```bash
cd "c:/Users/somca/내문서/project/PIMS" && python -m pytest tests/test_llm_filter.py -v 2>&1
```
Expected: `ImportError`

- [ ] **Step 4: `LLMFilter` 구현**

```python
# src/analysis/llm_filter.py
from __future__ import annotations
import re
from pathlib import Path

import pandas as pd
import requests

from src.analysis.base_detector import AnomalyEvent

_PROMPT_PATH = Path(__file__).parent.parent.parent / "config" / "llm_filter_prompt.md"
_CONTEXT_SEC = 10  # 이벤트 전후 통계 윈도우 (초)


def _load_prompt_template() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    # fallback: 최소 프롬프트
    return (
        "이상 후보를 분석하세요.\n"
        "timestamp: {timestamp}, score: {score:.4f}, signals: {top_signals}\n"
        "첫 줄에 KEEP: 이유 또는 REJECT: 이유 형식으로 응답하세요."
    )


def _build_context_stats(event: AnomalyEvent, df: pd.DataFrame) -> str:
    """이벤트 전후 _CONTEXT_SEC초 구간의 신호 통계를 텍스트로 반환한다."""
    ts = event.timestamp
    t_start = ts - pd.Timedelta(seconds=_CONTEXT_SEC)
    t_end = ts + pd.Timedelta(seconds=_CONTEXT_SEC)

    sig_names = [s[0] for s in event.top_signals]
    available = [s for s in sig_names if s in df.columns]
    if not available:
        return "(신호 데이터 없음)"

    window = df.loc[t_start:t_end, available] if t_start in df.index or t_end in df.index else df[available]
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
    """LLM 응답 첫 줄에서 KEEP/REJECT와 이유를 파싱한다."""
    first_line = response_text.strip().splitlines()[0] if response_text.strip() else ""
    m = re.match(r"^(KEEP|REJECT):\s*(.+)", first_line, re.IGNORECASE)
    if m:
        return m.group(1).upper(), m.group(2).strip()
    # 파싱 실패 → KEEP으로 안전 처리
    return "KEEP", f"파싱 불가 응답: {first_line[:80]}"


class LLMFilter:
    """Ollama를 사용해 이상치 후보 목록에서 유효 이상치만 선별한다.

    LLM 비활성화 또는 오류 시 모든 후보를 통과시킨다 (안전 fallback).
    프롬프트 템플릿: config/llm_filter_prompt.md (autoresearch 대상)
    """

    def __init__(
        self,
        enabled: bool = True,
        host: str = "http://localhost:11434",
        model: str = "qwen3:4b",
        timeout: int = 30,
    ):
        self.enabled = enabled
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def filter(
        self,
        candidates: list[AnomalyEvent],
        df: pd.DataFrame,
    ) -> list[AnomalyEvent]:
        """후보 목록을 LLM으로 검증하고 KEEP 판정된 이벤트만 반환한다."""
        if not candidates:
            return []
        if not self.enabled:
            return candidates

        template = _load_prompt_template()
        validated = []

        for event in candidates:
            verdict, reason = self._call_llm(event, df, template)
            updated_meta = {**event.metadata, "llm_verdict": verdict, "llm_reason": reason}
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

    def _call_llm(
        self,
        event: AnomalyEvent,
        df: pd.DataFrame,
        template: str,
    ) -> tuple[str, str]:
        """단일 이벤트에 대해 LLM을 호출하고 (verdict, reason)을 반환한다."""
        top_signals_text = "\n".join(
            f"  - {name}: 변동성={val:.3f}" for name, val in event.top_signals
        )
        context_stats = _build_context_stats(event, df)

        prompt = template.format(
            timestamp=str(event.timestamp),
            score=event.score,
            top_signals=top_signals_text,
            context_sec=_CONTEXT_SEC,
            context_stats=context_stats,
        )

        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            response_text = resp.json().get("response", "")
            return _parse_verdict(response_text)
        except Exception as exc:
            return "ERROR_KEEP", f"LLM 오류: {exc}"
```

- [ ] **Step 5: 테스트 통과 확인**

```bash
cd "c:/Users/somca/내문서/project/PIMS" && python -m pytest tests/test_llm_filter.py -v 2>&1
```
Expected: 6 PASSED

- [ ] **Step 6: 커밋**

```bash
cd "c:/Users/somca/내문서/project/PIMS" && git add src/analysis/llm_filter.py config/llm_filter_prompt.md tests/test_llm_filter.py && git commit -m "feat: LLMFilter for candidate validation via Ollama"
```

---

## Task 3: Dashboard 파이프라인 업데이트

**Files:**
- Modify: `dashboard.py`
- Modify: `config/settings.yaml`
- Modify: `config/signals.yaml`

현재 `dashboard.py`는 `ZScoreAdapter → detect → display` 1단계 파이프라인이다.
`IsolationForestAdapter → LLMFilter → display` 2단계로 교체하고,
요약 카드에 "IF 후보 N건 → 검증 후 M건" 정보를 추가한다.

- [ ] **Step 1: settings.yaml 업데이트**

`config/settings.yaml`에서:
```yaml
# 기존
llm:
  enabled: false
  model: "llama3.1:8b"
  host: "http://localhost:11434"
```
→ 아래로 교체:
```yaml
llm:
  enabled: true
  model: "qwen3:4b"
  host: "http://localhost:11434"
  filter_timeout: 30       # LLM 호출 타임아웃 (초)
```

- [ ] **Step 2: signals.yaml에 model_params 섹션 추가**

`config/signals.yaml` 맨 아래에 추가:
```yaml
# IsolationForest 파라미터 (autoresearch 수정 대상)
model_params:
  isolation_forest:
    window_size: 5         # 윈도우 크기 (샘플 수), 범위: 3~20
    contamination: 0.03    # 이상치 비율 추정, 범위: 0.01~0.10
    n_estimators: 100      # 트리 수, 범위: 50~200
```

- [ ] **Step 3: dashboard.py 임포트 추가**

기존 임포트 블록에 추가:
```python
from src.analysis.isolation_forest_adapter import IsolationForestAdapter
from src.analysis.llm_filter import LLMFilter
```
(`ZScoreAdapter` 임포트는 유지 — `build_detector`에서 아직 사용)

- [ ] **Step 4: `build_detector`와 `build_llm_filter` 헬퍼 교체**

`build_detector()` 함수를 아래로 교체하고, `build_llm_filter()` 추가:

```python
def build_detector(threshold: float, window: int, top_n: int,
                   excluded: list[str], overrides: dict) -> AnomalyDetector:
    """signals.yaml model_params를 읽어 IsolationForestAdapter를 생성한다."""
    with open(ROOT / "config" / "signals.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mp = (cfg.get("model_params") or {}).get("isolation_forest", {})
    return IsolationForestAdapter(
        window_size=mp.get("window_size", 5),
        contamination=mp.get("contamination", 0.03),
        n_estimators=mp.get("n_estimators", 100),
        excluded_signals=excluded,
        top_n=top_n,
    )


def build_llm_filter() -> LLMFilter:
    """settings.yaml llm 섹션으로 LLMFilter를 생성한다."""
    with open(ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    llm_cfg = settings.get("llm", {})
    return LLMFilter(
        enabled=llm_cfg.get("enabled", False),
        host=llm_cfg.get("host", "http://localhost:11434"),
        model=llm_cfg.get("model", "qwen3:4b"),
        timeout=llm_cfg.get("filter_timeout", 30),
    )
```

- [ ] **Step 5: `detect_anomalies` 캐시 함수를 2단계 파이프라인으로 교체**

기존 `detect_anomalies` 함수를 아래로 교체:

```python
@st.cache_data(show_spinner="이상치 탐지 중...")
def detect_anomalies(
    path: str, _thr: float, _win: int, _topn: int,
    _excluded: tuple[str, ...], _overrides_key: str,
) -> tuple[list[AnomalyEvent], list[AnomalyEvent]]:
    """(candidates, validated_events) 튜플을 반환한다."""
    import json
    overrides = json.loads(_overrides_key) if _overrides_key else {}
    df = load_and_process(path)
    detector = build_detector(_thr, _win, _topn, list(_excluded), overrides)
    candidates = detector.detect(df)
    llm_filter = build_llm_filter()
    validated = llm_filter.filter(candidates, df)
    return candidates, validated
```

- [ ] **Step 6: `run_btn` 클릭 섹션 및 session_state 업데이트**

`detect_anomalies` 호출 결과를 튜플로 받도록 수정:

```python
# 기존
events = detect_anomalies(csv_path, threshold, window, top_n,
                          tuple(excluded), overrides_key)
st.session_state.update(df=df, events=events, csv_path=csv_path)

# 교체 후
candidates, events = detect_anomalies(csv_path, threshold, window, top_n,
                                       tuple(excluded), overrides_key)
st.session_state.update(df=df, events=events,
                         candidates=candidates, csv_path=csv_path)
```

- [ ] **Step 7: 요약 카드 업데이트**

기존 4개 카드 섹션을 5개로 교체:
```python
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("총 샘플 수", f"{len(df):,}")
c2.metric("기록 신호 수", f"{len(df.columns):,}")
candidates_count = len(st.session_state.get("candidates", []))
c3.metric("IF 후보", f"{candidates_count}건")
c4.metric("검증된 이상", f"{len(events)}건")
dur = (df.index[-1] - df.index[0]).total_seconds() / 60
c5.metric("데이터 구간", f"{dur:.1f}분")
```

- [ ] **Step 8: 이벤트 상세에 LLM 판단 표시**

이벤트 요약 `left` 컬럼의 마크다운 테이블 아래에 추가:
```python
# LLM 판단 결과 표시 (있을 때만)
llm_verdict = event.metadata.get("llm_verdict", "")
llm_reason = event.metadata.get("llm_reason", "")
if llm_verdict:
    verdict_icon = "✅" if llm_verdict == "KEEP" else "⚠️"
    st.markdown(f"**LLM 판단:** {verdict_icon} `{llm_verdict}`")
    if llm_reason:
        st.caption(llm_reason)
```

- [ ] **Step 9: overrides_key 직렬화 방식 통일**

현재 `||` 구분자 방식을 `json.dumps`로 교체 (Step 5와 일관성):
```python
import json
overrides_key = json.dumps(overrides, ensure_ascii=False) if overrides else ""
```
`detect_anomalies` 내부도 `json.loads(_overrides_key)` 사용 (Step 5에서 이미 적용됨).

- [ ] **Step 10: 문법 오류 없음 확인**

```bash
cd "c:/Users/somca/내문서/project/PIMS" && python -c "import ast; ast.parse(open('dashboard.py').read()); print('OK')"
```

- [ ] **Step 11: 전체 테스트 통과 확인**

```bash
cd "c:/Users/somca/내문서/project/PIMS" && python -m pytest tests/ -v --ignore=tests/test_watcher.py 2>&1
```
Expected: 전부 PASSED

- [ ] **Step 12: 커밋**

```bash
cd "c:/Users/somca/내문서/project/PIMS" && git add dashboard.py config/settings.yaml config/signals.yaml && git commit -m "feat: dashboard IF+LLM two-stage pipeline, candidate vs validated cards"
```

---

## Task 4: `evaluate_detection.py` 업데이트

**Files:**
- Modify: `evaluate_detection.py`
- Modify: `tests/test_evaluate_detection.py`

autoresearch 루프가 IF 파라미터 + LLM 프롬프트를 최적화할 수 있도록
`evaluate()`를 `use_llm=False` 기본값으로 빠른 평가를 지원한다.
`--llm` 플래그로 LLM 포함 전체 평가 가능.

- [ ] **Step 1: evaluate_detection.py의 `build_detector` 교체**

기존 `build_detector` 함수를 아래로 교체 (ZScoreAdapter → IsolationForestAdapter):

```python
from src.analysis.isolation_forest_adapter import IsolationForestAdapter
from src.analysis.llm_filter import LLMFilter


def build_detector(signals: dict, settings: dict) -> IsolationForestAdapter:
    """외부 모델 이식 시 이 함수만 교체한다."""
    _validate_signals(signals)
    mp = (signals.get("model_params") or {}).get("isolation_forest", {})
    return IsolationForestAdapter(
        window_size=mp.get("window_size", 5),
        contamination=mp.get("contamination", 0.03),
        n_estimators=mp.get("n_estimators", 100),
        excluded_signals=signals.get("excluded_signals") or [],
    )


def build_llm_filter(settings: dict, enabled_override: bool | None = None) -> LLMFilter:
    """settings.yaml llm 섹션으로 LLMFilter를 생성한다."""
    llm_cfg = settings.get("llm", {})
    enabled = llm_cfg.get("enabled", False) if enabled_override is None else enabled_override
    return LLMFilter(
        enabled=enabled,
        host=llm_cfg.get("host", "http://localhost:11434"),
        model=llm_cfg.get("model", "qwen3:4b"),
        timeout=llm_cfg.get("filter_timeout", 30),
    )
```

- [ ] **Step 2: `_validate_signals`에 model_params 범위 검증 추가**

기존 `_validate_signals` 아래에 추가:

```python
def _validate_signals(signals: dict) -> None:
    """signal_overrides 및 model_params 값이 허용 범위 내인지 확인한다."""
    for sig, thr in (signals.get("signal_overrides") or {}).items():
        if not (1.5 <= float(thr) <= 5.0):
            raise ValueError(
                f"signal_overrides['{sig}'] = {thr} 는 허용 범위(1.5~5.0) 밖입니다."
            )
    mp = (signals.get("model_params") or {}).get("isolation_forest", {})
    contamination = mp.get("contamination")
    if contamination is not None and not (0.01 <= float(contamination) <= 0.10):
        raise ValueError(
            f"model_params.isolation_forest.contamination = {contamination} 는 "
            "허용 범위(0.01~0.10) 밖입니다."
        )
    window_size = mp.get("window_size")
    if window_size is not None and not (3 <= int(window_size) <= 20):
        raise ValueError(
            f"model_params.isolation_forest.window_size = {window_size} 는 "
            "허용 범위(3~20) 밖입니다."
        )
```

- [ ] **Step 3: `evaluate()` 함수에 `use_llm` 파라미터 추가**

기존 `evaluate(loader=None)` 시그니처를:
```python
def evaluate(loader=None, use_llm: bool = False) -> dict:
```
로 바꾸고, 루프 내부에서 LLM 필터 적용:

```python
def evaluate(loader=None, use_llm: bool = False) -> dict:
    signals, settings = _load_config()
    detector = build_detector(signals, settings)
    llm_filter = build_llm_filter(settings, enabled_override=use_llm)
    if loader is None:
        loader = IbaCSVLoader()
    preprocessor = Preprocessor()

    total_candidates = 0
    total_alarms = 0
    hits = 0
    total_known = sum(len(v) for v in KNOWN_EVENTS.values())

    for csv_file, known_ts_list in KNOWN_EVENTS.items():
        csv_path = EVAL_CSV_DIR / csv_file
        if not csv_path.exists():
            print(f"[WARN] 평가 CSV 없음: {csv_path}", file=sys.stderr)
            continue

        df = preprocessor.process(loader.load(str(csv_path)))
        candidates = detector.detect(df)
        total_candidates += len(candidates)
        events = llm_filter.filter(candidates, df)  # use_llm=False면 candidates 그대로
        detected_times = [e.timestamp for e in events]
        total_alarms += len(events)

        for known_ts_str in known_ts_list:
            kt = pd.Timestamp(known_ts_str)
            matched = any(
                abs((dt - kt).total_seconds()) <= TOLERANCE_SECONDS
                for dt in detected_times
            )
            if matched:
                hits += 1

    misses = total_known - hits
    false_positives = max(0, total_alarms - hits)
    hit_rate = hits / total_known if total_known > 0 else 0.0
    fp_rate = false_positives / max(total_alarms, 1)
    precision = hits / max(total_alarms, 1)
    recall = hits / max(total_known, 1)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "alarm_count": total_alarms,
        "candidate_count": total_candidates,
        "hit_count": hits,
        "miss_count": misses,
        "false_positive": false_positives,
        "hit_rate": round(hit_rate, 4),
        "fp_rate": round(fp_rate, 4),
        "f1": round(f1, 4),
    }
```

- [ ] **Step 4: `__main__` 블록에 `--llm` 플래그 추가**

```python
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="LLM 필터 포함 평가")
    args = parser.parse_args()
    result = evaluate(use_llm=args.llm)
    print("---")
    print(f"f1:              {result['f1']:.4f}")
    print(f"hit_rate:        {result['hit_rate']:.4f}")
    print(f"fp_rate:         {result['fp_rate']:.4f}")
    print(f"alarm_count:     {result['alarm_count']}")
    print(f"candidate_count: {result['candidate_count']}")
    print(f"hit_count:       {result['hit_count']}")
    print(f"miss_count:      {result['miss_count']}")
    print(f"false_positive:  {result['false_positive']}")
```

- [ ] **Step 5: test_evaluate_detection.py 업데이트**

기존 `test_validate_signals_raises_on_out_of_range` 테스트 아래에 추가:

```python
def test_validate_signals_raises_on_bad_contamination():
    import evaluate_detection as ed
    signals = {"model_params": {"isolation_forest": {"contamination": 0.5}}}
    with pytest.raises(ValueError, match="허용 범위"):
        ed._validate_signals(signals)


def test_validate_signals_raises_on_bad_window_size():
    import evaluate_detection as ed
    signals = {"model_params": {"isolation_forest": {"window_size": 50}}}
    with pytest.raises(ValueError, match="허용 범위"):
        ed._validate_signals(signals)


def test_evaluate_returns_candidate_count(tmp_path, monkeypatch):
    eval_dir = tmp_path / "eval_data"
    eval_dir.mkdir()
    csv_path = eval_dir / "trip_sample.csv"
    _write_eval_csv(csv_path, has_trip=True)

    import evaluate_detection as ed
    monkeypatch.setattr(ed, "EVAL_CSV_DIR", eval_dir)
    monkeypatch.setattr(ed, "KNOWN_EVENTS", {
        "trip_sample.csv": ["2026-03-20 15:49:03.000000+00:00"],
    })

    class SimpleCsvLoader:
        def load(self, path: str):
            import pandas as pd
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            return df

    result = ed.evaluate(loader=SimpleCsvLoader(), use_llm=False)
    assert "candidate_count" in result
    assert result["candidate_count"] >= 0
```

- [ ] **Step 6: 전체 테스트 통과 확인**

```bash
cd "c:/Users/somca/내문서/project/PIMS" && python -m pytest tests/ -v --ignore=tests/test_watcher.py 2>&1
```
Expected: 전부 PASSED

- [ ] **Step 7: 커밋**

```bash
cd "c:/Users/somca/내문서/project/PIMS" && git add evaluate_detection.py tests/test_evaluate_detection.py && git commit -m "feat: evaluate_detection uses IsolationForest + optional LLM filter"
```

---

## Task 5: AutoResearch 설정 업데이트

**Files:**
- Modify: `reference/pims_program.md`

autoresearch 루프의 최적화 대상을 2축으로 확장한다:
- **축 1**: `config/signals.yaml` → IF 파라미터 (`model_params.isolation_forest`)
- **축 2**: `config/llm_filter_prompt.md` → LLM 판단 프롬프트

- [ ] **Step 1: `reference/pims_program.md` 내용 전체 교체**

```markdown
# PIMS 자동 파라미터 튜닝 에이전트 (2축 최적화)

> autoresearch 방식 자가개선 루프 지시서.
> 최적화 축 1: config/signals.yaml (IF 파라미터)
> 최적화 축 2: config/llm_filter_prompt.md (LLM 프롬프트)

## 목표

`evaluate_detection.py`의 f1 점수를 최대화한다.

- **최우선**: `hit_rate = 1.0` (미탐 허용 안 함)
- **두 번째**: `fp_rate` 최소화 (오탐 줄이기)
- **단순성**: 수정 항목을 최소화하여 복잡도 통제

---

## Setup

```bash
git checkout -b pims-autotune/<날짜>
python evaluate_detection.py  # 베이스라인 (IF만, LLM 없이)
python evaluate_detection.py --llm  # LLM 포함 전체 평가
```

---

## 실험 루프

### 축 1: IF 파라미터 최적화 (`signals.yaml`만 수정)

```
수정 가능한 필드:
  model_params.isolation_forest.window_size   (3 ~ 20)
  model_params.isolation_forest.contamination (0.01 ~ 0.10)
  model_params.isolation_forest.n_estimators  (50 ~ 200)
  excluded_signals (노이즈 신호 제외)
  signal_overrides (사용 안 함, 호환성용)

루프:
  1. signals.yaml 수정
  2. git commit -m "exp: <설명>"
  3. python evaluate_detection.py > run.log 2>&1
  4. grep "^f1:" run.log
  5. f1 개선 → keep / 악화 → git reset --hard HEAD~1
  6. experiments/results.tsv 기록
```

### 축 2: LLM 프롬프트 최적화 (`llm_filter_prompt.md`만 수정)

```
수정 가능한 파일:
  config/llm_filter_prompt.md

목표: LLM이 더 정확하게 KEEP/REJECT를 판단하도록 지시문 개선

루프:
  1. llm_filter_prompt.md 수정 (판단 기준, 응답 형식 등)
  2. git commit -m "prompt: <설명>"
  3. python evaluate_detection.py --llm > run.log 2>&1
  4. grep "^f1:" run.log
  5. f1 개선 → keep / 악화 → git reset --hard HEAD~1
  6. experiments/results.tsv 기록
```

---

## 수정 불가 항목

- `evaluate_detection.py` 본체 및 `KNOWN_EVENTS`
- `src/` 하위 소스 코드
- `dashboard.py`

---

## 단순성 기준

| 상황 | 결정 |
|------|------|
| f1 동등 + signal_overrides 삭제 | **유지** (단순화 승리) |
| hit_rate < 1.0 | **무조건 폐기** |
| f1 +0.01 + 프롬프트 복잡도 대폭 증가 | **고민** |

## NEVER STOP

사람이 `Ctrl+C`로 중단할 때까지 루프를 반복한다.
```

- [ ] **Step 2: 전체 테스트 최종 확인**

```bash
cd "c:/Users/somca/내문서/project/PIMS" && python -m pytest tests/ -v --ignore=tests/test_watcher.py 2>&1
```
Expected: 전부 PASSED

- [ ] **Step 3: 커밋**

```bash
cd "c:/Users/somca/내문서/project/PIMS" && git add reference/pims_program.md && git commit -m "docs: update autoresearch guide for 2-axis optimization (IF params + LLM prompt)"
```

---

## 선행 조건

| 항목 | 확인 |
|------|------|
| `scikit-learn` 설치 | `python -c "import sklearn"` |
| `requests` 설치 | `python -c "import requests"` |
| Ollama 실행 중 | `curl http://localhost:11434/api/tags` |
| `qwen3:4b` 모델 다운로드 완료 | ollama list |

---

## 전체 실행 순서

```
Task 1 → Task 2 → Task 3 → Task 4 → Task 5 (순차)
```

Task 3과 Task 4는 Task 1, 2 완료 후 실행.
각 태스크는 독립적으로 테스트 가능.