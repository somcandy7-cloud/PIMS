# Anomaly Model Layer + AutoResearch Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Excel 데이터 지원 + 플러거블 이상치 모델 인터페이스 도입 + autoresearch 방식 자가개선 루프 구축

**Architecture:**
- `AnomalyDetector` ABC를 도입해 이상치 탐지 로직을 교체 가능하게 분리한다. 기존 `TripDetector`는 어댑터로 감싸 기본 구현이 된다. Excel 로더는 독립 클래스로 추가하고 대시보드는 파일 확장자로 로더를 자동 선택한다. autoresearch 루프는 `evaluate_detection.py`(고정 평가 함수) + `experiments/results.tsv`(실험 이력) + `reference/pims_program.md`(에이전트 지시서)로 구성된다.

**Tech Stack:** Python 3.10+, pandas, openpyxl, pytest, streamlit, PyYAML, git

---

## 파일 구조 변경 계획

```
신규 생성:
  src/analysis/base_detector.py          # AnomalyDetector ABC + AnomalyEvent
  src/analysis/zscore_adapter.py         # TripDetector → AnomalyDetector 어댑터
  src/data/excel_loader.py               # Excel 시계열 → DataFrame
  evaluate_detection.py                  # 고정 평가 함수 (수정 금지)
  experiments/.gitkeep                   # 실험 폴더
  experiments/eval_data/.gitkeep         # 평가용 CSV 세트 (고정)
  reference/pims_program.md              # 에이전트 자율 실험 지시서
  tests/test_base_detector.py
  tests/test_zscore_adapter.py
  tests/test_excel_loader.py
  tests/test_evaluate_detection.py

수정:
  dashboard.py                           # 로더 자동 선택 + AnomalyDetector 사용
  config/settings.yaml                   # detector 설정 섹션 추가
  .gitignore                             # experiments/results.tsv 제외
```

---

## Part A: Excel 로더 + 플러거블 이상치 모델

---

### Task 1: `AnomalyDetector` ABC 정의

**Files:**
- Create: `src/analysis/base_detector.py`
- Create: `tests/test_base_detector.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_base_detector.py
import pandas as pd
import numpy as np
import pytest
from src.analysis.base_detector import AnomalyDetector, AnomalyEvent


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
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/test_base_detector.py -v
```
Expected: `ImportError` (모듈 없음)

- [ ] **Step 3: ABC 구현**

```python
# src/analysis/base_detector.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class AnomalyEvent:
    """이상치 탐지 이벤트 공통 스키마."""
    timestamp: pd.Timestamp
    score: float                               # 이상 강도 (모델별 정규화)
    top_signals: list[tuple[str, float]]       # (신호명, 기여도)
    label: str = "anomaly"                     # 모델이 붙이는 레이블
    metadata: dict = field(default_factory=dict)


class AnomalyDetector(ABC):
    """이상치 탐지기 프로토콜.

    외부 모델을 이식할 때 이 ABC를 상속하여 detect()와 name()만 구현한다.
    """

    @abstractmethod
    def detect(self, df: pd.DataFrame) -> list[AnomalyEvent]:
        """전처리된 DataFrame을 받아 이상치 이벤트 목록을 반환한다."""

    @abstractmethod
    def name(self) -> str:
        """모델 식별자 (대시보드/로그에 표시)."""
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_base_detector.py -v
```
Expected: 3 PASSED

- [ ] **Step 5: 커밋**

```bash
git add src/analysis/base_detector.py tests/test_base_detector.py
git commit -m "feat: add AnomalyDetector ABC + AnomalyEvent dataclass"
```

---

### Task 2: `ZScoreAdapter` — 기존 TripDetector를 AnomalyDetector로 감싸기

**Files:**
- Create: `src/analysis/zscore_adapter.py`
- Create: `tests/test_zscore_adapter.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_zscore_adapter.py
import numpy as np
import pandas as pd
from src.analysis.zscore_adapter import ZScoreAdapter
from src.analysis.base_detector import AnomalyEvent


def _make_trip_df() -> pd.DataFrame:
    n_pre, n_trip, n_post = 300, 10, 100
    rng = np.random.default_rng(99)
    n = n_pre + n_trip + n_post
    t = pd.date_range("2026-03-20 15:49:00", periods=n, freq="10ms", tz="UTC")
    speed = np.concatenate([
        np.full(n_pre, 50.0) + rng.normal(0, 0.5, n_pre),
        np.linspace(50, 0, n_trip),
        np.zeros(n_post) + rng.normal(0, 0.1, n_post),
    ])
    current = np.concatenate([
        np.full(n_pre, 100.0) + rng.normal(0, 1, n_pre),
        np.linspace(100, 0, n_trip),
        np.zeros(n_post) + rng.normal(0, 0.2, n_post),
    ])
    return pd.DataFrame({"speed": speed, "current": current}, index=t)


def test_returns_anomaly_events():
    adapter = ZScoreAdapter()
    df = _make_trip_df()
    events = adapter.detect(df)
    assert len(events) >= 1
    assert all(isinstance(e, AnomalyEvent) for e in events)


def test_event_has_score_and_signals():
    adapter = ZScoreAdapter()
    events = adapter.detect(_make_trip_df())
    evt = events[0]
    assert evt.score > 0
    assert len(evt.top_signals) >= 1


def test_name():
    assert ZScoreAdapter().name() == "zscore"


def test_no_event_in_flat_data():
    rng = np.random.default_rng(0)
    t = pd.date_range("2026-03-20", periods=500, freq="10ms", tz="UTC")
    df = pd.DataFrame({
        "speed": np.full(500, 50.0) + rng.normal(0, 0.3, 500),
        "current": np.full(500, 100.0) + rng.normal(0, 0.5, 500),
    }, index=t)
    events = ZScoreAdapter().detect(df)
    assert len(events) == 0
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/test_zscore_adapter.py -v
```
Expected: `ImportError`

- [ ] **Step 3: 어댑터 구현**

```python
# src/analysis/zscore_adapter.py
from __future__ import annotations

import pandas as pd

from src.analysis.base_detector import AnomalyDetector, AnomalyEvent
from src.analysis.trip_detector import TripDetector


class ZScoreAdapter(AnomalyDetector):
    """TripDetector를 AnomalyDetector 프로토콜로 감싸는 어댑터.

    기본 모델. 외부 모델 이식 전까지 이 클래스가 대시보드·평가 함수에 사용된다.
    """

    def __init__(
        self,
        window: int = 50,
        roc_zscore_threshold: float = 3.0,
        top_n: int = 5,
        excluded_signals: list[str] | None = None,
        signal_overrides: dict[str, float] | None = None,
    ):
        self._detector = TripDetector(
            window=window,
            roc_zscore_threshold=roc_zscore_threshold,
            top_n=top_n,
            excluded_signals=excluded_signals,
            signal_overrides=signal_overrides,
        )

    def detect(self, df: pd.DataFrame) -> list[AnomalyEvent]:
        trip_events = self._detector.detect(df)
        return [
            AnomalyEvent(
                timestamp=e.timestamp,
                score=float(len(e.top_changed_signals)),  # 급변 신호 수를 점수로
                top_signals=e.top_changed_signals,
                label=e.trigger,
                metadata={"description": e.description, "window": e.window_size},
            )
            for e in trip_events
        ]

    def name(self) -> str:
        return "zscore"
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_zscore_adapter.py -v
```
Expected: 4 PASSED

- [ ] **Step 5: 커밋**

```bash
git add src/analysis/zscore_adapter.py tests/test_zscore_adapter.py
git commit -m "feat: ZScoreAdapter wraps TripDetector as AnomalyDetector"
```

---

### Task 3: `ExcelLoader` — Excel 시계열 데이터 로더

**Files:**
- Create: `src/data/excel_loader.py`
- Create: `tests/test_excel_loader.py`

목표: 첫 번째 열이 datetime인 표준 Excel 시계열 파일 → DataFrame (인덱스 = DatetimeIndex).

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_excel_loader.py
import io
import pandas as pd
import numpy as np
import pytest
import openpyxl
from pathlib import Path
from src.data.excel_loader import ExcelLoader


def _make_excel_bytes(sheet_name: str = "Sheet1") -> bytes:
    """테스트용 Excel 파일 바이트를 메모리에 생성한다."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name
    # 헤더
    ws.append(["timestamp", "speed", "current", "label_col"])
    # 데이터 10행
    base = pd.Timestamp("2026-01-01 00:00:00")
    for i in range(10):
        ts = (base + pd.Timedelta(seconds=i)).to_pydatetime()
        ws.append([ts, float(i * 2), float(100 - i), "ok"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


@pytest.fixture
def excel_file(tmp_path: Path) -> Path:
    p = tmp_path / "test.xlsx"
    p.write_bytes(_make_excel_bytes())
    return p


def test_loads_dataframe(excel_file):
    df = ExcelLoader().load(str(excel_file))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10


def test_index_is_datetime(excel_file):
    df = ExcelLoader().load(str(excel_file))
    assert isinstance(df.index, pd.DatetimeIndex)


def test_numeric_columns_only(excel_file):
    df = ExcelLoader().load(str(excel_file))
    # label_col(str) 은 제거되어야 한다
    assert "label_col" not in df.columns
    assert "speed" in df.columns


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        ExcelLoader().load("nonexistent.xlsx")
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/test_excel_loader.py -v
```
Expected: `ImportError`

- [ ] **Step 3: ExcelLoader 구현**

```python
# src/data/excel_loader.py
from __future__ import annotations
from pathlib import Path

import pandas as pd


class ExcelLoader:
    """일반 Excel 시계열 파일을 로드한다.

    규칙:
    - 첫 번째 열이 datetime (자동 파싱)
    - 나머지 수치형 컬럼만 유지 (비수치형 제거)
    - 인덱스 = DatetimeIndex

    외부 모델 이식 전 데이터 탐색 또는 레이블 데이터 로드에 사용한다.
    """

    def load(self, filepath: str, sheet_name: int | str = 0) -> pd.DataFrame:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(filepath)

        df = pd.read_excel(path, sheet_name=sheet_name, header=0, engine="openpyxl")

        if df.empty:
            return df

        # 첫 번째 컬럼을 인덱스로 (datetime 파싱)
        time_col = df.columns[0]
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.set_index(time_col)
        df.index.name = "timestamp"

        # 수치형 컬럼만 유지
        df = df.select_dtypes(include="number").astype(float)

        return df
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_excel_loader.py -v
```
Expected: 4 PASSED

- [ ] **Step 5: 커밋**

```bash
git add src/data/excel_loader.py tests/test_excel_loader.py
git commit -m "feat: ExcelLoader for generic time-series Excel files"
```

---

### Task 4: 대시보드 — 파일 타입 자동 감지 + AnomalyDetector 인터페이스 사용

**Files:**
- Modify: `dashboard.py`
- Modify: `config/settings.yaml`

현재 `dashboard.py`는 `IbaCSVLoader`와 `TripDetector`를 직접 호출한다.
이를 `ExcelLoader` 자동 선택 + `ZScoreAdapter` 사용으로 교체한다.

변경 범위를 최소화하기 위해 로더 선택 헬퍼 함수와 detector 생성 함수만 추가/교체한다.

- [ ] **Step 1: settings.yaml에 detector 섹션 추가**

`config/settings.yaml`에 아래 내용 추가:
```yaml
detector:
  name: "zscore"          # 사용할 탐지기 이름 (현재: zscore)
  # 외부 모델 이식 시 여기에 모델 경로/설정 추가
```

- [ ] **Step 2: `dashboard.py` 상단 임포트 교체**

기존:
```python
from src.data.loader import IbaCSVLoader
from src.analysis.trip_detector import TripDetector, TripEvent
```

교체 후:
```python
from src.data.loader import IbaCSVLoader
from src.data.excel_loader import ExcelLoader
from src.analysis.zscore_adapter import ZScoreAdapter
from src.analysis.base_detector import AnomalyDetector, AnomalyEvent
from src.analysis.trip_detector import TripEvent  # FaultClassifier가 아직 TripEvent를 씀
```

- [ ] **Step 3: 로더 선택 헬퍼 추가 (dashboard.py에 함수 삽입)**

`load_signal_config()` 함수 아래에 추가:
```python
def get_loader(filepath: str):
    """파일 확장자에 따라 적절한 로더를 반환한다."""
    if Path(filepath).suffix.lower() in {".xlsx", ".xls"}:
        return ExcelLoader()
    return IbaCSVLoader()


def build_detector(threshold: float, window: int, top_n: int,
                   excluded: list[str], overrides: dict) -> AnomalyDetector:
    """설정값으로 ZScoreAdapter를 생성한다."""
    return ZScoreAdapter(
        window=window,
        roc_zscore_threshold=threshold,
        top_n=top_n,
        excluded_signals=excluded,
        signal_overrides=overrides,
    )
```

- [ ] **Step 4: `detect_trips` 캐시 함수 교체**

기존 `detect_trips` 함수에서 `TripDetector` 직접 생성 → `build_detector` 사용으로 교체.
`TripEvent` 타입 → `AnomalyEvent` 타입으로 교체.

```python
@st.cache_data(show_spinner=False)
def load_and_detect(
    filepath: str,
    threshold: float,
    window: int,
    top_n: int,
    excluded: tuple[str, ...],
    overrides_json: str,
) -> tuple[pd.DataFrame, list[AnomalyEvent]]:
    import json
    overrides = json.loads(overrides_json)
    loader = get_loader(filepath)
    df_raw = loader.load(filepath)
    df = Preprocessor().process(df_raw)
    detector = build_detector(threshold, window, top_n, list(excluded), overrides)
    events = detector.detect(df)
    return df, events
```

사이드바에서 `run_btn` 클릭 시 이 함수를 호출하도록 연결.

- [ ] **Step 5: 대시보드 수동 테스트**

```bash
streamlit run dashboard.py
```
- CSV 파일 경로 입력 → 분석 실행 → 기존과 동일하게 동작 확인
- xlsx 파일 경로 입력 → Excel 로드 → 이상치 탐지 결과 표시 확인

- [ ] **Step 6: 커밋**

```bash
git add dashboard.py config/settings.yaml
git commit -m "feat: dashboard auto-selects loader by extension, uses AnomalyDetector interface"
```

---

### Task 5: 외부 모델 이식 가이드 문서 작성

**Files:**
- Create: `reference/how-to-plug-model.md`

모델 교체 방법을 3단계로 문서화한다.

- [ ] **Step 1: 가이드 작성**

```markdown
# 외부 이상치 모델 이식 가이드

## 1단계: AnomalyDetector 구현

src/analysis/base_detector.py의 AnomalyDetector를 상속한다.

```python
from src.analysis.base_detector import AnomalyDetector, AnomalyEvent
import pandas as pd

class MyModel(AnomalyDetector):
    def __init__(self, model_path: str):
        import pickle
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def detect(self, df: pd.DataFrame) -> list[AnomalyEvent]:
        scores = self.model.predict(df.values)
        events = []
        for i, score in enumerate(scores):
            if score > self.threshold:
                events.append(AnomalyEvent(
                    timestamp=df.index[i],
                    score=float(score),
                    top_signals=[],
                    label="anomaly",
                ))
        return events

    def name(self) -> str:
        return "my_model"
```

## 2단계: dashboard.py의 build_detector 수정

```python
def build_detector(...) -> AnomalyDetector:
    return MyModel(model_path="models/my_model.pkl")
```

## 3단계: evaluate_detection.py의 detector_factory 수정

evaluate_detection.py의 `build_detector()` 함수만 동일하게 교체한다.
평가 함수 자체는 수정하지 않는다.
```

- [ ] **Step 2: 커밋**

```bash
git add reference/how-to-plug-model.md
git commit -m "docs: add external model integration guide"
```

---

## Part B: AutoResearch 자가개선 루프

---

### Task 6: 평가 인프라 — `experiments/` 폴더 + `.gitignore`

**Files:**
- Create: `experiments/.gitkeep`
- Create: `experiments/eval_data/.gitkeep`
- Modify: `.gitignore`

> **Phase 0 선행 작업 (Task 7 전에 사람이 직접 해야 함):**
> 1. 대표 CSV 파일 2개 이상을 `experiments/eval_data/`에 복사
> 2. 각 CSV를 직접 열어 이상 발생 추정 시각을 확인 → `evaluate_detection.py`의 `KNOWN_EVENTS` 딕셔너리에 기록
> 3. 최소 기준: CSV 2개, 레이블 이벤트 3개 이상
>
> 레이블이 없으면 `KNOWN_EVENTS = {}`로 두고 베이스라인 측정 가능 (f1=0.0).
> 레이블 추가 후 재측정하면 된다.

- [ ] **Step 1: 디렉토리 생성**

```bash
mkdir -p experiments/eval_data
touch experiments/.gitkeep experiments/eval_data/.gitkeep
```

- [ ] **Step 2: .gitignore에 추가**

`.gitignore` 파일에 아래 항목 추가:
```
experiments/results.tsv
experiments/eval_data/*.csv
```
(평가 CSV는 민감 데이터일 수 있으므로 git에서 제외. `.gitkeep`은 포함됨.)

- [ ] **Step 3: 대표 CSV 복사 (사람이 직접)**

```bash
# 실제 ibaPDA CSV 파일을 eval_data에 복사
cp "C:/path/to/2603201549_oven.csv" experiments/eval_data/
cp "C:/path/to/2603201632_oven.csv" experiments/eval_data/
```

각 CSV를 열어 이상 발생 시각을 확인하고 메모해 둔다. (Task 7에서 KNOWN_EVENTS에 입력)

- [ ] **Step 4: 커밋**

```bash
git add experiments/.gitkeep experiments/eval_data/.gitkeep .gitignore
git commit -m "chore: add experiments/ folder structure for autoresearch loop"
```

---

### Task 7: `evaluate_detection.py` — 고정 평가 함수

**Files:**
- Create: `evaluate_detection.py`
- Create: `tests/test_evaluate_detection.py`

**중요:** 이 파일은 autoresearch 루프에서 절대 수정하지 않는다 (apply_guide.md §2 참조).
에이전트가 수정하는 것은 `config/signals.yaml`뿐이다.

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_evaluate_detection.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import openpyxl


def _write_eval_csv(path: Path, has_trip: bool = True):
    """테스트용 평가 CSV를 작성한다 (iba 포맷 시뮬레이션이 복잡하므로 단순 CSV 사용)."""
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


def test_evaluate_returns_dict(tmp_path, monkeypatch):
    """evaluate()가 올바른 키를 포함한 dict를 반환한다."""
    eval_dir = tmp_path / "eval_data"
    eval_dir.mkdir()
    csv_path = eval_dir / "trip_sample.csv"
    _write_eval_csv(csv_path, has_trip=True)

    # evaluate_detection.py의 상수를 monkeypatch로 교체
    import evaluate_detection as ed
    monkeypatch.setattr(ed, "EVAL_CSV_DIR", eval_dir)
    monkeypatch.setattr(ed, "KNOWN_EVENTS", {
        "trip_sample.csv": ["2026-03-20 15:49:03.000000+00:00"],
    })

    result = ed.evaluate()
    assert "f1" in result
    assert "hit_rate" in result
    assert "fp_rate" in result
    assert "alarm_count" in result
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/test_evaluate_detection.py -v
```
Expected: `ImportError`

- [ ] **Step 3: evaluate_detection.py 작성**

apply_guide.md §4.1을 기반으로 ZScoreAdapter를 사용하도록 작성:

```python
# evaluate_detection.py
# 고정 평가 스크립트 — 이 파일 자체는 에이전트가 수정하지 않는다.
# 에이전트가 수정하는 것은 config/signals.yaml 뿐이다.
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data.loader import IbaCSVLoader
from src.analysis.preprocessor import Preprocessor
from src.analysis.zscore_adapter import ZScoreAdapter

# ── 고정 상수 (이 섹션은 사람만 수정할 수 있다) ────────────────────────────────
EVAL_CSV_DIR = ROOT / "experiments" / "eval_data"

# 키: CSV 파일명, 값: 알려진 이상 발생 시각 목록 (ISO 8601, TZ 포함)
KNOWN_EVENTS: dict[str, list[str]] = {
    # 예시 — 실제 레이블 데이터로 채운다:
    # "2603201549_oven.csv": ["2026-03-20 15:52:00+09:00"],
}
TOLERANCE_SECONDS = 30
# ──────────────────────────────────────────────────────────────────────────────


def _load_config() -> tuple[dict, dict]:
    with open(ROOT / "config" / "signals.yaml", encoding="utf-8") as f:
        signals = yaml.safe_load(f)
    with open(ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    return signals, settings


def _validate_signals(signals: dict) -> None:
    """signal_overrides 임계값이 허용 범위(1.5~5.0) 내인지 확인한다."""
    for sig, thr in (signals.get("signal_overrides") or {}).items():
        if not (1.5 <= float(thr) <= 5.0):
            raise ValueError(
                f"signal_overrides['{sig}'] = {thr} 는 허용 범위(1.5~5.0) 밖입니다. "
                "git reset --hard 후 재시도하세요."
            )


def build_detector(signals: dict, settings: dict) -> ZScoreAdapter:
    _validate_signals(signals)
    ana = settings.get("analysis", {})
    return ZScoreAdapter(
        window=ana.get("trip_window", 50),
        roc_zscore_threshold=ana.get("trip_roc_zscore_threshold", 3.0),
        top_n=ana.get("top_n_signals", 5),
        excluded_signals=signals.get("excluded_signals") or [],
        signal_overrides=signals.get("signal_overrides") or {},
    )


def evaluate() -> dict:
    """
    고정 CSV 세트로 탐지 성능을 측정한다.

    반환값:
        alarm_count, hit_count, miss_count, false_positive, hit_rate, fp_rate, f1
    """
    signals, settings = _load_config()
    detector = build_detector(signals, settings)
    loader = IbaCSVLoader()
    preprocessor = Preprocessor()

    total_alarms = 0
    hits = 0
    total_known = sum(len(v) for v in KNOWN_EVENTS.values())

    for csv_file, known_ts_list in KNOWN_EVENTS.items():
        csv_path = EVAL_CSV_DIR / csv_file
        if not csv_path.exists():
            print(f"[WARN] 평가 CSV 없음: {csv_path}", file=sys.stderr)
            continue

        df = preprocessor.process(loader.load(str(csv_path)))
        events = detector.detect(df)
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
        "hit_count": hits,
        "miss_count": misses,
        "false_positive": false_positives,
        "hit_rate": round(hit_rate, 4),
        "fp_rate": round(fp_rate, 4),
        "f1": round(f1, 4),
    }


if __name__ == "__main__":
    result = evaluate()
    print("---")
    print(f"f1:            {result['f1']:.4f}")
    print(f"hit_rate:      {result['hit_rate']:.4f}")
    print(f"fp_rate:       {result['fp_rate']:.4f}")
    print(f"alarm_count:   {result['alarm_count']}")
    print(f"hit_count:     {result['hit_count']}")
    print(f"miss_count:    {result['miss_count']}")
    print(f"false_positive:{result['false_positive']}")
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_evaluate_detection.py -v
```
Expected: PASSED

- [ ] **Step 5: 커밋**

```bash
git add evaluate_detection.py tests/test_evaluate_detection.py
git commit -m "feat: fixed evaluation function for autoresearch loop"
```

---

### Task 8: 에이전트 지시서 + 베이스라인 측정

**Files:**
- Create: `reference/pims_program.md`
- Create: `experiments/results.tsv` (첫 행만, git 추적 안 함)

- [ ] **Step 1: pims_program.md 작성**

apply_guide.md §4.3 내용을 아래와 같이 작성:

```markdown
# PIMS 자동 파라미터 튜닝 에이전트

## 목표

config/signals.yaml의 임계값을 조정하여 evaluate_detection.py의 f1 점수를 최대화한다.
hit_rate = 1.0 유지가 최우선이다. fp_rate를 줄이는 것이 두 번째 목표다.

## Setup

1. 브랜치 생성: `git checkout -b pims-autotune/<날짜>`
2. 파일 읽기: `config/signals.yaml`, `config/settings.yaml`
3. 베이스라인 측정: `python evaluate_detection.py > run.log 2>&1`
4. `experiments/results.tsv` 초기화 (첫 행: 베이스라인 기록)

## 실험 루프 (LOOP FOREVER)

1. `git status` 확인
2. `config/signals.yaml`만 수정 (이 파일만)
3. `git commit -m "exp: <변경 내용 한 줄 설명>"`
4. `python evaluate_detection.py > run.log 2>&1`
5. `grep "^f1:" run.log` 로 결과 확인
6. f1 개선 시 → 브랜치 유지 (status: keep)
   f1 동등/악화 시 → `git reset --hard` (status: discard)
7. `experiments/results.tsv`에 한 행 기록

## 수정 가능한 항목 (signals.yaml만)

```yaml
signal_groups:
  <group_name>:
    zscore_threshold: 1.5 ~ 5.0  # 조정 가능
signal_overrides:
  "<신호명>": <임계값>             # 추가/삭제 가능
excluded_signals:
  - "<신호명>"                     # 추가/삭제 가능
```

settings.yaml의 `trip_window`, `trip_roc_zscore_threshold`도 조정 가능.

## 수정 불가 항목

- `evaluate_detection.py` 및 `KNOWN_EVENTS` 상수
- `src/` 하위 파이썬 소스
- `dashboard.py`

## 단순성 기준 (autoresearch §2 참조)

- f1 동등 + signal_overrides 삭제 → **유지** (단순화 승리)
- hit_rate 1.0 미만 → **무조건 폐기**
- f1 0.01 향상 + signal_overrides 5개 추가 → **고민 후 결정**

## NEVER STOP

사람이 중단할 때까지 루프를 반복한다.
```

- [ ] **Step 2: results.tsv 헤더 생성**

```bash
echo -e "commit\tf1\thit_rate\tfp_rate\talarm_count\tstatus\tdescription" \
  > experiments/results.tsv
```

- [ ] **Step 3: 베이스라인 측정**

```
# 평가 CSV와 KNOWN_EVENTS가 준비된 후 실행
python evaluate_detection.py
```

출력 예시:
```
---
f1:            0.0000
hit_rate:      0.0000
fp_rate:       0.0000
alarm_count:   0
```
(KNOWN_EVENTS가 비어 있으면 0. 레이블 후 재측정 필요.)

- [ ] **Step 4: 커밋**

```bash
git add reference/pims_program.md
git commit -m "docs: add agent instruction doc for autoresearch loop"
```

---

## 선행 조건 (구현 시작 전 확인)

| 항목 | 확인 방법 |
|------|----------|
| openpyxl 설치 | `python -c "import openpyxl"` |
| pytest 설치 | `pytest --version` |
| git 초기화 완료 | `git status` |
| 평가용 CSV 복사 | `experiments/eval_data/` 에 실제 CSV 배치 |
| KNOWN_EVENTS 레이블 | 각 CSV의 이상 발생 시각을 `evaluate_detection.py`에 기록 |

---

## 전체 실행 순서 요약

```
Part A (순차 실행):
  Task 1 → Task 2 → Task 3 → Task 4 → Task 5

Part B (Part A 완료 후, 또는 병렬 가능):
  Task 6 → Task 7 → Task 8
```

Part A와 Part B는 독립적이므로 병렬 서브에이전트로 실행 가능하다.
단, Task 7(evaluate_detection.py)은 ZScoreAdapter(Task 2)에 의존한다.