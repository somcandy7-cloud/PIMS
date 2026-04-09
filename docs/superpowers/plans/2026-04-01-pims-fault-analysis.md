# PIMS Fault Analysis Helper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** iba PDA CSV 파일을 자동 감시·분석하여 코크스공장 OVEN 이동차의 Trip 원인과 이상 신호를 비전문가도 이해할 수 있는 한국어 리포트로 출력하는 로컬 실행 헬퍼

**Architecture:** 폴더 감시 데몬(watchdog)이 신규 CSV를 감지하면 파이프라인(데이터 로드 → 이상치 탐지 → Trip 원인 분류 → 리포트 생성 → 알람)을 순서대로 실행한다. LLM은 옵션으로, 없으면 룰 기반 템플릿 리포트를 사용한다.

**Tech Stack:** Python 3.10+, pandas, numpy, watchdog, pyyaml, plyer(Windows 알림), ollama SDK(선택)

---

## 범위 확인

독립 서브시스템 3개로 구성:
1. **데이터 파이프라인** - CSV 로드 + 전처리
2. **분석 엔진** - Trip 탐지 + 이상치 분류
3. **출력 시스템** - 리포트 생성 + 알람

각 서브시스템은 단독으로 테스트 가능하다. 이 계획은 전체를 한 번에 구현한다.

---

## 파일 구조

```
PIMS/
├── src/
│   ├── data/
│   │   ├── loader.py           # iba CSV 청크 로더, 헤더 파싱
│   │   └── schema.py           # Signal 타입 정의 (dataclass)
│   ├── analysis/
│   │   ├── preprocessor.py     # 결측치 처리, 리샘플링, 정규화
│   │   ├── statistical.py      # z-score 기반 이상치 탐지
│   │   ├── signal_discovery.py # 수치형 신호 자동 분류 (속도/전류 역할 추정)
│   │   ├── trip_detector.py    # Trip 이벤트 탐지 (급변 신호 자동 탐지)
│   │   └── fault_classifier.py # 룰 기반 고장 유형 분류
│   ├── alarm/
│   │   ├── threshold_config.py # 임계값 YAML 로드
│   │   └── notifier.py         # Windows 토스트 + 로그 파일 알람
│   ├── report/
│   │   ├── template_reporter.py # 구조화된 텍스트 리포트 (LLM 없이)
│   │   └── llm_reporter.py      # Ollama 연동 자연어 리포트 (선택)
│   └── watcher/
│       └── file_watcher.py     # watchdog 기반 폴더 감시 데몬
├── config/
│   ├── signals.yaml            # 관심 신호 패턴 및 임계값 (컬럼명 몰라도 동작)
│   └── settings.yaml           # 시스템 설정 (감시 폴더 경로 등)
├── tests/
│   ├── conftest.py             # 공통 fixture (샘플 DataFrame 생성)
│   ├── test_loader.py
│   ├── test_statistical.py
│   ├── test_trip_detector.py
│   ├── test_fault_classifier.py
│   └── test_watcher.py
├── main.py                     # 진입점
├── requirements.txt
└── docs/superpowers/plans/     # 이 파일 위치
```

---

## Task 1: 프로젝트 초기 설정

**Files:**
- Create: `requirements.txt`
- Create: `config/signals.yaml`
- Create: `config/settings.yaml`
- Create: `src/__init__.py`, `src/data/__init__.py`, `src/analysis/__init__.py`, `src/alarm/__init__.py`, `src/report/__init__.py`, `src/watcher/__init__.py`

- [ ] **Step 1: 패키지 디렉토리 및 `__init__.py` 생성**

```bash
mkdir -p src/data src/analysis src/alarm src/report src/watcher tests config
touch src/__init__.py src/data/__init__.py src/analysis/__init__.py \
      src/alarm/__init__.py src/report/__init__.py src/watcher/__init__.py
```

> `from src.data.loader import ...` 같은 임포트가 동작하려면 `__init__.py`가 반드시 있어야 한다. Python 3.3+ namespace package라도 `pytest`가 루트에서 실행될 때 서브패키지를 못 찾는 경우가 있으므로 명시적으로 생성한다.

- [ ] **Step 2: requirements.txt 작성**

```text
pandas>=2.0
numpy>=1.24
watchdog>=3.0
pyyaml>=6.0
plyer>=2.1
pytest>=7.4
ollama>=0.1  # 선택적
```

- [ ] **Step 3: config/signals.yaml 작성**

실제 CSV에서 확인된 iba PDA Siemens S7 포맷 기준으로 작성한다.
정확한 신호 역할(속도/전류 등)은 아직 미확정이므로, 패턴 기반으로 관심 신호 그룹을 정의한다.
나중에 신호 역할이 파악되면 `known_roles` 항목에 추가한다.

```yaml
# iba PDA CSV (Siemens S7) 신호 설정
# columns: 정확한 컬럼명 (Row 1의 PLC 태그명과 일치해야 함)
# patterns: 컬럼명에 이 문자열이 포함되면 해당 그룹으로 분류

signal_groups:
  double_word:        # MD: 32비트 REAL, 아날로그 측정값 가능성 높음
    patterns: ["MD "]
    type: analog
    zscore_threshold: 3.0
  drive_db420:        # DB420: 구동부 1 인버터 데이터블록
    patterns: ["DB420.DBW"]
    type: analog
    zscore_threshold: 3.0
  drive_db421:        # DB421: 구동부 2 인버터 데이터블록
    patterns: ["DB421.DBW"]
    type: analog
    zscore_threshold: 3.0
  status_word:        # MW: 16비트 상태/설정 워드
    patterns: ["MW "]
    type: analog
    zscore_threshold: 4.0
  digital_input:      # IB: 디지털 입력 바이트
    patterns: ["IB "]
    type: digital
  digital_output:     # QB: 디지털 출력 바이트
    patterns: ["QB "]
    type: digital
  flag_byte:          # MB: 플래그/상태 바이트
    patterns: ["MB "]
    type: digital

# 신호 역할 확정 후 여기에 추가 (Trip 탐지에 직접 사용)
# known_roles:
#   speed_actual: "DB420.DBW 2"    # 실제 속도 신호 컬럼명
#   current_actual: "DB420.DBW 4"  # 실제 전류 신호 컬럼명
#   position: "MD 110"             # 위치 신호 컬럼명
known_roles: {}
```

- [ ] **Step 3: config/settings.yaml 작성**

```yaml
watch_folder: "C:/ibaData/exports"   # iba가 CSV를 내보내는 실제 폴더로 교체
output_folder: "C:/ibaData/reports"
log_file: "C:/ibaData/reports/alarm.log"
llm:
  enabled: false
  model: "llama3.1:8b"
  host: "http://localhost:11434"
analysis:
  zscore_threshold: 3.0              # 이상치 기준 z-score
  trip_window: 50                    # Trip 탐지 롤링 윈도우 (샘플 수)
  trip_roc_zscore_threshold: 4.0     # 변화율 z-score 임계값 (높을수록 민감도 낮음)
  top_n_signals: 5                   # 리포트에 표시할 급변 신호 최대 수

# 신호 역할 파악 후 아래 주석 해제 (Trip 정밀 탐지용)
# known_roles:
#   speed_actual: "DB420.DBW 2"
#   current_actual: "DB420.DBW 4"
#   position: "MD    110"
```

- [ ] **Step 5: 패키지 설치 확인**

```bash
pip install -r requirements.txt
python -c "import pandas, numpy, watchdog, yaml, plyer; print('OK')"
```
Expected: `OK`

- [ ] **Step 6: commit**

```bash
git add requirements.txt config/ src/__init__.py src/data/__init__.py \
        src/analysis/__init__.py src/alarm/__init__.py src/report/__init__.py \
        src/watcher/__init__.py
git commit -m "chore: initial project setup with config files and package structure"
```

---

## Task 2: 데이터 스키마 정의

**Files:**
- Create: `src/data/schema.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/conftest.py
import pandas as pd
import numpy as np
import pytest

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """15초 분량의 샘플 데이터 (10ms 샘플링)"""
    n = 1500
    t = pd.date_range("2026-03-20 15:49:00", periods=n, freq="10ms")
    return pd.DataFrame({
        "timestamp": t,
        "Speed_Act": np.concatenate([
            np.linspace(0, 50, 500),   # 가속
            np.full(700, 50),           # 정상 이동
            np.linspace(50, 0, 300),    # 감속 (Trip 전)
        ]),
        "Position_Act": np.cumsum(np.random.uniform(0.05, 0.15, n)),
        "Current_1": np.concatenate([
            np.random.normal(100, 5, 500),
            np.random.normal(80, 5, 500),
            np.random.normal(180, 10, 500),  # 전류 급증 (Trip)
        ]),
    }).set_index("timestamp")
```

```python
# tests/test_loader.py (Step 1용 skeleton)
from src.data.schema import SignalSchema

def test_signal_schema_has_required_fields():
    schema = SignalSchema(name="Speed_Act", unit="m/min", signal_type="analog")
    assert schema.name == "Speed_Act"
    assert schema.signal_type == "analog"
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/test_loader.py::test_signal_schema_has_required_fields -v
```
Expected: FAIL (ImportError)

- [ ] **Step 3: schema.py 구현**

```python
# src/data/schema.py
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class AlarmThreshold:
    high: float = float("inf")
    low: float = float("-inf")
    rate_of_change: float = float("inf")

@dataclass
class SignalSchema:
    name: str
    unit: str
    signal_type: Literal["analog", "digital"]
    alarm: AlarmThreshold = field(default_factory=AlarmThreshold)
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_loader.py::test_signal_schema_has_required_fields -v
```
Expected: PASS

- [ ] **Step 5: commit**

```bash
git add src/data/schema.py tests/
git commit -m "feat: add SignalSchema dataclass and test fixtures"
```

---

## Task 3: iba CSV 로더

**Files:**
- Create: `src/data/loader.py`
- Modify: `tests/test_loader.py`

iba Analyzer CSV 특성:
- 첫 번째 행: 신호명 (헤더)
- 두 번째 행: 단위 (있는 경우)
- 세 번째 행부터: 데이터 (첫 컬럼 = 타임스탬프)
- 파일 크기가 매우 크므로 청크 읽기 사용

- [ ] **Step 1: 실패하는 테스트 작성**

실제 iba PDA CSV 포맷을 모사한 테스트를 작성한다.

```python
# tests/test_loader.py (추가)
import pandas as pd
from src.data.loader import IbaCSVLoader

def _make_iba_csv(tmp_path, signal_names: list[str], n_rows: int = 5) -> str:
    """실제 iba PDA CSV 구조를 모사한 파일 생성."""
    sep = ";"
    header0 = sep.join(["Time"] + [f"[0:{i}]" for i in range(len(signal_names))])
    header1 = sep.join(["time"] + signal_names)
    unit_row = sep.join(["sec"] + [""] * len(signal_names))
    lines = [
        header0, header1, unit_row,
        "",  # 빈 행
        "Group_imageIndex_0;-1",
        "LicenseCustomer;TestPlant",
        "",  # 빈 행 (데이터 직전)
    ]
    base = pd.Timestamp("2026-03-12T15:49:15+09:00")
    for i in range(n_rows):
        ts = (base + pd.Timedelta(milliseconds=i * 100)).isoformat()
        vals = sep.join([ts] + [str(float(i * 10)) for _ in signal_names])
        lines.append(vals)
    path = tmp_path / "test_iba.csv"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)

def test_loader_returns_dataframe(tmp_path):
    path = _make_iba_csv(tmp_path, ["MD    110", "DB420.DBW 0"])
    loader = IbaCSVLoader()
    df = loader.load(path)
    assert isinstance(df, pd.DataFrame)
    assert "MD    110" in df.columns
    assert "DB420.DBW 0" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)

def test_loader_skips_metadata_rows(tmp_path):
    """Row 0 (iba 주소), 단위행, Group_imageIndex 등이 데이터에 포함되지 않아야 함."""
    path = _make_iba_csv(tmp_path, ["MD    110"], n_rows=3)
    loader = IbaCSVLoader()
    df = loader.load(path)
    # Group_imageIndex 같은 메타데이터 행이 데이터에 섞이면 행 수가 달라짐
    assert len(df) == 3

def test_loader_timestamp_is_index(tmp_path):
    path = _make_iba_csv(tmp_path, ["MD    110"])
    loader = IbaCSVLoader()
    df = loader.load(path)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index[0].year == 2026

def test_loader_large_file_chunked(tmp_path):
    """청크 읽기가 동일한 행 수를 반환하는지 확인."""
    path = _make_iba_csv(tmp_path, ["MD    110", "DB420.DBW 0"], n_rows=5000)
    loader = IbaCSVLoader(chunksize=1000)
    df = loader.load(path)
    assert len(df) == 5000
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/test_loader.py -v
```
Expected: 3개 FAIL (ImportError)

- [ ] **Step 3: loader.py 구현**

실제 확인된 iba PDA CSV 구조:
- Row 0: iba 채널 주소 (`Time;[0:0];[0:1];...`) → SKIP
- Row 1: PLC 태그명 (`time;MB 118;MD 110;DB420.DBW 0;...`) → **HEADER로 사용**
- Row 2: 단위 (`sec;;;...`) → SKIP
- Row 3: 빈 행 → SKIP
- Row 4~N: 메타데이터 (`Group_imageIndex_X;값`, `LicenseId;...` 등) → SKIP
- Row N+1: 빈 행 → SKIP
- Row N+2+: 실제 데이터 (`2026-03-12T15:49:15.730000+09:00;값;...`)
- 구분자: `;` (세미콜론)
- 타임스탬프: ISO 8601 형식 (`2026-03-12T15:49:15.730000+09:00`)

```python
# src/data/loader.py
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

_ISO_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


class IbaCSVLoader:
    """iba Analyzer 내보내기 CSV 로더 (Siemens S7 PLC, 세미콜론 구분).

    실제 iba PDA CSV 구조:
        Row 0   : iba 채널 주소  → SKIP
        Row 1   : PLC 태그명    → HEADER
        Row 2   : 단위 행       → SKIP
        Row 3~N : 메타데이터    → SKIP (Group_imageIndex, License 등)
        Row N+1+: 실제 데이터 (ISO 타임스탬프로 시작하는 첫 행부터)

    메모리 주의: concat 방식은 전체 DataFrame을 RAM에 올린다.
    파일이 수백 MB 이상이면 OOM 발생 가능. 해결책은 resample()로
    다운샘플링하거나 분석 구간을 좁히는 것.
    """

    SEP = ";"

    def __init__(self, chunksize: int = 50_000, encoding: str = "utf-8-sig"):
        self.chunksize = chunksize
        self.encoding = encoding

    def load(self, filepath: str) -> pd.DataFrame:
        path = Path(filepath)
        data_start = self._find_data_start(path)
        # Row 1 = PLC 태그명 헤더. skiprows = [0] + [2 .. data_start-1]
        skip = [0] + list(range(2, data_start))

        # 청크 읽기 후 concat → 전체 DataFrame을 메모리에 올림
        chunks = pd.read_csv(
            path,
            sep=self.SEP,
            skiprows=skip,
            header=0,
            chunksize=self.chunksize,
            encoding=self.encoding,
            low_memory=False,
        )
        df = pd.concat(chunks, ignore_index=True)
        return self._set_timestamp_index(df)

    def _find_data_start(self, path: Path) -> int:
        """ISO 타임스탬프로 시작하는 첫 번째 행 번호를 반환."""
        with open(path, encoding=self.encoding, errors="replace") as f:
            for i, line in enumerate(f):
                first = line.split(self.SEP)[0].strip()
                if _ISO_TS_RE.match(first):
                    return i
        return 23  # fallback

    def _set_timestamp_index(self, df: pd.DataFrame) -> pd.DataFrame:
        ts_col = df.columns[0]
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        return df.set_index(ts_col)
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_loader.py -v
```
Expected: 3개 PASS

- [ ] **Step 5: commit**

```bash
git add src/data/loader.py tests/test_loader.py
git commit -m "feat: implement IbaCSVLoader with chunked reading and unit-row detection"
```

---

## Task 4: 전처리기

**Files:**
- Create: `src/analysis/preprocessor.py`
- Create: `tests/test_preprocessor.py`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_preprocessor.py
import pandas as pd
import numpy as np
import pytest
from src.analysis.preprocessor import fill_missing, to_numeric_columns, resample

@pytest.fixture
def df_with_gaps():
    t = pd.date_range("2026-03-20 15:00", periods=5, freq="100ms")
    return pd.DataFrame({"Speed_Act": [1.0, None, None, 4.0, 5.0]}, index=t)

def test_fill_missing_forward_fills_nans(df_with_gaps):
    result = fill_missing(df_with_gaps)
    assert result["Speed_Act"].isna().sum() == 0
    assert result["Speed_Act"].iloc[1] == 1.0  # ffill

def test_to_numeric_converts_string_numbers():
    t = pd.date_range("2026-03-20", periods=3, freq="100ms")
    df = pd.DataFrame({"Speed_Act": ["10.0", "20.0", "bad"]}, index=t)
    result = to_numeric_columns(df, ["Speed_Act"])
    assert result["Speed_Act"].dtype == float
    assert pd.isna(result["Speed_Act"].iloc[2])

def test_to_numeric_ignores_missing_columns():
    t = pd.date_range("2026-03-20", periods=2, freq="100ms")
    df = pd.DataFrame({"Speed_Act": [1.0, 2.0]}, index=t)
    result = to_numeric_columns(df, ["Speed_Act", "NonExistent"])
    assert "NonExistent" not in result.columns

def test_resample_reduces_rows(df_with_gaps):
    filled = fill_missing(df_with_gaps)  # 100ms 간격
    result = resample(filled, freq="200ms")
    assert len(result) < len(filled)
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/test_preprocessor.py -v
```
Expected: 4개 FAIL (ImportError)

- [ ] **Step 3: preprocessor.py 구현**

```python
# src/analysis/preprocessor.py
import pandas as pd


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """앞 값으로 결측치 채우기 (PLC hold 특성 반영)."""
    return df.ffill().bfill()


def to_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """지정 컬럼을 숫자형으로 강제 변환. 변환 실패 → NaN."""
    result = df.copy()
    for col in columns:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def resample(df: pd.DataFrame, freq: str = "100ms") -> pd.DataFrame:
    """타임스탬프 인덱스 기준 리샘플링 (평균값).
    고주파 데이터를 분석 가능한 해상도로 낮출 때 사용."""
    return df.resample(freq).mean()
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_preprocessor.py -v
```
Expected: 4개 PASS

- [ ] **Step 5: commit**

```bash
git add src/analysis/preprocessor.py tests/test_preprocessor.py
git commit -m "feat: add preprocessor utilities with TDD"
```

---

## Task 5: 신호 자동 분류 (Signal Discovery)

**Files:**
- Create: `src/analysis/signal_discovery.py`
- Create: `tests/test_signal_discovery.py`

신호 역할(속도/전류/위치)을 사전에 알 수 없으므로, 데이터의 통계적 특성으로
수치형 신호를 분류하고 `signals.yaml`의 패턴 설정을 적용한다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_signal_discovery.py
import pandas as pd
import numpy as np
import pytest
from src.analysis.signal_discovery import (
    classify_numeric_columns,
    filter_by_patterns,
    SignalGroup,
)

@pytest.fixture
def wide_df():
    """MD / DB420 / MB 혼합 컬럼 DataFrame."""
    n = 500
    t = pd.date_range("2026-03-12 15:49:15", periods=n, freq="100ms", tz="Asia/Seoul")
    return pd.DataFrame({
        "MD    110": np.random.uniform(0, 100, n),   # 수치형 아날로그
        "DB420.DBW 0": np.random.uniform(0, 200, n),
        "MB    118": np.random.randint(0, 256, n).astype(float),  # 바이트
        "MW    200": np.random.randint(0, 65535, n).astype(float),
    }, index=t)

def test_filter_by_patterns_returns_matching_columns(wide_df):
    matched = filter_by_patterns(wide_df, patterns=["MD ", "DB420"])
    assert "MD    110" in matched
    assert "DB420.DBW 0" in matched
    assert "MB    118" not in matched

def test_classify_returns_signal_groups(wide_df):
    groups = classify_numeric_columns(wide_df)
    assert isinstance(groups, list)
    assert all(isinstance(g, SignalGroup) for g in groups)

def test_classify_detects_binary_columns():
    n = 300
    t = pd.date_range("2026-03-12 15:49", periods=n, freq="100ms")
    df = pd.DataFrame({
        "IB 4": np.random.randint(0, 2, n).astype(float),  # binary
        "MD 110": np.random.uniform(10, 100, n),            # continuous
    }, index=t)
    groups = classify_numeric_columns(df)
    binary_names = [g.name for g in groups if g.is_binary]
    continuous_names = [g.name for g in groups if not g.is_binary]
    assert "IB 4" in binary_names
    assert "MD 110" in continuous_names
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/test_signal_discovery.py -v
```
Expected: 3개 FAIL (ImportError)

- [ ] **Step 3: signal_discovery.py 구현**

```python
# src/analysis/signal_discovery.py
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class SignalGroup:
    name: str           # 컬럼명 (PLC 태그명)
    is_binary: bool     # True = 0/1 디지털, False = 아날로그
    unique_count: int
    mean: float
    std: float


def filter_by_patterns(df: pd.DataFrame, patterns: list[str]) -> list[str]:
    """컬럼명에 패턴 문자열이 포함된 컬럼명 목록 반환."""
    return [c for c in df.columns if any(p in c for p in patterns)]


def classify_numeric_columns(df: pd.DataFrame) -> list[SignalGroup]:
    """수치형 컬럼을 이진(디지털) / 연속(아날로그)으로 자동 분류."""
    groups: list[SignalGroup] = []
    for col in df.select_dtypes(include="number").columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        unique_vals = series.nunique()
        # 고유값 2개 이하 or 값이 모두 0/1 범위 → 이진 신호
        is_binary = unique_vals <= 2 or (series.min() >= 0 and series.max() <= 1)
        groups.append(SignalGroup(
            name=col,
            is_binary=bool(is_binary),
            unique_count=int(unique_vals),
            mean=float(series.mean()),
            std=float(series.std()),
        ))
    return groups
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_signal_discovery.py -v
```
Expected: 3개 PASS

- [ ] **Step 5: commit**

```bash
git add src/analysis/signal_discovery.py tests/test_signal_discovery.py
git commit -m "feat: add signal discovery for auto-classifying PLC columns without prior knowledge"
```

---

## Task 6: 통계적 이상치 탐지

**Files:**
- Create: `src/analysis/statistical.py`
- Create: `tests/test_statistical.py`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_statistical.py
import pandas as pd
import numpy as np
import pytest
from src.analysis.statistical import detect_anomalies, AnomalyResult

@pytest.fixture
def normal_series():
    np.random.seed(42)
    t = pd.date_range("2026-03-20 15:00", periods=3000, freq="100ms")
    return pd.Series(np.random.normal(50, 5, 3000), index=t, name="Speed_Act")

@pytest.fixture
def spike_series(normal_series):
    s = normal_series.copy()
    s.iloc[1500] = 200.0  # 명확한 스파이크
    return s

def test_no_anomalies_in_normal_data(normal_series):
    result = detect_anomalies(normal_series, zscore_threshold=3.0)
    assert isinstance(result, AnomalyResult)
    assert len(result.anomaly_indices) < 5  # 정규분포에서 3σ 초과는 극소수

def test_detects_spike(spike_series):
    result = detect_anomalies(spike_series, zscore_threshold=3.0)
    assert 1500 in result.anomaly_indices

def test_result_has_stats(normal_series):
    result = detect_anomalies(normal_series, zscore_threshold=3.0)
    assert result.mean is not None
    assert result.std > 0
    assert 0.0 <= result.anomaly_ratio <= 1.0
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/test_statistical.py -v
```
Expected: 3개 FAIL

- [ ] **Step 3: statistical.py 구현**

```python
# src/analysis/statistical.py
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class AnomalyResult:
    signal_name: str
    anomaly_indices: list[int]
    mean: float
    std: float
    anomaly_ratio: float
    zscore_threshold: float


def detect_anomalies(series: pd.Series, zscore_threshold: float = 3.0) -> AnomalyResult:
    """Rolling z-score 기반 이상치 탐지.
    
    전체 평균/표준편차 대신 rolling window를 쓰면 트렌드가 있는 신호에도 적용 가능.
    여기서는 단순 전체 통계 사용 (MVP).
    """
    clean = series.dropna()
    mean = float(clean.mean())
    std = float(clean.std())

    if std == 0:
        return AnomalyResult(
            signal_name=series.name or "",
            anomaly_indices=[],
            mean=mean,
            std=0.0,
            anomaly_ratio=0.0,
            zscore_threshold=zscore_threshold,
        )

    zscores = (clean - mean) / std
    anomaly_mask = zscores.abs() > zscore_threshold
    anomaly_positions = [int(clean.index.get_loc(idx)) for idx in clean.index[anomaly_mask]]

    return AnomalyResult(
        signal_name=series.name or "",
        anomaly_indices=anomaly_positions,
        mean=mean,
        std=std,
        anomaly_ratio=float(anomaly_mask.mean()),
        zscore_threshold=zscore_threshold,
    )
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_statistical.py -v
```
Expected: 3개 PASS

- [ ] **Step 5: commit**

```bash
git add src/analysis/statistical.py tests/test_statistical.py
git commit -m "feat: implement z-score anomaly detection with AnomalyResult"
```

---

## Task 6: Trip 이벤트 탐지

**Files:**
- Create: `src/analysis/trip_detector.py`
- Create: `tests/test_trip_detector.py`

코크스공장 OVEN 이동차 Trip 판단 기준:
1. **급변 탐지**: 아날로그 신호 중 단시간 내 변화량이 최대인 구간 탐지
2. **동시 급변**: 여러 신호가 동일 시점에 급변하면 Trip 가능성 높음
3. **신호 역할 확정 후**: `known_roles` 설정이 있으면 해당 신호로 정밀 탐지

> 이 탐지기는 사전에 어떤 신호가 속도/전류인지 몰라도 작동한다.
> Trip 분석 리포트에 "가장 급변한 신호 Top N"을 출력하면 운영자가 역할을 확정할 수 있다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_trip_detector.py
import pandas as pd
import numpy as np
import pytest
from src.analysis.trip_detector import TripDetector, TripEvent

@pytest.fixture
def trip_df():
    """Trip이 발생하는 시뮬레이션 데이터. 신호명은 실제 PLC 태그 형식 사용."""
    np.random.seed(0)
    n = 6000  # 10분, 100ms 샘플링
    t = pd.date_range("2026-03-12 15:40:00", periods=n, freq="100ms", tz="Asia/Seoul")
    # DB420.DBW 2 = 속도 역할 (이 테스트에서만 알고 있음, 실제는 모름)
    speed_sig = np.concatenate([
        np.full(3000, 400.0),          # 정상 이동
        np.linspace(400, 0, 50),       # Trip 순간 급감
        np.full(n - 3050, 0.0),        # 정지
    ])
    # DB420.DBW 4 = 전류 역할
    current_sig = np.concatenate([
        np.random.normal(800, 50, 3000),
        np.random.normal(1800, 50, 50),   # 전류 급증
        np.random.normal(50, 20, n - 3050),
    ])
    return pd.DataFrame({
        "DB420.DBW 2": speed_sig,
        "DB420.DBW 4": current_sig,
        "MD    110": np.random.normal(5000, 10, n),  # 위치 (느리게 변함)
    }, index=t)

def test_detects_trip_in_trip_data(trip_df):
    """신호명 지정 없이 자동 탐지."""
    detector = TripDetector()
    events = detector.detect(trip_df)
    assert len(events) >= 1

def test_trip_event_has_required_fields(trip_df):
    detector = TripDetector()
    events = detector.detect(trip_df)
    event = events[0]
    assert isinstance(event, TripEvent)
    assert event.timestamp is not None
    assert len(event.top_changed_signals) >= 1  # 급변 신호 목록 포함
    assert event.trigger in ("multi_signal_drop", "single_signal_drop", "spike")

def test_no_trip_in_normal_data():
    np.random.seed(42)
    n = 3000
    t = pd.date_range("2026-03-12 15:00", periods=n, freq="100ms", tz="Asia/Seoul")
    df = pd.DataFrame({
        "DB420.DBW 2": np.random.normal(400, 5, n),   # 안정적인 신호
        "DB420.DBW 4": np.random.normal(800, 20, n),
    }, index=t)
    detector = TripDetector()
    events = detector.detect(df)
    assert len(events) == 0
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/test_trip_detector.py -v
```
Expected: 3개 FAIL

- [ ] **Step 3: trip_detector.py 구현**

```python
# src/analysis/trip_detector.py
"""Trip 이벤트 탐지기. 사전에 신호 역할(속도/전류)을 몰라도 동작한다.

알고리즘:
1. 모든 아날로그(연속) 신호에 대해 rolling rate-of-change 계산
2. 변화율이 threshold를 초과하는 신호 중, 여러 신호가 동시에 급변하는 구간 탐지
3. 급변 구간을 TripEvent로 기록하고, 가장 많이 변한 신호 Top N을 포함
"""
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
import pandas as pd


@dataclass
class TripEvent:
    timestamp: pd.Timestamp
    trigger: Literal["multi_signal_drop", "single_signal_drop", "spike"]
    top_changed_signals: list[tuple[str, float]]  # (컬럼명, 변화량)
    description: str
    window_size: int


class TripDetector:
    def __init__(
        self,
        window: int = 50,              # 급변 탐지 윈도우 (샘플 수)
        roc_zscore_threshold: float = 4.0,  # 변화율 z-score 임계값
        min_signals_for_multi: int = 2,     # 동시 급변 최소 신호 수
        top_n: int = 5,                     # 리포트에 표시할 Top N 신호
        known_speed_col: str | None = None,  # signals.yaml known_roles에서 설정
        known_current_col: str | None = None,
    ):
        self.window = window
        self.roc_zscore_threshold = roc_zscore_threshold
        self.min_signals_for_multi = min_signals_for_multi
        self.top_n = top_n
        self.known_speed_col = known_speed_col
        self.known_current_col = known_current_col

    def detect(self, df: pd.DataFrame) -> list[TripEvent]:
        # 아날로그(연속) 신호만 선택: 고유값 > 10개인 수치형 컬럼
        analog_cols = [
            c for c in df.select_dtypes(include="number").columns
            if df[c].nunique() > 10
        ]
        if not analog_cols:
            return []

        # 각 신호의 rolling 변화율 계산
        roc = df[analog_cols].diff(self.window).abs()

        # 각 신호별 z-score 계산 (열 단위)
        means = roc.mean()
        stds = roc.std().replace(0, np.nan)
        zscores = (roc - means) / stds

        # 어떤 신호든 z-score 초과인 시점 탐지
        any_spike = (zscores > self.roc_zscore_threshold).any(axis=1)
        # 최초 발생 구간만 (연속된 True → 첫 번째만)
        rising_edge = any_spike & ~any_spike.shift(1).fillna(False)
        trip_times = df.index[rising_edge]

        events: list[TripEvent] = []
        for ts in trip_times:
            pos = df.index.get_loc(ts)
            # 해당 시점에서 가장 많이 변한 신호 Top N
            if pos < len(roc):
                row_roc = roc.iloc[pos]
                top = row_roc.nlargest(self.top_n)
                top_signals = [(col, float(val)) for col, val in top.items() if val > 0]
            else:
                top_signals = []

            n_spiked = int((zscores.iloc[pos] > self.roc_zscore_threshold).sum()) if pos < len(zscores) else 0
            if n_spiked >= self.min_signals_for_multi:
                trigger: Literal["multi_signal_drop", "single_signal_drop", "spike"] = "multi_signal_drop"
            else:
                trigger = "single_signal_drop"

            sig_names = ", ".join(s[0] for s in top_signals[:3])
            desc = (
                f"{n_spiked}개 신호 동시 급변 탐지. 주요 신호: {sig_names}. "
                "Trip 또는 급격한 상태 변화 가능성."
            )
            events.append(TripEvent(
                timestamp=ts,
                trigger=trigger,
                top_changed_signals=top_signals,
                description=desc,
                window_size=self.window,
            ))

        return events
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_trip_detector.py -v
```
Expected: 3개 PASS

- [ ] **Step 5: commit**

```bash
git add src/analysis/trip_detector.py tests/test_trip_detector.py
git commit -m "feat: implement TripDetector with speed-drop and current-spike detection"
```

---

## Task 7: 고장 분류기

**Files:**
- Create: `src/analysis/fault_classifier.py`
- Create: `tests/test_fault_classifier.py`

Trip 이후 어떤 점검을 해야 하는지 룰 기반으로 분류한다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_fault_classifier.py
import pytest
from src.analysis.fault_classifier import FaultClassifier, FaultReport
from src.analysis.trip_detector import TripEvent
import pandas as pd

@pytest.fixture
def current_spike_event():
    return TripEvent(
        timestamp=pd.Timestamp("2026-03-20 15:49:30"),
        trigger="current_spike",
        speed_before=40.0,
        speed_after=0.0,
        current_peak=210.0,
        description="전류 급증",
    )

@pytest.fixture
def speed_drop_event():
    return TripEvent(
        timestamp=pd.Timestamp("2026-03-20 15:49:30"),
        trigger="speed_drop",
        speed_before=35.0,
        speed_after=0.0,
        current_peak=85.0,
        description="속도 급감",
    )

def test_current_spike_suggests_overload(current_spike_event):
    clf = FaultClassifier()
    report = clf.classify(current_spike_event)
    assert isinstance(report, FaultReport)
    assert "전류" in report.fault_type or "과부하" in report.fault_type

def test_speed_drop_suggests_interlock(speed_drop_event):
    clf = FaultClassifier()
    report = clf.classify(speed_drop_event)
    assert "인터록" in report.fault_type or "비상정지" in report.fault_type

def test_report_has_inspection_items(current_spike_event):
    clf = FaultClassifier()
    report = clf.classify(current_spike_event)
    assert len(report.inspection_items) > 0
    assert all(isinstance(item, str) for item in report.inspection_items)
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/test_fault_classifier.py -v
```
Expected: 3개 FAIL

- [ ] **Step 3: fault_classifier.py 구현**

```python
# src/analysis/fault_classifier.py
from dataclasses import dataclass, field
from src.analysis.trip_detector import TripEvent


@dataclass
class FaultReport:
    fault_type: str
    severity: str            # "경고" / "주의" / "정보"
    inspection_items: list[str]
    root_cause_hypothesis: str


_RULES: list[dict] = [
    {
        "trigger": "current_spike",
        "fault_type": "과부하 / 전류 이상",
        "severity": "경고",
        "inspection_items": [
            "인버터 알람 코드 확인 및 기록",
            "모터 절연 저항 측정 (1MΩ 이상)",
            "동력 케이블 연결 상태 및 단락 여부 점검",
            "기계적 잠김(jam) 또는 이동 경로 내 장애물 확인",
            "냉각팬 및 방열판 청결 상태 확인",
        ],
        "hypothesis": "모터 과부하 또는 기계적 저항 증가로 보호 회로 동작.",
    },
    {
        "trigger": "speed_drop",
        "fault_type": "비상정지 / 인터록 동작",
        "severity": "주의",
        "inspection_items": [
            "PLC 인터록 비트 래치 상태 확인",
            "비상정지 버튼 눌림 여부 확인",
            "위치 리밋 스위치 동작 상태 확인",
            "안전 도어 및 가드 상태 점검",
            "통신 이상 여부 확인 (PLC-인버터 간)",
        ],
        "hypothesis": "인터록 조건 만족 또는 비상정지 입력에 의한 정상 보호 동작.",
    },
    {
        "trigger": "combined",
        "fault_type": "이동 중 과부하 Trip",
        "severity": "경고",
        "inspection_items": [
            "이동 경로 내 장애물 및 기계적 저항 확인",
            "구동 롤러/바퀴 마모 및 윤활 상태 점검",
            "인버터 열화 여부 및 출력 전압 파형 확인",
            "모터 베어링 온도 및 진동 점검",
            "PLC 인터록 래치 이력 확인",
        ],
        "hypothesis": "이동 중 기계적 과부하 또는 이물질 끼임으로 전류 초과 → 보호 정지.",
    },
]


class FaultClassifier:
    def classify(self, event: TripEvent) -> FaultReport:
        rule = next((r for r in _RULES if r["trigger"] == event.trigger), _RULES[1])
        return FaultReport(
            fault_type=rule["fault_type"],
            severity=rule["severity"],
            inspection_items=rule["inspection_items"],
            root_cause_hypothesis=rule["hypothesis"],
        )
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_fault_classifier.py -v
```
Expected: 3개 PASS

- [ ] **Step 5: commit**

```bash
git add src/analysis/fault_classifier.py tests/test_fault_classifier.py
git commit -m "feat: implement rule-based FaultClassifier with inspection checklists"
```

---

## Task 8: 알람 및 리포트 생성

**Files:**
- Create: `src/alarm/threshold_config.py`
- Create: `src/alarm/notifier.py`
- Create: `src/report/template_reporter.py`
- Create: `tests/test_reporter.py`
- Create: `tests/test_threshold_config.py`

- [ ] **Step 1: 실패하는 테스트 작성 (threshold_config + reporter)**

```python
# tests/test_threshold_config.py
import pytest
from pathlib import Path
from src.alarm.threshold_config import load_signal_schemas

def test_load_returns_empty_for_missing_file():
    result = load_signal_schemas("nonexistent.yaml")
    assert result == {}

def test_load_returns_schemas_from_yaml(tmp_path):
    yaml_content = """
signals:
  speed:
    columns: ["Speed_Act"]
    unit: "m/min"
    type: analog
    alarm:
      high: 100.0
      low: -100.0
"""
    cfg_file = tmp_path / "signals.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")
    result = load_signal_schemas(str(cfg_file))
    assert "Speed_Act" in result
    assert result["Speed_Act"].alarm.high == 100.0
```

```python
# tests/test_reporter.py
import pytest
from src.report.template_reporter import generate_report
from src.analysis.trip_detector import TripEvent
from src.analysis.fault_classifier import FaultClassifier
from src.analysis.statistical import AnomalyResult
import pandas as pd

def test_report_contains_trip_info():
    event = TripEvent(
        timestamp=pd.Timestamp("2026-03-20 15:49:30"),
        trigger="current_spike",
        speed_before=40.0, speed_after=0.0,
        current_peak=210.0, description="전류 급증"
    )
    fault = FaultClassifier().classify(event)
    anomaly = AnomalyResult("Speed_Act", [100, 200], 40.0, 5.0, 0.01, 3.0)
    report = generate_report("test.csv", [event], {"Speed_Act": anomaly}, [fault])
    assert "Trip #1" in report
    assert "15:49:30" in report
    assert len(report) > 200

def test_empty_report():
    report = generate_report("test.csv", [], {}, [])
    assert "Trip 이벤트 없음" in report
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/test_threshold_config.py tests/test_reporter.py -v
```
Expected: 4개 FAIL (ImportError)

- [ ] **Step 3: threshold_config.py 구현**

```python
# src/alarm/threshold_config.py
from pathlib import Path
import yaml
from src.data.schema import SignalSchema, AlarmThreshold


def load_signal_schemas(config_path: str = "config/signals.yaml") -> dict[str, SignalSchema]:
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    schemas: dict[str, SignalSchema] = {}
    for group_name, group in (raw.get("signals") or {}).items():
        for col in group.get("columns", []):
            alarm_cfg = group.get("alarm", {})
            schemas[col] = SignalSchema(
                name=col,
                unit=group.get("unit", ""),
                signal_type=group.get("type", "analog"),
                alarm=AlarmThreshold(
                    high=alarm_cfg.get("high", float("inf")),
                    low=alarm_cfg.get("low", float("-inf")),
                    rate_of_change=alarm_cfg.get("rate_of_change", float("inf")),
                ),
            )
    return schemas
```

- [ ] **Step 4: notifier.py 구현**

```python
# src/alarm/notifier.py
import logging
from pathlib import Path
from datetime import datetime


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("pims_alarm")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return logger


def send_alarm(title: str, message: str, log_file: str, use_toast: bool = True) -> None:
    logger = setup_logger(log_file)
    logger.warning(f"{title}: {message}")

    if use_toast:
        try:
            from plyer import notification
            notification.notify(title=title, message=message[:256], timeout=10)
        except Exception:
            pass  # plyer 미설치 시 무시
```

- [ ] **Step 5: template_reporter.py 구현**

```python
# src/report/template_reporter.py
from datetime import datetime
from pathlib import Path
from src.analysis.trip_detector import TripEvent
from src.analysis.fault_classifier import FaultReport
from src.analysis.statistical import AnomalyResult


def generate_report(
    source_file: str,
    trip_events: list[TripEvent],
    anomalies: dict[str, AnomalyResult],
    fault_reports: list[FaultReport],
) -> str:
    lines = [
        "=" * 60,
        "  PIMS 이동차 자동 분석 리포트",
        f"  생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  분석 파일: {Path(source_file).name}",
        "=" * 60,
        "",
    ]

    if not trip_events:
        lines.append("[ Trip 이벤트 없음 ] 분석 구간에서 Trip이 탐지되지 않았습니다.")
    else:
        lines.append(f"[ Trip 이벤트 {len(trip_events)}건 탐지 ]")
        for i, (event, report) in enumerate(zip(trip_events, fault_reports), 1):
            lines += [
                "",
                f"── Trip #{i} ──────────────────────────────",
                f"  발생 시각 : {event.timestamp}",
                f"  고장 유형 : {report.fault_type}",
                f"  심각도    : {report.severity}",
                f"  발생 전 속도: {event.speed_before:.1f} m/min → {event.speed_after:.1f} m/min",
                f"  전류 최대값: {event.current_peak:.1f} A",
                "",
                f"  [추정 원인]",
                f"  {report.root_cause_hypothesis}",
                "",
                "  [점검 항목]",
            ]
            for j, item in enumerate(report.inspection_items, 1):
                lines.append(f"    {j}. {item}")

    if anomalies:
        lines += ["", "[ 이상 신호 요약 ]"]
        for sig_name, result in anomalies.items():
            if result.anomaly_ratio > 0:
                lines.append(
                    f"  - {sig_name}: 이상치 {result.anomaly_ratio*100:.1f}% "
                    f"(평균 {result.mean:.2f}, 표준편차 {result.std:.2f})"
                )

    lines += ["", "=" * 60]
    return "\n".join(lines)


def save_report(report_text: str, output_folder: str, source_filename: str) -> str:
    folder = Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)
    stem = Path(source_filename).stem
    out_path = folder / f"{stem}_report_{datetime.now().strftime('%H%M%S')}.txt"
    out_path.write_text(report_text, encoding="utf-8")
    return str(out_path)
```

- [ ] **Step 6: 테스트 통과 확인**

```bash
pytest tests/test_threshold_config.py tests/test_reporter.py -v
```
Expected: 4개 PASS

- [ ] **Step 7: commit**

```bash
git add src/alarm/ src/report/template_reporter.py \
        tests/test_reporter.py tests/test_threshold_config.py
git commit -m "feat: add threshold config, notifier, and template report generator"
```

---

## Task 9: 파이프라인 조립

**Files:**
- Create: `src/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: 실패하는 통합 테스트 작성**

```python
# tests/test_pipeline.py
import pandas as pd
import numpy as np
from pathlib import Path
from src.pipeline import run_analysis


def _make_trip_csv(path: Path) -> None:
    """Trip이 포함된 최소 iba CSV 파일 생성."""
    n = 600  # 60초, 100ms 샘플링
    t = pd.date_range("2026-03-20 15:49:00", periods=n, freq="100ms")
    speed = np.concatenate([np.full(300, 40.0), np.linspace(40, 0, 50), np.full(250, 0.0)])
    current = np.concatenate([np.full(300, 80.0), np.full(50, 200.0), np.full(250, 5.0)])
    df = pd.DataFrame({"timestamp": t, "Speed_Act": speed, "Current_1": current})
    lines = ["timestamp,Speed_Act,Current_1"] + [
        f"{row.timestamp},{row.Speed_Act:.2f},{row.Current_1:.2f}"
        for _, row in df.iterrows()
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_pipeline_creates_report_file(tmp_path):
    csv_file = tmp_path / "test_trip.csv"
    _make_trip_csv(csv_file)
    output_folder = str(tmp_path / "output")
    log_file = str(tmp_path / "alarm.log")

    report_path = run_analysis(
        csv_path=str(csv_file),
        output_folder=output_folder,
        log_file=log_file,
        config_path="config/signals.yaml",
    )

    assert Path(report_path).exists()
    content = Path(report_path).read_text(encoding="utf-8")
    assert len(content) > 100


def test_pipeline_detects_trip_in_report(tmp_path):
    csv_file = tmp_path / "test_trip.csv"
    _make_trip_csv(csv_file)
    output_folder = str(tmp_path / "output")
    log_file = str(tmp_path / "alarm.log")

    report_path = run_analysis(
        csv_path=str(csv_file),
        output_folder=output_folder,
        log_file=log_file,
        config_path="config/signals.yaml",
    )

    content = Path(report_path).read_text(encoding="utf-8")
    assert "Trip" in content  # Trip 탐지 여부 확인
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/test_pipeline.py -v
```
Expected: 2개 FAIL (ImportError)

- [ ] **Step 3: pipeline.py 구현**

```python
# src/pipeline.py
"""전체 분석 파이프라인: 파일 경로 입력 → 리포트 + 알람 출력."""
from pathlib import Path
from src.data.loader import IbaCSVLoader
from src.analysis.preprocessor import fill_missing, to_numeric_columns
from src.analysis.statistical import detect_anomalies
from src.analysis.trip_detector import TripDetector
from src.analysis.fault_classifier import FaultClassifier
from src.alarm.threshold_config import load_signal_schemas
from src.alarm.notifier import send_alarm
from src.report.template_reporter import generate_report, save_report


def run_analysis(
    csv_path: str,
    output_folder: str,
    log_file: str,
    config_path: str = "config/signals.yaml",
    settings: dict | None = None,
) -> str:
    """CSV 파일 하나를 분석하고 리포트 파일 경로를 반환한다."""
    settings = settings or {}

    # 1. 데이터 로드
    loader = IbaCSVLoader()
    df = loader.load(csv_path)

    # 2. 전처리
    schemas = load_signal_schemas(config_path)
    numeric_cols = [c for c in schemas if c in df.columns]
    df = fill_missing(to_numeric_columns(df, numeric_cols))

    # 3. Trip 탐지 (신호 역할 사전지식 불필요 — 급변 신호 자동 탐지)
    known_roles = settings.get("known_roles", {})
    detector = TripDetector(
        window=settings.get("trip_window", 50),
        roc_zscore_threshold=settings.get("trip_roc_zscore_threshold", 4.0),
        known_speed_col=known_roles.get("speed_actual"),
        known_current_col=known_roles.get("current_actual"),
    )
    trip_events = detector.detect(df)

    # 4. 이상치 탐지
    zscore_threshold = settings.get("zscore_threshold", 3.0)
    anomalies = {
        col: detect_anomalies(df[col], zscore_threshold)
        for col in numeric_cols
        if col in df.columns
    }

    # 5. 고장 분류
    classifier = FaultClassifier()
    fault_reports = [classifier.classify(ev) for ev in trip_events]

    # 6. 리포트 생성 + 저장
    report_text = generate_report(csv_path, trip_events, anomalies, fault_reports)
    report_path = save_report(report_text, output_folder, csv_path)

    # 7. 알람 (Trip 또는 고이상치 비율 신호 존재 시)
    high_anomaly = [k for k, v in anomalies.items() if v.anomaly_ratio > 0.05]
    if trip_events:
        send_alarm(
            title="[PIMS] Trip 이벤트 탐지",
            message=f"{len(trip_events)}건 Trip. 리포트: {report_path}",
            log_file=log_file,
        )
    elif high_anomaly:
        send_alarm(
            title="[PIMS] 이상 신호 감지",
            message=f"이상 신호: {', '.join(high_anomaly[:3])}",
            log_file=log_file,
        )

    return report_path
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_pipeline.py -v
```
Expected: 2개 PASS

- [ ] **Step 5: commit**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "feat: assemble analysis pipeline from loader to alarm"
```

---

## Task 10: 폴더 감시 데몬

**Files:**
- Create: `src/watcher/file_watcher.py`
- Create: `tests/test_watcher.py`
- Create: `main.py`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_watcher.py
import time, threading
from pathlib import Path
import pytest
from src.watcher.file_watcher import CsvWatcher

def _wait_for(condition_fn, timeout: float = 3.0, interval: float = 0.1) -> bool:
    """조건이 True가 될 때까지 최대 timeout초 폴링."""
    elapsed = 0.0
    while elapsed < timeout:
        if condition_fn():
            return True
        time.sleep(interval)
        elapsed += interval
    return False

def test_watcher_calls_callback_on_new_file(tmp_path):
    received = []

    def on_new_file(path: str):
        received.append(path)

    watcher = CsvWatcher(watch_folder=str(tmp_path), callback=on_new_file)
    t = threading.Thread(target=watcher.start, daemon=True)
    t.start()
    time.sleep(0.5)  # 옵저버 시작 대기

    new_file = tmp_path / "test.csv"
    new_file.write_text("timestamp,Speed_Act\n2026-03-20,10.0\n")

    # 최대 3초 폴링 (sleep 고정보다 안정적)
    assert _wait_for(lambda: len(received) >= 1, timeout=3.0), \
        "watchdog이 3초 내에 파일을 감지하지 못함"
    watcher.stop()
    assert "test.csv" in received[0]

def test_watcher_ignores_non_csv(tmp_path):
    received = []
    watcher = CsvWatcher(watch_folder=str(tmp_path), callback=lambda p: received.append(p))
    t = threading.Thread(target=watcher.start, daemon=True)
    t.start()
    time.sleep(0.5)
    (tmp_path / "test.txt").write_text("not csv")
    time.sleep(0.8)
    watcher.stop()
    assert len(received) == 0
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/test_watcher.py -v
```
Expected: 2개 FAIL

- [ ] **Step 3: file_watcher.py 구현**

```python
# src/watcher/file_watcher.py
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent
from pathlib import Path
from typing import Callable
import time


class _CsvHandler(FileSystemEventHandler):
    def __init__(self, callback: Callable[[str], None]):
        self._callback = callback

    def on_created(self, event: FileCreatedEvent) -> None:
        if not event.is_directory and event.src_path.endswith(".csv"):
            self._callback(event.src_path)


class CsvWatcher:
    def __init__(self, watch_folder: str, callback: Callable[[str], None]):
        self._folder = watch_folder
        self._callback = callback
        self._observer = Observer()

    def start(self) -> None:
        handler = _CsvHandler(self._callback)
        self._observer.schedule(handler, self._folder, recursive=False)
        self._observer.start()
        try:
            while self._observer.is_alive():
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        self._observer.stop()
        self._observer.join()
```

- [ ] **Step 4: main.py 구현**

```python
# main.py
"""PIMS 이동차 이상 감지 데몬 진입점."""
import yaml
from pathlib import Path
from src.watcher.file_watcher import CsvWatcher
from src.pipeline import run_analysis


def main() -> None:
    settings_path = Path("config/settings.yaml")
    if not settings_path.exists():
        print("config/settings.yaml 파일이 없습니다. 생성 후 재실행하세요.")
        return

    with open(settings_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    watch_folder = cfg["watch_folder"]
    output_folder = cfg["output_folder"]
    log_file = cfg["log_file"]
    analysis_cfg = cfg.get("analysis", {})

    print(f"[PIMS] 감시 시작: {watch_folder}")
    print("[PIMS] 종료: Ctrl+C")

    def on_new_csv(path: str) -> None:
        print(f"[PIMS] 새 파일 감지: {path}")
        try:
            report_path = run_analysis(
                csv_path=path,
                output_folder=output_folder,
                log_file=log_file,
                settings=analysis_cfg,
            )
            print(f"[PIMS] 리포트 저장: {report_path}")
        except Exception as e:
            print(f"[PIMS] 분석 오류: {e}")

    watcher = CsvWatcher(watch_folder=watch_folder, callback=on_new_csv)
    watcher.start()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 테스트 통과 확인**

```bash
pytest tests/test_watcher.py -v
```
Expected: 2개 PASS

- [ ] **Step 6: 전체 테스트 통과 확인**

```bash
pytest tests/ -v
```
Expected: 전체 PASS

- [ ] **Step 7: commit**

```bash
git add src/watcher/ main.py tests/test_watcher.py
git commit -m "feat: add CsvWatcher daemon and main entry point"
```

---

## Task 11: 실제 데이터로 검증 (수동)

이 Task는 자동화 테스트가 아닌 수동 실행 검증이다.

- [ ] **Step 1: config/signals.yaml 실제 컬럼명으로 수정**

실제 CSV 파일의 헤더를 확인한다:
```bash
python -c "
import pandas as pd
df = pd.read_csv(r'PIMS/2603201549_oven.csv', nrows=2, encoding='utf-8-sig')
print(list(df.columns))
"
```
출력된 컬럼명을 `config/signals.yaml`의 `columns` 항목에 넣는다.

- [ ] **Step 2: 단일 파일 파이프라인 실행**

```bash
python -c "
from src.pipeline import run_analysis
report = run_analysis(
    csv_path=r'PIMS/2603201549_oven.csv',
    output_folder='output/',
    log_file='output/alarm.log',
)
print(open(report, encoding='utf-8').read())
"
```
Expected: 리포트 텍스트 출력, Trip 탐지 여부 확인

- [ ] **Step 3: 결과 검토 및 임계값 조정**

리포트를 보고 `config/settings.yaml`의 `zscore_threshold`, `trip_speed_threshold` 등 수치를 현실에 맞게 조정한다.

- [ ] **Step 4: commit**

```bash
git add config/signals.yaml config/settings.yaml
git commit -m "config: tune signal columns and thresholds from real data"
```

---

## Task 12 (선택): LLM 리포트 연동

> 이 Task는 Ollama가 설치된 후 진행한다. Task 1~11이 완전히 완료된 뒤 추가한다.

**Files:**
- Create: `src/report/llm_reporter.py`
- Modify: `src/pipeline.py` (llm_enabled 조건 추가)

- [ ] **Step 1: Ollama 설치 확인**

```bash
ollama list
```
모델 없으면: `ollama pull llama3.1:8b` 또는 `ollama pull qwen2.5:7b`

- [ ] **Step 2: llm_reporter.py 구현**

```python
# src/report/llm_reporter.py
import ollama
from src.analysis.trip_detector import TripEvent
from src.analysis.fault_classifier import FaultReport


SYSTEM_PROMPT = """당신은 제철소 코크스공장 OVEN 이동차(코크스 압출 설비) 유지보수 전문가입니다.
PLC 데이터 분석 결과를 비전문가가 이해할 수 있도록 명확한 한국어로 설명합니다.
기술 용어는 쉬운 말로 풀어서 설명하고, 점검해야 할 항목을 우선순위 순으로 정리합니다."""


def generate_llm_report(
    trip_events: list[TripEvent],
    fault_reports: list[FaultReport],
    model: str = "llama3.1:8b",
    host: str = "http://localhost:11434",
) -> str:
    if not trip_events:
        return "(LLM) 분석 구간 내 Trip 이벤트가 없어 추가 설명이 필요하지 않습니다."

    context_parts = []
    for i, (ev, fr) in enumerate(zip(trip_events, fault_reports), 1):
        context_parts.append(
            f"Trip #{i}: 시각={ev.timestamp}, 고장유형={fr.fault_type}, "
            f"발생 전 속도={ev.speed_before:.1f}m/min, 전류최대={ev.current_peak:.1f}A\n"
            f"추정원인: {fr.root_cause_hypothesis}"
        )

    user_msg = (
        "다음 PLC 분석 결과를 설비를 모르는 담당자도 이해할 수 있도록 설명해주세요.\n\n"
        + "\n\n".join(context_parts)
        + "\n\n비전문가를 위한 쉬운 설명과 우선 점검 항목을 알려주세요."
    )

    client = ollama.Client(host=host)
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    # ollama SDK 0.2+: response.message.content (속성 접근)
    # ollama SDK 0.1.x: response["message"]["content"] (딕셔너리)
    # 두 버전 모두 지원:
    msg = response.message if hasattr(response, "message") else response["message"]
    return msg.content if hasattr(msg, "content") else msg["content"]
```

- [ ] **Step 3: pipeline.py에 LLM 조건 추가**

`src/pipeline.py`의 `run_analysis` 함수 내 리포트 생성 부분에 추가:
```python
# 기존 template_reporter 뒤에 추가
llm_cfg = settings.get("llm", {})
if llm_cfg.get("enabled") and trip_events:
    from src.report.llm_reporter import generate_llm_report
    llm_text = generate_llm_report(
        trip_events, fault_reports,
        model=llm_cfg.get("model", "llama3.1:8b"),
        host=llm_cfg.get("host", "http://localhost:11434"),
    )
    report_text += "\n\n[ LLM 자연어 분석 ]\n" + llm_text
    # 리포트 재저장
    report_path = save_report(report_text, output_folder, csv_path)
```

- [ ] **Step 4: config/settings.yaml에서 LLM 활성화**

```yaml
llm:
  enabled: true
  model: "llama3.1:8b"
```

- [ ] **Step 5: 실행 테스트**

```bash
python -c "
from src.pipeline import run_analysis
run_analysis('PIMS/2603201549_oven.csv', 'output/', 'output/alarm.log',
             settings={'llm': {'enabled': True, 'model': 'llama3.1:8b'}})
"
```

- [ ] **Step 6: commit**

```bash
git add src/report/llm_reporter.py src/pipeline.py config/settings.yaml
git commit -m "feat: add optional Ollama LLM narrative report generation"
```

---

## 실행 방법 (완성 후)

```bash
# 1. 가상환경 활성화
python -m venv venv && source venv/Scripts/activate  # Windows

# 2. 의존성 설치
pip install -r requirements.txt

# 3. config/settings.yaml에서 watch_folder, output_folder 경로 설정

# 4. 데몬 시작 (iba가 CSV를 내보내는 폴더 감시)
python main.py

# 5. 단일 파일 분석 (테스트용)
python -c "from src.pipeline import run_analysis; run_analysis('파일경로.csv', 'output/', 'output/alarm.log')"
```

---

## 의존성 메모

- `watchdog` 3.x: Windows에서 `WindowsApiObserver` 자동 사용 → 반응 빠름
- `plyer`: Windows 토스트 알림 (`pip install plyer`)
- `ollama`: `pip install ollama` 후 별도 Ollama 앱 설치 필요 (https://ollama.com)
- 추천 로컬 모델 (한국어 지원):
  - `qwen2.5:7b` (한국어 성능 우수, 경량)
  - `llama3.1:8b` (균형 잡힌 성능)
  - `exaone3.5:7.8b` (LG AI, 한국어 특화)