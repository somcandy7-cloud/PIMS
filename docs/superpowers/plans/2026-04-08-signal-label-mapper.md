# Signal Label Mapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ibaPDA CSV의 PLC 태그 주소(`DB420.DBW 2`)를 3OVEN/4OVEN.xlsx의 변수명·설명과 자동 매핑하여, 대시보드에서 신호를 사람이 읽을 수 있는 이름으로 표시한다.

**Architecture:** `SignalLabelMapper`가 OVEN xlsx 파일 전체를 읽어 `{정규화된 태그 주소: (변수명, 설명)}` dict를 구성한다. 태그명 정규화(prefix 제거 + 공백 정규화)로 CSV와 XLSX 간 포맷 불일치를 해결한다. 대시보드는 매퍼를 선택적으로 사용하며, XLSX 파일이 없어도 기존 PLC 태그명으로 동작한다.

**Tech Stack:** Python 3.14, openpyxl, pandas, PyYAML, Streamlit

---

## 알려진 미해결 항목 (현장에서 수동 수정 필요)

아래 항목은 자동 매핑이 불가능하거나 현장 확인이 필요하다.  
`docs/known_gaps.md`에 상세 기록한다.

| # | 항목 | 원인 | 현장 조치 |
|---|------|------|-----------|
| G-1 | 한글 태그명 인코딩 손실 | ibaPDA CSV를 ASCII로 저장 시 CP949 한글 깨짐 (`3OVEN #A ??? LEVEL ???`) | ibaPDA 익스포트 설정에서 CP949 또는 UTF-8 선택 후 재수출 |
| G-2 | 3OVEN/4OVEN 중복 태그 | `DB420.DBW 0` 등 동일 주소가 두 파일에 모두 존재 | 파일 경로에 3OVEN/4OVEN 구분 정보를 포함하거나, 현장에서 어떤 호기 데이터인지 명시 |
| G-3 | DBX 비트 주소 형식 불일치 | CSV: `DB421.DBX 2.0` / XLSX: `DB421.DBX    2.0` (공백 다수) | 공백 정규화로 해결 시도하나, 비트 인덱스(`.0`~`.7`) 표기가 다를 경우 수동 확인 |
| G-4 | XLSX 미수록 태그 | `DB400`, `DB401`, `DB422`, `DB423` 일부 태그가 XLSX에 없음 | 최신 XLSX 문서 제공 또는 해당 태그 설명 별도 추가 |
| G-5 | 4OVEN 공백 없는 주소 | `DB420.DBW0` (공백 없음) vs CSV `DB420.DBW 0` | 공백 정규화 규칙으로 일부 해결; 완전 해결은 현장 XLSX 재정리 필요 |

---

## 파일 구조

| 파일 | 역할 | 상태 |
|------|------|------|
| `src/data/signal_label_mapper.py` | SignalLabelMapper 클래스 | **신규 생성** |
| `config/label_files.yaml` | OVEN xlsx 경로 설정 | **신규 생성** |
| `docs/known_gaps.md` | 미해결 항목 상세 기록 | **신규 생성** |
| `tests/test_signal_label_mapper.py` | 단위 테스트 | **신규 생성** |
| `dashboard.py` | 신호 레이블 표시 통합 | **수정** |

---

## Task 1: SignalLabelMapper 코어

**Files:**
- Create: `src/data/signal_label_mapper.py`
- Test: `tests/test_signal_label_mapper.py`

### 구현 대상

```python
# src/data/signal_label_mapper.py
from __future__ import annotations
from pathlib import Path
import re
import openpyxl

_PREFIX_RE = re.compile(r"^\[.*?\]")   # [3_CO_PLC_A] 등 제거
_SPACE_RE  = re.compile(r"\s+")         # 연속 공백 → 단일 공백


def _normalize_tag(tag: str) -> str:
    """PLC prefix 제거 + 공백 정규화."""
    tag = _PREFIX_RE.sub("", tag).strip()
    tag = _SPACE_RE.sub(" ", tag)
    return tag


class SignalLabelMapper:
    """OVEN xlsx에서 PLC 태그 주소 → (변수명, 설명) 매핑을 제공한다.

    xlsx 파일이 없거나 매핑이 없으면 None을 반환한다 (degraded gracefully).

    xlsx 구조 (각 시트의 컬럼):
        Col 0: PLC 태그 주소  예) DB420.DBW 2
        Col 1: 변수명          예) FR_HMI_W2
        Col 2: 데이터타입      예) INT
        Col 3: 설명            예) C/C STAND PIPE OPEN COUNT SET VALUE
    """

    def __init__(self, xlsx_paths: list[str | Path]):
        self._map: dict[str, tuple[str, str]] = {}  # {norm_tag: (var_name, description)}
        self._unmapped: list[str] = []               # 매핑 실패 태그 (런타임 기록용)
        for path in xlsx_paths:
            self._load_xlsx(Path(path))

    def _load_xlsx(self, path: Path) -> None:
        if not path.exists():
            return
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            for row in ws.iter_rows(min_row=2, values_only=True):
                tag_raw = row[0]
                var_name = row[1] if len(row) > 1 else None
                description = row[3] if len(row) > 3 else None
                if not tag_raw or not isinstance(tag_raw, str):
                    continue
                norm = _normalize_tag(tag_raw)
                if norm and norm not in self._map:
                    self._map[norm] = (
                        str(var_name) if var_name else "",
                        str(description) if description else "",
                    )
        wb.close()

    def lookup(self, tag: str) -> tuple[str, str] | None:
        """정규화된 태그명으로 (변수명, 설명)을 반환한다. 없으면 None."""
        norm = _normalize_tag(tag)
        result = self._map.get(norm)
        if result is None:
            self._unmapped.append(tag)
        return result

    def label(self, tag: str, fallback: str | None = None) -> str:
        """변수명을 반환한다. 없으면 fallback 또는 원래 tag."""
        hit = self.lookup(tag)
        if hit and hit[0]:
            return hit[0]
        return fallback if fallback is not None else tag

    def describe(self, tag: str) -> str:
        """설명을 반환한다. 없으면 빈 문자열."""
        hit = self.lookup(tag)
        return hit[1] if hit else ""

    @property
    def size(self) -> int:
        return len(self._map)

    @property
    def unmapped_tags(self) -> list[str]:
        """런타임 중 매핑되지 않은 태그 목록 (중복 포함)."""
        return list(self._unmapped)
```

- [ ] **Step 1-1: 테스트 파일 작성**

`tests/test_signal_label_mapper.py` 생성:

```python
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.data.signal_label_mapper import SignalLabelMapper, _normalize_tag


# ── _normalize_tag 단위 테스트 ────────────────────────────────────────────────

def test_normalize_removes_prefix():
    assert _normalize_tag("[3_CO_PLC_A]DB420.DBW 2") == "DB420.DBW 2"

def test_normalize_collapses_spaces():
    assert _normalize_tag("DB421.DBX    2.0") == "DB421.DBX 2.0"

def test_normalize_strips_leading_trailing():
    assert _normalize_tag("  DB420.DBW 0  ") == "DB420.DBW 0"

def test_normalize_no_space_variant():
    # 4OVEN 스타일: DB420.DBW0 → DB420.DBW0 (공백 없음 그대로)
    assert _normalize_tag("DB420.DBW0") == "DB420.DBW0"

def test_normalize_empty():
    assert _normalize_tag("") == ""


# ── SignalLabelMapper 단위 테스트 ─────────────────────────────────────────────

def _make_mapper_with_rows(rows: list[tuple]) -> SignalLabelMapper:
    """openpyxl을 mock해서 SignalLabelMapper를 생성하는 헬퍼."""
    mock_ws = MagicMock()
    mock_ws.iter_rows.return_value = rows

    mock_wb = MagicMock()
    mock_wb.sheetnames = ["Sheet1"]
    mock_wb.__getitem__ = lambda self, k: mock_ws
    mock_wb.close = MagicMock()

    with patch("src.data.signal_label_mapper.openpyxl.load_workbook", return_value=mock_wb), \
         patch("pathlib.Path.exists", return_value=True):
        mapper = SignalLabelMapper(["fake.xlsx"])
    return mapper


def test_lookup_found():
    mapper = _make_mapper_with_rows([
        ("DB420.DBW 2", "FR_HMI_W2", "INT", "STAND PIPE OPEN COUNT SET VALUE"),
    ])
    result = mapper.lookup("DB420.DBW 2")
    assert result == ("FR_HMI_W2", "STAND PIPE OPEN COUNT SET VALUE")


def test_lookup_with_prefix():
    mapper = _make_mapper_with_rows([
        ("DB420.DBW 2", "FR_HMI_W2", "INT", "DESC"),
    ])
    # CSV 태그명에 PLC prefix가 붙어 있어도 매핑되어야 한다
    result = mapper.lookup("[3_CO_PLC_A]DB420.DBW 2")
    assert result == ("FR_HMI_W2", "DESC")


def test_lookup_not_found_returns_none():
    mapper = _make_mapper_with_rows([
        ("DB420.DBW 2", "FR_HMI_W2", "INT", "DESC"),
    ])
    assert mapper.lookup("DB999.DBW 0") is None


def test_label_fallback():
    mapper = _make_mapper_with_rows([])
    assert mapper.label("DB999.DBW 0") == "DB999.DBW 0"
    assert mapper.label("DB999.DBW 0", fallback="UNKNOWN") == "UNKNOWN"


def test_describe_empty_when_not_found():
    mapper = _make_mapper_with_rows([])
    assert mapper.describe("DB999.DBW 0") == ""


def test_size():
    mapper = _make_mapper_with_rows([
        ("DB420.DBW 2", "V1", "INT", "D1"),
        ("DB420.DBW 4", "V2", "INT", "D2"),
    ])
    assert mapper.size == 2


def test_unmapped_tags_recorded():
    mapper = _make_mapper_with_rows([
        ("DB420.DBW 2", "V1", "INT", "D1"),
    ])
    mapper.lookup("DB999.DBW 0")
    mapper.lookup("DB888.DBW 0")
    assert "DB999.DBW 0" in mapper.unmapped_tags
    assert "DB888.DBW 0" in mapper.unmapped_tags


def test_duplicate_tag_first_wins():
    """동일 태그가 두 시트에 있으면 먼저 로드된 것이 우선한다."""
    mapper = _make_mapper_with_rows([
        ("DB420.DBW 2", "FIRST", "INT", "first desc"),
        ("DB420.DBW 2", "SECOND", "INT", "second desc"),
    ])
    assert mapper.lookup("DB420.DBW 2")[0] == "FIRST"


def test_xlsx_not_exists_graceful():
    """xlsx 파일이 없어도 예외 없이 동작한다."""
    with patch("pathlib.Path.exists", return_value=False):
        mapper = SignalLabelMapper(["nonexistent.xlsx"])
    assert mapper.size == 0
    assert mapper.lookup("DB420.DBW 2") is None


def test_none_tag_row_skipped():
    """태그 주소가 None인 행은 무시한다."""
    mapper = _make_mapper_with_rows([
        (None, "V1", "INT", "D1"),
        ("DB420.DBW 2", "V2", "INT", "D2"),
    ])
    assert mapper.size == 1
```

- [ ] **Step 1-2: 테스트 실패 확인**

```bash
cd "c:\Users\somca\내문서\project\PIMS"
python -m pytest tests/test_signal_label_mapper.py -v
```

Expected: `ModuleNotFoundError` 또는 `ImportError` (아직 구현 없음)

- [ ] **Step 1-3: 구현 작성**

위 `src/data/signal_label_mapper.py` 코드 그대로 작성.

- [ ] **Step 1-4: 테스트 통과 확인**

```bash
python -m pytest tests/test_signal_label_mapper.py -v
```

Expected: 13 passed

- [ ] **Step 1-5: G-5 no-space 정규화 추가 (선택적 개선)**

`_normalize_tag()` 함수에 아래 줄을 `_SPACE_RE.sub` 호출 **앞에** 추가한다:

```python
def _normalize_tag(tag: str) -> str:
    """PLC prefix 제거 + 공백 정규화 + no-space 주소 보정."""
    tag = _PREFIX_RE.sub("", tag).strip()
    # G-5: DB420.DBW0 → DB420.DBW 0 (4OVEN xlsx 공백 없는 주소 보정)
    tag = re.sub(r"(DB\w+\.DB[WDXB])(\d)", r"\1 \2", tag)
    tag = _SPACE_RE.sub(" ", tag)
    return tag
```

이에 따라 `test_normalize_no_space_variant` 테스트를 수정한다:

```python
def test_normalize_no_space_variant():
    # G-5 fix: DB420.DBW0 → DB420.DBW 0 으로 정규화되어야 한다
    assert _normalize_tag("DB420.DBW0") == "DB420.DBW 0"
```

테스트 재실행:
```bash
python -m pytest tests/test_signal_label_mapper.py -v
```
Expected: 13 passed (수정된 테스트 포함)

- [ ] **Step 1-6: 커밋**

```bash
git add src/data/signal_label_mapper.py tests/test_signal_label_mapper.py
git commit -m "feat: add SignalLabelMapper for OVEN xlsx signal labeling"
```

---

## Task 2: 설정 파일 + known_gaps 문서

**Files:**
- Create: `config/label_files.yaml`
- Create: `docs/known_gaps.md`

- [ ] **Step 2-1: `config/label_files.yaml` 생성**

```yaml
# OVEN 신호 라벨 파일 경로 목록
# 상대 경로: PIMS 루트 기준
# 파일이 없으면 라벨 기능이 비활성화(PLC 태그명 그대로 표시)됩니다.
label_files:
  - "3OVEN.xlsx"
  - "4OVEN.xlsx"
```

- [ ] **Step 2-2: `docs/known_gaps.md` 생성**

```markdown
# 신호 라벨 매핑 — 알려진 미해결 항목

마지막 업데이트: 2026-04-08

아래 항목은 자동 매핑이 불가능하여 **현장 수동 조치**가 필요합니다.

---

## G-1: 한글 태그명 인코딩 손실

**현상:**  
ibaPDA CSV의 일부 신호명이 ASCII 손실로 깨져 있습니다.  
예) `3OVEN #A ??? LEVEL ??? (MAX)` — 원문 한글이 `???`로 대체됨

**원인:**  
ibaPDA Analyzer가 CSV 익스포트 시 CP949 한글을 ASCII로 변환하면서 손실.

**현장 조치:**  
ibaPDA Analyzer → 데이터 익스포트 설정 → 인코딩을 `UTF-8` 또는 `CP949`(EUC-KR)로 변경 후 재수출.  
`IbaCSVLoader`의 `_detect_encoding()` 이 자동으로 CP949 시도하므로, 익스포트 인코딩만 맞추면 됩니다.

---

## G-2: 3OVEN / 4OVEN 중복 태그 (현재: 먼저 로드된 파일 우선)

**현상:**  
`DB420.DBW 0`, `DB421.DBW 0` 등 동일 주소가 3OVEN.xlsx과 4OVEN.xlsx 양쪽에 존재합니다.  
현재 구현은 `config/label_files.yaml`에 나열된 순서 중 먼저 로드된 것을 사용합니다.

**현장 조치 옵션:**  
1. 분석할 설비에 맞게 `config/label_files.yaml`에서 해당 파일만 지정  
   (3호기 데이터 분석 시: `label_files: ["3OVEN.xlsx"]`)  
2. 또는, XLSX 내 태그 주소에 PLC 장치 prefix를 포함하도록 XLSX 문서를 수정  
   (`[3_CO_PLC_A]DB420.DBW 0` 형식으로 통일)

---

## G-3: DBX 비트 주소 공백 불일치

**현상:**  
XLSX: `DB421.DBX    2.0` (탭/공백 다수)  
CSV: `DB421.DBX 2.0`  
→ 공백 정규화로 대부분 해결되나, 비트 인덱스 표기(`.0`~`.7`)가 다른 경우 매핑 실패.

**현장 조치:**  
XLSX의 DBX 주소를 `DB421.DBX 2.0` 형식으로 통일하여 재저장.

---

## G-4: XLSX 미수록 태그

**현상:**  
CSV에 등장하는 `DB400`, `DB401`, `DB110`, `DB422`, `DB423` 계열 일부 태그가  
3OVEN/4OVEN.xlsx에 수록되어 있지 않아 매핑이 되지 않습니다.

**현장 조치:**  
해당 DB 블록의 변수 목록 문서를 xlsx 형식으로 추가 작성 후  
`config/label_files.yaml`의 `label_files` 목록에 추가.

---

## G-5: 4OVEN 공백 없는 주소 표기

**현상:**  
4OVEN.xlsx 일부 시트의 태그 주소가 `DB420.DBW0` (공백 없음) 형식.  
CSV는 `DB420.DBW 0` (공백 있음).  
→ 정규화 규칙이 **공백 정규화**만 수행하므로, 공백 자체가 없는 경우 미매핑됩니다.

**현장 조치:**  
4OVEN.xlsx를 열어 해당 시트의 태그 주소에 공백을 추가하여 `DB420.DBW 0` 형식으로 통일.  
또는 `_normalize_tag()` 함수에 추가 규칙을 적용합니다:
```python
# DBW/DBD/DBB/DBX 뒤에 공백 강제 삽입
tag = re.sub(r"(DB\w+\.DB[WDXB])(\d)", r"\1 \2", tag)
```
이 규칙은 `test_normalize_no_space_variant` 테스트를 수정한 후 적용합니다.
```

- [ ] **Step 2-3: 커밋**

```bash
git add config/label_files.yaml docs/known_gaps.md
git commit -m "docs: add label_files config and known_gaps for signal mapping"
```

---

## Task 3: 대시보드 통합

**Files:**
- Modify: `dashboard.py`

신호 표시가 필요한 4곳을 수정한다:
1. 매퍼 로드 (앱 시작 시 1회)
2. 급변 신호 Bar Chart — y축 레이블에 변수명 추가
3. 신호 추이 차트 라디오 버튼 — 변수명 표시
4. 신호 통계 테이블 — 변수명 + 설명 컬럼 추가

### 추가할 헬퍼 함수

```python
# dashboard.py 상단부 (import 직후)
import yaml
from src.data.signal_label_mapper import SignalLabelMapper

@st.cache_resource
def load_label_mapper() -> SignalLabelMapper | None:
    """OVEN xlsx 라벨 매퍼를 로드한다. 파일 없으면 None."""
    cfg_path = ROOT / "config" / "label_files.yaml"
    if not cfg_path.exists():
        return None
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths = [ROOT / p for p in (cfg.get("label_files") or [])]
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return SignalLabelMapper(existing)


def display_name(tag: str, mapper: SignalLabelMapper | None) -> str:
    """태그명 → '변수명 (태그주소)' 형식. 매퍼 없거나 미매핑이면 태그명만."""
    if mapper is None:
        return tag
    var = mapper.label(tag, fallback=None)
    if var and var != tag:
        return f"{var} ({tag})"
    return tag
```

- [ ] **Step 3-1: 매퍼 로드 + 헬퍼 함수 추가**

`dashboard.py`에서 다음 두 가지를 수정한다.

**① import 블록 (파일 상단, 기존 `from src.data.signal_label_mapper` 줄 추가)**

기존 import 섹션 (`from src.analysis.fault_classifier import FaultClassifier` 아래)에 추가:
```python
from src.data.signal_label_mapper import SignalLabelMapper
```

**② 헬퍼 함수 추가** (`SIGNALS_YAML = ROOT / "config" / "signals.yaml"` 바로 아래에 삽입):

```python
@st.cache_resource
def load_label_mapper() -> "SignalLabelMapper | None":
    cfg_path = ROOT / "config" / "label_files.yaml"
    if not cfg_path.exists():
        return None
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths = [ROOT / p for p in (cfg.get("label_files") or [])]
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return SignalLabelMapper(existing)


def display_name(tag: str, mapper: "SignalLabelMapper | None") -> str:
    if mapper is None:
        return tag
    var = mapper.label(tag, fallback=None)
    if var and var != tag:
        return f"{var} ({tag})"
    return tag
```

**③ `mapper` 변수 초기화** — `dashboard.py`에서 `df: pd.DataFrame = st.session_state.get("df")` 줄 **바로 위**에 추가:
```python
mapper = load_label_mapper()
```

`ROOT`는 `dashboard.py` 상단의 `ROOT = Path(__file__).parent`로 이미 정의되어 있다. `yaml`도 이미 import되어 있다.

- [ ] **Step 3-2: 급변 신호 Bar Chart y축 레이블 수정**

`dashboard.py`에서 `sig_names = [s[0] for s in event.top_signals]` 줄 **바로 아래**에 한 줄 추가:

```python
sig_names = [s[0] for s in event.top_signals]
sig_display_names = [display_name(s, mapper) for s in sig_names]   # ← 추가
```

그리고 `fig_bar` 정의에서 `y=sig_names[::-1]` 을 `y=sig_display_names[::-1]` 로 교체:

```python
# 수정 전
fig_bar = go.Figure(go.Bar(
    x=sig_mags[::-1], y=sig_names[::-1],
    ...
))

# 수정 후 (y= 파라미터만 변경, 나머지 그대로)
fig_bar = go.Figure(go.Bar(
    x=sig_mags[::-1], y=sig_display_names[::-1],
    ...
))
```

- [ ] **Step 3-3: 신호 선택 라디오 레이블 수정**

```python
# 수정 전
sel_sig = st.radio("신호 선택 →", options=available, ...)

# 수정 후
available_display = {s: display_name(s, mapper) for s in available}
sel_sig = st.radio(
    "신호 선택 →",
    options=available,
    format_func=lambda s: available_display[s],
    ...
)
```

- [ ] **Step 3-4: 신호 통계 테이블에 설명 컬럼 추가**

```python
# 기존 c1에 태그명 대신 display_name 사용
c1.markdown(f"**{display_name(sig, mapper)}**")

# 설명이 있으면 바로 아래 caption으로 표시
desc = mapper.describe(sig) if mapper else ""
if desc:
    c1.caption(desc)
```

- [ ] **Step 3-5: 이벤트 요약 테이블 — 급변 신호 수 옆에 신호 목록 표시**

IF 이벤트 fallback 섹션(trip_event 없는 경우)에서:

```python
# 기존
| 급변 신호 수 | {len(event.top_signals)}개 |

# 수정 후
top_sig_str = ", ".join(display_name(s[0], mapper) for s in event.top_signals[:3])
if len(event.top_signals) > 3:
    top_sig_str += f" 외 {len(event.top_signals)-3}개"
```

그리고 테이블에 행 추가:
```
| 주요 신호 | {top_sig_str} |
```

- [ ] **Step 3-6: import 정리 + 전체 실행 확인**

```bash
python -m py_compile dashboard.py && echo "OK"
```

- [ ] **Step 3-7: 커밋**

```bash
git add dashboard.py
git commit -m "feat: show signal labels from OVEN xlsx in dashboard"
```

---

## Task 4: 전체 테스트 통과 확인

- [ ] **Step 4-1: 전체 테스트 실행**

```bash
python -m pytest tests/ --ignore=tests/test_watcher.py -v
```

Expected: 모든 기존 테스트 통과 + 신규 13개 포함 79+ passed

- [ ] **Step 4-2: 미매핑 태그 현황 출력 스크립트 실행 확인**

아래 스크립트로 실제 CSV의 몇 % 신호가 매핑되는지 확인할 수 있다  
(테스트가 아니므로 직접 실행):

```python
# 임시 확인용 (커밋 불필요)
import sys
sys.path.insert(0, ".")
from src.data.signal_label_mapper import SignalLabelMapper
from src.data.loader import IbaCSVLoader
import yaml
from pathlib import Path

ROOT = Path(".")
with open(ROOT / "config" / "label_files.yaml") as f:
    cfg = yaml.safe_load(f)
paths = [ROOT / p for p in cfg["label_files"]]
mapper = SignalLabelMapper(paths)

df = IbaCSVLoader().load("2603201549_oven.csv")
total = len(df.columns)
mapped = sum(1 for c in df.columns if mapper.lookup(c) is not None)
print(f"매핑됨: {mapped}/{total} ({mapped/total*100:.1f}%)")
print(f"미매핑 상위 20개:")
for c in df.columns:
    if mapper.lookup(c) is None:
        print(f"  {c}")
```

Expected: 매핑률과 미매핑 태그 목록 출력

- [ ] **Step 4-3: 최종 커밋**

Step 4-2의 임시 확인 스크립트는 커밋하지 않는다. 변경된 파일만 명시적으로 스테이징:

```bash
git add src/data/signal_label_mapper.py \
        tests/test_signal_label_mapper.py \
        config/label_files.yaml \
        docs/known_gaps.md \
        dashboard.py
git commit -m "feat: signal label mapper integration complete"
```