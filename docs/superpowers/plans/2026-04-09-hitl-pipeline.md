# HITL 기반 능동 학습 파이프라인 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 새 CSV가 들어오면 가동 신호를 스스로 발견하고 → 관리자 1회 확인 후 설비 프로파일로 저장 → 이후 자동 적용하는 HITL 파이프라인을 구축한다.

**Architecture:**
```
새 CSV 입력
    │
    ├─ 설비 프로파일 캐시 hit?
    │      Yes → OperationFilter(cached_conditions)
    │      No  → AutoOperationDiscovery.discover(df)
    │                → 이진 신호(ON 10~90%) + 이중봉 아날로그 신호 자동 탐지
    │                → "이 조건들로 가동 판별하겠습니다" 관리자 확인
    │                → equipment_profiles.yaml에 저장
    │
    ↓
OperationFilter(conditions)   ← 조건 목록 기반, N개 AND
    ↓
RollingFeatureExtractor       ← rolling mean/std/roc 트렌드 특징
    ↓
IsolationForestAdapter        ← 이상 탐지 (기존 코드 재사용)
    ↓
AnomalyContextFormatter       ← LLM/관리자용 JSON+텍스트
    ↓
LLMFilter (1차 자동)          ← 기존 코드 재사용
    ↓
관리자 O/X 피드백
    ↓
FeedbackStore (SQLite)        ← 재학습 뼈대
```

기존 `AnomalyDetector` / `LLMBackend` 인터페이스는 수정 없이 재사용한다.

**Tech Stack:** Python 3.11+, pandas, scikit-learn (IsolationForest, KMeans), sqlite3 (표준 라이브러리), PyYAML, pytest, 기존 LLMBackend (Ollama / OpenAI)

---

## 파일 맵

| 상태 | 경로 | 역할 |
|------|------|------|
| 신규 | `src/utils/operation_discovery.py` | 가동 신호 자동 탐색 (이진/이중봉 분석) |
| 신규 | `src/utils/operation_filter.py` | 조건 목록 기반 AND 필터 |
| 신규 | `src/utils/rolling_features.py` | rolling mean/std/roc 특징 추출 |
| 신규 | `src/utils/context_formatter.py` | 이상 이벤트 → LLM/관리자용 JSON+텍스트 |
| 신규 | `src/services/equipment_profile_store.py` | 설비 프로파일 YAML 캐시 저장/조회 |
| 신규 | `src/services/feedback_store.py` | SQLite O/X 피드백 DB + 재학습 뼈대 |
| 신규 | `config/equipment_profiles.yaml` | 설비별 가동 조건 캐시 (자동 생성) |
| 신규 | `config/llm_hitl_prompt.md` | HITL 전용 LLM 프롬프트 |
| 신규 | `hitl_pipeline.py` | 메인 HITL 실행 스크립트 |
| 신규 | `tests/test_operation_discovery.py` | AutoOperationDiscovery 단위 테스트 |
| 신규 | `tests/test_operation_filter.py` | OperationFilter 단위 테스트 |
| 신규 | `tests/test_rolling_features.py` | RollingFeatureExtractor 단위 테스트 |
| 신규 | `tests/test_context_formatter.py` | AnomalyContextFormatter 단위 테스트 |
| 신규 | `tests/test_equipment_profile_store.py` | EquipmentProfileStore 단위 테스트 |
| 신규 | `tests/test_feedback_store.py` | FeedbackStore 단위 테스트 |
| 수정 | `config/settings.yaml` | hitl 섹션 추가 |

---

## Task 1: AutoOperationDiscovery

**역할:** 새 CSV(DataFrame)를 입력받아 가동 판별에 쓸 수 있는 신호와 임계값을 자동으로 찾는다.

**알고리즘:**
- **이진 후보(제어 명령)**: 값이 0/1만 존재하고, 전체 데이터 중 ON(=1) 비율이 10~90%인 신호. 비율이 이 범위 안이면 설비가 켜졌다 꺼지는 구간이 모두 있다는 뜻. 조건: `column == 1.0`
- **이중봉 후보(물리량)**: 아날로그 신호 중 분포의 중앙 밀집도(center density)가 낮은 신호. KMeans(k=2)로 두 클러스터 중심을 찾고, 두 중심의 중간값을 임계값으로 설정. 조건: `column > threshold`
- 두 후보를 각각 상위 N개씩 신뢰도 점수 순으로 반환한다. (신뢰도: 이진은 0.5와의 거리 기반, 이중봉은 클러스터 분리도 기반)

**Files:**
- Create: `src/utils/operation_discovery.py`
- Test: `tests/test_operation_discovery.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_operation_discovery.py
import numpy as np
import pandas as pd
import pytest
from src.utils.operation_discovery import AutoOperationDiscovery, OperationCondition


def _make_df_with_known_signals(n: int = 200) -> pd.DataFrame:
    """
    가동 판별이 명확한 인공 데이터:
      - 'ctrl_on'  : 0/1 이진, ON 비율 50%  → 이진 후보
      - 'speed'    : 0 또는 300 (이중봉)    → 이중봉 후보
      - 'noise'    : 정규분포 (단봉)         → 후보 아님
      - 'always_0' : 항상 0                 → 후보 아님 (변화 없음)
    """
    t = pd.date_range("2026-03-12", periods=n, freq="2s", tz="UTC")
    rng = np.random.default_rng(42)
    ctrl = np.array([1.0 if i < n // 2 else 0.0 for i in range(n)])
    speed = np.where(ctrl == 1.0,
                     rng.normal(300, 5, n),
                     rng.normal(0, 1, n))
    return pd.DataFrame({
        "ctrl_on":   ctrl,
        "speed":     speed,
        "noise":     rng.normal(50, 5, n),
        "always_0":  np.zeros(n),
    }, index=t)


def test_discovers_binary_candidate():
    df = _make_df_with_known_signals()
    conditions = AutoOperationDiscovery().discover(df)
    binary_cols = [c.column for c in conditions if c.op == "=="]
    assert "ctrl_on" in binary_cols


def test_discovers_bimodal_candidate():
    df = _make_df_with_known_signals()
    conditions = AutoOperationDiscovery().discover(df)
    bimodal_cols = [c.column for c in conditions if c.op == ">"]
    assert "speed" in bimodal_cols


def test_noise_signal_not_selected():
    df = _make_df_with_known_signals()
    conditions = AutoOperationDiscovery().discover(df)
    all_cols = [c.column for c in conditions]
    assert "noise" not in all_cols


def test_always_zero_not_selected():
    df = _make_df_with_known_signals()
    conditions = AutoOperationDiscovery().discover(df)
    all_cols = [c.column for c in conditions]
    assert "always_0" not in all_cols


def test_returns_operation_condition_objects():
    df = _make_df_with_known_signals()
    conditions = AutoOperationDiscovery().discover(df)
    assert len(conditions) >= 1
    for c in conditions:
        assert isinstance(c, OperationCondition)
        assert c.op in ("==", "!=", ">", ">=", "<", "<=")
        assert isinstance(c.value, float)
        assert 0.0 <= c.confidence <= 1.0


def test_bimodal_threshold_between_clusters():
    """이중봉 신호의 임계값은 두 클러스터 사이에 있어야 한다."""
    df = _make_df_with_known_signals()
    conditions = AutoOperationDiscovery().discover(df)
    speed_cond = next((c for c in conditions if c.column == "speed"), None)
    assert speed_cond is not None
    # speed는 ~0 클러스터와 ~300 클러스터 → 임계값은 0~300 사이
    assert 10.0 < speed_cond.value < 290.0


def test_empty_df_returns_empty():
    df = pd.DataFrame(columns=["a", "b"])
    conditions = AutoOperationDiscovery().discover(df)
    assert conditions == []
```

- [ ] **Step 2: 테스트 실패 확인**

```
pytest tests/test_operation_discovery.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: 구현**

```python
# src/utils/operation_discovery.py
"""가동 상태 판별 신호 자동 탐색 모듈.

새 CSV DataFrame을 입력받아 이진 신호(제어 명령 후보)와
이중봉 아날로그 신호(물리량 후보)를 자동으로 찾고
OperationCondition 목록으로 반환한다.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


@dataclass
class OperationCondition:
    """단일 가동 판별 조건.

    Attributes
    ----------
    column     : DataFrame 컬럼명
    op         : 비교 연산자 문자열 ('==', '!=', '>', '>=', '<', '<=')
    value      : 비교 기준값
    confidence : 탐지 신뢰도 0.0~1.0 (높을수록 확실한 후보)
    source     : 탐지 방법 ('binary' or 'bimodal')
    """
    column: str
    op: str
    value: float
    confidence: float
    source: str

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "op": self.op,
            "value": self.value,
            "confidence": round(self.confidence, 4),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OperationCondition":
        return cls(**d)


# ── 탐지 파라미터 기본값 ────────────────────────────────────────────────────
_BINARY_ON_RATIO_MIN  = 0.10   # ON 비율 하한 (10% 미만이면 항상 꺼짐에 가까움)
_BINARY_ON_RATIO_MAX  = 0.90   # ON 비율 상한 (90% 초과면 항상 켜짐에 가까움)
_BIMODAL_CENTER_MAX   = 0.30   # 중앙 밀집도 상한 (이 이하면 이중봉으로 판단)
_BIMODAL_MIN_STD_RATIO = 0.05  # std/range 최솟값 (너무 작으면 계단 신호로 처리)
_TOP_BINARY_N         = 3      # 최대 반환할 이진 후보 수
_TOP_BIMODAL_N        = 3      # 최대 반환할 이중봉 후보 수
_MIN_SAMPLES          = 30     # 분석 최소 샘플 수


class AutoOperationDiscovery:
    """DataFrame을 분석해 가동 판별 조건을 자동으로 생성한다."""

    def __init__(
        self,
        binary_on_min: float = _BINARY_ON_RATIO_MIN,
        binary_on_max: float = _BINARY_ON_RATIO_MAX,
        bimodal_center_max: float = _BIMODAL_CENTER_MAX,
        top_binary_n: int = _TOP_BINARY_N,
        top_bimodal_n: int = _TOP_BIMODAL_N,
    ):
        self.binary_on_min    = binary_on_min
        self.binary_on_max    = binary_on_max
        self.bimodal_center_max = bimodal_center_max
        self.top_binary_n     = top_binary_n
        self.top_bimodal_n    = top_bimodal_n

    def discover(self, df: pd.DataFrame) -> list[OperationCondition]:
        """DataFrame에서 가동 판별 후보 조건 목록을 반환한다.

        이진 후보와 이중봉 후보를 합쳐서 신뢰도 내림차순으로 정렬한다.
        """
        if df.empty:
            return []

        numeric_df = df.select_dtypes(include="number")
        if numeric_df.empty:
            return []

        binary_conds  = self._find_binary(numeric_df)
        bimodal_conds = self._find_bimodal(numeric_df)

        # 이진 후보에 포함된 컬럼은 이중봉 탐지에서 제외 (중복 방지)
        binary_cols = {c.column for c in binary_conds}
        bimodal_conds = [c for c in bimodal_conds if c.column not in binary_cols]

        combined = binary_conds[:self.top_binary_n] + bimodal_conds[:self.top_bimodal_n]
        combined.sort(key=lambda c: c.confidence, reverse=True)
        return combined

    # ── 이진 신호 탐지 ────────────────────────────────────────────────────────

    def _find_binary(self, df: pd.DataFrame) -> list[OperationCondition]:
        """0/1만 존재하고 ON 비율이 적절한 신호를 탐지한다."""
        results = []
        for col in df.columns:
            s = df[col].dropna()
            if len(s) < _MIN_SAMPLES:
                continue

            unique_vals = set(s.unique())
            if not unique_vals.issubset({0.0, 1.0}):
                continue

            on_ratio = float((s == 1.0).mean())
            if not (self.binary_on_min <= on_ratio <= self.binary_on_max):
                continue

            # 신뢰도: ON 비율이 50%에 가까울수록 가동/비가동 구간이 균형 있게 존재
            confidence = 1.0 - abs(on_ratio - 0.5) / 0.5
            results.append(OperationCondition(
                column=col,
                op="==",
                value=1.0,
                confidence=confidence,
                source="binary",
            ))

        results.sort(key=lambda c: c.confidence, reverse=True)
        return results

    # ── 이중봉 아날로그 신호 탐지 ─────────────────────────────────────────────

    def _find_bimodal(self, df: pd.DataFrame) -> list[OperationCondition]:
        """두 개의 뚜렷한 클러스터를 가진 아날로그 신호를 탐지한다."""
        results = []
        for col in df.columns:
            s = df[col].dropna()
            if len(s) < _MIN_SAMPLES:
                continue

            val_range = float(s.max() - s.min())
            if val_range == 0:
                continue

            # 변동이 너무 작으면 의미 있는 이중봉이 아님
            if s.std() / val_range < _BIMODAL_MIN_STD_RATIO:
                continue

            center_density = self._center_density(s)
            if center_density >= self.bimodal_center_max:
                continue

            threshold, separation = self._kmeans_threshold(s)
            if threshold is None:
                continue

            # 신뢰도: 중앙 밀집도가 낮을수록 + 클러스터 분리도가 클수록 높음
            confidence = (1.0 - center_density) * min(separation, 1.0)
            results.append(OperationCondition(
                column=col,
                op=">",
                value=float(threshold),
                confidence=float(confidence),
                source="bimodal",
            ))

        results.sort(key=lambda c: c.confidence, reverse=True)
        return results

    @staticmethod
    def _center_density(s: pd.Series) -> float:
        """중앙값 ±IQR/4 구간에 속하는 비율. 낮을수록 이중봉에 가깝다."""
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return 1.0
        mid = s.median()
        band = iqr / 4
        return float(((s >= mid - band) & (s <= mid + band)).mean())

    @staticmethod
    def _kmeans_threshold(s: pd.Series) -> tuple[float | None, float]:
        """KMeans(k=2)로 두 클러스터 중심을 찾고 (중간값, 분리도)를 반환한다.

        분리도 = |center1 - center2| / std(전체)
        클러스터가 명확히 분리되지 않으면 (None, 0.0) 반환.
        """
        X = s.values.reshape(-1, 1)
        try:
            km = KMeans(n_clusters=2, random_state=42, n_init=5)
            km.fit(X)
            c0, c1 = sorted(km.cluster_centers_.flatten())
            threshold = (c0 + c1) / 2.0
            separation = abs(c1 - c0) / (s.std() + 1e-9)
            # 분리도가 1.0 미만이면 두 클러스터가 겹쳐 있음 → 신뢰 불가
            if separation < 1.0:
                return None, 0.0
            return threshold, separation
        except Exception:
            return None, 0.0
```

- [ ] **Step 4: 테스트 통과 확인**

```
pytest tests/test_operation_discovery.py -v
```
Expected: 7개 PASSED

- [ ] **Step 5: 커밋**

```bash
git add src/utils/operation_discovery.py tests/test_operation_discovery.py
git commit -m "feat: add AutoOperationDiscovery for data-driven running-state detection"
```

---

## Task 2: OperationFilter (조건 목록 기반)

**역할:** `OperationCondition` 목록을 받아 모든 조건을 AND로 묶어 가동 구간을 필터링한다. 조건 개수 제한 없음.

**Files:**
- Create: `src/utils/operation_filter.py`
- Test: `tests/test_operation_filter.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_operation_filter.py
import pandas as pd
import numpy as np
import pytest
from src.utils.operation_discovery import OperationCondition
from src.utils.operation_filter import OperationFilter


def _make_df() -> pd.DataFrame:
    t = pd.date_range("2026-03-12", periods=5, freq="2s", tz="UTC")
    return pd.DataFrame({
        "ctrl":  [0.0, 1.0, 1.0, 1.0, 0.0],
        "speed": [0.0, 50_000.0, 150_000.0, 200_000.0, 0.0],
        "temp":  [20.0, 180.0, 200.0, 195.0, 20.0],
    }, index=t)


def _conds_two() -> list[OperationCondition]:
    return [
        OperationCondition("ctrl", "==", 1.0, 0.9, "binary"),
        OperationCondition("speed", ">", 100_000.0, 0.8, "bimodal"),
    ]


def test_and_filter_keeps_correct_rows():
    df = _make_df()
    result = OperationFilter(_conds_two()).filter(df)
    # ctrl==1 AND speed>100k → 행 2, 3만 통과
    assert len(result) == 2
    assert result["temp"].tolist() == [200.0, 195.0]


def test_three_conditions_and():
    df = _make_df()
    conds = _conds_two() + [
        OperationCondition("temp", ">", 190.0, 0.7, "bimodal"),
    ]
    result = OperationFilter(conds).filter(df)
    # ctrl==1 AND speed>100k AND temp>190 → 행 2만 통과
    assert len(result) == 1


def test_all_stopped_returns_empty():
    t = pd.date_range("2026-03-12", periods=3, freq="2s", tz="UTC")
    df = pd.DataFrame({
        "ctrl":  [0.0, 0.0, 0.0],
        "speed": [0.0, 0.0, 0.0],
    }, index=t)
    result = OperationFilter(_conds_two()).filter(df)
    assert result.empty


def test_missing_column_raises():
    t = pd.date_range("2026-03-12", periods=2, freq="2s", tz="UTC")
    df = pd.DataFrame({"other": [1.0, 2.0]}, index=t)
    with pytest.raises(KeyError):
        OperationFilter(_conds_two()).filter(df)


def test_supported_operators():
    t = pd.date_range("2026-03-12", periods=3, freq="2s", tz="UTC")
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=t)
    ops_expected = [
        (">",  2.0, [3.0]),
        (">=", 2.0, [2.0, 3.0]),
        ("<",  2.0, [1.0]),
        ("<=", 2.0, [1.0, 2.0]),
        ("==", 2.0, [2.0]),
        ("!=", 2.0, [1.0, 3.0]),
    ]
    for op, val, expected in ops_expected:
        cond = OperationCondition("v", op, val, 1.0, "binary")
        result = OperationFilter([cond]).filter(df)
        assert result["v"].tolist() == expected, f"op={op} failed"


def test_empty_conditions_returns_all_rows():
    df = _make_df()
    result = OperationFilter([]).filter(df)
    assert len(result) == len(df)
```

- [ ] **Step 2: 테스트 실패 확인**

```
pytest tests/test_operation_filter.py -v
```

- [ ] **Step 3: 구현**

```python
# src/utils/operation_filter.py
"""조건 목록 기반 가동 상태 필터.

AutoOperationDiscovery 또는 EquipmentProfileStore에서 받은
OperationCondition 목록을 모두 AND로 묶어 가동 구간만 반환한다.
"""
from __future__ import annotations
import operator as _op

import pandas as pd

from src.utils.operation_discovery import OperationCondition

_OPS: dict[str, object] = {
    "==": _op.eq,
    "!=": _op.ne,
    ">":  _op.gt,
    ">=": _op.ge,
    "<":  _op.lt,
    "<=": _op.le,
}


class OperationFilter:
    """OperationCondition 목록을 AND로 결합하는 가동 구간 필터."""

    def __init__(self, conditions: list[OperationCondition]):
        self.conditions = conditions

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """모든 조건을 AND로 묶어 통과한 행만 반환한다. 원본은 수정하지 않는다."""
        if not self.conditions:
            return df.copy()

        mask = pd.Series(True, index=df.index)
        for cond in self.conditions:
            col_data = df[cond.column]   # KeyError → 의도적으로 전파
            op_func = _OPS.get(cond.op)
            if op_func is None:
                raise ValueError(f"지원하지 않는 연산자: {cond.op!r}")
            mask &= op_func(col_data, cond.value)

        return df.loc[mask].copy()
```

- [ ] **Step 4: 테스트 통과 확인**

```
pytest tests/test_operation_filter.py -v
```
Expected: 6개 PASSED

- [ ] **Step 5: 커밋**

```bash
git add src/utils/operation_filter.py tests/test_operation_filter.py
git commit -m "feat: add OperationFilter with N-condition AND logic"
```

---

## Task 3: EquipmentProfileStore

**역할:** 설비별로 탐지된(또는 관리자가 확인한) 가동 조건을 `config/equipment_profiles.yaml`에 저장하고 조회한다. 같은 설비 ID가 있으면 재탐색을 건너뛴다.

**설비 ID 규칙:** CSV 파일명에서 타임스탬프를 제거한 패턴 사용.  
예) `2603201549_oven.csv` → `oven`, `2603211200_loco_A.csv` → `loco_A`

**Files:**
- Create: `src/services/equipment_profile_store.py`
- Test: `tests/test_equipment_profile_store.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_equipment_profile_store.py
import pytest
from pathlib import Path
from src.utils.operation_discovery import OperationCondition
from src.services.equipment_profile_store import EquipmentProfileStore


@pytest.fixture
def store(tmp_path):
    return EquipmentProfileStore(profile_path=str(tmp_path / "profiles.yaml"))


def _conds() -> list[OperationCondition]:
    return [
        OperationCondition("ctrl", "==", 1.0, 0.9, "binary"),
        OperationCondition("speed", ">", 100_000.0, 0.8, "bimodal"),
    ]


def test_save_and_load(store):
    store.save("oven", _conds(), confirmed=True)
    loaded = store.load("oven")
    assert loaded is not None
    assert len(loaded) == 2
    assert loaded[0].column == "ctrl"


def test_load_unknown_returns_none(store):
    assert store.load("unknown_equipment") is None


def test_is_confirmed(store):
    store.save("oven", _conds(), confirmed=True)
    assert store.is_confirmed("oven") is True


def test_unconfirmed_profile(store):
    store.save("oven", _conds(), confirmed=False)
    assert store.is_confirmed("oven") is False


def test_overwrite_updates_profile(store):
    store.save("oven", _conds(), confirmed=False)
    new_conds = [OperationCondition("new_col", ">", 50.0, 0.95, "bimodal")]
    store.save("oven", new_conds, confirmed=True)
    loaded = store.load("oven")
    assert len(loaded) == 1
    assert loaded[0].column == "new_col"


def test_extract_equipment_id():
    assert EquipmentProfileStore.extract_id("2603201549_oven.csv") == "oven"
    assert EquipmentProfileStore.extract_id("2603211200_loco_A.csv") == "loco_A"
    assert EquipmentProfileStore.extract_id("no_timestamp.csv") == "no_timestamp"
```

- [ ] **Step 2: 테스트 실패 확인**

```
pytest tests/test_equipment_profile_store.py -v
```

- [ ] **Step 3: 구현**

```python
# src/services/equipment_profile_store.py
"""설비별 가동 판별 조건 프로파일 YAML 저장소.

파일 형식 (config/equipment_profiles.yaml):
  oven:
    confirmed: true
    conditions:
      - column: "[3_CO_PLC_B]QB 42"
        op: "=="
        value: 1.0
        confidence: 0.87
        source: binary
      - column: "MD    356_1"
        op: ">"
        value: 148500.0
        confidence: 0.76
        source: bimodal
  loco_A:
    confirmed: false
    conditions: ...
"""
from __future__ import annotations
import re
from pathlib import Path

import yaml

from src.utils.operation_discovery import OperationCondition

_DEFAULT_PATH = Path(__file__).parent.parent.parent / "config" / "equipment_profiles.yaml"
_TS_PREFIX_RE = re.compile(r"^\d{10}_(.+)$")   # 예: 2603201549_oven → oven


class EquipmentProfileStore:
    """설비 프로파일 YAML을 읽고 쓰는 저장소."""

    def __init__(self, profile_path: str | Path | None = None):
        self.profile_path = Path(profile_path) if profile_path else _DEFAULT_PATH

    # ── 공개 API ─────────────────────────────────────────────────────────────

    def load(self, equipment_id: str) -> list[OperationCondition] | None:
        """저장된 조건 목록을 반환한다. 없으면 None."""
        data = self._read()
        profile = data.get(equipment_id)
        if profile is None:
            return None
        return [OperationCondition.from_dict(c) for c in profile["conditions"]]

    def save(
        self,
        equipment_id: str,
        conditions: list[OperationCondition],
        confirmed: bool = False,
    ) -> None:
        """조건 목록을 저장한다. 기존 항목이 있으면 덮어쓴다."""
        data = self._read()
        data[equipment_id] = {
            "confirmed": confirmed,
            "conditions": [c.to_dict() for c in conditions],
        }
        self._write(data)

    def is_confirmed(self, equipment_id: str) -> bool:
        """관리자가 조건을 확인했으면 True."""
        data = self._read()
        profile = data.get(equipment_id)
        return bool(profile and profile.get("confirmed", False))

    # ── 유틸리티 ─────────────────────────────────────────────────────────────

    @staticmethod
    def extract_id(csv_filename: str) -> str:
        """CSV 파일명에서 설비 ID를 추출한다.

        '2603201549_oven.csv' → 'oven'
        'no_timestamp.csv'   → 'no_timestamp'
        """
        stem = Path(csv_filename).stem          # 확장자 제거
        m = _TS_PREFIX_RE.match(stem)
        return m.group(1) if m else stem

    # ── 내부 I/O ──────────────────────────────────────────────────────────────

    def _read(self) -> dict:
        if not self.profile_path.exists():
            return {}
        with open(self.profile_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _write(self, data: dict) -> None:
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.profile_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
```

- [ ] **Step 4: 테스트 통과 확인**

```
pytest tests/test_equipment_profile_store.py -v
```
Expected: 7개 PASSED

- [ ] **Step 5: 커밋**

```bash
git add src/services/equipment_profile_store.py tests/test_equipment_profile_store.py
git commit -m "feat: add EquipmentProfileStore for per-equipment condition caching"
```

---

## Task 4: RollingFeatureExtractor

**Files:**
- Create: `src/utils/rolling_features.py`
- Test: `tests/test_rolling_features.py`

**설계 근거:**
현재 데이터: ~2.1 sec/row. 원본 7,308열을 IF에 넣으면 `N열 × window_size` 차원 폭발.
rolling mean/std/roc로 같은 행 수를 유지하면서 트렌드를 압축한 뒤 IF window_size=1 적용.

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_rolling_features.py
import pandas as pd
import numpy as np
from src.utils.rolling_features import RollingFeatureExtractor


def _make_df(n: int = 50) -> pd.DataFrame:
    t = pd.date_range("2026-03-12", periods=n, freq="2s", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "sig_a": np.full(n, 100.0) + rng.normal(0, 1, n),
        "sig_b": np.linspace(0, 50, n),
    }, index=t)


def test_output_has_same_index():
    result = RollingFeatureExtractor(window=5).transform(_make_df())
    assert list(result.index) == list(_make_df().index)


def test_column_names_contain_suffix():
    cols = RollingFeatureExtractor(window=5).transform(_make_df()).columns.tolist()
    assert any("_rmean" in c for c in cols)
    assert any("_rstd" in c for c in cols)
    assert any("_roc" in c for c in cols)


def test_no_nan_after_transform():
    result = RollingFeatureExtractor(window=5).transform(_make_df(n=30))
    assert result.isna().sum().sum() == 0


def test_constant_signal_zero_rstd():
    t = pd.date_range("2026-03-12", periods=20, freq="2s", tz="UTC")
    df = pd.DataFrame({"flat": np.ones(20)}, index=t)
    result = RollingFeatureExtractor(window=5).transform(df)
    assert (result["flat_rstd"] == 0.0).all()
```

- [ ] **Step 2: 테스트 실패 확인**

```
pytest tests/test_rolling_features.py -v
```

- [ ] **Step 3: 구현**

```python
# src/utils/rolling_features.py
from __future__ import annotations
import pandas as pd


class RollingFeatureExtractor:
    """아날로그 신호를 rolling 통계(mean/std/roc)로 변환한다.

    window 기본값 30행 ≈ 63초 (2.1sec/row 기준).
    5분 트렌드가 필요하면 window=143으로 설정.
    """
    _EPS = 1e-9

    def __init__(self, window: int = 30):
        self.window = window

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out: dict[str, pd.Series] = {}
        for col in df.columns:
            s = df[col]
            rolled = s.rolling(window=self.window, min_periods=1)
            out[f"{col}_rmean"] = rolled.mean()
            out[f"{col}_rstd"]  = rolled.std().fillna(0.0)
            shifted = s.shift(self.window).ffill().bfill()
            out[f"{col}_roc"]   = (s - shifted) / (shifted.abs() + self._EPS)
        return pd.DataFrame(out, index=df.index).ffill().bfill().fillna(0.0)
```

- [ ] **Step 4: 테스트 통과 확인**

```
pytest tests/test_rolling_features.py -v
```
Expected: 4개 PASSED

- [ ] **Step 5: 커밋**

```bash
git add src/utils/rolling_features.py tests/test_rolling_features.py
git commit -m "feat: add RollingFeatureExtractor for trend-based IF input"
```

---

## Task 5: AnomalyContextFormatter

**Files:**
- Create: `src/utils/context_formatter.py`
- Test: `tests/test_context_formatter.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_context_formatter.py
import json
import pandas as pd
import numpy as np
import pytest
from src.agents.base_detector import AnomalyEvent
from src.utils.context_formatter import AnomalyContextFormatter


def _make_event() -> AnomalyEvent:
    return AnomalyEvent(
        timestamp=pd.Timestamp("2026-03-12 06:54:00+00:00"),
        score=-0.35,
        top_signals=[("sig_a", 0.8), ("sig_b", 0.4)],
        label="if_candidate",
        metadata={"window_size": 30},
    )


def _make_df(ts: pd.Timestamp, n: int = 300) -> pd.DataFrame:
    start = ts - pd.Timedelta(minutes=5)
    t = pd.date_range(start, periods=n, freq="2s", tz="UTC")
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "sig_a":       rng.normal(100, 5, n),
        "sig_b":       rng.normal(50, 2, n),
        "MD    348_1": rng.normal(150_000, 10_000, n),
    }, index=t)


def test_returns_required_keys():
    event = _make_event()
    result = AnomalyContextFormatter().format(event, _make_df(event.timestamp))
    assert "json_payload" in result
    assert "text_summary" in result


def test_json_payload_valid():
    event = _make_event()
    payload = json.loads(
        AnomalyContextFormatter().format(event, _make_df(event.timestamp))["json_payload"]
    )
    for key in ("timestamp", "score_level", "top_signals", "context_stats"):
        assert key in payload


def test_cross_signal_in_summary():
    event = _make_event()
    result = AnomalyContextFormatter().format(event, _make_df(event.timestamp))
    assert "MD    348_1" in result["text_summary"]


def test_missing_cross_signal_no_error():
    event = _make_event()
    t = pd.date_range("2026-03-12 06:49:00", periods=50, freq="2s", tz="UTC")
    df = pd.DataFrame({"sig_a": np.ones(50)}, index=t)
    result = AnomalyContextFormatter().format(event, df)
    assert "json_payload" in result
```

- [ ] **Step 2: 테스트 실패 확인**

```
pytest tests/test_context_formatter.py -v
```

- [ ] **Step 3: 구현**

```python
# src/utils/context_formatter.py
from __future__ import annotations
import json

import pandas as pd

from src.agents.base_detector import AnomalyEvent

_CROSS_SIGNALS = ["MD    348_1"]
_CONTEXT_MIN   = 5


def _score_level(score: float) -> str:
    if score <= -0.5:  return "매우 강함 (상위 1% 수준)"
    if score <= -0.3:  return "강함 (명확한 이상 신호)"
    if score <= -0.15: return "보통 (이상 가능성 있음)"
    return "약함 (경계 수준)"


def _window_stats(df: pd.DataFrame, ts: pd.Timestamp, cols: list[str]) -> dict:
    t0 = ts - pd.Timedelta(minutes=_CONTEXT_MIN)
    t1 = ts + pd.Timedelta(minutes=_CONTEXT_MIN)
    stats: dict[str, dict] = {}
    for col in [c for c in cols if c in df.columns]:
        try:
            s = df.loc[t0:t1, col].dropna()
        except Exception:
            s = df[col].dropna()
        if len(s) == 0:
            continue
        trend = float(s.iloc[-1] - s.iloc[0]) if len(s) >= 2 else 0.0
        stats[col] = {
            "mean": round(float(s.mean()), 3),
            "std":  round(float(s.std()), 3) if len(s) > 1 else 0.0,
            "min":  round(float(s.min()), 3),
            "max":  round(float(s.max()), 3),
            "trend": round(trend, 3),
        }
    return stats


class AnomalyContextFormatter:
    def __init__(
        self,
        cross_signals: list[str] | None = None,
        context_minutes: int = _CONTEXT_MIN,
    ):
        self.cross_signals = cross_signals or _CROSS_SIGNALS
        self.context_minutes = context_minutes

    def format(self, event: AnomalyEvent, df: pd.DataFrame) -> dict[str, str]:
        ts      = event.timestamp
        level   = _score_level(event.score)
        sig_names = [s[0] for s in event.top_signals]
        ctx     = _window_stats(df, ts, sig_names + self.cross_signals)

        payload = {
            "timestamp": str(ts),
            "score":     round(event.score, 4),
            "score_level": level,
            "top_signals": [
                {"name": n, "variability": round(v, 4)}
                for n, v in event.top_signals
            ],
            "context_window_minutes": self.context_minutes,
            "context_stats": ctx,
        }

        lines = [
            "[이상 탐지 리포트]",
            f"  발생 시각    : {ts}",
            f"  이상 강도    : {level}  (score={event.score:.4f})",
            f"  주요 신호    : {', '.join(sig_names)}",
            f"  분석 구간    : ±{self.context_minutes}분",
            "", "  [신호별 트렌드 통계]",
        ]
        for col, s in ctx.items():
            lines.append(
                f"  {col:<35} mean={s['mean']:.2f}  std={s['std']:.2f}"
                f"  [{s['min']:.2f}~{s['max']:.2f}]  trend={s['trend']:+.2f}"
            )

        return {
            "json_payload": json.dumps(payload, ensure_ascii=False, indent=2),
            "text_summary": "\n".join(lines),
        }
```

- [ ] **Step 4: 테스트 통과 확인**

```
pytest tests/test_context_formatter.py -v
```
Expected: 4개 PASSED

- [ ] **Step 5: 커밋**

```bash
git add src/utils/context_formatter.py tests/test_context_formatter.py
git commit -m "feat: add AnomalyContextFormatter for LLM and admin review"
```

---

## Task 6: FeedbackStore

**Files:**
- Create: `src/services/feedback_store.py`
- Test: `tests/test_feedback_store.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

```
pytest tests/test_feedback_store.py -v
```

- [ ] **Step 3: 구현**

```python
# src/services/feedback_store.py
from __future__ import annotations
import json
import sqlite3
from pathlib import Path

import pandas as pd

from src.agents.base_detector import AnomalyEvent

_DEFAULT_DB = "feedback.db"
_MIN_RETRAIN = 20


class FeedbackStore:
    """관리자 O/X 피드백 SQLite 저장소.

    admin_label:
      'O' = 실제 이상 확정  → 이상 데이터 DB 가중 학습 대상
      'X' = 정상 패턴 확정  → 정상 데이터 DB 추가
    """

    def __init__(self, db_path: str = _DEFAULT_DB):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT NOT NULL,
                    score       REAL NOT NULL,
                    top_signals TEXT,
                    admin_label TEXT NOT NULL,
                    reason      TEXT,
                    llm_verdict TEXT,
                    created_at  TEXT DEFAULT (datetime('now'))
                )
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def save(self, event: AnomalyEvent, label: str, reason: str = "") -> None:
        if label not in ("O", "X"):
            raise ValueError(f"label은 'O' 또는 'X'여야 합니다. 입력: {label!r}")
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO feedback (timestamp, score, top_signals, admin_label, reason, llm_verdict) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (str(event.timestamp), event.score,
                 json.dumps(event.top_signals, ensure_ascii=False),
                 label, reason, event.metadata.get("llm_verdict")),
            )

    def load_labeled(self) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query(
                "SELECT id, timestamp, score, top_signals, admin_label, reason, created_at "
                "FROM feedback ORDER BY timestamp",
                conn,
            )

    def retrain_scaffold(self) -> dict:
        """재학습 가능 여부와 라벨 통계를 반환한다. 실제 재학습 로직은 TODO."""
        df = self.load_labeled()
        n_normal  = int((df["admin_label"] == "X").sum())
        n_anomaly = int((df["admin_label"] == "O").sum())
        total = n_normal + n_anomaly
        ready = total >= _MIN_RETRAIN
        return {
            "total_labeled": total,
            "normal_count":  n_normal,
            "anomaly_count": n_anomaly,
            "ready":         ready,
            "message": (
                f"재학습 준비 완료 ({total}개 라벨 확보)"
                if ready
                else f"재학습까지 {_MIN_RETRAIN - total}개 더 필요"
            ),
            # TODO: retrain 시 활용
            # - label='X': IF 정상 훈련 데이터로 추가 → 오탐 감소
            # - label='O': contamination 재산정 기준점
        }
```

- [ ] **Step 4: 테스트 통과 확인**

```
pytest tests/test_feedback_store.py -v
```
Expected: 4개 PASSED

- [ ] **Step 5: 커밋**

```bash
git add src/services/feedback_store.py tests/test_feedback_store.py
git commit -m "feat: add FeedbackStore with SQLite and retrain scaffold"
```

---

## Task 7: LLM HITL 프롬프트 + settings.yaml

**Files:**
- Create: `config/llm_hitl_prompt.md`
- Modify: `config/settings.yaml`

- [ ] **Step 1: llm_hitl_prompt.md 작성**

```markdown
# config/llm_hitl_prompt.md
# Role & Objective
당신은 산업 데이터 이상치 탐색(Anomaly Detection)을 보조하고 모델을 고도화하는
'AI 데이터 분석 에이전트'입니다.
목표: 1차 ML 모델이 탐지한 이상치 후보군의 맥락을 분석하여 관리자에게 보고하고,
관리자의 피드백(O/X)을 학습 데이터베이스에 반영하여 전처리 및 ML 모델의 성능을
지속적으로 개선하는 것입니다.

# Constraints & Rules
1. 분석 대상은 반드시 전처리 단계에서 '안정적인 가동 중'으로 판별된 구간에 한정합니다.
   - [가동 판단 기준]: AutoOperationDiscovery 또는 equipment_profiles.yaml의 조건 전부 충족
2. 타임 윈도우는 전후 5분 트렌드 기준으로 분석하며, 단기 스파이크(노이즈)와
   경향성 이탈(이상 징후)을 명확히 구분하십시오.
3. 추측성 원인 분석 시 반드시 "추측입니다" / "가능성이 있습니다"라고 명시하고
   데이터(분포, 통계량) 기반 근거를 함께 제시하십시오.
4. 관리자 판단이 데이터 경향성과 다를 경우 그 이유를 지적하십시오.
   맹목적 동의 금지.

# Input Context
- 발생 시각: {timestamp}
- 이상 강도: {score_level}  (raw score: {score})
- 주요 급변 신호:
{top_signals}

## 전후 {context_window_minutes}분 신호 통계

{context_stats}

# Process

## Step 1: 1차 ML 결과 분석
- 주요 신호의 trend 값(구간 내 변화량)으로 이상 경향성을 판단합니다.
- 교차 검증 신호(동일 설비 연동 채널)를 비교하여 심각도를 상/중/하로 분류합니다.
  * 상: 복수 신호 동시 급변, 물리적 비정상 범위
  * 중: 주요 신호 이탈 확인, 연동 신호 안정
  * 하: 단일 신호만 변화

## Step 2: 관리자 브리핑
아래 포맷으로 브리핑하십시오:
  [발생 일시 및 탐지 구간]:
  [주요 이상 지표 및 트렌드 이탈 수준]:
  [교차 검증 결과 및 예상 원인 (데이터 근거, 추측 명시)]:
  [심각도 판정]: 상/중/하

## Step 3: 판정 및 피드백 요청
브리핑 마지막 줄을 반드시 아래 형식으로 마무리하십시오:

VERDICT: KEEP: {판단 이유 한 줄}
또는
VERDICT: REJECT: {판단 이유 한 줄}
```

- [ ] **Step 2: settings.yaml에 hitl 섹션 추가**

`config/settings.yaml` 하단에 추가:

```yaml
hitl:
  rolling:
    window_rows: 30        # ~63초 (2.1sec/row 기준)
  detector:
    contamination: 0.02
    n_estimators: 100
    window_size: 1         # rolling 변환 후 단일 포인트 분석
    top_n: 10
  discovery:
    top_binary_n: 3        # 이진 후보 최대 반환 수
    top_bimodal_n: 3       # 이중봉 후보 최대 반환 수
  feedback_db: "feedback.db"
  llm_prompt: "config/llm_hitl_prompt.md"
  context_minutes: 5
  min_retrain_samples: 20
```

- [ ] **Step 3: 커밋**

```bash
git add config/llm_hitl_prompt.md config/settings.yaml
git commit -m "config: add HITL prompt and hitl settings section"
```

---

## Task 8: hitl_pipeline.py (메인 스크립트)

**Files:**
- Create: `hitl_pipeline.py`

- [ ] **Step 1: 구현**

```python
# hitl_pipeline.py
"""HITL(Human-in-the-Loop) 능동 학습 파이프라인.

사용법:
  python hitl_pipeline.py --file 2603201549_oven.csv
  python hitl_pipeline.py --file 2603201549_oven.csv --no-interactive
  python hitl_pipeline.py --retrain-status
  python hitl_pipeline.py --show-profile oven
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).parent


def _load_settings() -> dict:
    with open(BASE_DIR / "config" / "settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_llm_prompt(template: str, payload: dict) -> str:
    top_signals_text = "\n".join(
        f"  - {s['name']}: 변동성={s['variability']:.4f}"
        for s in payload["top_signals"]
    )
    context_stats_text = "\n".join(
        f"  {col}: mean={v['mean']:.2f}, std={v['std']:.2f}, "
        f"범위=[{v['min']:.2f}~{v['max']:.2f}], trend={v['trend']:+.2f}"
        for col, v in payload["context_stats"].items()
    )
    return template.format(
        timestamp=payload["timestamp"],
        score_level=payload["score_level"],
        score=payload["score"],
        top_signals=top_signals_text,
        context_window_minutes=payload["context_window_minutes"],
        context_stats=context_stats_text,
    )


# ── Step 1: 가동 조건 확보 (캐시 또는 자동 탐색) ─────────────────────────────

def _get_operation_conditions(df, equipment_id: str, hitl_cfg: dict):
    """캐시 hit이면 바로 반환, miss이면 AutoDiscovery 후 관리자 확인."""
    from src.services.equipment_profile_store import EquipmentProfileStore
    from src.utils.operation_discovery import AutoOperationDiscovery

    disc_cfg = hitl_cfg.get("discovery", {})
    store = EquipmentProfileStore()

    cached = store.load(equipment_id)
    if cached and store.is_confirmed(equipment_id):
        print(f"  [프로파일 캐시 hit] '{equipment_id}' — 저장된 조건 {len(cached)}개 적용")
        return cached

    # 캐시 miss 또는 미확인 → 자동 탐색
    print(f"  [프로파일 없음] '{equipment_id}' — 가동 신호 자동 탐색 중...")
    discoverer = AutoOperationDiscovery(
        top_binary_n=int(disc_cfg.get("top_binary_n", 3)),
        top_bimodal_n=int(disc_cfg.get("top_bimodal_n", 3)),
    )
    conditions = discoverer.discover(df)

    if not conditions:
        print("  [경고] 가동 판별 후보 신호를 찾지 못했습니다. 전체 데이터로 진행합니다.")
        return []

    # 발견 결과 출력
    print(f"\n  발견된 가동 판별 후보 조건 ({len(conditions)}개):")
    for i, c in enumerate(conditions, 1):
        print(f"    {i}. [{c.source}] {c.column} {c.op} {c.value:.2f}  (신뢰도 {c.confidence:.2f})")

    # 관리자 확인
    ans = input("\n  이 조건들로 가동 상태를 판별하겠습니다. 확인하시겠습니까? [Y=확인 / N=건너뜀]: ").strip().upper()
    confirmed = ans == "Y"
    store.save(equipment_id, conditions, confirmed=confirmed)

    if confirmed:
        print(f"  → '{equipment_id}' 프로파일 저장 완료.")
    else:
        print("  → 미확인 상태로 저장. 이번 실행에서는 발견된 조건을 그대로 사용합니다.")

    return conditions


# ── 관리자 피드백 루프 ──────────────────────────────────────────────────────

def _interactive_feedback(event, context: dict, store) -> None:
    print("\n" + "=" * 70)
    print(context["text_summary"])
    print("=" * 70)

    while True:
        ans = input("\n이 탐지 결과가 실제 이상입니까? [O=이상 확정 / X=정상 패턴 / S=건너뜀]: ").strip().upper()
        if ans in ("O", "X", "S"):
            break
        print("  O, X, S 중 하나를 입력하세요.")

    if ans == "S":
        print("  → 건너뜀.")
        return

    reason = input("  판단 근거를 입력하세요 (엔터=생략): ").strip()
    store.save(event, label=ans, reason=reason)
    print(f"  → '{ans}' 피드백 저장 완료.")


# ── 메인 파이프라인 ────────────────────────────────────────────────────────

def run_hitl_pipeline(csv_path: str, interactive: bool = True) -> list[dict]:
    from src.services.loader import IbaCSVLoader
    from src.utils.preprocessor import Preprocessor
    from src.utils.operation_filter import OperationFilter
    from src.utils.rolling_features import RollingFeatureExtractor
    from src.agents.isolation_forest_adapter import IsolationForestAdapter
    from src.agents.llm_filter import LLMFilter
    from src.agents.llm_backends import build_llm_backend
    from src.utils.context_formatter import AnomalyContextFormatter
    from src.services.feedback_store import FeedbackStore
    from src.services.equipment_profile_store import EquipmentProfileStore

    settings  = _load_settings()
    hitl_cfg  = settings.get("hitl", {})
    det_cfg   = hitl_cfg.get("detector", {})
    roll_cfg  = hitl_cfg.get("rolling", {})
    equipment_id = EquipmentProfileStore.extract_id(Path(csv_path).name)

    # ── Step 1: 로드 + 수치 전처리 ──────────────────────────────────────────
    print(f"\n[Step 1] CSV 로드: {csv_path}")
    df_raw = Preprocessor().process(IbaCSVLoader().load(csv_path))
    print(f"  전체: {len(df_raw)}행 × {len(df_raw.columns)}열")

    # ── Step 1-B: 가동 조건 확보 (캐시 or 자동탐색) ─────────────────────────
    if interactive:
        conditions = _get_operation_conditions(df_raw, equipment_id, hitl_cfg)
    else:
        # 비인터랙티브: 캐시가 있으면 사용, 없으면 자동 탐색만 (확인 생략)
        from src.services.equipment_profile_store import EquipmentProfileStore
        from src.utils.operation_discovery import AutoOperationDiscovery
        store_ep = EquipmentProfileStore()
        conditions = store_ep.load(equipment_id) or \
                     AutoOperationDiscovery().discover(df_raw)

    # ── Step 1-C: 가동 구간 필터링 ──────────────────────────────────────────
    df_running = OperationFilter(conditions).filter(df_raw)
    pct = len(df_running) / max(len(df_raw), 1) * 100
    print(f"  가동 구간: {len(df_running)}행 ({pct:.1f}%)")

    if df_running.empty:
        print("  [경고] 가동 구간 없음 — 파이프라인 종료.")
        return []

    # ── Step 2: Rolling 특징 추출 → IF 탐지 ────────────────────────────────
    print(f"\n[Step 2] Rolling 특징 추출 (window={roll_cfg.get('window_rows', 30)}행)")
    df_feat = RollingFeatureExtractor(
        window=int(roll_cfg.get("window_rows", 30))
    ).transform(df_running)

    candidates = IsolationForestAdapter(
        window_size=int(det_cfg.get("window_size", 1)),
        contamination=float(det_cfg.get("contamination", 0.02)),
        n_estimators=int(det_cfg.get("n_estimators", 100)),
        top_n=int(det_cfg.get("top_n", 10)),
    ).detect(df_feat)
    print(f"  IF 탐지 후보: {len(candidates)}건")

    if not candidates:
        print("  이상 후보 없음.")
        return []

    # ── Step 2-B: LLM 1차 자동 필터 ────────────────────────────────────────
    backend = build_llm_backend(settings.get("llm", {}))
    if backend:
        print(f"\n[Step 2-B] LLM 1차 필터 ({backend.name()})...")
        events = LLMFilter(backend).filter(candidates, df_running)
        print(f"  LLM KEEP: {len(events)}건")
    else:
        events = candidates
        print("  LLM 비활성화 — 전체 후보를 관리자 검토로 전달")

    # ── Step 3: 컨텍스트 포맷팅 ─────────────────────────────────────────────
    print(f"\n[Step 3] 컨텍스트 포맷팅 (±{hitl_cfg.get('context_minutes', 5)}분)")
    formatter = AnomalyContextFormatter(
        context_minutes=int(hitl_cfg.get("context_minutes", 5))
    )
    contexts = []
    for event in events:
        ctx = formatter.format(event, df_running)
        ctx["event"]   = event
        ctx["payload"] = json.loads(ctx["json_payload"])
        contexts.append(ctx)

    # ── Step 4: 관리자 피드백 루프 ──────────────────────────────────────────
    fb_store = FeedbackStore(
        db_path=str(BASE_DIR / hitl_cfg.get("feedback_db", "feedback.db"))
    )

    if interactive:
        print(f"\n[Step 4] 관리자 피드백 루프 ({len(events)}건)")
        for i, ctx in enumerate(contexts, 1):
            print(f"\n  [{i}/{len(events)}]", end="")
            _interactive_feedback(ctx["event"], ctx, fb_store)

        status = fb_store.retrain_scaffold()
        print(f"\n[재학습 상태] {status['message']}")
    else:
        for ctx in contexts:
            print(ctx["text_summary"])

    return contexts


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PIMS HITL 파이프라인")
    parser.add_argument("--file",            help="분석할 CSV 파일 경로")
    parser.add_argument("--no-interactive",  action="store_true")
    parser.add_argument("--retrain-status",  action="store_true")
    parser.add_argument("--show-profile",    metavar="EQUIPMENT_ID")
    args = parser.parse_args()

    settings = _load_settings()
    hitl_cfg = settings.get("hitl", {})

    if args.retrain_status:
        from src.services.feedback_store import FeedbackStore
        status = FeedbackStore(
            db_path=str(BASE_DIR / hitl_cfg.get("feedback_db", "feedback.db"))
        ).retrain_scaffold()
        print(json.dumps(status, ensure_ascii=False, indent=2))

    elif args.show_profile:
        from src.services.equipment_profile_store import EquipmentProfileStore
        store = EquipmentProfileStore()
        conds = store.load(args.show_profile)
        if conds is None:
            print(f"'{args.show_profile}' 프로파일 없음.")
        else:
            confirmed = store.is_confirmed(args.show_profile)
            print(f"설비: {args.show_profile}  (confirmed={confirmed})")
            for c in conds:
                print(f"  {c.column} {c.op} {c.value:.2f}  [{c.source}, conf={c.confidence:.2f}]")

    elif args.file:
        run_hitl_pipeline(args.file, interactive=not args.no_interactive)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 비인터랙티브 모드 동작 확인**

```bash
python hitl_pipeline.py --file 2603201549_oven.csv --no-interactive
```

예상 출력:
```
[Step 1] CSV 로드: 2603201549_oven.csv
  전체: 285행 × ...열
  [프로파일 없음] 'oven' — 가동 신호 자동 탐색 중...
  발견된 가동 판별 후보 조건 (N개):
    1. [binary] ... == 1.00  (신뢰도 ...)
    2. [bimodal] ... > ...   (신뢰도 ...)
  가동 구간: N행 (N.N%)
[Step 2] Rolling 특징 추출 ...
  IF 탐지 후보: N건
[이상 탐지 리포트]
  발생 시각: ...
```

- [ ] **Step 3: 커밋**

```bash
git add hitl_pipeline.py
git commit -m "feat: add HITL pipeline with auto-discovery, caching, and feedback loop"
```

---

## Task 9: 전체 테스트 통과 확인

- [ ] **Step 1: 전체 테스트 실행**

```bash
pytest tests/ -v --tb=short 2>&1 | head -80
```

Expected: 신규 6개 테스트 파일 포함 전체 PASSED

- [ ] **Step 2: 린트 확인**

```bash
python -m py_compile \
  src/utils/operation_discovery.py \
  src/utils/operation_filter.py \
  src/utils/rolling_features.py \
  src/utils/context_formatter.py \
  src/services/equipment_profile_store.py \
  src/services/feedback_store.py \
  hitl_pipeline.py
echo "문법 오류 없음"
```

- [ ] **Step 3: 최종 커밋**

```bash
git add -A
git commit -m "test: verify all HITL pipeline tests pass"
```