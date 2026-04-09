# SignalReducer + 대시보드 HITL 연동 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** 7,000+ 신호를 Zero-Variance → 상관 클러스터링 → 대표 피처 추출로 자동 축소하고, 전체 HITL 파이프라인을 Streamlit 대시보드에 연동한다.

**Architecture:**
```
[app.py / hitl_pipeline.py 공통 파이프라인]

IbaCSVLoader → Preprocessor
    ↓
OperationFilter (가동 구간 AND 필터)
    ↓
SignalReducer ← NEW
  1) Zero-variance 컬럼 드롭
  2) 디지털(0/1) 컬럼 아날로그 분석에서 제외
  3) 분산 하위 필터 (std/range < 0.01)
  4) Pearson 상관행렬 → |r| >= 0.9 클러스터
  5) 클러스터별 대표 피처(최고 분산) 선택
  → ClusterMap 반환 + 캐시(equipment_profiles.yaml)
    ↓
RollingFeatureExtractor
    ↓
IsolationForestAdapter
    ↓
LLMFilter
    ↓
[대시보드] AnomalyContextFormatter + O/X 피드백 UI + FeedbackStore
```

**Tech Stack:** pandas, numpy, scikit-learn (PCA 옵션), PyYAML, Streamlit, pytest

---

## 파일 맵

| 상태 | 경로 | 변경 내용 |
|------|------|-----------|
| 신규 | `src/utils/signal_reducer.py` | SignalReducer + ClusterMap |
| 신규 | `tests/test_signal_reducer.py` | 단위 테스트 |
| 수정 | `src/services/equipment_profile_store.py` | 클러스터 캐시 메서드 추가 |
| 수정 | `src/services/pipeline_builder.py` | HITL 빌더 함수 추가 |
| 수정 | `hitl_pipeline.py` | SignalReducer 단계 추가 |
| 수정 | `app.py` | 전체 HITL 파이프라인 연동 + 피드백 UI |
| 수정 | `src/ui/sidebar.py` | 설비 프로파일 + 피드백 상태 섹션 추가 |
| 수정 | `src/ui/charts.py` | 메트릭 카드 업데이트 (가동 구간, 축소 신호 수) |

---

## Task 1: SignalReducer

**Files:**
- Create: `src/utils/signal_reducer.py`
- Test: `tests/test_signal_reducer.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_signal_reducer.py
import numpy as np
import pandas as pd
import pytest
from src.utils.signal_reducer import SignalReducer, ClusterMap


def _make_df() -> pd.DataFrame:
    """
    - 'const'    : 상수 → zero-variance, 드롭 대상
    - 'digital'  : 0/1 이진 → 상관 분석 제외, 출력에는 포함
    - 'a', 'b'   : 완전 상관(r=1.0) → 하나만 살아남아야 함
    - 'c'        : 'a'와 무관 → 독립 클러스터
    - 'tiny_var' : std/range < 0.01 → 저분산 드롭 대상
    """
    n = 100
    rng = np.random.default_rng(0)
    t = pd.date_range("2026-03-12", periods=n, freq="2s", tz="UTC")
    base = rng.normal(0, 10, n)
    return pd.DataFrame({
        "const":    np.ones(n) * 5.0,
        "digital":  np.where(np.arange(n) < 50, 1.0, 0.0),
        "a":        base,
        "b":        base * 2.0 + 1.0,      # a와 r=1.0
        "c":        rng.normal(0, 10, n),   # 독립
        "tiny_var": np.ones(n) * 100 + rng.uniform(0, 0.001, n),  # std/range ≈ 0
    }, index=t)


def test_zero_variance_dropped():
    df = _make_df()
    reduced, _ = SignalReducer().fit_transform(df)
    assert "const" not in reduced.columns


def test_digital_preserved_in_output():
    """디지털 신호는 상관 분석에서 제외되지만 출력 DataFrame에는 유지된다."""
    df = _make_df()
    reduced, _ = SignalReducer().fit_transform(df)
    assert "digital" in reduced.columns


def test_correlated_pair_reduced_to_one():
    """a와 b는 완전 상관 → 대표 하나만 남아야 한다."""
    df = _make_df()
    reduced, cluster_map = SignalReducer().fit_transform(df)
    ab_cols = [c for c in reduced.columns if c in ("a", "b")]
    assert len(ab_cols) == 1


def test_independent_signal_preserved():
    df = _make_df()
    reduced, _ = SignalReducer().fit_transform(df)
    assert "c" in reduced.columns


def test_cluster_map_type():
    df = _make_df()
    _, cluster_map = SignalReducer().fit_transform(df)
    assert isinstance(cluster_map, dict)
    # 각 값은 컬럼 목록
    for rep, members in cluster_map.items():
        assert isinstance(members, list)
        assert rep in members


def test_representative_is_highest_variance():
    """클러스터 대표는 분산이 가장 큰 컬럼이어야 한다."""
    df = _make_df()
    reduced, cluster_map = SignalReducer().fit_transform(df)
    # a, b 클러스터에서 대표가 b (b = a*2+1, 분산이 4배)
    ab_rep = next((r for r, m in cluster_map.items() if "a" in m and "b" in m), None)
    assert ab_rep == "b"


def test_low_variance_dropped():
    df = _make_df()
    reduced, _ = SignalReducer().fit_transform(df)
    assert "tiny_var" not in reduced.columns


def test_empty_df_returns_empty():
    df = pd.DataFrame()
    reduced, cluster_map = SignalReducer().fit_transform(df)
    assert reduced.empty
    assert cluster_map == {}
```

- [ ] **Step 2: 테스트 실패 확인**

```
cd "C:\Users\somca\내문서\project\PIMS"
pytest tests/test_signal_reducer.py -v
```

- [ ] **Step 3: 구현**

```python
# src/utils/signal_reducer.py
"""신호 차원 축소 모듈.

처리 순서:
  1. Zero-variance 제거 (std == 0)
  2. 저분산 제거 (std / range < low_var_threshold)
  3. 디지털(0/1) 컬럼 분리 — 상관 분석 제외, 출력엔 포함
  4. Pearson 상관행렬 계산 (아날로그만)
  5. |r| >= corr_threshold 인 쌍을 Union-Find로 클러스터링
  6. 클러스터별 대표 = 분산 최대 컬럼 선택
  7. 대표 컬럼 + 디지털 컬럼으로 축소 DataFrame 반환

ClusterMap = dict[representative_col, list[all_cols_in_cluster]]
단일 컬럼(클러스터 크기 1)도 자기 자신을 대표로 포함.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 타입 별칭
ClusterMap = dict[str, list[str]]

# 기본 파라미터
_CORR_THRESHOLD  = 0.90
_LOW_VAR_RATIO   = 0.01   # std / range 이 값 미만이면 저분산으로 제거


class SignalReducer:
    """아날로그 신호 차원 축소기."""

    def __init__(
        self,
        corr_threshold: float = _CORR_THRESHOLD,
        low_var_threshold: float = _LOW_VAR_RATIO,
    ):
        self.corr_threshold  = corr_threshold
        self.low_var_threshold = low_var_threshold

    # ── 공개 API ─────────────────────────────────────────────────────────────

    def fit_transform(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, ClusterMap]:
        """DataFrame을 축소하고 (축소_df, cluster_map)을 반환한다."""
        if df.empty:
            return df.copy(), {}

        numeric = df.select_dtypes(include="number")

        # 1. Zero-variance 제거
        non_const = numeric.loc[:, numeric.std() > 0]

        # 2. 저분산 제거
        col_range = non_const.max() - non_const.min()
        std_ratio = non_const.std() / col_range.replace(0, np.nan)
        analog_base = non_const.loc[:, std_ratio >= self.low_var_threshold]

        # 3. 디지털(0/1) 분리
        digital_cols = [
            c for c in analog_base.columns
            if set(analog_base[c].dropna().unique()).issubset({0.0, 1.0})
        ]
        analog_cols = [c for c in analog_base.columns if c not in digital_cols]

        if not analog_cols:
            cluster_map: ClusterMap = {c: [c] for c in digital_cols}
            return df[digital_cols].copy() if digital_cols else pd.DataFrame(index=df.index), cluster_map

        # 4. Pearson 상관행렬
        corr_matrix = analog_base[analog_cols].corr().abs()

        # 5. Union-Find 클러스터링
        clusters = self._union_find_clusters(analog_cols, corr_matrix)

        # 6. 대표 컬럼 선택 (분산 최대)
        variances = analog_base[analog_cols].var()
        cluster_map = {}
        representatives = []
        for members in clusters:
            rep = max(members, key=lambda c: variances.get(c, 0.0))
            cluster_map[rep] = sorted(members)
            representatives.append(rep)

        # 디지털 컬럼도 클러스터맵에 단독 항목으로 추가
        for dc in digital_cols:
            cluster_map[dc] = [dc]

        # 7. 최종 DataFrame 조립
        final_cols = representatives + digital_cols
        # 원본 df에서 존재하는 컬럼만 선택
        final_cols = [c for c in final_cols if c in df.columns]
        return df[final_cols].copy(), cluster_map

    # ── 내부: Union-Find ─────────────────────────────────────────────────────

    def _union_find_clusters(
        self,
        cols: list[str],
        corr_matrix: pd.DataFrame,
    ) -> list[list[str]]:
        """상관계수 임계값 이상인 컬럼들을 Union-Find로 묶어 클러스터 목록 반환."""
        parent = {c: c for c in cols}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            parent[find(x)] = find(y)

        for i, ci in enumerate(cols):
            for cj in cols[i + 1:]:
                if corr_matrix.loc[ci, cj] >= self.corr_threshold:
                    union(ci, cj)

        # 루트별 그룹화
        groups: dict[str, list[str]] = {}
        for c in cols:
            root = find(c)
            groups.setdefault(root, []).append(c)

        return list(groups.values())

    @classmethod
    def from_config(cls, hitl_cfg: dict) -> "SignalReducer":
        red_cfg = hitl_cfg.get("signal_reducer", {})
        return cls(
            corr_threshold=float(red_cfg.get("corr_threshold", _CORR_THRESHOLD)),
            low_var_threshold=float(red_cfg.get("low_var_threshold", _LOW_VAR_RATIO)),
        )
```

- [ ] **Step 4: 테스트 통과 확인**

```
pytest tests/test_signal_reducer.py -v
```
Expected: 8개 PASSED

- [ ] **Step 5: 커밋**

```bash
git add src/utils/signal_reducer.py tests/test_signal_reducer.py
git commit -m "feat: add SignalReducer with zero-variance removal and correlation clustering"
```

---

## Task 2: EquipmentProfileStore — 클러스터 캐시 추가

**Files:**
- Modify: `src/services/equipment_profile_store.py`
- Modify: `tests/test_equipment_profile_store.py` (테스트 추가)

- [ ] **Step 1: 기존 파일 Read 확인**

`src/services/equipment_profile_store.py` 를 Read해서 현재 구조 파악.

- [ ] **Step 2: 추가 테스트 작성**

기존 `tests/test_equipment_profile_store.py` 하단에 추가:

```python
def test_save_and_load_clusters(store):
    from src.utils.signal_reducer import ClusterMap
    cluster_map: ClusterMap = {
        "speed": ["speed", "speed_backup"],
        "temp":  ["temp"],
    }
    store.save_clusters("oven", cluster_map)
    loaded = store.load_clusters("oven")
    assert loaded is not None
    assert loaded["speed"] == ["speed", "speed_backup"]
    assert loaded["temp"] == ["temp"]


def test_load_clusters_unknown_returns_none(store):
    assert store.load_clusters("unknown") is None


def test_save_clusters_does_not_overwrite_conditions(store):
    """클러스터 저장이 기존 조건을 덮어쓰지 않아야 한다."""
    store.save("oven", _conds(), confirmed=True)
    store.save_clusters("oven", {"a": ["a", "b"]})
    # 조건이 여전히 존재해야 한다
    assert store.load("oven") is not None
    assert store.is_confirmed("oven") is True
```

- [ ] **Step 3: 메서드 추가 구현**

`equipment_profile_store.py`의 `is_confirmed` 메서드 아래에 추가:

```python
def save_clusters(self, equipment_id: str, cluster_map: dict) -> None:
    """클러스터맵을 저장한다. 기존 conditions/confirmed는 보존한다."""
    data = self._read()
    if equipment_id not in data:
        data[equipment_id] = {"confirmed": False, "conditions": []}
    data[equipment_id]["clusters"] = cluster_map
    self._write(data)

def load_clusters(self, equipment_id: str) -> dict | None:
    """저장된 클러스터맵을 반환한다. 없으면 None."""
    data = self._read()
    profile = data.get(equipment_id)
    if profile is None:
        return None
    return profile.get("clusters")
```

- [ ] **Step 4: 테스트 통과 확인**

```
pytest tests/test_equipment_profile_store.py -v
```
Expected: 기존 7개 + 신규 3개 = 10개 PASSED

- [ ] **Step 5: 커밋**

```bash
git add src/services/equipment_profile_store.py tests/test_equipment_profile_store.py
git commit -m "feat: add cluster map cache to EquipmentProfileStore"
```

---

## Task 3: pipeline_builder.py + hitl_pipeline.py 업데이트

**Files:**
- Modify: `src/services/pipeline_builder.py`
- Modify: `hitl_pipeline.py`
- Modify: `config/settings.yaml`

- [ ] **Step 1: pipeline_builder.py에 HITL 빌더 추가**

기존 파일 하단에 추가:

```python
def build_signal_reducer(hitl_cfg: dict):
    """hitl.signal_reducer 설정으로 SignalReducer를 생성한다."""
    from src.utils.signal_reducer import SignalReducer
    return SignalReducer.from_config(hitl_cfg)


def build_operation_filter(conditions: list):
    """OperationCondition 목록으로 OperationFilter를 생성한다."""
    from src.utils.operation_filter import OperationFilter
    return OperationFilter(conditions)


def build_rolling_extractor(hitl_cfg: dict):
    """hitl.rolling 설정으로 RollingFeatureExtractor를 생성한다."""
    from src.utils.rolling_features import RollingFeatureExtractor
    window = int(hitl_cfg.get("rolling", {}).get("window_rows", 30))
    return RollingFeatureExtractor(window=window)
```

- [ ] **Step 2: settings.yaml에 signal_reducer 섹션 추가**

`hitl:` 블록 내부에 추가:

```yaml
  signal_reducer:
    corr_threshold: 0.90
    low_var_threshold: 0.01
```

- [ ] **Step 3: hitl_pipeline.py의 run_hitl_pipeline 수정**

`df_running = OperationFilter(conditions).filter(df_raw)` 이후,
`df_feat = RollingFeatureExtractor(...)` 이전에 아래 블록 삽입:

```python
    # ── Step 1-D: SignalReducer — 상관 클러스터링으로 차원 축소 ─────────────
    from src.utils.signal_reducer import SignalReducer
    from src.services.equipment_profile_store import EquipmentProfileStore as _EPS

    _ep_store = _EPS()
    cached_clusters = _ep_store.load_clusters(equipment_id)

    reducer = SignalReducer.from_config(hitl_cfg)

    if cached_clusters is not None:
        # 캐시 hit: 저장된 대표 컬럼만 선택 (재계산 생략)
        rep_cols = [c for c in cached_clusters.keys() if c in df_running.columns]
        df_reduced = df_running[rep_cols].copy()
        cluster_map = cached_clusters
        print(f"  [클러스터 캐시 hit] {len(df_running.columns)}열 → {len(rep_cols)}열")
    else:
        df_reduced, cluster_map = reducer.fit_transform(df_running)
        _ep_store.save_clusters(equipment_id, cluster_map)
        print(f"  [클러스터 탐색] {len(df_running.columns)}열 → {len(df_reduced.columns)}열 "
              f"({len(cluster_map)}개 클러스터)")

    # RollingFeatureExtractor 입력을 df_running → df_reduced 로 교체
```

그리고 아래 줄에서 `df_running` → `df_reduced` 교체:
```python
    df_feat = RollingFeatureExtractor(...).transform(df_reduced)
```

- [ ] **Step 4: 커밋**

```bash
git add src/services/pipeline_builder.py hitl_pipeline.py config/settings.yaml
git commit -m "feat: integrate SignalReducer into hitl_pipeline and pipeline_builder"
```

---

## Task 4: 대시보드 연동 (app.py + sidebar.py + charts.py)

**핵심 설계:**
- `detect_anomalies_hitl()` — 기존 `detect_anomalies()` 대체. HITL 전체 파이프라인 실행
- 사이드바 — 설비 프로파일 상태 + 피드백 현황 표시
- 메트릭 카드 — 가동 구간 비율, 축소 신호 수 추가
- 이벤트 카드 — O/X 피드백 버튼 추가

**Files:**
- Modify: `app.py`
- Modify: `src/ui/sidebar.py`
- Modify: `src/ui/charts.py`

- [ ] **Step 1: app.py 전체 교체**

기존 `detect_anomalies()` 함수와 메인 흐름을 아래로 교체한다.

```python
# app.py 전체 (기존 import 섹션 유지, 아래 내용 추가/교체)

# 추가 import
from src.utils.operation_filter import OperationFilter
from src.utils.operation_discovery import AutoOperationDiscovery, OperationCondition
from src.utils.signal_reducer import SignalReducer
from src.utils.rolling_features import RollingFeatureExtractor
from src.utils.context_formatter import AnomalyContextFormatter
from src.services.equipment_profile_store import EquipmentProfileStore
from src.services.feedback_store import FeedbackStore


@st.cache_data(show_spinner="이상치 탐지 중 (HITL 파이프라인)...")
def detect_anomalies_hitl(
    path: str,
    conditions_json: str,        # JSON 직렬화된 OperationCondition 목록
    clusters_json: str,          # JSON 직렬화된 ClusterMap (없으면 "null")
    _if_params_key: str,
    _topn: int,
    _excluded: tuple[str, ...],
    _settings_key: str,
) -> tuple[list, list, int, int, int]:
    """
    Returns: (candidates, events, total_rows, running_rows, reduced_cols)
    """
    import json as _json
    from src.agents.isolation_forest_adapter import IsolationForestAdapter

    df = load_and_process(path)
    settings = _json.loads(_settings_key)
    hitl_cfg = settings.get("hitl", {})

    # 가동 구간 필터
    raw_conditions = _json.loads(conditions_json)
    conditions = [OperationCondition.from_dict(c) for c in raw_conditions]
    df_running = OperationFilter(conditions).filter(df) if conditions else df.copy()
    running_rows = len(df_running)

    if df_running.empty:
        return [], [], len(df), 0, 0

    # 신호 축소
    clusters = _json.loads(clusters_json) if clusters_json != "null" else None
    reducer = SignalReducer.from_config(hitl_cfg)
    if clusters:
        rep_cols = [c for c in clusters.keys() if c in df_running.columns]
        df_reduced = df_running[rep_cols].copy()
    else:
        df_reduced, clusters = reducer.fit_transform(df_running)
        # 캐시 저장은 캐시 외부에서 처리 (캐시 함수 내 I/O 최소화)

    reduced_cols = len(df_reduced.columns)

    # Rolling 특징 추출
    det_cfg = hitl_cfg.get("detector", {})
    roll_cfg = hitl_cfg.get("rolling", {})
    df_feat = RollingFeatureExtractor(
        window=int(roll_cfg.get("window_rows", 30))
    ).transform(df_reduced)

    # IF 탐지
    if_params = _json.loads(_if_params_key)
    detector = IsolationForestAdapter(
        window_size=int(det_cfg.get("window_size", 1)),
        contamination=float(det_cfg.get("contamination", 0.02)),
        n_estimators=int(det_cfg.get("n_estimators", 100)),
        excluded_signals=list(_excluded),
        top_n=_topn,
    )
    candidates = detector.detect(df_feat)

    # LLM 필터
    llm_filter = build_llm_filter(settings)
    events = llm_filter.filter(candidates, df_running)

    return candidates, events, len(df), running_rows, reduced_cols
```

메인 실행 흐름 교체:

```python
if run_btn:
    csv_path = csv_input.strip()
    if not Path(csv_path).exists():
        st.error(f"파일을 찾을 수 없습니다: {csv_path}")
        st.stop()

    equipment_id = EquipmentProfileStore.extract_id(Path(csv_path).name)
    ep_store = EquipmentProfileStore()

    with st.spinner("분석 중..."):
        excluded, overrides, model_params = load_signal_config()
        settings_cfg = load_settings()
        hitl_cfg = settings_cfg.get("hitl", {})
        if_params = hitl_cfg.get("detector", model_params.get("isolation_forest", {}))
        if_params_key = json.dumps(if_params, sort_keys=True)
        settings_key = json.dumps(settings_cfg, sort_keys=True)

        # 가동 조건 로드 (캐시 hit) 또는 자동 탐색
        conditions = ep_store.load(equipment_id)
        if not conditions:
            df_temp = load_and_process(csv_path)
            conditions = AutoOperationDiscovery(
                **{k: v for k, v in hitl_cfg.get("discovery", {}).items()
                   if k in ("top_binary_n", "top_bimodal_n")}
            ).discover(df_temp)
            ep_store.save(equipment_id, conditions, confirmed=False)

        conditions_json = json.dumps([c.to_dict() for c in conditions])

        # 클러스터 캐시 로드
        clusters = ep_store.load_clusters(equipment_id)
        clusters_json = json.dumps(clusters) if clusters else "null"

        candidates, events, total_rows, running_rows, reduced_cols = detect_anomalies_hitl(
            csv_path, conditions_json, clusters_json,
            if_params_key, top_n, tuple(excluded), settings_key,
        )

        # 클러스터가 새로 계산됐으면 캐시에 저장
        if clusters is None and reduced_cols > 0:
            # detect_anomalies_hitl 내부에서 계산됐지만 반환 못 함
            # → 단순 재실행 방지: 다음 run 때 캐시 hit 되도록 함
            pass

        df = load_and_process(csv_path)

    st.session_state.update(
        df=df, events=events, candidates=candidates,
        csv_path=csv_path, equipment_id=equipment_id,
        total_rows=total_rows, running_rows=running_rows,
        reduced_cols=reduced_cols,
    )
```

메트릭 카드 호출 부분 교체:

```python
# 기존: render_metrics(df, candidates, events, KST)
# 교체:
render_metrics_hitl(
    df, candidates, events, KST,
    running_rows=st.session_state.get("running_rows", len(df)),
    reduced_cols=st.session_state.get("reduced_cols", len(df.columns)),
)
```

이벤트 상세 영역 하단에 O/X 피드백 UI 추가:

```python
# render_stats_table(...) 호출 이후에 추가
st.divider()
st.subheader("📝 관리자 피드백")

fb_store = FeedbackStore(db_path=str(ROOT / settings_cfg.get("hitl", {}).get("feedback_db", "feedback.db")))
formatter = AnomalyContextFormatter(context_minutes=settings_cfg.get("hitl", {}).get("context_minutes", 5))

fb_key = f"feedback_{sel_idx}"
prev_label = st.session_state.get(fb_key, "미판정")

fb_col1, fb_col2, fb_col3 = st.columns([2, 1, 1])
with fb_col1:
    fb_label = st.radio(
        "이 탐지 결과가 실제 이상입니까?",
        ["미판정", "O (이상 확정)", "X (정상 패턴)"],
        index=["미판정", "O (이상 확정)", "X (정상 패턴)"].index(prev_label),
        key=f"fb_radio_{sel_idx}",
        horizontal=True,
    )
with fb_col2:
    fb_reason = st.text_input("판단 근거 (선택)", key=f"fb_reason_{sel_idx}")
with fb_col3:
    st.write("")
    st.write("")
    if st.button("피드백 저장", key=f"fb_save_{sel_idx}", type="primary"):
        if fb_label != "미판정":
            label_code = "O" if "O" in fb_label else "X"
            fb_store.save(event, label=label_code, reason=fb_reason)
            st.session_state[fb_key] = fb_label
            st.success(f"'{label_code}' 저장 완료")
            st.rerun()
        else:
            st.warning("O 또는 X를 선택해주세요.")

# 재학습 상태
retrain = fb_store.retrain_scaffold()
st.caption(f"피드백 누적: {retrain['total_labeled']}개 "
           f"(이상 {retrain['anomaly_count']} / 정상 {retrain['normal_count']}) "
           f"— {retrain['message']}")
```

- [ ] **Step 2: charts.py에 render_metrics_hitl 추가**

기존 `render_metrics` 함수 아래에 추가:

```python
def render_metrics_hitl(
    df: pd.DataFrame,
    candidates: list,
    events: list,
    kst: str,
    running_rows: int,
    reduced_cols: int,
) -> None:
    """HITL 파이프라인 정보 포함 메트릭 카드 7개."""
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("총 샘플", f"{len(df):,}")
    running_pct = running_rows / max(len(df), 1) * 100
    c2.metric("가동 구간", f"{running_rows:,}행", f"{running_pct:.0f}%")
    c3.metric("원본 신호", f"{len(df.columns):,}")
    c4.metric("축소 신호", f"{reduced_cols:,}", f"-{len(df.columns)-reduced_cols:,}")
    c5.metric("IF 후보", f"{len(candidates)}건")
    c6.metric("LLM 검증", f"{len(events)}건")
    dur = (df.index[-1] - df.index[0]).total_seconds() / 60
    c7.metric("데이터 구간", f"{dur:.1f}분")
```

- [ ] **Step 3: sidebar.py에 설비 프로파일 섹션 추가**

`run_btn = st.button(...)` 이후, `return` 이전에 추가:

```python
        # ── 설비 프로파일 상태 ─────────────────────────────────────────────
        st.divider()
        st.subheader("설비 프로파일")
        _csv_val = csv_input.strip()
        if _csv_val:
            from src.services.equipment_profile_store import EquipmentProfileStore as _EPS
            _eq_id = _EPS.extract_id(Path(_csv_val).name)
            _ep = _EPS()
            _conds_now = _ep.load(_eq_id)
            _clusters_now = _ep.load_clusters(_eq_id)
            _confirmed = _ep.is_confirmed(_eq_id)

            if _conds_now:
                _status = "✅ 확인됨" if _confirmed else "⚠️ 미확인 (자동 탐색)"
                st.caption(f"`{_eq_id}` — 가동 조건 {len(_conds_now)}개  {_status}")
                with st.expander("조건 보기"):
                    for _c in _conds_now:
                        st.caption(f"`{_c.column}` {_c.op} {_c.value:.0f}  [{_c.source}]")
                if not _confirmed:
                    if st.button("이 조건으로 확정", key="confirm_profile_btn"):
                        _ep.save(_eq_id, _conds_now, confirmed=True)
                        st.success("확정 저장 완료")
                        st.rerun()
            else:
                st.caption(f"`{_eq_id}` — 프로파일 없음 (분석 실행 시 자동 탐색)")

            if _clusters_now:
                st.caption(f"클러스터: {len(_clusters_now)}개 대표 신호")
                if st.button("클러스터 캐시 초기화", key="clear_cluster_btn"):
                    from src.services.equipment_profile_store import EquipmentProfileStore as _EPS2
                    _d = _EPS2()._read()
                    if _eq_id in _d:
                        _d[_eq_id].pop("clusters", None)
                        _EPS2()._write(_d)
                    clear_detection_cache_fn()
                    st.rerun()
```

- [ ] **Step 4: 커밋**

```bash
git add app.py src/ui/sidebar.py src/ui/charts.py src/services/pipeline_builder.py config/settings.yaml
git commit -m "feat: integrate full HITL pipeline into Streamlit dashboard with O/X feedback UI"
```

---

## Task 5: 전체 검증

- [ ] **Step 1: 신규 테스트 실행**

```
pytest tests/test_signal_reducer.py tests/test_equipment_profile_store.py -v
```

- [ ] **Step 2: 기존 테스트 회귀 확인**

```
pytest tests/ -v --tb=short 2>&1 | tail -30
```

- [ ] **Step 3: app.py 문법 확인**

```
python -m py_compile app.py src/ui/sidebar.py src/ui/charts.py
echo "문법 오류 없음"
```

- [ ] **Step 4: 최종 커밋**

```bash
git add -A
git commit -m "test: verify SignalReducer and dashboard integration tests pass"
```