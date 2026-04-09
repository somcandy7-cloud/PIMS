# Per-Device-Group HITL Pipeline 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 컬럼명 브래킷 접두사(`[3_CO_PLC_B]`, `[3_ECS_PLC_B]` 등)로 신호를 장치 그룹으로 분류하고, 그룹별로 독립된 가동 조건·신호 축소·이상 탐지를 수행한다.

**Architecture:**
기존 "전역 단일 가동 조건 → 전체 신호 일괄 분석" 구조를 "컬럼명 파싱 → 장치 그룹 분리 → 그룹별 독립 파이프라인 → 결과 집계"로 전환한다.
각 그룹은 자체 가동 조건, 클러스터 캐시, 이상 탐지 결과를 독립적으로 보유한다.
브래킷 없는 컬럼(`MB 118`, `MD 356` 등)은 `__default__` 그룹으로 취급한다.

**Tech Stack:** pandas, scikit-learn, PyYAML, Streamlit, pytest

---

## 배경 (구현자 필독)

### 데이터 구조
- CSV 7,308개 신호 × 285행 (약 2.1초/행)
- 컬럼명 형태:
  - `[3_CO_PLC_B]QB 42` → 장치 그룹 `3_CO_PLC_B`
  - `[3_ECS_PLC_B]MB    31` → 장치 그룹 `3_ECS_PLC_B`
  - `[3_P_CAR_1]MB     291` → 장치 그룹 `3_P_CAR_1`
  - `MB    118`, `MD    356_1` → `__default__` 그룹

### 기존 구조의 문제
`MB 118 > 15.5` 라는 단일 조건으로 전체 7,308개 신호를 필터링 → 서로 다른 설비의 신호가 혼재된 데이터에서 잘못된 가동 구간 판별.

### 목표 구조
```
DeviceGroupParser.parse(columns)
    ↓
{group_id: [column_list], ...}
    ↓ (그룹별 독립 루프)
AutoOperationDiscovery.discover(group_df) → 그룹 전용 가동 조건
OperationFilter(group_conditions).filter(group_df) → 그룹 가동 구간
SignalReducer.fit_transform(running_group_df) → 그룹 내 축소
RollingFeatureExtractor + IsolationForestAdapter → 그룹 이상 탐지
    ↓
event.metadata['group_id'] = group_id 태깅 후 전체 결과 집계
```

### YAML 스키마 (equipment_profiles.yaml)
```yaml
oven:
  groups:
    3_CO_PLC_B:
      confirmed: false
      conditions:
        - {column: "[3_CO_PLC_B]QB 42", op: "==", value: 1.0, confidence: 0.9, source: "binary"}
      clusters:
        "[3_CO_PLC_B]QB 42": ["[3_CO_PLC_B]QB 42"]
    __default__:
      confirmed: false
      conditions: []
      clusters: null
```
기존 `oven.conditions` / `oven.clusters` 최상위 키는 유지(기존 테스트 호환). 신규 로직은 `oven.groups.<group_id>` 아래만 사용.

---

## 파일 맵

| 상태 | 경로 | 변경 내용 |
|------|------|-----------|
| 신규 | `src/utils/device_group_parser.py` | 컬럼명 파싱 → 그룹 분류 |
| 신규 | `tests/test_device_group_parser.py` | 단위 테스트 |
| 수정 | `src/services/equipment_profile_store.py` | 그룹별 조건·클러스터 메서드 추가 |
| 수정 | `tests/test_equipment_profile_store.py` | 그룹 메서드 테스트 추가 |
| 수정 | `hitl_pipeline.py` | 그룹별 루프로 전환 |
| 수정 | `app.py` | `detect_anomalies_hitl` 시그니처 변경 + 그룹 프로파일 연동 |
| 수정 | `src/ui/sidebar.py` | 그룹별 프로파일 상태 표시 |
| 수정 | `src/ui/charts.py` | 메트릭 카드에 그룹 수 추가 |

---

## Task 1: DeviceGroupParser

**Files:**
- Create: `src/utils/device_group_parser.py`
- Test: `tests/test_device_group_parser.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_device_group_parser.py
import pytest
from src.utils.device_group_parser import DeviceGroupParser


def test_bracketed_column_uses_prefix():
    groups = DeviceGroupParser.parse(["[3_CO_PLC_B]QB 42"])
    assert "3_CO_PLC_B" in groups
    assert "[3_CO_PLC_B]QB 42" in groups["3_CO_PLC_B"]


def test_unbracketed_column_goes_to_default():
    groups = DeviceGroupParser.parse(["MB    118", "MD    356_1"])
    assert DeviceGroupParser.DEFAULT_GROUP in groups
    assert len(groups[DeviceGroupParser.DEFAULT_GROUP]) == 2


def test_mixed_columns_split_correctly():
    cols = ["[3_CO_PLC_B]QB 42", "[3_ECS_PLC_B]MB    31", "MB    118"]
    groups = DeviceGroupParser.parse(cols)
    assert set(groups.keys()) == {"3_CO_PLC_B", "3_ECS_PLC_B", DeviceGroupParser.DEFAULT_GROUP}


def test_empty_columns_returns_empty():
    assert DeviceGroupParser.parse([]) == {}


def test_group_id_single_column():
    assert DeviceGroupParser.group_id("[3_CO_PLC_B]QB 42") == "3_CO_PLC_B"
    assert DeviceGroupParser.group_id("MB    118") == DeviceGroupParser.DEFAULT_GROUP


def test_multiple_signals_same_group():
    cols = ["[3_CO_PLC_B]QB 42", "[3_CO_PLC_B]QB 43", "[3_CO_PLC_B]MB    10"]
    groups = DeviceGroupParser.parse(cols)
    assert len(groups["3_CO_PLC_B"]) == 3
```

- [ ] **Step 2: 테스트 실패 확인**

```
cd "C:\Users\somca\내문서\project\PIMS"
pytest tests/test_device_group_parser.py -v
```
Expected: ImportError (모듈 없음)

- [ ] **Step 3: 구현**

```python
# src/utils/device_group_parser.py
from __future__ import annotations

import re
from collections import defaultdict

_PREFIX_RE = re.compile(r'^\[([^\]]+)\]')


class DeviceGroupParser:
    """컬럼명 브래킷 접두사로 신호를 장치 그룹으로 분류한다.

    Examples
    --------
    '[3_CO_PLC_B]QB 42'  → group_id='3_CO_PLC_B'
    'MB    118'           → group_id=DEFAULT_GROUP ('__default__')
    """

    DEFAULT_GROUP: str = "__default__"

    @staticmethod
    def parse(columns: list[str]) -> dict[str, list[str]]:
        """컬럼 목록을 장치 그룹별로 분류한다.

        Returns
        -------
        dict[group_id, list[column_name]]
        """
        groups: dict[str, list[str]] = defaultdict(list)
        for col in columns:
            m = _PREFIX_RE.match(col)
            gid = m.group(1) if m else DeviceGroupParser.DEFAULT_GROUP
            groups[gid].append(col)
        return dict(groups)

    @staticmethod
    def group_id(column: str) -> str:
        """단일 컬럼의 그룹 ID를 반환한다."""
        m = _PREFIX_RE.match(column)
        return m.group(1) if m else DeviceGroupParser.DEFAULT_GROUP
```

- [ ] **Step 4: 테스트 통과 확인**

```
pytest tests/test_device_group_parser.py -v
```
Expected: 6개 PASSED

- [ ] **Step 5: 커밋**

```bash
git add src/utils/device_group_parser.py tests/test_device_group_parser.py
git commit -m "feat: add DeviceGroupParser for column prefix-based device grouping"
```

---

## Task 2: EquipmentProfileStore — 그룹별 메서드

**Files:**
- Modify: `src/services/equipment_profile_store.py`
- Modify: `tests/test_equipment_profile_store.py`

기존 `save`/`load`/`save_clusters`/`load_clusters` 메서드는 유지 (기존 테스트 통과 필요).
신규 메서드는 YAML의 `data[equipment_id]["groups"][group_id]` 경로를 사용한다.

- [ ] **Step 1: 추가 테스트 작성**

기존 `tests/test_equipment_profile_store.py` 하단에 추가:

```python
# ── 그룹별 메서드 테스트 ────────────────────────────────────────────────────

def test_save_and_load_group(store):
    conds = [OperationCondition("[grp]QB 1", "==", 1.0, 0.9, "binary")]
    store.save_group("oven", "3_CO_PLC_B", conds, confirmed=False)
    loaded = store.load_group("oven", "3_CO_PLC_B")
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded[0].column == "[grp]QB 1"


def test_load_group_unknown_returns_none(store):
    assert store.load_group("oven", "nonexistent_group") is None


def test_is_group_confirmed(store):
    conds = [OperationCondition("c", "==", 1.0, 1.0, "binary")]
    store.save_group("oven", "g1", conds, confirmed=True)
    assert store.is_group_confirmed("oven", "g1") is True
    store.save_group("oven", "g2", conds, confirmed=False)
    assert store.is_group_confirmed("oven", "g2") is False


def test_save_group_does_not_affect_other_groups(store):
    conds = [OperationCondition("c", "==", 1.0, 1.0, "binary")]
    store.save_group("oven", "g1", conds, confirmed=True)
    store.save_group("oven", "g2", conds, confirmed=False)
    assert store.is_group_confirmed("oven", "g1") is True
    assert store.is_group_confirmed("oven", "g2") is False


def test_save_and_load_group_clusters(store):
    cm = {"rep": ["rep", "other"]}
    store.save_group_clusters("oven", "3_CO_PLC_B", cm)
    loaded = store.load_group_clusters("oven", "3_CO_PLC_B")
    assert loaded is not None
    assert loaded["rep"] == ["rep", "other"]


def test_load_group_clusters_unknown_returns_none(store):
    assert store.load_group_clusters("oven", "nonexistent") is None


def test_load_all_groups(store):
    conds = [OperationCondition("c", "==", 1.0, 1.0, "binary")]
    store.save_group("oven", "g1", conds, confirmed=True)
    store.save_group("oven", "g2", conds, confirmed=False)
    all_groups = store.load_all_groups("oven")
    assert set(all_groups.keys()) == {"g1", "g2"}


def test_group_methods_do_not_affect_top_level_conditions(store):
    """그룹 메서드가 기존 top-level save/load에 영향을 주면 안 된다."""
    store.save("oven", _conds(), confirmed=True)
    store.save_group("oven", "g1", [], confirmed=False)
    loaded = store.load("oven")
    assert loaded is not None
    assert len(loaded) == 2


def test_clear_group_clusters(store):
    """clear_group_clusters가 clusters만 제거하고 conditions는 보존한다."""
    conds = [OperationCondition("c", "==", 1.0, 1.0, "binary")]
    store.save_group("oven", "g1", conds, confirmed=True)
    store.save_group_clusters("oven", "g1", {"rep": ["rep"]})
    store.clear_group_clusters("oven", "g1")
    assert store.load_group_clusters("oven", "g1") is None
    # conditions 보존 확인
    assert store.load_group("oven", "g1") is not None
```

- [ ] **Step 2: 테스트 실패 확인**

```
pytest tests/test_equipment_profile_store.py -v -k "group"
```
Expected: AttributeError (메서드 없음)

- [ ] **Step 3: 메서드 구현**

`equipment_profile_store.py`의 `load_clusters` 메서드 아래에 추가:

```python
    # ── 그룹별 메서드 ──────────────────────────────────────────────────────────

    def save_group(
        self,
        equipment_id: str,
        group_id: str,
        conditions: list[OperationCondition],
        confirmed: bool = False,
    ) -> None:
        """그룹별 가동 조건을 저장한다. 다른 그룹에는 영향 없음."""
        data = self._read()
        eq = data.setdefault(equipment_id, {})
        groups = eq.setdefault("groups", {})
        existing = groups.get(group_id, {})
        groups[group_id] = {
            **existing,
            "confirmed": confirmed,
            "conditions": [c.to_dict() for c in conditions],
        }
        self._write(data)

    def load_group(
        self, equipment_id: str, group_id: str
    ) -> list[OperationCondition] | None:
        """그룹별 가동 조건을 반환한다. 없으면 None."""
        data = self._read()
        group = data.get(equipment_id, {}).get("groups", {}).get(group_id)
        if group is None:
            return None
        return [OperationCondition.from_dict(c) for c in group.get("conditions", [])]

    def is_group_confirmed(self, equipment_id: str, group_id: str) -> bool:
        data = self._read()
        group = data.get(equipment_id, {}).get("groups", {}).get(group_id)
        return bool(group and group.get("confirmed", False))

    def save_group_clusters(
        self, equipment_id: str, group_id: str, cluster_map: dict
    ) -> None:
        """그룹별 클러스터맵을 저장한다."""
        data = self._read()
        eq = data.setdefault(equipment_id, {})
        groups = eq.setdefault("groups", {})
        existing = groups.get(group_id, {})
        groups[group_id] = {**existing, "clusters": cluster_map}
        self._write(data)

    def load_group_clusters(
        self, equipment_id: str, group_id: str
    ) -> dict | None:
        """그룹별 클러스터맵을 반환한다. 없으면 None."""
        data = self._read()
        group = data.get(equipment_id, {}).get("groups", {}).get(group_id)
        if group is None:
            return None
        return group.get("clusters")

    def load_all_groups(self, equipment_id: str) -> dict[str, dict]:
        """장치의 모든 그룹 프로파일 {group_id: {confirmed, conditions, clusters}} 반환."""
        data = self._read()
        return data.get(equipment_id, {}).get("groups", {})

    def clear_group_clusters(self, equipment_id: str, group_id: str) -> None:
        """그룹 클러스터맵만 삭제한다. conditions/confirmed는 보존."""
        data = self._read()
        group = data.get(equipment_id, {}).get("groups", {}).get(group_id)
        if group is not None:
            group.pop("clusters", None)
            self._write(data)
```

- [ ] **Step 4: 테스트 통과 확인**

```
pytest tests/test_equipment_profile_store.py -v
```
Expected: 기존 10개 + 신규 9개 = 19개 PASSED

- [ ] **Step 5: 커밋**

```bash
git add src/services/equipment_profile_store.py tests/test_equipment_profile_store.py
git commit -m "feat: add per-group condition and cluster methods to EquipmentProfileStore"
```

---

## Task 3: hitl_pipeline.py — 그룹별 루프로 전환

**Files:**
- Modify: `hitl_pipeline.py`

현재 `run_hitl_pipeline`의 Step 1-B(가동 조건 확보)·Step 1-C(필터링)·Step 1-D(SignalReducer)를
그룹별 루프로 교체한다.

- [ ] **Step 1: `_get_group_conditions` 헬퍼 작성**

기존 `_get_operation_conditions` 함수 아래에 추가:

```python
def _get_group_conditions(
    group_df,
    equipment_id: str,
    group_id: str,
    hitl_cfg: dict,
    interactive: bool,
    store,                 # EquipmentProfileStore 인스턴스를 외부에서 주입 (루프마다 재생성 방지)
):
    """그룹별 가동 조건을 캐시에서 로드하거나 자동 탐색한다."""
    from src.utils.operation_discovery import AutoOperationDiscovery

    disc_cfg = hitl_cfg.get("discovery", {})

    cached = store.load_group(equipment_id, group_id)
    if cached is not None and store.is_group_confirmed(equipment_id, group_id):
        print(f"    [캐시 hit] '{group_id}' — 저장된 조건 {len(cached)}개")
        return cached

    discoverer = AutoOperationDiscovery(
        top_binary_n=int(disc_cfg.get("top_binary_n", 1)),
        top_bimodal_n=int(disc_cfg.get("top_bimodal_n", 1)),
    )
    conditions = discoverer.discover(group_df)

    if not conditions:
        store.save_group(equipment_id, group_id, [], confirmed=False)
        return []

    if interactive:
        print(f"\n    [{group_id}] 발견된 가동 조건 ({len(conditions)}개):")
        for i, c in enumerate(conditions, 1):
            print(f"      {i}. [{c.source}] {c.column} {c.op} {c.value:.2f}  (신뢰도={c.confidence:.2f})")
        ans = input(f"\n    '{group_id}' 조건 확정? [Y/N/S(건너뜀)]: ").strip().upper()
        confirmed = (ans == "Y")
        if ans == "S":
            return []
    else:
        confirmed = False

    store.save_group(equipment_id, group_id, conditions, confirmed=confirmed)
    return conditions
```

- [ ] **Step 2: `run_hitl_pipeline` 내부 그룹별 루프로 교체**

`run_hitl_pipeline` 함수에서 Step 1-B부터 Step 2까지를 아래로 교체한다.
(Step 1 로드+전처리, Step 3 컨텍스트, Step 4 피드백 루프는 유지)

교체 전 범위: `# Step 1-B` ~ `df_feat = RollingFeatureExtractor(...).transform(df_reduced)` 끝까지.

```python
    # Step 1-B ~ 2: 그룹별 독립 파이프라인
    from src.utils.device_group_parser import DeviceGroupParser
    from src.utils.signal_reducer import SignalReducer
    from src.utils.operation_filter import OperationFilter
    from src.utils.rolling_features import RollingFeatureExtractor
    from src.agents.isolation_forest_adapter import IsolationForestAdapter
    from src.services.equipment_profile_store import EquipmentProfileStore

    ep_store = EquipmentProfileStore()   # 루프 밖에서 한 번만 생성
    groups = DeviceGroupParser.parse(df_raw.columns.tolist())
    reducer_cfg = hitl_cfg
    det_cfg = hitl_cfg.get("detector", {})
    roll_cfg = hitl_cfg.get("rolling", {})

    all_candidates: list = []

    print(f"\n[Step 1-B~2] 장치 그룹 {len(groups)}개 개별 분석")

    for group_id, group_cols in groups.items():
        if len(group_cols) < 5:
            # 신호가 5개 미만인 소형 그룹은 의미 있는 분석 불가 — 건너뜀
            continue

        group_df = df_raw[group_cols]
        print(f"\n  ── 그룹: {group_id} ({len(group_cols)}개 신호) ──")

        # 가동 조건 확보 (ep_store 주입으로 루프마다 재생성 방지)
        conditions = _get_group_conditions(
            group_df, equipment_id, group_id, hitl_cfg, interactive, ep_store
        )

        # 가동 구간 필터
        df_running = OperationFilter(conditions).filter(group_df) if conditions else group_df.copy()
        pct = len(df_running) / max(len(group_df), 1) * 100
        print(f"    가동 구간: {len(df_running)}행 ({pct:.1f}%)")

        if df_running.empty:
            continue

        # SignalReducer
        cached_clusters = ep_store.load_group_clusters(equipment_id, group_id)
        sr = SignalReducer.from_config(reducer_cfg)
        if cached_clusters is not None:
            rep_cols = [c for c in cached_clusters if c in df_running.columns]
            df_reduced = df_running[rep_cols].copy()
        else:
            df_reduced, cluster_map = sr.fit_transform(df_running)
            ep_store.save_group_clusters(equipment_id, group_id, cluster_map)
        print(f"    축소: {len(group_cols)}→{len(df_reduced.columns)}열")

        if df_reduced.empty:
            continue

        # Rolling 특징 + IF 탐지
        df_feat = RollingFeatureExtractor(
            window=int(roll_cfg.get("window_rows", 30))
        ).transform(df_reduced)

        candidates = IsolationForestAdapter(
            window_size=int(det_cfg.get("window_size", 1)),
            contamination=float(det_cfg.get("contamination", 0.02)),
            n_estimators=int(det_cfg.get("n_estimators", 100)),
            top_n=int(det_cfg.get("top_n", 5)),
        ).detect(df_feat)

        # 그룹 태깅
        for ev in candidates:
            ev.metadata["group_id"] = group_id

        print(f"    IF 후보: {len(candidates)}건")
        all_candidates.extend(candidates)

    candidates = all_candidates
    print(f"\n  전체 IF 후보: {len(candidates)}건")
```

그리고 이후 LLM 필터 호출에서 `df_running` → `df_raw` 로 변경 (컨텍스트 포맷팅에 전체 데이터 사용):
```python
    events = LLMFilter(backend).filter(candidates, df_raw)
    ...
    for event in events:
        ctx = formatter.format(event, df_raw)
```

- [ ] **Step 3: 수동 확인 (hitl_pipeline CLI)**

```bash
python hitl_pipeline.py --file 2603201549_oven.csv --no-interactive
```
Expected: 그룹별 분석 로그 출력, 에러 없음

- [ ] **Step 4: 커밋**

```bash
git add hitl_pipeline.py
git commit -m "refactor: per-device-group loop in hitl_pipeline replacing global running condition"
```

---

## Task 4: app.py — 그룹별 프로파일 연동

**Files:**
- Modify: `app.py`

`detect_anomalies_hitl` 시그니처에서 `conditions_json, clusters_json` 제거 →
`groups_profile_json`(그룹별 조건+클러스터 직렬화)으로 교체.

- [ ] **Step 1: `detect_anomalies_hitl` 교체**

기존 함수를 아래로 완전 교체:

```python
@st.cache_data(show_spinner="이상치 탐지 중 (그룹별 HITL 파이프라인)...")
def detect_anomalies_hitl(
    path: str,
    groups_profile_json: str,   # {group_id: {conditions:[...], clusters:{...}|null}}
    _if_params_key: str,
    _topn: int,
    _excluded: tuple[str, ...],
    _settings_key: str,
) -> tuple[list, list, int, int, int]:
    """
    Returns: (candidates, events, total_rows, running_rows, reduced_cols)
    running_rows = 가동 구간이 있는 그룹들의 행 수 합산 (중복 포함)
    reduced_cols = 전체 그룹 대표 신호 수 합산
    """
    from src.agents.isolation_forest_adapter import IsolationForestAdapter
    from src.utils.device_group_parser import DeviceGroupParser
    from src.utils.signal_reducer import SignalReducer

    df = load_and_process(path)
    settings = json.loads(_settings_key)
    hitl_cfg = settings.get("hitl", {})
    det_cfg = hitl_cfg.get("detector", {})
    roll_cfg = hitl_cfg.get("rolling", {})
    if_params = json.loads(_if_params_key)

    groups_profile: dict = json.loads(groups_profile_json)
    device_groups = DeviceGroupParser.parse(df.columns.tolist())

    total_running_rows = 0
    total_reduced_cols = 0
    all_candidates: list = []

    for group_id, group_cols in device_groups.items():
        if len(group_cols) < 5:
            continue

        group_df = df[group_cols]
        profile = groups_profile.get(group_id, {})
        raw_conds = profile.get("conditions") or []
        conditions = [OperationCondition.from_dict(c) for c in raw_conds]

        df_running = OperationFilter(conditions).filter(group_df) if conditions else group_df.copy()
        if df_running.empty:
            continue

        total_running_rows += len(df_running)

        clusters = profile.get("clusters")
        if clusters:
            rep_cols = [c for c in clusters if c in df_running.columns]
            df_reduced = df_running[rep_cols].copy()
        else:
            # 클러스터 저장은 @st.cache_data 외부(run_btn 블록)에서 수행한다.
            # 여기서는 계산만 한다.
            df_reduced, _ = SignalReducer.from_config(hitl_cfg).fit_transform(df_running)

        total_reduced_cols += len(df_reduced.columns)

        if df_reduced.empty:
            continue

        df_feat = RollingFeatureExtractor(
            window=int(roll_cfg.get("window_rows", 30))
        ).transform(df_reduced)

        candidates = IsolationForestAdapter(
            window_size=int(det_cfg.get("window_size", 1)),
            contamination=float(det_cfg.get("contamination", 0.02)),
            n_estimators=int(if_params.get("n_estimators", 100)),
            excluded_signals=list(_excluded),
            top_n=_topn,
        ).detect(df_feat)

        for ev in candidates:
            ev.metadata["group_id"] = group_id

        all_candidates.extend(candidates)

    llm_filter = build_llm_filter(settings)
    events = llm_filter.filter(all_candidates, df)

    return all_candidates, events, len(df), total_running_rows, total_reduced_cols
```

- [ ] **Step 2: 메인 실행 블록 업데이트**

기존 `if run_btn:` 블록에서 `conditions_json`, `clusters_json` 조립 부분을 아래로 교체:

```python
        # 그룹별 프로파일 조립
        all_groups = ep_store.load_all_groups(equipment_id)
        groups_profile = {}
        device_groups = {}
        try:
            df_temp = load_and_process(csv_path)
            from src.utils.device_group_parser import DeviceGroupParser
            device_groups = DeviceGroupParser.parse(df_temp.columns.tolist())
        except Exception:
            device_groups = {}

        disc_cfg = hitl_cfg.get("discovery", {})
        sr = SignalReducer.from_config(hitl_cfg)

        for group_id, group_cols in device_groups.items():
            if len(group_cols) < 5:
                continue
            gp = all_groups.get(group_id, {})
            raw_conds = gp.get("conditions")

            if raw_conds is None:
                # 미탐색 그룹 → 자동 탐색 후 저장
                group_df = df_temp[group_cols]
                new_conds = AutoOperationDiscovery(
                    top_binary_n=int(disc_cfg.get("top_binary_n", 1)),
                    top_bimodal_n=int(disc_cfg.get("top_bimodal_n", 1)),
                ).discover(group_df)
                ep_store.save_group(equipment_id, group_id, new_conds, confirmed=False)
                raw_conds = [c.to_dict() for c in new_conds]
                clusters = None
            else:
                clusters = gp.get("clusters")

            # 클러스터 미캐시 → run_btn 블록에서 사전 계산 후 저장
            # (@st.cache_data 내부에서 파일 I/O 금지 원칙 준수)
            if clusters is None and raw_conds:
                from src.utils.operation_discovery import OperationCondition as _OC
                conds_objs = [_OC.from_dict(c) for c in raw_conds]
                df_grp = df_temp[group_cols]
                df_running_grp = OperationFilter(conds_objs).filter(df_grp) if conds_objs else df_grp.copy()
                if not df_running_grp.empty:
                    _, clusters = sr.fit_transform(df_running_grp)
                    ep_store.save_group_clusters(equipment_id, group_id, clusters)

            groups_profile[group_id] = {"conditions": raw_conds, "clusters": clusters}

        groups_profile_json = json.dumps(groups_profile, sort_keys=True)

        candidates, events, total_rows, running_rows, reduced_cols = detect_anomalies_hitl(
            csv_path, groups_profile_json,
            if_params_key, top_n, tuple(excluded), settings_key,
        )
        group_count = len([g for g, cols in device_groups.items() if len(cols) >= 5])
        df = load_and_process(csv_path)

    st.session_state.update(
        df=df, events=events, candidates=candidates,
        csv_path=csv_path, equipment_id=equipment_id,
        total_rows=total_rows, running_rows=running_rows,
        reduced_cols=reduced_cols, group_count=group_count,
    )
    st.rerun()
```

- [ ] **Step 3: 문법 검사**

```bash
python -m py_compile app.py
echo "OK"
```

- [ ] **Step 4: 커밋**

```bash
git add app.py
git commit -m "refactor: per-group profile assembly in detect_anomalies_hitl"
```

---

## Task 5: sidebar.py + charts.py — 그룹 인식 UI

**Files:**
- Modify: `src/ui/sidebar.py`
- Modify: `src/ui/charts.py`

- [ ] **Step 1: sidebar.py 설비 프로파일 섹션 교체**

기존 "설비 프로파일" 섹션(run_btn 이후 블록 전체)을 아래로 교체:

```python
        # ── 설비 프로파일 (그룹별) ─────────────────────────────────────────
        st.divider()
        st.subheader("설비 프로파일")
        _csv_val = csv_input.strip()
        if _csv_val:
            from src.services.equipment_profile_store import EquipmentProfileStore as _EPS
            from src.utils.device_group_parser import DeviceGroupParser as _DGP
            _eq_id = _EPS.extract_id(Path(_csv_val).name)
            _ep = _EPS()
            _all_groups = _ep.load_all_groups(_eq_id)

            if _all_groups:
                st.caption(f"`{_eq_id}` — {len(_all_groups)}개 장치 그룹")
                for _gid, _gdata in _all_groups.items():
                    _gconds = _gdata.get("conditions") or []
                    _gconf = _gdata.get("confirmed", False)
                    _gclusters = _gdata.get("clusters")
                    _status = "확인됨" if _gconf else "미확인"
                    with st.expander(f"`{_gid}` — 조건 {len(_gconds)}개  [{_status}]"):
                        for _c in _gconds:
                            st.caption(
                                f"`{_c['column']}` {_c['op']} {_c['value']:.0f}"
                                f"  [{_c['source']}]"
                            )
                        if not _gconf and _gconds:
                            if st.button("확정", key=f"confirm_{_gid}"):
                                from src.utils.operation_discovery import OperationCondition as _OC
                                _ep.save_group(
                                    _eq_id, _gid,
                                    [_OC.from_dict(c) for c in _gconds],
                                    confirmed=True,
                                )
                                st.rerun()
                        if _gclusters:
                            st.caption(f"클러스터: {len(_gclusters)}개 대표 신호")
                            if st.button("클러스터 초기화", key=f"clr_clust_{_gid}"):
                                _ep.clear_group_clusters(_eq_id, _gid)
                                clear_detection_cache_fn()
                                st.rerun()
            else:
                st.caption(f"`{_eq_id}` — 프로파일 없음 (분석 실행 시 그룹별 자동 탐색)")
```

- [ ] **Step 2: charts.py `render_metrics_hitl` 업데이트**

7번째 컬럼(데이터 구간) 앞에 그룹 수 컬럼 추가하여 8컬럼으로 변경:

```python
def render_metrics_hitl(
    df: pd.DataFrame,
    candidates: list,
    events: list,
    kst: str,
    running_rows: int,
    reduced_cols: int,
    group_count: int = 0,
) -> None:
    """HITL 파이프라인 정보 포함 메트릭 카드 8개."""
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("총 샘플", f"{len(df):,}")
    running_pct = running_rows / max(len(df), 1) * 100
    c2.metric("가동 구간", f"{running_rows:,}행", f"{running_pct:.0f}%")
    c3.metric("장치 그룹", f"{group_count}개" if group_count else "-")
    c4.metric("원본 신호", f"{len(df.columns):,}")
    c5.metric("축소 신호", f"{reduced_cols:,}", f"-{len(df.columns)-reduced_cols:,}")
    c6.metric("IF 후보", f"{len(candidates)}건")
    c7.metric("LLM 검증", f"{len(events)}건")
    dur = (df.index[-1] - df.index[0]).total_seconds() / 60
    c8.metric("데이터 구간", f"{dur:.1f}분")
```

- [ ] **Step 3: app.py에서 group_count 전달**

`app.py`의 `render_metrics_hitl(...)` 호출을 업데이트:
(`group_count`는 Task 4에서 이미 `st.session_state`에 저장됨)

```python
render_metrics_hitl(
    df, candidates, events, KST,
    running_rows=st.session_state.get("running_rows", len(df)),
    reduced_cols=st.session_state.get("reduced_cols", len(df.columns)),
    group_count=st.session_state.get("group_count", 0),
)
```

- [ ] **Step 4: 문법 검사**

```bash
python -m py_compile app.py src/ui/sidebar.py src/ui/charts.py
echo "OK"
```

- [ ] **Step 5: 커밋**

```bash
git add src/ui/sidebar.py src/ui/charts.py app.py
git commit -m "feat: per-group profile display in sidebar and group count metric in dashboard"
```

---

## Task 6: 전체 검증

- [ ] **Step 1: 단위 테스트**

```
pytest tests/test_device_group_parser.py tests/test_equipment_profile_store.py -v
```
Expected: 전부 PASSED

- [ ] **Step 2: 회귀 테스트**

```
pytest tests/ -v --tb=short 2>&1 | tail -30
```
Expected: 기존 테스트 회귀 없음

- [ ] **Step 3: 프로파일 초기화 후 대시보드 실행**

```bash
# equipment_profiles.yaml 초기화
echo "{}" > config/equipment_profiles.yaml

streamlit run app.py
```

확인 항목:
- 사이드바: 그룹별 조건 목록 표시
- 메트릭: 장치 그룹 수 표시
- 이벤트 카드: `group_id` 태그 표시

- [ ] **Step 4: 최종 커밋**

```bash
git add config/equipment_profiles.yaml
git commit -m "chore: reset equipment profiles for per-group pipeline"
```