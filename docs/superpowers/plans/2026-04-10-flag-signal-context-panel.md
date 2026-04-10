# Flag 신호 컨텍스트 패널 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 이상 이벤트 상세 화면에서 탐지 시점의 flag 신호(IB/QB/MB/DBB) 상태를 테이블로 표시해, 엔지니어가 "그때 어떤 명령이 내려갔고 어떤 스위치가 눌렸는지" 확인할 수 있게 한다.

**Architecture:**
measurement 신호로 탐지한 이상 이벤트에 대해, 동일 그룹의 flag 신호들을 이벤트 시점 ±N초 윈도우로 슬라이싱해 값 변화가 있는 것만 표시한다. `charts.py`에 `render_flag_context_panel()` 함수를 추가하고 `app.py`에서 호출한다. flag 신호 원본 df는 session_state의 전체 df(measurement + flag 모두 포함)에서 직접 슬라이싱한다.

**Tech Stack:** pandas, Streamlit, 기존 `SignalTypeFilter`, `SignalLabelMapper`

---

## 파일 맵

| 상태 | 경로 | 변경 내용 |
|------|------|-----------|
| 신규 | `tests/test_flag_context.py` | `get_flag_context()` 단위 테스트 |
| 신규 | `src/utils/flag_context.py` | 이벤트 시점 flag 신호 추출 로직 |
| 수정 | `src/ui/charts.py` | `render_flag_context_panel()` 추가 |
| 수정 | `app.py` | 이벤트 상세 섹션에 패널 호출 추가 |

---

## Task 1: `get_flag_context()` — flag 신호 상태 추출

**Files:**
- Create: `src/utils/flag_context.py`
- Test: `tests/test_flag_context.py`

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_flag_context.py
import numpy as np
import pandas as pd
import pytest
from src.utils.flag_context import get_flag_context

def _make_df(n: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2026-03-20 10:00:00", periods=n, freq="2s", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        # measurement 신호
        "[3_P_CAR_1]DB420.DBW   2": rng.normal(100, 5, n),
        # flag 신호 — 이벤트 시점 전후에 변화 있음
        "[3_P_CAR_1]QB      11": np.where(np.arange(n) < 30, 0.0, 1.0),
        "[3_P_CAR_1]IB       1": np.ones(n) * 3.0,   # 변화 없음 → 제외 대상
        "[3_P_CAR_1]MB       1": np.where(np.arange(n) < 28, 0.0, 255.0),
        # 다른 그룹 flag — group_id 필터로 제외되어야 함
        "[3_CO_PLC_A]MB      43": rng.normal(0, 1, n),
    }, index=idx)

def test_returns_only_flag_signals():
    df = _make_df()
    ts = df.index[30]
    result = get_flag_context(df, ts, group_id="3_P_CAR_1", context_sec=30)
    for col in result["signal"]:
        assert "DB420.DBW" not in col

def test_filters_to_same_group():
    df = _make_df()
    ts = df.index[30]
    result = get_flag_context(df, ts, group_id="3_P_CAR_1", context_sec=30)
    for col in result["signal"]:
        assert "3_CO_PLC_A" not in col

def test_excludes_no_change_signals():
    """이벤트 윈도우 내에서 값이 변하지 않은 flag는 제외한다."""
    df = _make_df()
    ts = df.index[30]
    result = get_flag_context(df, ts, group_id="3_P_CAR_1", context_sec=30)
    assert not any("IB" in s for s in result["signal"].values)

def test_changed_signals_included():
    df = _make_df()
    ts = df.index[30]
    result = get_flag_context(df, ts, group_id="3_P_CAR_1", context_sec=30)
    signals = result["signal"].tolist()
    assert any("QB" in s for s in signals)
    assert any("MB" in s for s in signals)

def test_columns_present():
    df = _make_df()
    ts = df.index[30]
    result = get_flag_context(df, ts, group_id="3_P_CAR_1", context_sec=30)
    for col in ("signal", "val_at_event", "val_before", "changed", "label"):
        assert col in result.columns

def test_empty_when_no_flags():
    idx = pd.date_range("2026-03-20", periods=10, freq="2s", tz="UTC")
    df = pd.DataFrame({"[grp]DB420.DBW 1": range(10)}, index=idx)
    result = get_flag_context(df, idx[5], group_id="grp", context_sec=10)
    assert result.empty

def test_no_group_returns_all_flags():
    """group_id=None 이면 그룹 필터 없이 전체 flag 반환."""
    df = _make_df()
    ts = df.index[30]
    result_all   = get_flag_context(df, ts, group_id=None, context_sec=30)
    result_group = get_flag_context(df, ts, group_id="3_P_CAR_1", context_sec=30)
    assert len(result_all) >= len(result_group)
```

- [ ] **Step 2: 테스트 실패 확인**

```
pytest tests/test_flag_context.py -v
```
Expected: ImportError (모듈 없음)

- [ ] **Step 3: 구현**

```python
# src/utils/flag_context.py
"""이상 이벤트 시점의 flag 신호 상태 추출."""
from __future__ import annotations

import re

import pandas as pd

from src.utils.signal_type_filter import classify_signal

_PREFIX_RE = re.compile(r'^\[([^\]]+)\]')


def get_flag_context(
    df: pd.DataFrame,
    event_ts: pd.Timestamp,
    group_id: str | None,
    context_sec: int = 30,
    mapper=None,
) -> pd.DataFrame:
    """이벤트 시점 ±context_sec 구간에서 변화가 있는 flag 신호를 반환한다.

    Parameters
    ----------
    df          : 원본 전체 DataFrame (measurement + flag 신호 모두 포함)
    event_ts    : 이상 이벤트 타임스탬프
    group_id    : 필터링할 장치 그룹 ID. None 이면 전체.
    context_sec : 이벤트 전후 ±초
    mapper      : SignalLabelMapper (선택, 라벨 표시용)

    Returns
    -------
    DataFrame columns:
        signal, val_before, val_at_event, val_after, changed, label
    정렬: changed(변화 큰 순) → signal
    """
    if df.empty:
        return pd.DataFrame()

    # flag 신호만 선택
    flag_cols = [c for c in df.columns if classify_signal(c) == "flag"]
    if not flag_cols:
        return pd.DataFrame()

    # 그룹 필터
    if group_id is not None:
        filtered = []
        for c in flag_cols:
            m = _PREFIX_RE.match(c)
            if m and m.group(1) == group_id:
                filtered.append(c)
            elif not m and group_id == "__default__":
                filtered.append(c)
        flag_cols = filtered

    if not flag_cols:
        return pd.DataFrame()

    t_start = event_ts - pd.Timedelta(seconds=context_sec)
    t_end   = event_ts + pd.Timedelta(seconds=context_sec)

    try:
        window = df.loc[t_start:t_end, flag_cols]
    except Exception:
        window = df[flag_cols]

    if window.empty:
        return pd.DataFrame()

    rows = []
    for col in flag_cols:
        s = window[col].dropna()
        if len(s) == 0:
            continue

        # 윈도우 내 값 변화 없으면 제외
        if s.nunique() <= 1:
            continue

        before_s = s[s.index <= event_ts]
        after_s  = s[s.index >  event_ts]

        val_before   = float(before_s.iloc[-2]) if len(before_s) >= 2 else (float(before_s.iloc[0]) if len(before_s) == 1 else float("nan"))
        val_at_event = float(before_s.iloc[-1]) if len(before_s) >= 1 else float("nan")
        val_after    = float(after_s.iloc[0])   if len(after_s)  >= 1 else float("nan")

        changed = (
            abs(val_at_event - val_before)
            if not (pd.isna(val_at_event) or pd.isna(val_before))
            else 0.0
        )

        label = ""
        if mapper:
            hit = mapper.lookup(col)
            if hit:
                label = hit[0] or hit[1]

        rows.append({
            "signal"       : col,
            "val_before"   : round(val_before,   3),
            "val_at_event" : round(val_at_event, 3),
            "val_after"    : round(val_after,    3),
            "changed"      : round(changed,      3),
            "label"        : label,
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    return result.sort_values(["changed", "signal"], ascending=[False, True]).reset_index(drop=True)
```

- [ ] **Step 4: 테스트 통과 확인**

```
pytest tests/test_flag_context.py -v
```
Expected: 7개 PASSED

- [ ] **Step 5: 커밋**

```bash
git add src/utils/flag_context.py tests/test_flag_context.py
git commit -m "feat: add get_flag_context for flag signal state extraction at anomaly time"
```

---

## Task 2: `render_flag_context_panel()` — 대시보드 패널

**Files:**
- Modify: `src/ui/charts.py`

- [ ] **Step 1: `render_flag_context_panel` 추가**

`charts.py` 하단에 추가:

```python
def render_flag_context_panel(
    df: pd.DataFrame,
    event,
    group_id: str | None,
    context_sec: int,
    mapper,
) -> None:
    """이상 시점 flag 신호 상태 패널을 렌더링한다."""
    from src.utils.flag_context import get_flag_context

    flag_df = get_flag_context(
        df, event.timestamp, group_id=group_id,
        context_sec=context_sec, mapper=mapper,
    )

    st.subheader("제어·상태 신호 (이벤트 시점)")

    if flag_df.empty:
        st.caption("이 그룹에서 이벤트 시점에 변화한 제어/상태 신호가 없습니다.")
        return

    display = flag_df.copy()
    display["신호"] = display.apply(
        lambda r: f"{r['label']}  `{r['signal']}`" if r["label"] else f"`{r['signal']}`",
        axis=1,
    )
    display["이전"] = display["val_before"]
    display["이벤트시점"] = display["val_at_event"]
    display["이후"] = display["val_after"]
    display["변화량"] = display["changed"].apply(lambda v: f"{v:+.1f}" if v != 0 else "-")

    st.dataframe(
        display[["신호", "이전", "이벤트시점", "이후", "변화량"]],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"변화 감지 신호 {len(flag_df)}개 · 표시 구간 ±{context_sec}초")
```

- [ ] **Step 2: 문법 확인**

```
python -m py_compile src/ui/charts.py && echo "OK"
```

- [ ] **Step 3: 커밋**

```bash
git add src/ui/charts.py
git commit -m "feat: add render_flag_context_panel to charts"
```

---

## Task 3: `app.py` — 이벤트 상세 섹션에 패널 연결

**Files:**
- Modify: `app.py`

현재 이벤트 상세 섹션 흐름:
```
render_event_summary_and_bar(...)   ← top_signals bar + 신호 선택
st.divider()
[trend / llm / alarm 3컬럼]
render_stats_table(...)
```

`render_stats_table(...)` 호출 **아래**에 flag 패널을 추가한다.

- [ ] **Step 1: import 추가**

`app.py` charts import 블록에 `render_flag_context_panel` 추가

- [ ] **Step 2: 패널 호출 추가**

`render_stats_table(...)` 호출 바로 아래에:

```python
# flag 신호 컨텍스트 패널 — 이벤트 시점의 제어/상태 신호 변화
group_id_for_event = event.metadata.get("group_id")
render_flag_context_panel(
    df=df_raw,
    event=event,
    group_id=group_id_for_event,
    context_sec=context_minutes * 60,
    mapper=mapper,
)
```

> `df_raw`는 SignalTypeFilter 적용 전 원본 df (flag 포함). `context_minutes`는 기존 hitl 설정값.

- [ ] **Step 3: 문법 확인**

```
python -m py_compile app.py && echo "OK"
```

- [ ] **Step 4: 커밋**

```bash
git add app.py
git commit -m "feat: show flag signal context panel in anomaly event detail"
```

---

## Task 4: 전체 검증

- [ ] **Step 1: 전체 테스트**

```
pytest tests/test_flag_context.py tests/test_signal_type_filter.py -v
```
Expected: 전체 PASSED

- [ ] **Step 2: 기존 테스트 회귀 확인**

```
pytest tests/ -q --tb=short 2>&1 | tail -20
```

- [ ] **Step 3: 최종 커밋**

```bash
git add -A
git commit -m "feat: flag signal context panel — show control/status state at anomaly time"
```
