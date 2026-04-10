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
        filtered: list[str] = []
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

    rows: list[dict] = []
    for col in flag_cols:
        s = window[col].dropna()
        if len(s) == 0:
            continue

        # 윈도우 내 값 변화 없으면 제외
        if s.nunique() <= 1:
            continue

        before_s = s[s.index <= event_ts]
        after_s  = s[s.index >  event_ts]

        val_before = (
            float(before_s.iloc[-2]) if len(before_s) >= 2
            else (float(before_s.iloc[0]) if len(before_s) == 1 else float("nan"))
        )
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
    return result.sort_values(
        ["changed", "signal"], ascending=[False, True]
    ).reset_index(drop=True)
