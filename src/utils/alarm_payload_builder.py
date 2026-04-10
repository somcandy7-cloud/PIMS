"""이상 이벤트를 사내 알람 로직 LLM용 JSON 페이로드로 변환하는 유틸리티."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.agents.base_detector import AnomalyEvent

_CONTEXT_SEC = 60          # 이벤트 전후 ±60초를 이상 구간으로 사용
_BASELINE_RATIO = 0.3      # 전체 데이터 앞 30% 를 정상 기준으로 사용

_ROLL_SUFFIXES = {
    "_rstd":  "StdDev Spike",
    "_roc":   "ChangeRate Increase",
    "_rmean": "Mean Deviation",
}


def _base_sig(col: str) -> tuple[str, str]:
    """rolling 접미사 제거 → (base_name, anomaly_type)."""
    for sfx, kind in _ROLL_SUFFIXES.items():
        if col.endswith(sfx):
            return col[: -len(sfx)], kind
    return col, "Value Anomaly"


def _if_score_to_probability(score: float) -> float:
    """IF score(-inf~0)를 0~1 이상 확률로 변환.

    IF score는 음수이며 -0.5 이하면 거의 확실한 이상.
    score=-0.5 → 1.0, score=0 → 0.0 으로 선형 매핑 후 클램프.
    """
    return round(min(1.0, max(0.0, -score / 0.5)), 4)


def _get_signal_stats(
    df: pd.DataFrame,
    sig: str,
    event_ts: pd.Timestamp,
) -> dict[str, float | None]:
    """이벤트 시점 신호 통계 계산."""
    if sig not in df.columns:
        return {"value": None, "baseline": None, "deviation": None,
                "stddev": None, "delta": None}

    n_baseline = max(10, int(len(df) * _BASELINE_RATIO))
    baseline_s = df[sig].iloc[:n_baseline].dropna()

    # 이벤트 전후 ±CONTEXT_SEC 구간
    t_start = event_ts - pd.Timedelta(seconds=_CONTEXT_SEC)
    t_end   = event_ts + pd.Timedelta(seconds=_CONTEXT_SEC)
    try:
        window = df.loc[t_start:t_end, sig].dropna()
    except Exception:
        window = pd.Series(dtype=float)

    # 이벤트 시점 값
    try:
        ev_series = df.loc[:event_ts, sig].dropna()
        value = round(float(ev_series.iloc[-1]), 4) if len(ev_series) > 0 else None
    except Exception:
        value = None

    baseline = round(float(baseline_s.mean()), 4) if len(baseline_s) > 0 else None
    stddev   = round(float(baseline_s.std()),  4) if len(baseline_s) > 1 else None
    deviation = round(value - baseline, 4) if (value is not None and baseline is not None) else None

    # 5초 변화율
    try:
        t5 = event_ts - pd.Timedelta(seconds=5)
        s5 = df.loc[t5:event_ts, sig].dropna()
        delta = round(float(s5.iloc[-1] - s5.iloc[0]), 4) if len(s5) >= 2 else None
    except Exception:
        delta = None

    return {
        "value":     value,
        "baseline":  baseline,
        "deviation": deviation,
        "stddev":    stddev,
        "delta_5s":  delta,
    }


def _get_operating_conditions(
    df: pd.DataFrame,
    event_ts: pd.Timestamp,
    group_id: str | None,
    mapper=None,
) -> dict[str, Any]:
    """이벤트 시점의 flag 신호 상태를 dict로 반환."""
    from src.utils.flag_context import get_flag_context

    flag_df = get_flag_context(
        df, event_ts,
        group_id=group_id,
        context_sec=_CONTEXT_SEC,
        mapper=mapper,
    )
    if flag_df.empty:
        return {}

    result: dict[str, Any] = {}
    for _, row in flag_df.iterrows():
        # 라벨이 있으면 라벨, 없으면 신호명 사용
        key = row["label"] if row["label"] else row["signal"]
        val = row["val_at_event"]
        # 정수처럼 생긴 값은 int로, boolean-like(0/1)는 bool로
        if not math.isnan(val):
            if val in (0.0, 1.0):
                result[key] = bool(int(val))
            elif val == int(val):
                result[key] = int(val)
            else:
                result[key] = round(val, 4)
        else:
            result[key] = None
    return result


def build_alarm_payload(
    event: AnomalyEvent,
    df: pd.DataFrame,
    equipment_id: str,
    kst: str = "Asia/Seoul",
    mapper=None,
) -> dict[str, Any]:
    """AnomalyEvent를 사내 알람 로직 LLM용 JSON 페이로드로 변환한다.

    Parameters
    ----------
    event        : KEEP 판정된 이상 이벤트
    df           : 원본 전체 DataFrame (measurement + flag 포함)
    equipment_id : 장비 ID (파일명에서 추출)
    kst          : 타임존 문자열
    mapper       : SignalLabelMapper (선택)

    Returns
    -------
    dict : 사내 LLM에 전달할 JSON 구조
    """
    ts_kst = event.timestamp.tz_convert(kst)
    t_start_kst = (event.timestamp - pd.Timedelta(seconds=_CONTEXT_SEC)).tz_convert(kst)
    t_end_kst   = (event.timestamp + pd.Timedelta(seconds=_CONTEXT_SEC)).tz_convert(kst)

    # 주요 신호 (top_signals[0])
    primary_sig_raw, primary_mag = event.top_signals[0] if event.top_signals else ("", 0.0)
    base_name, _ = _base_sig(primary_sig_raw)

    # 신호 라벨
    sig_label = base_name
    if mapper:
        hit = mapper.lookup(base_name)
        if hit:
            sig_label = hit[0] or hit[1] or base_name

    # 이상 타입 목록 (top_signals 접미사 기준)
    anomaly_types: list[str] = []
    seen_types: set[str] = set()
    for sig, _ in event.top_signals[:5]:
        _, kind = _base_sig(sig)
        if kind not in seen_types:
            anomaly_types.append(kind)
            seen_types.add(kind)

    # 신호 통계
    stats = _get_signal_stats(df, base_name, event.timestamp)

    # 가동 조건 (flag 신호 상태)
    group_id = event.metadata.get("group_id")
    op_conds = _get_operating_conditions(df, event.timestamp, group_id, mapper)

    payload: dict[str, Any] = {
        "equipment": equipment_id,
        "signal": sig_label,
        "signal_address": base_name,
        "timestamp": {
            "start": t_start_kst.strftime("%Y-%m-%d %H:%M:%S"),
            "end":   t_end_kst.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "anomaly_features": {
            "value":     stats["value"],
            "baseline":  stats["baseline"],
            "deviation": stats["deviation"],
            "stddev":    stats["stddev"],
            "delta_5s":  stats["delta_5s"],
            "type":      anomaly_types if anomaly_types else ["Unknown"],
        },
        "operating_condition": op_conds if op_conds else {"(no flag signals detected)": None},
        "ai_model_info": {
            "model":         "IsolationForest",
            "anomaly_score": _if_score_to_probability(event.score),
            "raw_if_score":  round(float(event.score), 6),
            "llm_verdict":   event.metadata.get("llm_verdict", ""),
        },
    }
    return payload
