"""대시보드 차트 및 표시 컴포넌트."""
from __future__ import annotations
from typing import Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.agents.base_detector import AnomalyEvent
from src.agents.fault_classifier import FaultClassifier
from src.services.config_service import load_signal_config, save_signal_config
from src.utils.formatters import display_name


def render_metrics(
    df: pd.DataFrame,
    candidates: list[AnomalyEvent],
    events: list[AnomalyEvent],
    kst: str,
) -> None:
    """상단 요약 메트릭 카드 5개를 렌더링한다."""
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("총 샘플 수", f"{len(df):,}")
    c2.metric("기록 신호 수", f"{len(df.columns):,}")
    c3.metric("IF 후보", f"{len(candidates)}건")
    c4.metric("LLM 검증 이상", f"{len(events)}건")
    dur = (df.index[-1] - df.index[0]).total_seconds() / 60
    c5.metric("데이터 구간", f"{dur:.1f}분")


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
    running_pct = min(100.0, running_rows / max(len(df), 1) * 100)
    c2.metric("가동 구간", f"{running_rows:,}행", f"{running_pct:.0f}%")
    c3.metric("장치 그룹", f"{group_count}개" if group_count else "-")
    c4.metric("원본 신호", f"{len(df.columns):,}")
    c5.metric("축소 신호", f"{reduced_cols:,}", f"-{len(df.columns)-reduced_cols:,}")
    c6.metric("IF 후보", f"{len(candidates)}건")
    c7.metric("LLM 검증", f"{len(events)}건")
    dur = (df.index[-1] - df.index[0]).total_seconds() / 60
    c8.metric("데이터 구간", f"{dur:.1f}분")


def render_event_selector(
    events: list[AnomalyEvent],
    kst: str,
) -> tuple[int, AnomalyEvent, pd.Timestamp]:
    """이상 이벤트 선택 드롭다운을 렌더링하고 (sel_idx, event, ts_kst)를 반환한다."""
    event_labels = [
        f"이상 {i+1:02d} | {ev.timestamp.tz_convert(kst).strftime('%H:%M:%S')} | score={ev.score:.4f}"
        + (f" [{ev.metadata.get('llm_verdict')}]" if ev.metadata.get("llm_verdict") else "")
        for i, ev in enumerate(events)
    ]
    sel_idx = st.selectbox(
        "이상 이벤트 선택",
        range(len(events)),
        format_func=lambda i: event_labels[i],
    )
    event = events[sel_idx]
    ts_kst = event.timestamp.tz_convert(kst)
    return sel_idx, event, ts_kst


def render_event_summary_and_bar(
    event: AnomalyEvent,
    mapper,
    classifier: FaultClassifier,
    df: pd.DataFrame,
    sel_idx: int,
    ts_kst: pd.Timestamp,
    kst: str,
) -> str | None:
    """이벤트 요약(left) + 급변 신호 bar chart + 신호 라디오(right)를 렌더링한다.

    선택된 신호명을 반환한다. 신호 없으면 None.
    """
    trip_event = event.metadata.get("trip_event")
    left, right = st.columns([1, 2])

    with left:
        st.subheader("📋 이벤트 요약")
        if trip_event is not None:
            report = classifier.classify(trip_event)
            sev_icon = {"높음": "🔴", "중간": "🟡", "낮음": "🟢"}.get(report.severity, "⚪")
            st.markdown(f"""
| 항목 | 내용 |
|------|------|
| 발생 시각 (KST) | `{ts_kst.strftime('%Y-%m-%d %H:%M:%S')}` |
| 장애 유형 | {report.fault_type} |
| 심각도 | {sev_icon} {report.severity} |
| 급변 신호 수 | {len(event.top_signals)}개 |
""")
            st.markdown("**원인 추정**")
            st.info(report.root_cause_hypothesis)
            st.markdown("**점검 항목**")
            for i, item in enumerate(report.inspection_items, 1):
                st.markdown(f"{i}. {item}")
        else:
            top_sig_str = ", ".join(display_name(s[0], mapper) for s in event.top_signals[:3])
            if len(event.top_signals) > 3:
                top_sig_str += f" 외 {len(event.top_signals)-3}개"
            st.markdown(f"""
| 항목 | 내용 |
|------|------|
| 발생 시각 (KST) | `{ts_kst.strftime('%Y-%m-%d %H:%M:%S')}` |
| 이상치 점수 | {event.score:.4f} |
| 탐지 모델 | Isolation Forest |
| 급변 신호 수 | {len(event.top_signals)}개 |
| 주요 신호 | {top_sig_str} |
""")
            llm_verdict = event.metadata.get("llm_verdict", "")
            llm_reason = event.metadata.get("llm_reason", "")
            if llm_verdict:
                st.markdown(f"**LLM 판정**: `{llm_verdict}`")
            if llm_reason:
                st.info(llm_reason)

    with right:
        st.subheader("📊 급변 신호 Top N — 신호를 선택하면 아래 추이 차트 갱신")
        sig_names = [s[0] for s in event.top_signals]
        sig_mags  = [s[1] for s in event.top_signals]
        sig_display_names = [display_name(s, mapper) for s in sig_names]

        fig_bar = go.Figure(go.Bar(
            x=sig_mags[::-1], y=sig_display_names[::-1],
            orientation="h", marker_color="crimson",
            text=[f"{v:.2f}" for v in sig_mags[::-1]], textposition="outside",
        ))
        fig_bar.update_layout(
            xaxis_title="변화량 (절대값)",
            height=max(250, len(sig_names) * 45),
            margin=dict(l=10, r=60, t=10, b=30),
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="bar_chart")

        # top_signals 컬럼명은 rolling suffix(_rmean/_rstd/_roc)가 붙은 특징명.
        # 원본 df 컬럼과 대조하려면 suffix를 제거해 기본 신호명으로 복원한다.
        _ROLL_SUFFIXES = ("_rmean", "_rstd", "_roc")

        def _base_sig(col: str) -> str:
            for sfx in _ROLL_SUFFIXES:
                if col.endswith(sfx):
                    return col[: -len(sfx)]
            return col

        seen: set[str] = set()
        available: list[str] = []
        for s in sig_names:
            base = _base_sig(s)
            if base not in seen and base in df.columns:
                seen.add(base)
                available.append(base)

        if available:
            available_display = {s: display_name(s, mapper) for s in available}
            sel_sig = st.radio(
                "신호 선택 →",
                options=available,
                format_func=lambda s: available_display[s],
                horizontal=True,
                key=f"sig_radio_{sel_idx}",
            )
        else:
            sel_sig = None

    return sel_sig


def render_trend_chart(
    df: pd.DataFrame,
    event: AnomalyEvent,
    sel_sig: str,
    context_sec: int,
    mapper,
    kst: str,
) -> None:
    """선택 신호의 추이 차트(이벤트 전후 ±context_sec)를 렌더링한다."""
    ts_kst_trip = event.timestamp.tz_convert(kst)
    t_start = ts_kst_trip - pd.Timedelta(seconds=context_sec)
    t_end   = ts_kst_trip + pd.Timedelta(seconds=context_sec)
    window_df = df.loc[t_start:t_end]

    n_baseline = max(10, int(len(df) * 0.3))
    baseline_df = df.iloc[:n_baseline]

    series = window_df[sel_sig].dropna()
    base_s = baseline_df[sel_sig].dropna() if sel_sig in baseline_df.columns else pd.Series(dtype=float)

    b_mean = float(base_s.mean()) if len(base_s) > 0 else float("nan")
    b_std  = float(base_s.std())  if len(base_s) > 1 else 0.0

    fig = go.Figure()

    if len(series) > 0:
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            mode="lines", name=sel_sig,
            line=dict(color="royalblue", width=2),
        ))

        x0, x1 = series.index[0].isoformat(), series.index[-1].isoformat()

        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[b_mean, b_mean],
            mode="lines", name=f"정상 평균 ({b_mean:.2f})",
            line=dict(color="green", width=1, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[b_mean + 2*b_std, b_mean + 2*b_std],
            mode="lines", name=f"μ+2σ ({b_mean+2*b_std:.2f})",
            line=dict(color="orange", width=1, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[b_mean - 2*b_std, b_mean - 2*b_std],
            mode="lines", name=f"μ-2σ ({b_mean-2*b_std:.2f})",
            line=dict(color="orange", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(0,200,0,0.07)",
        ))

        _ts_iso = ts_kst_trip.isoformat()
        fig.add_shape(type="line", x0=_ts_iso, x1=_ts_iso, y0=0, y1=1,
                      xref="x", yref="paper",
                      line=dict(color="red", width=2))
        fig.add_annotation(x=_ts_iso, y=0.97, xref="x", yref="paper",
                           text="이상", showarrow=False,
                           xanchor="left", font=dict(color="red", size=12))

    fig.update_layout(
        title=f"{sel_sig}  |  정상 평균={b_mean:.2f}  σ={b_std:.2f}",
        xaxis_title="시각 (KST)",
        yaxis_title="값",
        height=400,
        legend=dict(orientation="h", y=-0.25),
        margin=dict(t=50, b=80),
    )
    st.plotly_chart(fig, use_container_width=True, key="trend_chart")


def render_stats_table(
    df: pd.DataFrame,
    event: AnomalyEvent,
    mapper,
    sel_idx: int,
    context_sec: int,
    kst: str,
    clear_detection_cache_fn: Callable[[], None],
) -> None:
    """급변 신호 통계 테이블 + 제외/임계값 조정 버튼을 렌더링한다."""
    st.subheader("📊 급변 신호 통계")

    ts_kst_trip = event.timestamp.tz_convert(kst)
    t_start = ts_kst_trip - pd.Timedelta(seconds=context_sec)
    t_end   = ts_kst_trip + pd.Timedelta(seconds=context_sec)
    window_df = df.loc[t_start:t_end]

    n_baseline = max(10, int(len(df) * 0.3))
    baseline_df = df.iloc[:n_baseline]

    excl_cfg, ovrd_cfg, _ = load_signal_config()

    for sig, mag in event.top_signals:
        if sig not in df.columns:
            continue
        base_s2 = baseline_df[sig].dropna() if sig in baseline_df.columns else pd.Series(dtype=float)
        w_s = window_df[sig].dropna() if sig in window_df.columns else pd.Series(dtype=float)
        trip_val = float(w_s.iloc[0]) if len(w_s) > 0 else float("nan")

        with st.container():
            c1, c2, c3, c4, c5, c_ex, c_thr = st.columns([3, 2, 2, 2, 2, 1, 1])
            c1.markdown(f"**{display_name(sig, mapper)}**")
            desc = mapper.describe(sig) if mapper else ""
            if desc:
                c1.caption(desc)
            c2.metric("이벤트 시점 값", f"{trip_val:.2f}" if not pd.isna(trip_val) else "-")
            c3.metric("정상 평균",
                      f"{base_s2.mean():.2f}" if len(base_s2) > 0 else "-")
            c4.metric("정상 σ",
                      f"{base_s2.std():.2f}" if len(base_s2) > 1 else "-")
            c5.metric("변화량", f"{mag:.2f}")

            if sig not in excl_cfg:
                if c_ex.button("제외", key=f"excl_{sel_idx}_{sig}",
                               help="이 신호를 탐지에서 제외"):
                    excl_cfg.append(sig)
                    save_signal_config(excl_cfg, ovrd_cfg)
                    clear_detection_cache_fn()
                    st.rerun()
            else:
                c_ex.caption("(제외됨)")

            cur_thr = ovrd_cfg.get(sig, 3.0)
            new_thr = cur_thr + 0.5
            if c_thr.button(f"+0.5", key=f"thr_{sel_idx}_{sig}",
                            help=f"임계값 {cur_thr:.1f} → {new_thr:.1f}로 상향"):
                ovrd_cfg[sig] = round(new_thr, 1)
                save_signal_config(excl_cfg, ovrd_cfg)
                clear_detection_cache_fn()
                st.rerun()


def render_timeline(
    df: pd.DataFrame,
    events: list[AnomalyEvent],
    sel_idx: int,
    available: list[str],
    kst: str,
) -> None:
    """전체 이상 이벤트 타임라인 차트를 렌더링한다."""
    st.subheader("🕐 전체 이상 이벤트 타임라인")

    ref_sig = (available[0] if available else
               (df.columns[0] if len(df.columns) > 0 else None))

    fig_tl = go.Figure()
    if ref_sig and ref_sig in df.columns:
        fig_tl.add_trace(go.Scatter(
            x=df.index, y=df[ref_sig].values,
            mode="lines", name=ref_sig,
            line=dict(color="lightblue", width=1), opacity=0.7,
        ))

    for i, ev in enumerate(events):
        _ev_kst = ev.timestamp.tz_convert(kst)
        _ev_iso = _ev_kst.isoformat()
        is_sel  = (i == sel_idx)
        color   = "red" if is_sel else "orange"
        fig_tl.add_shape(type="line", x0=_ev_iso, x1=_ev_iso, y0=0, y1=1,
                         xref="x", yref="paper",
                         line=dict(color=color, width=3 if is_sel else 1.5))
        fig_tl.add_annotation(x=_ev_iso, y=1, xref="x", yref="paper",
                              text=f"E{i+1}", showarrow=False,
                              xanchor="center", yanchor="bottom",
                              font=dict(color=color, size=11))

    fig_tl.update_layout(
        xaxis_title="시각 (KST)", yaxis_title=ref_sig or "",
        height=280, margin=dict(t=30, b=40), showlegend=False,
    )
    st.plotly_chart(fig_tl, use_container_width=True, key="timeline_chart")
