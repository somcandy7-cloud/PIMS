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


def render_detection_reasoning(event: AnomalyEvent) -> None:
    """IF 판단 근거와 LLM 검증 결과를 사용자 친화적으로 표시한다."""

    # ── rolling 접미사 → 한국어 해석 ───────────────────────────────────────
    _SUFFIX_LABEL: dict[str, str] = {
        "_rstd": "구간 변동성 급증",
        "_roc":  "급격한 변화율",
        "_rmean":"평균값 이탈",
    }

    def _decode_signal(col: str) -> tuple[str, str]:
        """(base_name, 변화 유형) 반환."""
        for sfx, label in _SUFFIX_LABEL.items():
            if col.endswith(sfx):
                return col[: -len(sfx)], label
        return col, "값 이상"

    # ── IF 점수 해석 ────────────────────────────────────────────────────────
    s = event.score
    if s <= -0.5:
        score_label, score_color = "매우 강함", "🔴"
    elif s <= -0.3:
        score_label, score_color = "강함", "🟠"
    elif s <= -0.15:
        score_label, score_color = "보통", "🟡"
    else:
        score_label, score_color = "경계 수준", "🟢"

    contamination = event.metadata.get("contamination", 0.02)

    # ── LLM 결과 ────────────────────────────────────────────────────────────
    llm_verdict = event.metadata.get("llm_verdict", "")
    llm_reason  = event.metadata.get("llm_reason", "")
    llm_backend = event.metadata.get("llm_backend", "")

    st.subheader("탐지 과정 요약")
    c_if, c_llm = st.columns(2, gap="large")

    # ── IF 패널 ─────────────────────────────────────────────────────────────
    with c_if:
        st.markdown("**Isolation Forest 판단**")
        st.markdown(
            f"{score_color} 이상 점수: **{s:.4f}** ({score_label})  \n"
            f"전체 데이터 중 상위 **{contamination*100:.0f}%** 이상치 기준으로 분류됨"
        )
        st.caption("주요 이상 원인 신호:")
        for sig, mag in event.top_signals[:5]:
            base, kind = _decode_signal(sig)
            st.markdown(f"- `{base}` — **{kind}** (변화량 {mag:.3f})")
        with st.expander("Isolation Forest란?", expanded=False):
            st.caption(
                "각 데이터 포인트를 무작위로 고립시킬 때 '몇 번 만에 고립되는가'를 기준으로 "
                "이상치를 판단합니다. 적은 횟수로 고립될수록 이상치일 가능성이 높습니다. "
                f"현재 설정: contamination={contamination*100:.0f}% "
                "(전체 데이터의 이 비율을 이상으로 간주)"
            )

    # ── LLM 패널 ────────────────────────────────────────────────────────────
    with c_llm:
        st.markdown("**LLM 검증 결과**")
        if not llm_verdict:
            st.caption("LLM 결과 없음")
        elif llm_verdict == "KEEP":
            st.success(f"✅ 이상 확정 (KEEP)")
            st.markdown(f"**판단 근거:** {llm_reason}")
        elif llm_verdict == "REJECT":
            st.info(f"⬜ 정상으로 판단 (REJECT)")
            st.markdown(f"**판단 근거:** {llm_reason}")
        elif llm_verdict == "SKIP_KEEP":
            st.warning("⏭ LLM 검증 생략 — IF 점수 기준 통과")
            st.caption(llm_reason)
        else:  # ERROR_KEEP
            st.warning(f"⚠ LLM 오류 — 안전하게 포함 처리")
            st.caption(llm_reason)

        if llm_backend:
            st.caption(f"모델: {llm_backend}")

        with st.expander("LLM 역할이란?", expanded=False):
            st.caption(
                "IF가 통계적으로 '이상'으로 분류한 후보를 LLM이 2차 검토합니다. "
                "전후 구간의 신호 트렌드·통계를 함께 보고 '단순 노이즈인지 vs 실제 이상인지'를 "
                "판단해 최종 이벤트 목록을 정제합니다."
            )


def render_flag_context_panel(
    df: pd.DataFrame,
    event: AnomalyEvent,
    group_id: str | None,
    context_sec: int,
    mapper,
) -> None:
    """이상 시점 flag 신호 상태 패널을 렌더링한다.

    Parameters
    ----------
    df          : 원본 전체 DataFrame (measurement + flag 포함)
    event       : 이상 이벤트
    group_id    : 이벤트가 속한 장치 그룹 ID
    context_sec : 이벤트 전후 ±초 (표시 구간)
    mapper      : SignalLabelMapper (라벨 표시용, None 가능)
    """
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
