from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.agents.base_detector import AnomalyEvent


def render_anomaly_score_board(
    *,
    events: list[AnomalyEvent],
    selected_idx: int,
    kst: str,
    max_items: int = 10,
) -> int:
    """이상치 TOP 점수 보드(실제 events 기반)를 렌더링하고 선택 index를 반환한다."""
    st.markdown('<div class="ia-section-title">Top Anomaly Scores</div>', unsafe_allow_html=True)
    if not events:
        st.info("표시할 이상치가 없습니다.")
        return selected_idx

    ranked = sorted(
        list(enumerate(events)),
        key=lambda x: abs(float(x[1].score)),
        reverse=True,
    )[: max(1, max_items)]
    max_abs = max(abs(float(ev.score)) for _, ev in ranked) or 1.0

    new_selected = selected_idx
    for idx, ev in ranked:
        ts = ev.timestamp.tz_convert(kst).strftime("%H:%M:%S")
        ratio = min(1.0, abs(float(ev.score)) / max_abs)
        st.markdown('<div class="ia-row">', unsafe_allow_html=True)
        c1, c2 = st.columns([5, 1])
        with c1:
            st.caption(f"이상 {idx+1:02d} | {ts} | score={ev.score:.4f}")
            st.progress(ratio, text=f"severity {ratio*100:.1f}%")
        with c2:
            if st.button("선택", key=f"pick_ev_{idx}", use_container_width=True):
                new_selected = idx
        st.markdown("</div>", unsafe_allow_html=True)
    return new_selected


def render_event_image_panel(
    *,
    root: Path,
    event: AnomalyEvent,
    kst: str,
) -> None:
    """이미지 패널(선택 이벤트 요약 포함)."""
    st.markdown('<div class="ia-section-title">Anomaly Image View</div>', unsafe_allow_html=True)
    img_path = root / "dashboard_UI" / "screen.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
    else:
        st.info("이미지 가이드 파일이 없어 기본 뷰만 표시합니다.")

    ts = event.timestamp.tz_convert(kst).strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(
        f"""
<div class="ia-kv">
  <strong>발생 시각:</strong> {ts}<br/>
  <strong>점수:</strong> {event.score:.4f}<br/>
  <strong>그룹:</strong> {event.metadata.get("group_id", "-")}<br/>
  <strong>LLM 판정:</strong> {event.metadata.get("llm_verdict", "-")}
</div>
        """,
        unsafe_allow_html=True,
    )


def render_llm_chat_panel(event: AnomalyEvent, sel_idx: int) -> None:
    """선택 이벤트 기준 LLM 근거/대화 패널."""
    st.markdown('<div class="ia-section-title">Architect Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="ia-glass">', unsafe_allow_html=True)

    chat_key = f"llm_chat_{sel_idx}"
    if chat_key not in st.session_state:
        seed_reason = event.metadata.get("llm_reason", "").strip()
        if not seed_reason:
            seed_reason = "현재 이벤트에 대한 상세 LLM 근거가 비어 있습니다. 설정/키/모델 상태를 점검해 주세요."
        st.session_state[chat_key] = [
            {"role": "assistant", "content": seed_reason},
        ]

    for msg in st.session_state[chat_key]:
        role = "assistant" if msg["role"] == "assistant" else "user"
        with st.chat_message(role):
            st.write(msg["content"])

    user_prompt = st.chat_input("이 이상치 근거를 추가로 질문하세요.", key=f"chat_input_{sel_idx}")
    if user_prompt:
        st.session_state[chat_key].append({"role": "user", "content": user_prompt})
        answer = (
            "대화형 후속 분석은 다음 단계에서 실제 LLM 재호출로 연결할 예정입니다. "
            f"현재 이벤트 판정은 `{event.metadata.get('llm_verdict', 'N/A')}`이며, "
            "상단 근거 문장을 기준으로 검증해 주세요."
        )
        st.session_state[chat_key].append({"role": "assistant", "content": answer})
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def render_alarm_logic_panel(
    *,
    event: AnomalyEvent,
    signal_overrides: dict[str, float],
) -> None:
    """선택 이벤트의 상위 신호 기준 알람 로직 리스트."""
    st.markdown('<div class="ia-section-title">Alarm Execution Logic</div>', unsafe_allow_html=True)
    if not event.top_signals:
        st.info("알람 로직을 만들 상위 신호가 없습니다.")
        return

    for sig, mag in event.top_signals[:10]:
        thr = float(signal_overrides.get(sig, 3.0))
        state = "ARMED" if abs(float(mag)) >= thr else "WAITING"
        st.markdown('<div class="ia-row">', unsafe_allow_html=True)
        c1, c2 = st.columns([4, 1])
        with c1:
            st.write(f"**{sig}**")
            st.caption(f"Logic: abs(contribution) >= {thr:.2f} | value={float(mag):.4f}")
        with c2:
            st.write(f"`{state}`")
        st.markdown("</div>", unsafe_allow_html=True)


def render_alarm_payload_panel(
    *,
    event: AnomalyEvent,
    df: pd.DataFrame,
    equipment_id: str,
    kst: str,
    mapper=None,
    sel_idx: int = 0,
) -> None:
    """사내 알람 로직 LLM에 전달할 JSON 페이로드를 생성하고 표시한다.

    JSON을 복사해서 사내 LLM에 붙여넣으면 알람 로직 추천을 받을 수 있다.
    """
    import json
    from src.utils.alarm_payload_builder import build_alarm_payload

    st.subheader("🔔 알람 로직 LLM 페이로드")
    st.caption(
        "아래 JSON을 복사해서 사내 알람 로직 LLM에 전달하면 "
        "기존 알람 로직 고도화 방안 또는 신규 로직 제안을 받을 수 있습니다."
    )

    with st.spinner("페이로드 생성 중..."):
        try:
            payload = build_alarm_payload(
                event=event,
                df=df,
                equipment_id=equipment_id,
                kst=kst,
                mapper=mapper,
            )
            payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception as exc:
            st.error(f"페이로드 생성 실패: {exc}")
            return

    # 필드 요약 표시
    af = payload.get("anomaly_features", {})
    ai = payload.get("ai_model_info", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("주요 신호", payload.get("signal", "-")[:20])
    c2.metric("이상 점수", f"{ai.get('anomaly_score', 0):.2f}")
    c3.metric("이상 편차", f"{af.get('deviation') or 0:.2f}")
    c4.metric("이상 타입", ", ".join(af.get("type", ["-"]))[:20])

    # JSON 코드 블록 (st.code는 자동으로 복사 버튼 포함)
    st.code(payload_json, language="json")

    # 가동 조건 플래그가 비어있으면 안내
    op = payload.get("operating_condition", {})
    if "(no flag signals detected)" in op:
        st.info(
            "💡 `operating_condition` 항목이 비어 있습니다. "
            "이 시점에 변화한 flag 신호가 없거나, 그룹 정보가 없습니다. "
            "필요 시 수동으로 채워서 전달하세요."
        )


