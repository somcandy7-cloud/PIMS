"""PIMS 장애 분석 대시보드.

실행:
    streamlit run app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.agents.base_detector import AnomalyEvent
from src.agents.fault_classifier import FaultClassifier
from src.services.analysis_report_exporter import (
    build_report_frames,
    frames_to_excel_bytes,
    save_report_file,
)
from src.services.config_service import load_settings, load_signal_config
from src.services.equipment_profile_store import EquipmentProfileStore
from src.services.feedback_store import FeedbackStore
from src.services.pipeline_builder import build_llm_filter, get_loader
from src.services.signal_label_mapper import SignalLabelMapper
from src.ui.charts import (
    render_event_summary_and_bar,
    render_metrics_hitl,
    render_stats_table,
    render_timeline,
    render_trend_chart,
)
from src.ui.industrial_panels import (
    render_alarm_logic_panel,
    render_anomaly_score_board,
    render_event_image_panel,
    render_llm_chat_panel,
)
from src.ui.sidebar import render_sidebar
from src.ui.theme import apply_dashboard_theme
from src.utils.device_group_parser import DeviceGroupParser
from src.utils.operation_discovery import AutoOperationDiscovery, OperationCondition
from src.utils.operation_filter import OperationFilter
from src.utils.preprocessor import Preprocessor
from src.utils.rolling_features import RollingFeatureExtractor
from src.utils.signal_reducer import SignalReducer

KST = "Asia/Seoul"

st.set_page_config(page_title="PIMS 장애 분석", page_icon="⚡", layout="wide")
apply_dashboard_theme()


@st.cache_resource
def load_label_mapper() -> SignalLabelMapper | None:
    """OVEN xlsx 라벨 매퍼를 로드한다. 파일이 없으면 None."""
    cfg_path = ROOT / "config" / "label_files.yaml"
    if not cfg_path.exists():
        return None
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    paths = [ROOT / p for p in (cfg.get("label_files") or [])]
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return SignalLabelMapper(existing)


@st.cache_data(show_spinner="데이터 로드 중...")
def load_and_process(path: str) -> pd.DataFrame:
    df = get_loader(path).load(path)
    df = Preprocessor().process(df)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(KST)
    else:
        df.index = df.index.tz_localize(KST)
    return df


@st.cache_data(show_spinner="이상치 탐지 중 (그룹별 HITL 파이프라인)...")
def detect_anomalies_hitl(
    path: str,
    groups_profile_json: str,
    _if_params_key: str,
    _topn: int,
    _excluded: tuple[str, ...],
    _settings_key: str,
) -> tuple[list[AnomalyEvent], list[AnomalyEvent], int, int, int]:
    """Returns: (candidates, events, total_rows, running_rows, reduced_cols)."""
    from src.agents.isolation_forest_adapter import IsolationForestAdapter

    df = load_and_process(path)
    settings = json.loads(_settings_key)
    hitl_cfg = settings.get("hitl", {})
    det_cfg = hitl_cfg.get("detector", {})
    roll_cfg = hitl_cfg.get("rolling", {})
    if_params = json.loads(_if_params_key)

    groups_profile: dict = json.loads(groups_profile_json)
    device_groups = DeviceGroupParser.parse(df.columns.tolist())

    total_reduced_cols = 0
    all_candidates: list[AnomalyEvent] = []
    running_index_union = pd.Index([])
    all_feature_frames: list[pd.DataFrame] = []

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

        running_index_union = running_index_union.union(df_running.index)

        clusters = profile.get("clusters")
        if clusters:
            rep_cols = [c for c in clusters if c in df_running.columns]
            df_reduced = df_running[rep_cols].copy()
        else:
            df_reduced, _ = SignalReducer.from_config(hitl_cfg).fit_transform(df_running)

        total_reduced_cols += len(df_reduced.columns)
        if df_reduced.empty:
            continue

        df_feat = RollingFeatureExtractor(window=int(roll_cfg.get("window_rows", 30))).transform(df_reduced)
        all_feature_frames.append(df_feat)

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

    if all_feature_frames:
        llm_input_df = pd.concat(all_feature_frames, axis=1)
        llm_input_df = llm_input_df.loc[:, ~llm_input_df.columns.duplicated()]
        llm_input_df = llm_input_df.sort_index()
    else:
        llm_input_df = df

    llm_filter = build_llm_filter(settings)
    events = llm_filter.filter(all_candidates, llm_input_df)

    return all_candidates, events, len(df), len(running_index_union), total_reduced_cols


csv_input, top_n, run_btn = render_sidebar(ROOT, clear_detection_cache_fn=detect_anomalies_hitl.clear)

st.title("PIMS 장애 분석 대시보드")

if not run_btn and "df" not in st.session_state:
    st.info("사이드바에서 CSV 파일 경로를 입력하고 **분석 시작하기**를 눌러주세요.")
    st.stop()

if run_btn:
    csv_path = csv_input.strip()
    if not csv_path:
        st.error("분석할 파일 경로를 입력하거나 파일을 업로드해 주세요.")
        st.stop()

    csv_file = Path(csv_path)
    if not csv_file.exists():
        st.error(f"파일을 찾을 수 없습니다: {csv_path}")
        st.stop()
    if not csv_file.is_file():
        st.error(f"파일이 아니라 폴더입니다: {csv_path}")
        st.stop()
    if csv_file.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
        st.error(f"지원하지 않는 확장자입니다: {csv_file.suffix}")
        st.stop()

    equipment_id = EquipmentProfileStore.extract_id(csv_file.name)
    ep_store = EquipmentProfileStore()

    with st.spinner("분석 중..."):
        excluded, _, model_params = load_signal_config()
        settings_cfg = load_settings()
        hitl_cfg = settings_cfg.get("hitl", {})
        if_params = hitl_cfg.get("detector", model_params.get("isolation_forest", {}))
        if_params_key = json.dumps(if_params, sort_keys=True)
        settings_key = json.dumps(settings_cfg, sort_keys=True)

        df_temp = load_and_process(csv_path)
        device_groups = DeviceGroupParser.parse(df_temp.columns.tolist())
        all_groups = ep_store.load_all_groups(equipment_id)
        disc_cfg = hitl_cfg.get("discovery", {})
        sr = SignalReducer.from_config(hitl_cfg)

        groups_profile: dict = {}
        for group_id, group_cols in device_groups.items():
            if len(group_cols) < 5:
                continue

            gp = all_groups.get(group_id, {})
            raw_conds = gp.get("conditions")

            if raw_conds is None:
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

            if clusters is None and raw_conds:
                conds_objs = [OperationCondition.from_dict(c) for c in raw_conds]
                df_grp = df_temp[group_cols]
                df_running_grp = OperationFilter(conds_objs).filter(df_grp) if conds_objs else df_grp.copy()
                if not df_running_grp.empty:
                    _, clusters = sr.fit_transform(df_running_grp)
                    ep_store.save_group_clusters(equipment_id, group_id, clusters)

            groups_profile[group_id] = {"conditions": raw_conds, "clusters": clusters}

        groups_profile_json = json.dumps(groups_profile, sort_keys=True)
        candidates, events, total_rows, running_rows, reduced_cols = detect_anomalies_hitl(
            csv_path,
            groups_profile_json,
            if_params_key,
            top_n,
            tuple(excluded),
            settings_key,
        )
        group_count = len([g for g, cols in device_groups.items() if len(cols) >= 5])
        df = load_and_process(csv_path)

    st.session_state.update(
        df=df,
        events=events,
        candidates=candidates,
        csv_path=csv_path,
        equipment_id=equipment_id,
        total_rows=total_rows,
        running_rows=running_rows,
        reduced_cols=reduced_cols,
        group_count=group_count,
        groups_profile=groups_profile,
        group_signal_counts={g: len(cols) for g, cols in device_groups.items() if len(cols) >= 5},
    )
    st.rerun()

mapper = load_label_mapper()
df: pd.DataFrame | None = st.session_state.get("df")
events: list[AnomalyEvent] = st.session_state.get("events", [])
candidates: list[AnomalyEvent] = st.session_state.get("candidates", [])
if df is None:
    st.stop()

render_metrics_hitl(
    df,
    candidates,
    events,
    KST,
    running_rows=st.session_state.get("running_rows", len(df)),
    reduced_cols=st.session_state.get("reduced_cols", len(df.columns)),
    group_count=st.session_state.get("group_count", 0),
)

report_frames = build_report_frames(
    df=df,
    source_file=st.session_state.get("csv_path", ""),
    candidates=candidates,
    events=events,
    running_rows=st.session_state.get("running_rows", len(df)),
    reduced_cols=st.session_state.get("reduced_cols", len(df.columns)),
    group_count=st.session_state.get("group_count", 0),
    group_signal_counts=st.session_state.get("group_signal_counts", {}),
    groups_profile=st.session_state.get("groups_profile", {}),
    kst=KST,
)
report_bytes = frames_to_excel_bytes(report_frames)
report_name = f"pims_report_{pd.Timestamp.now(tz=KST).strftime('%Y%m%d_%H%M%S')}.xlsx"

st.subheader("검증 리포트")
rc1, rc2 = st.columns([1, 1])
with rc1:
    if st.button("엑셀 파일 저장", key="save_excel_report_btn"):
        saved_path = save_report_file(
            root=ROOT,
            source_file=st.session_state.get("csv_path", "analysis.csv"),
            excel_bytes=report_bytes,
        )
        st.session_state["last_report_path"] = str(saved_path)
        st.success(f"저장 완료: {saved_path}")
with rc2:
    st.download_button(
        "엑셀 다운로드",
        data=report_bytes,
        file_name=report_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_excel_report_btn",
        use_container_width=True,
    )

last_saved = st.session_state.get("last_report_path")
if last_saved:
    st.caption(f"최근 저장 파일: {last_saved}")

with st.expander("리포트 미리보기 (대시보드에서 바로 확인)"):
    tab1, tab2, tab3, tab4 = st.tabs(["요약", "장치 그룹", "IF 후보", "LLM 검증"])
    with tab1:
        st.dataframe(report_frames["Summary"], use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(report_frames["DeviceGroups"], use_container_width=True, hide_index=True)
    with tab3:
        st.dataframe(report_frames["IFCandidates"], use_container_width=True, hide_index=True)
    with tab4:
        st.dataframe(report_frames["LLMVerified"], use_container_width=True, hide_index=True)

st.divider()
if not events:
    st.success("현재는 이상 이벤트가 없습니다.")
    st.stop()

settings_cfg = load_settings()
hitl_cfg = settings_cfg.get("hitl", {})
fb_store = FeedbackStore(db_path=str(ROOT / hitl_cfg.get("feedback_db", "feedback.db")))
_, signal_overrides, _ = load_signal_config()

st.subheader("실시간 이상치 검증")
top_left, top_right = st.columns([1.2, 1], gap="large")
selected_idx = int(st.session_state.get("selected_event_idx", 0))
selected_idx = min(max(selected_idx, 0), len(events) - 1)

with top_left:
    picked_idx = render_anomaly_score_board(
        events=events,
        selected_idx=selected_idx,
        kst=KST,
        max_items=max(1, top_n),
    )
if picked_idx != selected_idx:
    st.session_state["selected_event_idx"] = picked_idx
    st.rerun()

sel_idx = st.selectbox(
    "이상 이벤트 선택",
    options=list(range(len(events))),
    index=selected_idx,
    format_func=lambda i: (
        f"이상 {i+1:02d} | "
        f"{events[i].timestamp.tz_convert(KST).strftime('%H:%M:%S')} | "
        f"score={events[i].score:.4f} "
        f"[{events[i].metadata.get('llm_verdict', '')}]"
    ),
    key="event_picker_v2",
)
st.session_state["selected_event_idx"] = sel_idx
event = events[sel_idx]
ts_kst = event.timestamp.tz_convert(KST)

with top_right:
    render_event_image_panel(root=ROOT, event=event, kst=KST)
    q1, q2 = st.columns(2)
    if q1.button("O (실제 이상)", key=f"quick_o_{sel_idx}", type="primary", use_container_width=True):
        fb_store.save(event, label="O", reason="quick_validation")
        st.success("O 라벨 저장 완료")
    if q2.button("X (오탐)", key=f"quick_x_{sel_idx}", use_container_width=True):
        fb_store.save(event, label="X", reason="quick_validation")
        st.success("X 라벨 저장 완료")

classifier = FaultClassifier()
sel_sig = render_event_summary_and_bar(event, mapper, classifier, df, sel_idx, ts_kst, KST)
st.divider()

st.subheader("신호 추이 / LLM 근거 / 알람 로직")
context_sec = st.slider("표시 구간 (이벤트 기준 ±초)", 10, 300, 60, 10, key="ctx_slider")
c_trend, c_chat, c_alarm = st.columns([1, 1, 1], gap="large")

with c_trend:
    if sel_sig:
        render_trend_chart(df, event, sel_sig, context_sec, mapper, KST)
    else:
        st.info("상단에서 신호를 선택하면 추이 그래프가 표시됩니다.")

with c_chat:
    render_llm_chat_panel(event, sel_idx)

with c_alarm:
    render_alarm_logic_panel(event=event, signal_overrides=signal_overrides)

if sel_sig:
    render_stats_table(
        df,
        event,
        mapper,
        sel_idx,
        context_sec,
        KST,
        clear_detection_cache_fn=detect_anomalies_hitl.clear,
    )

st.divider()
st.subheader("관리자 피드백")
fb_key = f"feedback_{sel_idx}"
prev_label = st.session_state.get(fb_key, "미판정")
fc1, fc2, fc3 = st.columns([3, 1, 1])
with fc1:
    fb_label = st.radio(
        "탐지 결과가 실제 이상입니까?",
        ["미판정", "O (이상 확정)", "X (정상 오탐)"],
        index=["미판정", "O (이상 확정)", "X (정상 오탐)"].index(prev_label),
        key=f"fb_radio_{sel_idx}",
        horizontal=True,
    )
with fc2:
    fb_reason = st.text_input("판단 근거 (선택)", key=f"fb_reason_{sel_idx}")
with fc3:
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

retrain = fb_store.retrain_scaffold()
st.caption(
    f"피드백 누적: {retrain['total_labeled']}개"
    f"(이상 {retrain['anomaly_count']} / 정상 {retrain['normal_count']}) "
    f"→ {retrain['message']}"
)

st.divider()
available = [s[0] for s in event.top_signals if s[0] in df.columns]
render_timeline(df, events, sel_idx, available, KST)
