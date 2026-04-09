"""PIMS 장애 분석 대시보드.

실행: streamlit run app.py
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
from src.services.config_service import load_settings, load_signal_config
from src.services.pipeline_builder import build_llm_filter, get_loader
from src.services.signal_label_mapper import SignalLabelMapper
from src.ui.charts import (
    render_event_selector,
    render_event_summary_and_bar,
    render_metrics_hitl,
    render_stats_table,
    render_timeline,
    render_trend_chart,
)
from src.ui.sidebar import render_sidebar
from src.utils.preprocessor import Preprocessor
from src.utils.operation_filter import OperationFilter
from src.utils.operation_discovery import AutoOperationDiscovery, OperationCondition
from src.utils.signal_reducer import SignalReducer
from src.utils.rolling_features import RollingFeatureExtractor
from src.utils.context_formatter import AnomalyContextFormatter
from src.services.equipment_profile_store import EquipmentProfileStore
from src.services.feedback_store import FeedbackStore
from src.utils.device_group_parser import DeviceGroupParser

KST = "Asia/Seoul"

# ── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="PIMS 장애 분석", page_icon="⚡", layout="wide")


# ── 캐시된 리소스 / 데이터 ────────────────────────────────────────────────────
@st.cache_resource
def load_label_mapper() -> "SignalLabelMapper | None":
    """OVEN xlsx 라벨 매퍼를 로드한다. 파일 없으면 None."""
    cfg_path = ROOT / "config" / "label_files.yaml"
    if not cfg_path.exists():
        return None
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
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
    groups_profile_json: str,   # {group_id: {conditions:[...], clusters:{...}|null}}
    _if_params_key: str,
    _topn: int,
    _excluded: tuple[str, ...],
    _settings_key: str,
) -> tuple[list, list, int, int, int]:
    """
    Returns: (candidates, events, total_rows, running_rows, reduced_cols)
    running_rows = 그룹들의 가동 행 수 합산
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


# ── 사이드바 ──────────────────────────────────────────────────────────────────
csv_input, top_n, run_btn = render_sidebar(ROOT, clear_detection_cache_fn=detect_anomalies_hitl.clear)

# ── 메인 영역 ─────────────────────────────────────────────────────────────────
st.title("PIMS 장애 분석 대시보드")

if not run_btn and "df" not in st.session_state:
    st.info("사이드바에서 CSV 파일 경로를 입력하고 **분석 실행**을 클릭하세요.")
    st.stop()

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

            # 클러스터 미캐시 → 사전 계산 후 저장 (@st.cache_data 내부 파일 I/O 금지 원칙)
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

mapper = load_label_mapper()
df: pd.DataFrame = st.session_state.get("df")
events: list[AnomalyEvent] = st.session_state.get("events", [])
candidates: list[AnomalyEvent] = st.session_state.get("candidates", [])
if df is None:
    st.stop()

# ── 요약 카드 ─────────────────────────────────────────────────────────────────
render_metrics_hitl(
    df, candidates, events, KST,
    running_rows=st.session_state.get("running_rows", len(df)),
    reduced_cols=st.session_state.get("reduced_cols", len(df.columns)),
    group_count=st.session_state.get("group_count", 0),
)
st.divider()

if not events:
    st.success("탐지된 이상 이벤트가 없습니다.")
    st.stop()

# ── 이상 이벤트 선택 ──────────────────────────────────────────────────────────
classifier = FaultClassifier()
sel_idx, event, ts_kst = render_event_selector(events, KST)

# ── 이벤트 요약 + 급변 신호 bar chart + 신호 선택 라디오 ──────────────────────
sel_sig = render_event_summary_and_bar(event, mapper, classifier, df, sel_idx, ts_kst, KST)
st.divider()

# ── 신호 추이 차트 ────────────────────────────────────────────────────────────
st.subheader("📈 신호 추이 (이상 이벤트 전후)")
context_sec = st.slider("표시 구간 (이벤트 기준 ±초)", 10, 300, 60, 10, key="ctx_slider")

if sel_sig:
    render_trend_chart(df, event, sel_sig, context_sec, mapper, KST)
    render_stats_table(df, event, mapper, sel_idx, context_sec, KST,
                       clear_detection_cache_fn=detect_anomalies_hitl.clear)

    # ── 관리자 피드백 ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("관리자 피드백")

    _settings_cfg = load_settings()
    _hitl_cfg = _settings_cfg.get("hitl", {})
    fb_store = FeedbackStore(
        db_path=str(ROOT / _hitl_cfg.get("feedback_db", "feedback.db"))
    )
    formatter = AnomalyContextFormatter(
        context_minutes=int(_hitl_cfg.get("context_minutes", 5))
    )

    fb_key = f"feedback_{sel_idx}"
    prev_label = st.session_state.get(fb_key, "미판정")

    fb_col1, fb_col2, fb_col3 = st.columns([3, 1, 1])
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

    retrain = fb_store.retrain_scaffold()
    st.caption(
        f"피드백 누적: {retrain['total_labeled']}개 "
        f"(이상 {retrain['anomaly_count']} / 정상 {retrain['normal_count']}) "
        f"— {retrain['message']}"
    )

# ── 전체 이상 타임라인 ────────────────────────────────────────────────────────
st.divider()
available = [s[0] for s in event.top_signals if s[0] in df.columns]
render_timeline(df, events, sel_idx, available, KST)
