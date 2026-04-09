"""사이드바 UI 컴포넌트."""
from __future__ import annotations
from pathlib import Path
from typing import Callable

import streamlit as st
import yaml

from src.services.config_service import (
    SETTINGS_YAML,
    load_signal_config,
    save_llm_config,
    save_signal_config,
)


def render_sidebar(
    root: Path,
    clear_detection_cache_fn: Callable[[], None],
) -> tuple[str, int, bool]:
    """사이드바를 렌더링하고 (csv_path, top_n, run_btn) 을 반환한다."""

    with st.sidebar:
        st.title("⚡ PIMS 분석")
        csv_input = st.text_input(
            "데이터 파일 경로 (.csv / .xlsx)",
            value=str(root / "2603201549_oven.csv"),
            help="iba CSV 또는 Excel 시계열 파일의 전체 경로",
        )
        st.divider()
        st.subheader("탐지 설정")
        top_n = st.slider("급변 신호 표시 개수", 3, 20, 5)
        st.divider()
        st.subheader("LLM 설정")

        with open(SETTINGS_YAML, encoding="utf-8") as _f:
            _settings_now = yaml.safe_load(_f)
        _llm_now = _settings_now.get("llm", {})

        _BACKEND_OPTIONS = ["OpenAI (GPT)", "Ollama (로컬)", "비활성화"]
        _backend_map = {"openai": 0, "ollama": 1, "disabled": 2}
        _current_backend_key = (_llm_now.get("backend") or "disabled").lower()
        _current_idx = _backend_map.get(_current_backend_key, 2)

        _selected_backend = st.radio(
            "백엔드", _BACKEND_OPTIONS, index=_current_idx, key="llm_backend_radio"
        )

        _new_llm_cfg = dict(_llm_now)

        if _selected_backend == "OpenAI (GPT)":
            _new_llm_cfg["backend"] = "openai"
            _oai = _llm_now.get("openai") or {}
            _oai_models = ["gpt-4o", "gpt-4o-mini", "o3-mini"]
            _cur_model = _oai.get("model", "gpt-4o")
            _sel_model_idx = _oai_models.index(_cur_model) if _cur_model in _oai_models else 0
            _sel_model = st.selectbox("모델", _oai_models, index=_sel_model_idx, key="oai_model")
            _new_llm_cfg["openai"] = {**_oai, "model": _sel_model}
            st.caption("API 키: `OPENAI_API_KEY` 환경변수")
        elif _selected_backend == "Ollama (로컬)":
            _new_llm_cfg["backend"] = "ollama"
            _olla = _llm_now.get("ollama") or {}
            _olla_model = st.text_input("모델명", value=_olla.get("model", "qwen3:4b"), key="olla_model")
            _olla_host = st.text_input(
                "호스트", value=_olla.get("host", "http://localhost:11434"), key="olla_host"
            )
            _new_llm_cfg["ollama"] = {**_olla, "model": _olla_model, "host": _olla_host}
        else:
            _new_llm_cfg["backend"] = "disabled"
            st.caption("LLM 필터 비활성화 — IF 후보 전체를 알람으로 처리")

        st.caption("백엔드 변경 후 반드시 '저장' 버튼을 누른 뒤 '분석 실행'을 클릭하세요.")

        if st.button("LLM 설정 저장", key="save_llm_btn"):
            save_llm_config(_new_llm_cfg)
            st.cache_data.clear()
            st.success("저장됨")
            st.rerun()

        _active_backend = _llm_now.get("backend", "disabled")
        if _active_backend == "openai":
            _active_model = (_llm_now.get("openai") or {}).get("model", "?")
            st.info(f"현재: OpenAI / {_active_model}")
        elif _active_backend == "ollama":
            _active_model = (_llm_now.get("ollama") or {}).get("model", "?")
            st.info(f"현재: Ollama / {_active_model}")
        else:
            st.warning("현재: LLM 비활성화")

        st.divider()
        run_btn = st.button("분석 실행", type="primary", use_container_width=True)

        # ── 설비 프로파일 (그룹별) ─────────────────────────────────────────────
        st.divider()
        st.subheader("설비 프로파일")
        _csv_val = csv_input.strip()
        if _csv_val:
            from src.services.equipment_profile_store import EquipmentProfileStore as _EPS
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
                            st.caption(f"`{_c['column']}` {_c['op']} {_c.get('value', 0):.0f}  [{_c.get('source', '?')}]")
                        if not _gconf and _gconds:
                            if st.button("확정", key=f"confirm_{_gid}"):
                                from src.utils.operation_discovery import OperationCondition as _OC
                                _ep.save_group(_eq_id, _gid, [_OC.from_dict(c) for c in _gconds], confirmed=True)
                                st.rerun()
                        if _gclusters:
                            st.caption(f"클러스터: {len(_gclusters)}개 대표 신호")
                            if st.button("클러스터 초기화", key=f"clr_clust_{_gid}"):
                                _ep.clear_group_clusters(_eq_id, _gid)
                                clear_detection_cache_fn()
                                st.rerun()
            else:
                st.caption(f"`{_eq_id}` — 프로파일 없음 (분석 실행 시 그룹별 자동 탐색)")

        # ── 제외 신호 관리 ────────────────────────────────────────────────────
        _excl, _ovrd, _ = load_signal_config()
        if _excl or _ovrd:
            st.divider()
            st.subheader("신호 설정")
            if _excl:
                st.caption(f"제외 중인 신호: {len(_excl)}개")
                for sig in _excl:
                    col_a, col_b = st.columns([3, 1])
                    col_a.caption(sig)
                    if col_b.button("복원", key=f"restore_{sig}"):
                        _excl.remove(sig)
                        save_signal_config(_excl, _ovrd)
                        clear_detection_cache_fn()
                        st.rerun()
            if _ovrd:
                st.caption(f"임계값 조정 중인 신호: {len(_ovrd)}개")
                for sig, thr in list(_ovrd.items()):
                    col_a, col_b = st.columns([3, 1])
                    col_a.caption(f"{sig}: {thr:.1f}")
                    if col_b.button("초기화", key=f"reset_{sig}"):
                        del _ovrd[sig]
                        save_signal_config(_excl, _ovrd)
                        clear_detection_cache_fn()
                        st.rerun()

    return csv_input, top_n, run_btn
