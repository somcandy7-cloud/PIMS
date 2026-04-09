from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable

import streamlit as st
import yaml

from src.services.config_service import (
    SETTINGS_YAML,
    load_signal_config,
    save_hitl_top_n,
    save_llm_config,
    save_signal_config,
)


def _clamp_top_n(value: int) -> int:
    return max(0, min(20, int(value)))


def render_sidebar(
    root: Path,
    clear_detection_cache_fn: Callable[[], None],
) -> tuple[str, int, bool]:
    """사이드바를 렌더링하고 (csv_path, top_n, run_btn)을 반환한다."""
    with st.sidebar:
        st.title("PIMS 분석")

        uploaded = st.file_uploader(
            "데이터 파일 업로드 (.csv / .xlsx)",
            type=["csv", "xlsx"],
            key="uploaded_data_file",
            help="Railway에서는 로컬 경로 대신 업로드를 사용하세요.",
        )

        default_local = str(root / "반출데이터" / "2603201549_oven.csv")
        use_default = (not os.getenv("RAILWAY_ENVIRONMENT")) and Path(default_local).exists()
        default_input = default_local if use_default else ""
        if "csv_path_input" not in st.session_state:
            st.session_state["csv_path_input"] = default_input

        st.text_input(
            "데이터 파일 경로 (.csv / .xlsx)",
            key="csv_path_input",
            help="로컬 실행 시 절대경로를 직접 입력해도 됩니다.",
        )

        if uploaded is not None:
            upload_dir = root / ".runtime_uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            file_bytes = bytes(uploaded.getbuffer())
            digest = hashlib.sha1(file_bytes).hexdigest()[:10]
            src_name = Path(uploaded.name)
            upload_name = f"{src_name.stem}_{digest}{src_name.suffix.lower()}"
            upload_path = upload_dir / upload_name
            if not upload_path.exists():
                upload_path.write_bytes(file_bytes)
            st.session_state["csv_path_input"] = str(upload_path)
            st.caption(f"업로드 파일 사용 중: `{uploaded.name}` → `{upload_name}`")

        csv_input = str(st.session_state.get("csv_path_input", "")).strip()

        with open(SETTINGS_YAML, encoding="utf-8") as f:
            settings_now = yaml.safe_load(f) or {}

        st.divider()
        st.subheader("탐지 설정")
        hitl_now = settings_now.get("hitl", {}) or {}
        detector_now = hitl_now.get("detector", {}) or {}
        saved_top_n = _clamp_top_n(detector_now.get("top_n", 5))
        top_n = st.slider("상위 신호 개수", 0, 20, saved_top_n, key="top_n_slider")

        st.divider()
        st.subheader("LLM 설정")
        llm_now = settings_now.get("llm", {}) or {}

        backend_options = ["OpenAI", "Ollama", "비활성화"]
        backend_map = {"openai": 0, "ollama": 1, "disabled": 2}
        current_backend = (llm_now.get("backend") or "disabled").lower()
        selected_backend = st.radio(
            "백엔드",
            backend_options,
            index=backend_map.get(current_backend, 2),
            key="llm_backend_radio",
        )

        new_llm_cfg = dict(llm_now)
        if selected_backend == "OpenAI":
            new_llm_cfg["backend"] = "openai"
            oai = llm_now.get("openai") or {}
            models = ["gpt-4o", "gpt-4o-mini", "o3-mini"]
            current_model = oai.get("model", "gpt-4o-mini")
            model_index = models.index(current_model) if current_model in models else 1
            selected_model = st.selectbox("모델", models, index=model_index, key="oai_model")
            new_llm_cfg["openai"] = {
                **oai,
                "model": selected_model,
                "api_key_env": oai.get("api_key_env", "OPENAI_API_KEY"),
                "timeout": int(oai.get("timeout", 60)),
                "max_tokens": int(oai.get("max_tokens", 256)),
            }
            st.caption("Railway Variables에 OPENAI_API_KEY를 설정하세요.")
        elif selected_backend == "Ollama":
            new_llm_cfg["backend"] = "ollama"
            olla = llm_now.get("ollama") or {}
            ollama_model = st.text_input("모델", value=olla.get("model", "qwen3:4b"), key="olla_model")
            ollama_host = st.text_input("호스트", value=olla.get("host", "http://localhost:11434"), key="olla_host")
            new_llm_cfg["ollama"] = {
                **olla,
                "model": ollama_model,
                "host": ollama_host,
                "timeout": int(olla.get("timeout", 30)),
            }
        else:
            new_llm_cfg["backend"] = "disabled"

        if st.button("설정 저장", key="save_all_settings_btn", use_container_width=True):
            save_llm_config(new_llm_cfg)
            save_hitl_top_n(top_n)
            st.cache_data.clear()
            st.success("설정이 저장되었습니다.")
            st.rerun()

        st.divider()
        run_btn = st.button("분석 시작하기", type="primary", use_container_width=True)

        st.divider()
        st.subheader("장치 프로필")
        csv_val = csv_input.strip()
        if csv_val:
            from src.services.equipment_profile_store import EquipmentProfileStore

            eq_id = EquipmentProfileStore.extract_id(Path(csv_val).name)
            ep_store = EquipmentProfileStore()
            all_groups = ep_store.load_all_groups(eq_id)

            if all_groups:
                st.caption(f"`{eq_id}` · 그룹 {len(all_groups)}개")
                for gid, gdata in all_groups.items():
                    gconds = gdata.get("conditions") or []
                    gconf = bool(gdata.get("confirmed", False))
                    gclusters = gdata.get("clusters")
                    status = "확정" if gconf else "미확정"
                    with st.expander(f"{gid} · 조건 {len(gconds)}개 [{status}]"):
                        for cond in gconds:
                            st.caption(
                                f"`{cond.get('column', '?')}` {cond.get('op', '?')} "
                                f"{cond.get('value', 0)} [{cond.get('source', '?')}]"
                            )
                        if gclusters:
                            st.caption(f"클러스터 대표 신호: {len(gclusters)}개")
                            if st.button("클러스터 초기화", key=f"clr_cluster_{gid}"):
                                ep_store.clear_group_clusters(eq_id, gid)
                                clear_detection_cache_fn()
                                st.rerun()
            else:
                st.caption(f"`{eq_id}` 프로필 없음 (분석 시 자동 생성)")

        excl, ovrd, _ = load_signal_config()
        if excl or ovrd:
            st.divider()
            st.subheader("분석 제외/오버라이드")

            if excl:
                st.caption(f"제외 신호: {len(excl)}개")
                for sig in list(excl):
                    col_a, col_b = st.columns([3, 1])
                    col_a.caption(sig)
                    if col_b.button("복원", key=f"restore_{sig}"):
                        excl.remove(sig)
                        save_signal_config(excl, ovrd)
                        clear_detection_cache_fn()
                        st.rerun()

            if ovrd:
                st.caption(f"임계값 변경: {len(ovrd)}개")
                for sig, thr in list(ovrd.items()):
                    col_a, col_b = st.columns([3, 1])
                    col_a.caption(f"{sig}: {thr:.1f}")
                    if col_b.button("초기화", key=f"reset_{sig}"):
                        del ovrd[sig]
                        save_signal_config(excl, ovrd)
                        clear_detection_cache_fn()
                        st.rerun()

    return csv_input, top_n, run_btn
