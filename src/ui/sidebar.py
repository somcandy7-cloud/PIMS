from __future__ import annotations

import os
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
    """Render sidebar and return (csv_path, top_n, run_btn)."""

    with st.sidebar:
        st.title("PIMS Dashboard")

        default_input_path = "" if os.getenv("RAILWAY_ENVIRONMENT") else str(root / "2603201549_oven.csv")
        csv_input = st.text_input(
            "Data file path (.csv / .xlsx)",
            value=default_input_path,
            help="Use upload on Railway, or local absolute path on your PC.",
        )

        st.divider()
        st.subheader("Detection")
        top_n = st.slider("Top-N signals", 3, 20, 5)

        st.divider()
        st.subheader("LLM")

        with open(SETTINGS_YAML, encoding="utf-8") as f:
            settings_now = yaml.safe_load(f)
        llm_now = settings_now.get("llm", {})

        backend_options = ["OpenAI", "Ollama", "Disabled"]
        backend_map = {"openai": 0, "ollama": 1, "disabled": 2}
        current_backend = (llm_now.get("backend") or "disabled").lower()
        selected_backend = st.radio(
            "Backend",
            backend_options,
            index=backend_map.get(current_backend, 2),
            key="llm_backend_radio",
        )

        new_llm_cfg = dict(llm_now)

        if selected_backend == "OpenAI":
            new_llm_cfg["backend"] = "openai"
            oai = llm_now.get("openai") or {}
            oai_models = ["gpt-4o", "gpt-4o-mini", "o3-mini"]
            current_model = oai.get("model", "gpt-4o-mini")
            model_index = oai_models.index(current_model) if current_model in oai_models else 1
            selected_model = st.selectbox("Model", oai_models, index=model_index, key="oai_model")
            new_llm_cfg["openai"] = {
                **oai,
                "model": selected_model,
                "api_key_env": oai.get("api_key_env", "OPENAI_API_KEY"),
                "timeout": int(oai.get("timeout", 60)),
                "max_tokens": int(oai.get("max_tokens", 256)),
            }
            st.caption("Set OPENAI_API_KEY in Railway Variables.")

        elif selected_backend == "Ollama":
            new_llm_cfg["backend"] = "ollama"
            olla = llm_now.get("ollama") or {}
            ollama_model = st.text_input("Model", value=olla.get("model", "qwen3:4b"), key="olla_model")
            ollama_host = st.text_input("Host", value=olla.get("host", "http://localhost:11434"), key="olla_host")
            new_llm_cfg["ollama"] = {
                **olla,
                "model": ollama_model,
                "host": ollama_host,
                "timeout": int(olla.get("timeout", 30)),
            }

        else:
            new_llm_cfg["backend"] = "disabled"

        if st.button("Save LLM settings", key="save_llm_btn"):
            save_llm_config(new_llm_cfg)
            st.cache_data.clear()
            st.success("Saved")
            st.rerun()

        st.divider()
        run_btn = st.button("Run analysis", type="primary", use_container_width=True)

        excl, ovrd, _ = load_signal_config()
        if excl or ovrd:
            st.divider()
            st.subheader("Signal config")

            if excl:
                st.caption(f"Excluded signals: {len(excl)}")
                for sig in list(excl):
                    col_a, col_b = st.columns([3, 1])
                    col_a.caption(sig)
                    if col_b.button("Restore", key=f"restore_{sig}"):
                        excl.remove(sig)
                        save_signal_config(excl, ovrd)
                        clear_detection_cache_fn()
                        st.rerun()

            if ovrd:
                st.caption(f"Threshold overrides: {len(ovrd)}")
                for sig, thr in list(ovrd.items()):
                    col_a, col_b = st.columns([3, 1])
                    col_a.caption(f"{sig}: {thr:.1f}")
                    if col_b.button("Reset", key=f"reset_{sig}"):
                        del ovrd[sig]
                        save_signal_config(excl, ovrd)
                        clear_detection_cache_fn()
                        st.rerun()

    return csv_input, top_n, run_btn
