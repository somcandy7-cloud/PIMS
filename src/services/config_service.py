"""설정 파일(signals.yaml, settings.yaml) 읽기/쓰기 서비스."""
from __future__ import annotations
from pathlib import Path

import yaml

_ROOT = Path(__file__).parent.parent.parent
SIGNALS_YAML = _ROOT / "config" / "signals.yaml"
SETTINGS_YAML = _ROOT / "config" / "settings.yaml"


def load_signal_config() -> tuple[list[str], dict[str, float], dict]:
    """signals.yaml에서 excluded_signals, signal_overrides, model_params 읽기."""
    with open(SIGNALS_YAML, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    excluded = cfg.get("excluded_signals") or []
    overrides = cfg.get("signal_overrides") or {}
    model_params = cfg.get("model_params") or {}
    return excluded, overrides, model_params


def save_signal_config(excluded: list[str], overrides: dict[str, float]) -> None:
    """excluded_signals, signal_overrides를 signals.yaml에 저장."""
    with open(SIGNALS_YAML, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["excluded_signals"] = excluded
    cfg["signal_overrides"] = overrides
    with open(SIGNALS_YAML, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def save_llm_config(llm_cfg: dict) -> None:
    """settings.yaml의 llm 섹션을 업데이트한다."""
    with open(SETTINGS_YAML, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["llm"] = llm_cfg
    with open(SETTINGS_YAML, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def load_settings() -> dict:
    """settings.yaml 전체를 dict로 반환한다."""
    with open(SETTINGS_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)
