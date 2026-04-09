"""분석 파이프라인 컴포넌트 팩토리 함수."""
from __future__ import annotations
from pathlib import Path


def get_loader(filepath: str):
    """파일 확장자에 따라 적절한 로더를 반환한다."""
    from src.services.loader import IbaCSVLoader
    from src.services.excel_loader import ExcelLoader
    if Path(filepath).suffix.lower() in {".xlsx", ".xls"}:
        return ExcelLoader()
    return IbaCSVLoader()


def build_if_detector(if_params: dict, excluded: list[str], top_n: int):
    """signals.yaml model_params로 IsolationForestAdapter를 생성한다."""
    from src.agents.isolation_forest_adapter import IsolationForestAdapter
    return IsolationForestAdapter(
        window_size=if_params.get("window_size", 5),
        contamination=if_params.get("contamination", 0.03),
        n_estimators=if_params.get("n_estimators", 100),
        excluded_signals=excluded,
        top_n=top_n,
    )


def build_llm_filter(settings: dict):
    """settings.yaml llm 섹션으로 LLMFilter를 생성한다."""
    from src.agents.llm_backends import build_llm_backend
    from src.agents.llm_filter import LLMFilter
    llm_cfg = settings.get("llm", {})
    backend = build_llm_backend(llm_cfg)
    # Large CSV can produce many IF candidates; cap per-run LLM calls to avoid timeouts.
    max_candidates = int(llm_cfg.get("max_candidates_per_run", 10))
    return LLMFilter(
        backend=backend,
        max_candidates_per_run=max_candidates if max_candidates > 0 else None,
    )


def build_signal_reducer(hitl_cfg: dict):
    """hitl.signal_reducer 설정으로 SignalReducer를 생성한다."""
    from src.utils.signal_reducer import SignalReducer
    return SignalReducer.from_config(hitl_cfg)


def build_operation_filter(conditions: list):
    """OperationCondition 목록으로 OperationFilter를 생성한다."""
    from src.utils.operation_filter import OperationFilter
    return OperationFilter(conditions)


def build_rolling_extractor(hitl_cfg: dict):
    """hitl.rolling 설정으로 RollingFeatureExtractor를 생성한다."""
    from src.utils.rolling_features import RollingFeatureExtractor
    window = int(hitl_cfg.get("rolling", {}).get("window_rows", 30))
    return RollingFeatureExtractor(window=window)
