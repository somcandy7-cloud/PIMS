# evaluate_detection.py
# 고정 평가 스크립트 — 이 파일 자체는 에이전트가 수정하지 않는다.
# 에이전트가 수정하는 것은 config/signals.yaml 뿐이다.
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data.loader import IbaCSVLoader
from src.analysis.preprocessor import Preprocessor
from src.analysis.isolation_forest_adapter import IsolationForestAdapter
from src.analysis.llm_filter import LLMFilter
from src.analysis.llm_backends import build_llm_backend

# ── 고정 상수 (이 섹션은 사람만 수정할 수 있다) ────────────────────────────────
EVAL_CSV_DIR = ROOT / "experiments" / "eval_data"

# 키: CSV 파일명, 값: 알려진 이상 발생 시각 목록 (ISO 8601, TZ 포함)
KNOWN_EVENTS: dict[str, list[str]] = {
    # 레이블 추가 예시:
    # "2603201549_oven.csv": ["2026-03-20 15:52:00+09:00"],
}
TOLERANCE_SECONDS = 30
# ──────────────────────────────────────────────────────────────────────────────


def _load_config() -> tuple[dict, dict]:
    with open(ROOT / "config" / "signals.yaml", encoding="utf-8") as f:
        signals = yaml.safe_load(f)
    with open(ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    return signals, settings


def _validate_signals(signals: dict) -> None:
    """signal_overrides 임계값이 허용 범위(1.5~5.0) 내인지 확인한다."""
    for sig, thr in (signals.get("signal_overrides") or {}).items():
        if not (1.5 <= float(thr) <= 5.0):
            raise ValueError(
                f"signal_overrides['{sig}'] = {thr} 는 허용 범위(1.5~5.0) 밖입니다. "
                "git reset --hard 후 재시도하세요."
            )
    # NEW: validate isolation_forest params
    if_params = (signals.get("model_params") or {}).get("isolation_forest") or {}
    contamination = if_params.get("contamination", 0.03)
    if not (0.001 <= float(contamination) <= 0.5):
        raise ValueError(
            f"model_params.isolation_forest.contamination = {contamination} "
            "는 허용 범위(0.001~0.5) 밖입니다."
        )
    window_size = if_params.get("window_size", 5)
    if not (1 <= int(window_size) <= 50):
        raise ValueError(
            f"model_params.isolation_forest.window_size = {window_size} "
            "는 허용 범위(1~50) 밖입니다."
        )


def build_detector(signals: dict, settings: dict) -> "IsolationForestAdapter":
    """signals.yaml + settings.yaml로 IF 탐지기를 생성한다. 외부 모델 이식 시 이 함수만 교체."""
    _validate_signals(signals)
    if_params = (signals.get("model_params") or {}).get("isolation_forest") or {}
    excluded = signals.get("excluded_signals") or []
    ana = settings.get("analysis", {})
    top_n = ana.get("top_n_signals", 5)
    return IsolationForestAdapter(
        window_size=int(if_params.get("window_size", 5)),
        contamination=float(if_params.get("contamination", 0.03)),
        n_estimators=int(if_params.get("n_estimators", 100)),
        excluded_signals=excluded,
        top_n=top_n,
    )


def build_llm_filter(settings: dict) -> "LLMFilter":
    """settings.yaml llm 섹션으로 LLMFilter를 생성한다."""
    llm_cfg = settings.get("llm", {})
    backend = build_llm_backend(llm_cfg)
    return LLMFilter(backend=backend)


def evaluate(loader=None, use_llm: bool = False) -> dict:
    """
    고정 CSV 세트로 탐지 성능을 측정한다.

    Parameters
    ----------
    loader : optional
        load(path: str) -> pd.DataFrame 메서드를 가진 객체.
        None이면 IbaCSVLoader()를 사용한다.
    use_llm : bool
        True이면 LLM 필터를 적용한다 (느림). False이면 IF 단독 평가.

    반환값:
        alarm_count, candidate_count, hit_count, miss_count, false_positive,
        hit_rate, fp_rate, f1
    """
    signals, settings = _load_config()
    detector = build_detector(signals, settings)
    llm_filter = build_llm_filter(settings) if use_llm else None
    if loader is None:
        loader = IbaCSVLoader()
    preprocessor = Preprocessor()

    total_alarms = 0
    total_candidates = 0
    hits = 0
    total_known = sum(len(v) for v in KNOWN_EVENTS.values())

    for csv_file, known_ts_list in KNOWN_EVENTS.items():
        csv_path = EVAL_CSV_DIR / csv_file
        if not csv_path.exists():
            print(f"[WARN] 평가 CSV 없음: {csv_path}", file=sys.stderr)
            continue

        df = preprocessor.process(loader.load(str(csv_path)))
        candidates = detector.detect(df)
        total_candidates += len(candidates)

        if llm_filter is not None:
            events = llm_filter.filter(candidates, df)
        else:
            events = candidates
        total_alarms += len(events)

        detected_times = [e.timestamp for e in events]
        for known_ts_str in known_ts_list:
            kt = pd.Timestamp(known_ts_str)
            matched = any(
                abs((dt - kt).total_seconds()) <= TOLERANCE_SECONDS
                for dt in detected_times
            )
            if matched:
                hits += 1

    misses = total_known - hits
    false_positives = max(0, total_alarms - hits)
    hit_rate = hits / total_known if total_known > 0 else 0.0
    fp_rate = false_positives / max(total_alarms, 1)
    precision = hits / max(total_alarms, 1)
    recall = hits / max(total_known, 1)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "alarm_count": total_alarms,
        "candidate_count": total_candidates,
        "hit_count": hits,
        "miss_count": misses,
        "false_positive": false_positives,
        "hit_rate": round(hit_rate, 4),
        "fp_rate": round(fp_rate, 4),
        "f1": round(f1, 4),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PIMS 탐지 성능 평가")
    parser.add_argument("--llm", action="store_true", help="LLM 필터 적용 (느림)")
    args = parser.parse_args()

    result = evaluate(use_llm=args.llm)
    print("---")
    print(f"f1:             {result['f1']:.4f}")
    print(f"hit_rate:       {result['hit_rate']:.4f}")
    print(f"fp_rate:        {result['fp_rate']:.4f}")
    print(f"alarm_count:    {result['alarm_count']}")
    print(f"candidate_count:{result['candidate_count']}")
    print(f"hit_count:      {result['hit_count']}")
    print(f"miss_count:     {result['miss_count']}")
    print(f"false_positive: {result['false_positive']}")
