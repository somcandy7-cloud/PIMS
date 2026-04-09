# src/agents/zscore_adapter.py
from __future__ import annotations

import pandas as pd

from src.agents.base_detector import AnomalyDetector, AnomalyEvent
from src.agents.trip_detector import TripDetector


class ZScoreAdapter(AnomalyDetector):
    """TripDetector를 AnomalyDetector 프로토콜로 감싸는 어댑터.

    기본 모델. 외부 모델 이식 전까지 이 클래스가 대시보드·평가 함수에 사용된다.
    """

    def __init__(
        self,
        window: int = 50,
        roc_zscore_threshold: float = 3.0,
        top_n: int = 5,
        excluded_signals: list[str] | None = None,
        signal_overrides: dict[str, float] | None = None,
    ):
        self._detector = TripDetector(
            window=window,
            roc_zscore_threshold=roc_zscore_threshold,
            top_n=top_n,
            excluded_signals=excluded_signals,
            signal_overrides=signal_overrides,
        )

    def detect(self, df: pd.DataFrame) -> list[AnomalyEvent]:
        trip_events = self._detector.detect(df)
        return [
            AnomalyEvent(
                timestamp=e.timestamp,
                score=float(len(e.top_changed_signals)),  # 급변 신호 수를 점수로
                top_signals=e.top_changed_signals,
                label=e.trigger,
                metadata={
                    "description": e.description,
                    "window": e.window_size,
                    "trip_event": e,
                },
            )
            for e in trip_events
        ]

    def name(self) -> str:
        return "zscore"
