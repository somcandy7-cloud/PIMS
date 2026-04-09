# src/agents/base_detector.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class AnomalyEvent:
    """이상치 탐지 이벤트 공통 스키마."""
    timestamp: pd.Timestamp
    score: float                               # 이상 강도 (모델별 정규화)
    top_signals: list[tuple[str, float]]       # (신호명, 기여도)
    label: str = "anomaly"                     # 모델이 붙이는 레이블
    metadata: dict = field(default_factory=dict)


class AnomalyDetector(ABC):
    """이상치 탐지기 프로토콜.

    외부 모델을 이식할 때 이 ABC를 상속하여 detect()와 name()만 구현한다.
    """

    @abstractmethod
    def detect(self, df: pd.DataFrame) -> list[AnomalyEvent]:
        """전처리된 DataFrame을 받아 이상치 이벤트 목록을 반환한다."""

    @abstractmethod
    def name(self) -> str:
        """모델 식별자 (대시보드/로그에 표시)."""
