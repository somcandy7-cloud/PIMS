# src/agents/isolation_forest_adapter.py
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.agents.base_detector import AnomalyDetector, AnomalyEvent


class IsolationForestAdapter(AnomalyDetector):
    """윈도우드 Isolation Forest를 AnomalyDetector 인터페이스로 감싼다.

    각 윈도우(window_size 샘플)를 flat vector로 만들어 IF에 입력한다.
    anomaly(-1)로 분류된 윈도우 끝 시점을 AnomalyEvent로 반환한다.
    score = IF anomaly score (낮을수록 이상함, 음수).
    """

    def __init__(
        self,
        window_size: int = 5,
        contamination: float = 0.03,
        n_estimators: int = 100,
        excluded_signals: list[str] | None = None,
        top_n: int = 5,
    ):
        self.window_size = window_size
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.excluded_signals: set[str] = set(excluded_signals or [])
        self.top_n = top_n

    def detect(self, df: pd.DataFrame) -> list[AnomalyEvent]:
        analog_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c not in self.excluded_signals
        ]
        if not analog_cols or len(df) < self.window_size:
            return []

        matrix = df[analog_cols].values  # (N, M)
        N, M = matrix.shape

        # 윈도우드 특징 행렬: (N - W + 1, M * W)
        windowed = np.array([
            matrix[i: i + self.window_size].flatten()
            for i in range(N - self.window_size + 1)
        ])

        iso = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=42,
            n_jobs=1,
        )
        preds = iso.fit_predict(windowed)    # 1: normal, -1: anomaly
        scores = iso.score_samples(windowed) # 낮을수록 이상

        target_indices = list(range(self.window_size - 1, N))

        events: list[AnomalyEvent] = []
        for i, (pred, score) in enumerate(zip(preds, scores)):
            if pred != -1:
                continue

            df_idx = target_indices[i]
            ts = df.index[df_idx]

            # 윈도우 내 컬럼별 표준편차 → top_n 기여 신호
            contrib_window = max(self.window_size, 5)
            start_idx = max(0, df_idx - contrib_window // 2)
            end_idx = min(N, start_idx + contrib_window)
            start_idx = max(0, end_idx - contrib_window)
            window_slice = matrix[start_idx:end_idx]  # (W, M)
            col_stds = window_slice.std(axis=0)
            top_idx = np.argsort(col_stds)[::-1][: self.top_n]
            top_signals = [(analog_cols[j], float(col_stds[j])) for j in top_idx]

            events.append(
                AnomalyEvent(
                    timestamp=ts,
                    score=float(score),
                    top_signals=top_signals,
                    label="if_candidate",
                    metadata={
                        "window_size": self.window_size,
                        "contamination": self.contamination,
                    },
                )
            )
        return events

    def name(self) -> str:
        return "isolation_forest"
