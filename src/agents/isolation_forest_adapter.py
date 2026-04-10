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

    @staticmethod
    def _ablation_importance(
        iso: IsolationForest,
        windowed: np.ndarray,
        anomaly_indices: list[int],
        window_size: int,
        n_orig_cols: int,
    ) -> np.ndarray:
        """Ablation 방식으로 feature별 IF 점수 기여도를 계산한다.

        각 feature를 학습 데이터 평균값으로 대체했을 때 anomaly score 변화량을
        측정한다.  양수(+) = 해당 feature가 이상치 점수를 끌어내리는 원인.

        Returns
        -------
        np.ndarray of shape (n_anomalies, n_orig_cols)
            원본 컬럼 단위로 집계된 기여도 행렬 (정규화 전).
        """
        n_flat = windowed.shape[1]
        train_means = windowed.mean(axis=0)  # (n_flat,)

        anomaly_samples = windowed[anomaly_indices]          # (A, n_flat)
        orig_scores = iso.score_samples(anomaly_samples)     # (A,)

        # 각 feature dimension을 학습 평균으로 교체한 배치를 한꺼번에 계산
        # batch shape: (A * n_flat, n_flat)
        batch = np.repeat(anomaly_samples, n_flat, axis=0)   # (A*n_flat, n_flat)
        for j in range(n_flat):
            rows = np.arange(len(anomaly_indices)) * n_flat + j
            batch[rows, j] = train_means[j]

        perturbed_scores = iso.score_samples(batch)          # (A*n_flat,)
        perturbed_scores = perturbed_scores.reshape(len(anomaly_indices), n_flat)

        # contribution: 제거 시 점수 상승량 (양수 = 이상 원인)
        contrib = perturbed_scores - orig_scores[:, None]    # (A, n_flat)

        # windowed flatten (W*M) → 원본 컬럼 (M) 집계
        col_contrib = np.zeros((len(anomaly_indices), n_orig_cols))
        for k in range(window_size):
            offset = k * n_orig_cols
            col_contrib += contrib[:, offset: offset + n_orig_cols]

        return col_contrib

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
        anomaly_win_indices = [i for i, p in enumerate(preds) if p == -1]

        if not anomaly_win_indices:
            return []

        # 모든 이상치에 대해 ablation 기여도 일괄 계산
        col_contribs = self._ablation_importance(
            iso, windowed, anomaly_win_indices, self.window_size, M
        )  # (A, M)

        events: list[AnomalyEvent] = []
        for rank, i in enumerate(anomaly_win_indices):
            df_idx = target_indices[i]
            ts = df.index[df_idx]
            score = float(scores[i])

            col_imp = np.maximum(col_contribs[rank], 0)  # 음수(정상 기여) 제거
            imp_sum = col_imp.sum()
            if imp_sum > 0:
                col_imp /= imp_sum

            top_idx = np.argsort(col_imp)[::-1][: self.top_n]
            top_signals = [(analog_cols[j], float(col_imp[j])) for j in top_idx]

            events.append(
                AnomalyEvent(
                    timestamp=ts,
                    score=score,
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
