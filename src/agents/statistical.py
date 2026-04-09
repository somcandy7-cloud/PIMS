from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AnomalyResult:
    timestamp: pd.Timestamp
    column: str
    value: float
    zscore: float


class ZScoreDetector:
    """컬럼별 z-score로 정적 이상치를 탐지한다."""

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold

    def detect(self, df: pd.DataFrame) -> list[AnomalyResult]:
        results: list[AnomalyResult] = []
        for col in df.select_dtypes(include="number").columns:
            series = df[col].dropna()
            if len(series) < 2:
                continue
            mean = series.mean()
            std = series.std()
            if std == 0:
                continue
            zscores = (series - mean) / std
            flagged = zscores[zscores.abs() > self.threshold]
            for ts, z in flagged.items():
                results.append(
                    AnomalyResult(
                        timestamp=ts,
                        column=col,
                        value=float(series.loc[ts]),
                        zscore=float(z),
                    )
                )
        return results
