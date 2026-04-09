from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


@dataclass
class OperationCondition:
    column: str
    op: str          # '==', '!=', '>', '>=', '<', '<='
    value: float
    confidence: float  # 0.0~1.0
    source: str        # 'binary' or 'bimodal'

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "op": self.op,
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OperationCondition":
        return cls(
            column=d["column"],
            op=d["op"],
            value=float(d["value"]),
            confidence=float(d["confidence"]),
            source=d["source"],
        )


class AutoOperationDiscovery:
    _MIN_SAMPLES: int = 30

    def __init__(
        self,
        binary_on_min: float = 0.10,
        binary_on_max: float = 0.90,
        bimodal_center_max: float = 0.30,
        top_binary_n: int = 3,
        top_bimodal_n: int = 3,
    ) -> None:
        self.binary_on_min = binary_on_min
        self.binary_on_max = binary_on_max
        self.bimodal_center_max = bimodal_center_max
        self.top_binary_n = top_binary_n
        self.top_bimodal_n = top_bimodal_n

    def discover(self, df: pd.DataFrame) -> list[OperationCondition]:
        """이진 후보 + 이중봉 후보를 합쳐 신뢰도 내림차순 반환.
        이진 후보가 이중봉 후보에도 포함되면 중복 제거(이진 우선).
        """
        if df.empty:
            return []

        binary_candidates = self._find_binary(df)
        bimodal_candidates = self._find_bimodal(df)

        binary_cols = {c.column for c in binary_candidates}

        # 중복 제거: 이진 우선
        filtered_bimodal = [c for c in bimodal_candidates if c.column not in binary_cols]

        # top_n 적용
        binary_top = sorted(binary_candidates, key=lambda c: c.confidence, reverse=True)[: self.top_binary_n]
        bimodal_top = sorted(filtered_bimodal, key=lambda c: c.confidence, reverse=True)[: self.top_bimodal_n]

        combined = binary_top + bimodal_top
        combined.sort(key=lambda c: c.confidence, reverse=True)
        return combined

    def _find_binary(self, df: pd.DataFrame) -> list[OperationCondition]:
        """조건:
        - 값이 {0.0, 1.0}만 존재
        - ON(=1.0) 비율이 binary_on_min ~ binary_on_max 사이
        - 최소 30샘플 필요
        신뢰도: 1.0 - abs(on_ratio - 0.5) / 0.5
        """
        results: list[OperationCondition] = []

        for col in df.columns:
            s = df[col].dropna()
            if len(s) < self._MIN_SAMPLES:
                continue
            unique_vals = set(s.unique())
            if not unique_vals.issubset({0.0, 1.0}):
                continue
            if unique_vals != {0.0, 1.0}:
                # 값이 하나만 존재하는 경우 (예: always_0)
                continue
            on_ratio = float((s == 1.0).mean())
            if not (self.binary_on_min <= on_ratio <= self.binary_on_max):
                continue
            confidence = 1.0 - abs(on_ratio - 0.5) / 0.5
            results.append(
                OperationCondition(
                    column=col,
                    op="==",
                    value=1.0,
                    confidence=float(confidence),
                    source="binary",
                )
            )

        return results

    def _find_bimodal(self, df: pd.DataFrame) -> list[OperationCondition]:
        """조건:
        - 수치형
        - range > 0, std/range >= 0.05
        - center_density < bimodal_center_max
        - KMeans(k=2) 분리도 >= 1.0
        신뢰도: (1 - center_density) * min(separation, 1.0)
        """
        results: list[OperationCondition] = []

        for col in df.select_dtypes(include=[np.number]).columns:
            s = df[col].dropna()
            if len(s) < self._MIN_SAMPLES:
                continue

            col_range = float(s.max() - s.min())
            if col_range <= 0:
                continue

            col_std = float(s.std())
            if col_std / col_range < 0.05:
                continue

            within_ratio, threshold, separation = self._analyze_bimodal(s)
            if within_ratio >= self.bimodal_center_max:
                continue

            if separation < 1.0 or threshold is None:
                continue

            confidence = (1.0 - within_ratio) * min(separation, 1.0)
            results.append(
                OperationCondition(
                    column=col,
                    op=">",
                    value=float(threshold),
                    confidence=float(confidence),
                    source="bimodal",
                )
            )

        return results

    @staticmethod
    def _analyze_bimodal(s: pd.Series) -> tuple[float, float | None, float]:
        """
        KMeans(k=2)를 1회 실행해 (within_ratio, threshold, separation)를 반환한다.

        Returns
        -------
        within_ratio : float  — KMeans 기반 중앙 밀집도 지표 (낮을수록 이중봉)
        threshold    : float | None  — 두 클러스터 중간값 (실패 시 None)
        separation   : float  — 클러스터 분리도 (낮으면 0.0)
        """
        try:
            X = s.values.reshape(-1, 1)
            km = KMeans(n_clusters=2, random_state=42, n_init=5)
            km.fit(X)
            labels = km.labels_
            c0, c1 = sorted(km.cluster_centers_.flatten())
            threshold = (c0 + c1) / 2.0
            separation = abs(c1 - c0) / (s.std() + 1e-9)

            # within-cluster std ratio (낮을수록 클러스터가 뭉쳐 있음 = 이중봉에 유리)
            within_stds = [s[labels == i].std() for i in range(2)]
            within_ratio = float(np.mean(within_stds)) / (s.std() + 1e-9)

            return within_ratio, threshold, separation
        except Exception:
            return 1.0, None, 0.0
