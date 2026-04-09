from __future__ import annotations

import pandas as pd
import numpy as np

ClusterMap = dict[str, list[str]]  # {대표컬럼: [전체멤버]}


class SignalReducer:
    """
    데이터프레임에서 불필요한 신호를 제거하고 상관 클러스터를 대표 신호 하나로 축약한다.

    Parameters
    ----------
    corr_threshold : float
        절댓값 피어슨 상관계수가 이 값 이상이면 같은 클러스터로 묶는다.
    low_var_threshold : float
        컬럼의 std가 전체 컬럼 std 중앙값의 이 비율 미만이면 저분산으로 제거한다.
    """

    def __init__(self, corr_threshold: float = 0.90, low_var_threshold: float = 0.01) -> None:
        self.corr_threshold = corr_threshold
        self.low_var_threshold = low_var_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ClusterMap]:
        """
        처리 순서:
        1. numeric 컬럼만 선택
        2. std == 0 제거 (zero-variance)
        3. std < median_std * low_var_threshold 제거 (저분산)
        4. 0/1 이진 컬럼 분리 (digital_cols) → 상관 분석 제외
        5. 나머지 analog_cols에 대해 corr().abs() 계산
        6. Union-Find로 |r| >= corr_threshold 쌍 클러스터링
        7. 클러스터별 대표 = var() 최대 컬럼
        8. 반환: df[representatives + digital_cols], cluster_map
           - cluster_map에 digital_cols도 단독 항목으로 포함
        """
        if df.empty:
            return df, {}

        # 1. numeric 컬럼만
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            return pd.DataFrame(index=df.index), {}

        # 2. zero-variance 제거 (std == 0)
        non_const = numeric.loc[:, numeric.std() > 0]
        if non_const.empty:
            return pd.DataFrame(index=df.index), {}

        # 3. 저분산 제거: std < median_std * low_var_threshold
        col_stds = non_const.std()
        median_std = col_stds.median()
        if median_std > 0:
            low_var_mask = col_stds >= median_std * self.low_var_threshold
        else:
            low_var_mask = col_stds > 0
        base = non_const.loc[:, low_var_mask]

        if base.empty:
            return pd.DataFrame(index=df.index), {}

        # 4. 0/1 이진 컬럼 분리
        digital_cols: list[str] = []
        analog_cols: list[str] = []
        for col in base.columns:
            unique_vals = set(base[col].dropna().unique())
            if unique_vals.issubset({0.0, 1.0}):
                digital_cols.append(col)
            else:
                analog_cols.append(col)

        # 5. analog 컬럼 상관행렬
        cluster_map: ClusterMap = {}
        representatives: list[str] = []

        if analog_cols:
            analog_df = base[analog_cols]
            corr_matrix = analog_df.corr().abs()

            # 6. Union-Find 클러스터링
            clusters = self._union_find_clusters(analog_cols, corr_matrix)

            # 7. 클러스터별 대표 = var() 최대
            for cluster in clusters:
                variances = analog_df[cluster].var()
                rep = str(variances.idxmax())
                cluster_map[rep] = sorted(cluster)
                representatives.append(rep)

        # digital 컬럼도 cluster_map에 단독 항목으로 추가
        for col in digital_cols:
            cluster_map[col] = [col]

        # 8. 결과 데이터프레임 조합
        output_cols = representatives + digital_cols
        result_df = df[output_cols]
        return result_df, cluster_map

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _union_find_clusters(
        self, cols: list[str], corr_matrix: pd.DataFrame
    ) -> list[list[str]]:
        """path compression union-find로 클러스터 목록 반환."""
        parent: dict[str, str] = {c: c for c in cols}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression (halving)
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        n = len(cols)
        for i in range(n):
            for j in range(i + 1, n):
                ci, cj = cols[i], cols[j]
                if corr_matrix.loc[ci, cj] >= self.corr_threshold:
                    union(ci, cj)

        # 루트별로 멤버 수집
        from collections import defaultdict
        groups: dict[str, list[str]] = defaultdict(list)
        for c in cols:
            groups[find(c)].append(c)

        return list(groups.values())

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, hitl_cfg: dict) -> "SignalReducer":
        """hitl.signal_reducer 섹션에서 생성."""
        sr_cfg = hitl_cfg.get("signal_reducer", {})
        return cls(
            corr_threshold=sr_cfg.get("corr_threshold", 0.90),
            low_var_threshold=sr_cfg.get("low_var_threshold", 0.01),
        )
