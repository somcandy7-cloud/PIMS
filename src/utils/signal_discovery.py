from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SignalGroup:
    name: str
    is_binary: bool
    unique_count: int
    mean: float
    std: float


def filter_by_patterns(df: pd.DataFrame, patterns: list[str]) -> list[str]:
    """패턴 문자열 중 하나라도 포함된 컬럼명 목록 반환."""
    return [c for c in df.columns if any(p in c for p in patterns)]


def classify_numeric_columns(df: pd.DataFrame) -> list[SignalGroup]:
    """수치형 컬럼을 analog/digital로 자동 분류."""
    groups: list[SignalGroup] = []
    for col in df.select_dtypes(include="number").columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        unique_count = int(series.nunique())
        is_binary = unique_count <= 2 or (
            float(series.min()) >= 0.0 and float(series.max()) <= 1.0
        )
        groups.append(
            SignalGroup(
                name=col,
                is_binary=bool(is_binary),
                unique_count=unique_count,
                mean=float(series.mean()),
                std=float(series.std()),
            )
        )
    return groups
