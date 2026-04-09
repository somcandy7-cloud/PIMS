from __future__ import annotations
import pandas as pd


class RollingFeatureExtractor:
    """아날로그 신호를 rolling 통계(mean/std/roc)로 변환한다.

    window 기본값 30행 ≈ 63초 (2.1sec/row 기준).
    """
    _EPS = 1e-9

    def __init__(self, window: int = 30):
        self.window = window

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out: dict[str, pd.Series] = {}
        for col in df.columns:
            s = df[col]
            rolled = s.rolling(window=self.window, min_periods=1)
            out[f"{col}_rmean"] = rolled.mean()
            out[f"{col}_rstd"]  = rolled.std().fillna(0.0)
            shifted = s.shift(self.window).ffill().bfill()
            out[f"{col}_roc"]   = (s - shifted) / (shifted.abs() + self._EPS)
        return pd.DataFrame(out, index=df.index).ffill().bfill().fillna(0.0)
