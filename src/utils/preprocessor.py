import pandas as pd


class Preprocessor:
    """iba CSV 로드 후 분석 전 전처리.

    - 비수치형 컬럼 제거
    - 결측치 전방 채우기 → 후방 채우기 → 0 채우기
    - float64 강제 변환
    """

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        # 수치형 컬럼만 유지
        df = df.select_dtypes(include="number").astype(float)

        # 결측치 처리: ffill → bfill → 0
        df = df.ffill().bfill().fillna(0.0)

        return df
