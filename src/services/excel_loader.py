# src/services/excel_loader.py
from __future__ import annotations
from pathlib import Path

import pandas as pd


class ExcelLoader:
    """일반 Excel 시계열 파일을 로드한다.

    규칙:
    - 첫 번째 열이 datetime (자동 파싱)
    - 나머지 수치형 컬럼만 유지 (비수치형 제거)
    - 인덱스 = DatetimeIndex

    외부 모델 이식 전 데이터 탐색 또는 레이블 데이터 로드에 사용한다.
    """

    def load(self, filepath: str, sheet_name: int | str = 0) -> pd.DataFrame:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(filepath)

        df = pd.read_excel(path, sheet_name=sheet_name, header=0, engine="openpyxl")

        if df.empty:
            return df

        # 첫 번째 컬럼을 인덱스로 (datetime 파싱)
        time_col = df.columns[0]
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.set_index(time_col)
        df.index.name = "timestamp"

        # 수치형 컬럼만 유지
        df = df.select_dtypes(include="number").astype(float)

        return df
