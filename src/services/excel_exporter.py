"""iba CSV → Excel 변환 서비스."""
from __future__ import annotations
from pathlib import Path

import pandas as pd

KST = "Asia/Seoul"


def csv_to_excel(
    csv_path: str,
    output_path: str | None = None,
    label_mapper=None,
) -> str:
    """iba CSV 파일을 파싱해 Excel(.xlsx)로 저장한다.

    Args:
        csv_path: iba PDA CSV 파일 경로
        output_path: 저장할 .xlsx 경로. None이면 CSV와 같은 폴더에 같은 이름으로 저장.
        label_mapper: SignalLabelMapper 인스턴스. 있으면 '변수명 (태그)' 형식 컬럼명 적용.

    Returns:
        저장된 xlsx 파일의 절대 경로 문자열.
    """
    from src.services.loader import IbaCSVLoader
    from src.utils.preprocessor import Preprocessor

    csv_path = Path(csv_path)
    if output_path is None:
        output_path = csv_path.with_suffix(".xlsx")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 로드 + 전처리
    raw_df = IbaCSVLoader().load(str(csv_path))
    original_tags = raw_df.columns.tolist()

    df = Preprocessor().process(raw_df)

    # 2. UTC → KST 변환
    if df.index.tz is not None:
        df.index = df.index.tz_convert(KST)
    else:
        df.index = df.index.tz_localize(KST)
    df.index.name = "timestamp_KST"
    df.index = df.index.tz_localize(None)  # Excel은 tz-aware datetime 미지원

    # 3. 컬럼명 변환 (라벨 매퍼가 있을 때)
    display_cols = {}
    for tag in original_tags:
        if label_mapper is not None:
            var = label_mapper.label(tag, fallback=None)
            display_cols[tag] = f"{var} ({tag})" if var and var != tag else tag
        else:
            display_cols[tag] = tag
    df = df.rename(columns=display_cols)

    # 4. Excel 저장
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # 시트 1: 파싱된 시계열 데이터
        df.to_excel(writer, sheet_name="데이터")

        # 시트 2: 신호 목록 (태그주소 ↔ 변수명 ↔ 설명)
        if label_mapper is not None:
            meta_rows = [
                {
                    "태그주소": tag,
                    "변수명": label_mapper.label(tag, fallback=""),
                    "설명": label_mapper.describe(tag),
                }
                for tag in original_tags
            ]
        else:
            meta_rows = [{"태그주소": tag} for tag in original_tags]

        pd.DataFrame(meta_rows).to_excel(writer, sheet_name="신호목록", index=False)

    return str(output_path.resolve())
