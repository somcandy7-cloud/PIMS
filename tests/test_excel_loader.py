# tests/test_excel_loader.py
import io
import pandas as pd
import numpy as np
import pytest
import openpyxl
from pathlib import Path
from src.services.excel_loader import ExcelLoader


def _make_excel_bytes(sheet_name: str = "Sheet1") -> bytes:
    """테스트용 Excel 파일 바이트를 메모리에 생성한다."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name
    # 헤더
    ws.append(["timestamp", "speed", "current", "label_col"])
    # 데이터 10행
    base = pd.Timestamp("2026-01-01 00:00:00")
    for i in range(10):
        ts = (base + pd.Timedelta(seconds=i)).to_pydatetime()
        ws.append([ts, float(i * 2), float(100 - i), "ok"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


@pytest.fixture
def excel_file(tmp_path: Path) -> Path:
    p = tmp_path / "test.xlsx"
    p.write_bytes(_make_excel_bytes())
    return p


def test_loads_dataframe(excel_file):
    df = ExcelLoader().load(str(excel_file))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10


def test_index_is_datetime(excel_file):
    df = ExcelLoader().load(str(excel_file))
    assert isinstance(df.index, pd.DatetimeIndex)


def test_numeric_columns_only(excel_file):
    df = ExcelLoader().load(str(excel_file))
    # label_col(str) 은 제거되어야 한다
    assert "label_col" not in df.columns
    assert "speed" in df.columns


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        ExcelLoader().load("nonexistent.xlsx")
