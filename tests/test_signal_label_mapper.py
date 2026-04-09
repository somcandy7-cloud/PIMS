import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.services.signal_label_mapper import SignalLabelMapper, _normalize_tag


# ── _normalize_tag 단위 테스트 ────────────────────────────────────────────────

def test_normalize_removes_prefix():
    assert _normalize_tag("[3_CO_PLC_A]DB420.DBW 2") == "DB420.DBW 2"

def test_normalize_collapses_spaces():
    assert _normalize_tag("DB421.DBX    2.0") == "DB421.DBX 2.0"

def test_normalize_strips_leading_trailing():
    assert _normalize_tag("  DB420.DBW 0  ") == "DB420.DBW 0"

def test_normalize_no_space_variant():
    # G-5 fix: DB420.DBW0 → DB420.DBW 0 으로 정규화되어야 한다
    assert _normalize_tag("DB420.DBW0") == "DB420.DBW 0"

def test_normalize_empty():
    assert _normalize_tag("") == ""


# ── SignalLabelMapper 단위 테스트 ─────────────────────────────────────────────

def _make_mapper_with_rows(rows: list[tuple]) -> SignalLabelMapper:
    """openpyxl을 mock해서 SignalLabelMapper를 생성하는 헬퍼."""
    mock_ws = MagicMock()
    mock_ws.iter_rows.return_value = rows

    mock_wb = MagicMock()
    mock_wb.sheetnames = ["Sheet1"]
    mock_wb.__getitem__ = lambda self, k: mock_ws
    mock_wb.close = MagicMock()

    with patch("src.data.signal_label_mapper.openpyxl.load_workbook", return_value=mock_wb), \
         patch("pathlib.Path.exists", return_value=True):
        mapper = SignalLabelMapper(["fake.xlsx"])
    return mapper


def test_lookup_found():
    mapper = _make_mapper_with_rows([
        ("DB420.DBW 2", "FR_HMI_W2", "INT", "STAND PIPE OPEN COUNT SET VALUE"),
    ])
    result = mapper.lookup("DB420.DBW 2")
    assert result == ("FR_HMI_W2", "STAND PIPE OPEN COUNT SET VALUE")


def test_lookup_with_prefix():
    mapper = _make_mapper_with_rows([
        ("DB420.DBW 2", "FR_HMI_W2", "INT", "DESC"),
    ])
    result = mapper.lookup("[3_CO_PLC_A]DB420.DBW 2")
    assert result == ("FR_HMI_W2", "DESC")


def test_lookup_not_found_returns_none():
    mapper = _make_mapper_with_rows([
        ("DB420.DBW 2", "FR_HMI_W2", "INT", "DESC"),
    ])
    assert mapper.lookup("DB999.DBW 0") is None


def test_label_fallback():
    mapper = _make_mapper_with_rows([])
    assert mapper.label("DB999.DBW 0") == "DB999.DBW 0"
    assert mapper.label("DB999.DBW 0", fallback="UNKNOWN") == "UNKNOWN"


def test_describe_empty_when_not_found():
    mapper = _make_mapper_with_rows([])
    assert mapper.describe("DB999.DBW 0") == ""


def test_size():
    mapper = _make_mapper_with_rows([
        ("DB420.DBW 2", "V1", "INT", "D1"),
        ("DB420.DBW 4", "V2", "INT", "D2"),
    ])
    assert mapper.size == 2


def test_unmapped_tags_recorded():
    mapper = _make_mapper_with_rows([
        ("DB420.DBW 2", "V1", "INT", "D1"),
    ])
    mapper.lookup("DB999.DBW 0")
    mapper.lookup("DB888.DBW 0")
    assert "DB999.DBW 0" in mapper.unmapped_tags
    assert "DB888.DBW 0" in mapper.unmapped_tags


def test_duplicate_tag_first_wins():
    mapper = _make_mapper_with_rows([
        ("DB420.DBW 2", "FIRST", "INT", "first desc"),
        ("DB420.DBW 2", "SECOND", "INT", "second desc"),
    ])
    assert mapper.lookup("DB420.DBW 2")[0] == "FIRST"


def test_xlsx_not_exists_graceful():
    with patch("pathlib.Path.exists", return_value=False):
        mapper = SignalLabelMapper(["nonexistent.xlsx"])
    assert mapper.size == 0
    assert mapper.lookup("DB420.DBW 2") is None


def test_none_tag_row_skipped():
    mapper = _make_mapper_with_rows([
        (None, "V1", "INT", "D1"),
        ("DB420.DBW 2", "V2", "INT", "D2"),
    ])
    assert mapper.size == 1
