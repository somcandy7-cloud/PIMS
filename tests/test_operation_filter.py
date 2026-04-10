# tests/test_operation_filter.py
import pandas as pd
import numpy as np
import pytest
from src.utils.operation_discovery import OperationCondition
from src.utils.operation_filter import OperationFilter


def _make_df() -> pd.DataFrame:
    t = pd.date_range("2026-03-12", periods=5, freq="2s", tz="UTC")
    return pd.DataFrame({
        "ctrl":  [0.0, 1.0, 1.0, 1.0, 0.0],
        "speed": [0.0, 50_000.0, 150_000.0, 200_000.0, 0.0],
        "temp":  [20.0, 180.0, 200.0, 195.0, 20.0],
    }, index=t)


def _conds_two() -> list[OperationCondition]:
    return [
        OperationCondition("ctrl", "==", 1.0, 0.9, "binary"),
        OperationCondition("speed", ">", 100_000.0, 0.8, "bimodal"),
    ]


def test_and_filter_keeps_correct_rows():
    df = _make_df()
    result = OperationFilter(_conds_two()).filter(df)
    assert len(result) == 2
    assert result["temp"].tolist() == [200.0, 195.0]


def test_three_conditions_and():
    df = _make_df()
    conds = _conds_two() + [
        OperationCondition("temp", ">", 190.0, 0.7, "bimodal"),
    ]
    result = OperationFilter(conds).filter(df)
    # ctrl==1, speed>100000, temp>190: rows idx=2(200°) and idx=3(195°) 모두 통과 → 2행
    assert len(result) == 2


def test_all_stopped_returns_empty():
    t = pd.date_range("2026-03-12", periods=3, freq="2s", tz="UTC")
    df = pd.DataFrame({
        "ctrl":  [0.0, 0.0, 0.0],
        "speed": [0.0, 0.0, 0.0],
    }, index=t)
    result = OperationFilter(_conds_two()).filter(df)
    assert result.empty


def test_missing_column_skipped():
    """조건 컬럼이 없으면 해당 조건은 스킵한다 (CSV 전환 시 동작)."""
    t = pd.date_range("2026-03-12", periods=2, freq="2s", tz="UTC")
    df = pd.DataFrame({"other": [1.0, 2.0]}, index=t)
    # 컬럼이 없으면 조건 스킵 → 전체 반환
    result = OperationFilter(_conds_two()).filter(df)
    assert len(result) == 2


# ── warmup / cooldown 버퍼 테스트 ─────────────────────────────────────────────

def _make_long_df(stopped: int = 10, running: int = 80, freq: str = "1s") -> pd.DataFrame:
    """가동(ctrl=1) 구간이 중간에 있는 DataFrame 생성."""
    n = stopped + running + stopped
    t = pd.date_range("2026-03-12", periods=n, freq=freq, tz="UTC")
    ctrl  = [0.0] * stopped + [1.0] * running + [0.0] * stopped
    speed = [0.0] * stopped + [150_000.0] * running + [0.0] * stopped
    return pd.DataFrame({"ctrl": ctrl, "speed": speed}, index=t)


def test_warmup_trim():
    """warmup_sec 동안 가동 블록 앞부분을 제외한다."""
    df = _make_long_df()
    conds = [OperationCondition("ctrl", "==", 1.0, 0.9, "binary")]
    result_no_buf = OperationFilter(conds).filter(df)
    result_warmup = OperationFilter(conds, warmup_sec=5).filter(df)
    assert len(result_warmup) == len(result_no_buf) - 5


def test_cooldown_trim():
    """cooldown_sec 동안 가동 블록 뒷부분을 제외한다."""
    df = _make_long_df()
    conds = [OperationCondition("ctrl", "==", 1.0, 0.9, "binary")]
    result_no_buf = OperationFilter(conds).filter(df)
    result_cool = OperationFilter(conds, cooldown_sec=5).filter(df)
    assert len(result_cool) == len(result_no_buf) - 5


def test_warmup_and_cooldown_trim():
    """warmup + cooldown 동시 적용."""
    df = _make_long_df()
    conds = [OperationCondition("ctrl", "==", 1.0, 0.9, "binary")]
    result_no_buf = OperationFilter(conds).filter(df)
    result = OperationFilter(conds, warmup_sec=5, cooldown_sec=5).filter(df)
    assert len(result) == len(result_no_buf) - 10


def test_multiple_blocks_each_trimmed():
    """복수 가동 블록이 있을 때 각 블록마다 트리밍 적용."""
    # 블록1: [10..29], 블록2: [40..59] (각 20행, 1초 간격)
    n = 70
    t = pd.date_range("2026-03-12", periods=n, freq="1s", tz="UTC")
    ctrl = [0.0]*10 + [1.0]*20 + [0.0]*10 + [1.0]*20 + [0.0]*10
    df = pd.DataFrame({"ctrl": ctrl}, index=t)
    conds = [OperationCondition("ctrl", "==", 1.0, 0.9, "binary")]

    result_no_buf = OperationFilter(conds).filter(df)          # 40행
    result = OperationFilter(conds, warmup_sec=3, cooldown_sec=3).filter(df)
    # 각 블록에서 앞3 + 뒤3 = 6 제거 × 2블록 = 12 제거
    assert len(result) == len(result_no_buf) - 12


def test_buffer_larger_than_block_returns_empty():
    """버퍼가 블록 전체를 덮으면 빈 DataFrame 반환."""
    df = _make_long_df(stopped=5, running=10)  # ctrl=1 구간 10행
    conds = [OperationCondition("ctrl", "==", 1.0, 0.9, "binary")]
    result = OperationFilter(conds, warmup_sec=6, cooldown_sec=6).filter(df)
    # 10행 중 앞6 + 뒤6 = 12 → 전체 제거
    assert result.empty


def test_zero_buffer_unchanged():
    """버퍼 0이면 기존 동작과 동일."""
    df = _make_long_df()
    conds = [OperationCondition("ctrl", "==", 1.0, 0.9, "binary")]
    result_default = OperationFilter(conds).filter(df)
    result_zero    = OperationFilter(conds, warmup_sec=0, cooldown_sec=0).filter(df)
    assert len(result_zero) == len(result_default)


def test_supported_operators():
    t = pd.date_range("2026-03-12", periods=3, freq="2s", tz="UTC")
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=t)
    ops_expected = [
        (">",  2.0, [3.0]),
        (">=", 2.0, [2.0, 3.0]),
        ("<",  2.0, [1.0]),
        ("<=", 2.0, [1.0, 2.0]),
        ("==", 2.0, [2.0]),
        ("!=", 2.0, [1.0, 3.0]),
    ]
    for op, val, expected in ops_expected:
        cond = OperationCondition("v", op, val, 1.0, "binary")
        result = OperationFilter([cond]).filter(df)
        assert result["v"].tolist() == expected, f"op={op} failed"


def test_empty_conditions_returns_all_rows():
    df = _make_df()
    result = OperationFilter([]).filter(df)
    assert len(result) == len(df)
