import pandas as pd
import numpy as np
import pytest
from src.utils.signal_type_filter import SignalTypeFilter, classify_signal


def _df(cols: list[str], n: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(0, 1, (n, len(cols))), columns=cols)


# ── classify_signal ─────────────────────────────────────────────────────────

def test_mb_is_flag():
    assert classify_signal("[3_CO_PLC_A]MB      43") == "flag"

def test_qb_is_flag():
    assert classify_signal("[3_CO_PLC_B]QB      22") == "flag"

def test_ib_is_flag():
    assert classify_signal("[3_C_CAR_1]IB       9") == "flag"

def test_dbb_is_flag():
    assert classify_signal("[3_CO_PLC_A]DB421.DBB   3") == "flag"

def test_dbw_is_measurement():
    assert classify_signal("[3_P_CAR_1]DB420.DBW   2") == "measurement"

def test_dbd_is_measurement():
    assert classify_signal("DB420.DBD   10") == "measurement"

def test_no_prefix_mb_is_flag():
    assert classify_signal("MB    118") == "flag"

def test_no_prefix_dbw_is_measurement():
    assert classify_signal("DB420.DBW    4") == "measurement"

def test_unknown_kept():
    # 분류 불명인 신호는 보수적으로 포함
    assert classify_signal("MD    348_1") == "unknown"


# ── SignalTypeFilter ─────────────────────────────────────────────────────────

def test_flags_removed():
    cols = [
        "[3_CO_PLC_A]MB      43",
        "[3_CO_PLC_A]QB      15",
        "[3_C_CAR_1]IB       9",
        "[3_CO_PLC_A]DB421.DBB   3",
    ]
    df = _df(cols)
    result, stats = SignalTypeFilter().filter(df)
    assert result.empty or len(result.columns) == 0
    assert stats["excluded"] == 4

def test_measurements_kept():
    cols = [
        "[3_P_CAR_1]DB420.DBW   2",
        "DB420.DBD   10",
    ]
    df = _df(cols)
    result, stats = SignalTypeFilter().filter(df)
    assert list(result.columns) == cols
    assert stats["kept"] == 2
    assert stats["excluded"] == 0

def test_mixed_columns():
    cols = [
        "[3_CO_PLC_A]MB      43",        # flag
        "[3_P_CAR_1]DB420.DBW   2",      # measurement
        "[3_CO_PLC_B]DB421.DBB   61",    # flag
        "DB420.DBD   10",                # measurement
    ]
    df = _df(cols)
    result, stats = SignalTypeFilter().filter(df)
    assert stats["kept"] == 2
    assert stats["excluded"] == 2
    assert "[3_P_CAR_1]DB420.DBW   2" in result.columns
    assert "DB420.DBD   10" in result.columns

def test_empty_df():
    result, stats = SignalTypeFilter().filter(pd.DataFrame())
    assert result.empty
    assert stats["excluded"] == 0

def test_unknown_signals_kept():
    """분류 불명 신호는 보수적으로 유지한다."""
    cols = ["MD    348_1", "some_unknown_signal"]
    df = _df(cols)
    result, stats = SignalTypeFilter().filter(df)
    assert stats["kept"] == 2

def test_index_preserved():
    idx = pd.date_range("2026-03-12", periods=5, freq="2s", tz="UTC")
    df = pd.DataFrame({"DB420.DBW 1": range(5), "MB 1": range(5)}, index=idx)
    result, _ = SignalTypeFilter().filter(df)
    assert list(result.index) == list(idx)
