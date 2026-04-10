import numpy as np
import pandas as pd
import pytest
from src.utils.flag_context import get_flag_context


def _make_df(n: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2026-03-20 10:00:00", periods=n, freq="2s", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        # measurement 신호
        "[3_P_CAR_1]DB420.DBW   2": rng.normal(100, 5, n),
        # flag 신호 — 이벤트 시점 전후에 변화 있음
        "[3_P_CAR_1]QB      11": np.where(np.arange(n) < 30, 0.0, 1.0),
        "[3_P_CAR_1]IB       1": np.ones(n) * 3.0,   # 변화 없음 → 제외 대상
        "[3_P_CAR_1]MB       1": np.where(np.arange(n) < 28, 0.0, 255.0),
        # 다른 그룹 flag — group_id 필터로 제외되어야 함
        "[3_CO_PLC_A]MB      43": rng.normal(0, 1, n),
    }, index=idx)


def test_returns_only_flag_signals():
    df = _make_df()
    ts = df.index[30]
    result = get_flag_context(df, ts, group_id="3_P_CAR_1", context_sec=30)
    for col in result["signal"]:
        assert "DB420.DBW" not in col


def test_filters_to_same_group():
    df = _make_df()
    ts = df.index[30]
    result = get_flag_context(df, ts, group_id="3_P_CAR_1", context_sec=30)
    for col in result["signal"]:
        assert "3_CO_PLC_A" not in col


def test_excludes_no_change_signals():
    """이벤트 윈도우 내에서 값이 변하지 않은 flag는 제외한다."""
    df = _make_df()
    ts = df.index[30]
    result = get_flag_context(df, ts, group_id="3_P_CAR_1", context_sec=30)
    assert not any("IB" in s for s in result["signal"].values)


def test_changed_signals_included():
    df = _make_df()
    ts = df.index[30]
    result = get_flag_context(df, ts, group_id="3_P_CAR_1", context_sec=30)
    signals = result["signal"].tolist()
    assert any("QB" in s for s in signals)
    assert any("MB" in s for s in signals)


def test_columns_present():
    df = _make_df()
    ts = df.index[30]
    result = get_flag_context(df, ts, group_id="3_P_CAR_1", context_sec=30)
    for col in ("signal", "val_at_event", "val_before", "changed", "label"):
        assert col in result.columns


def test_empty_when_no_flags():
    idx = pd.date_range("2026-03-20", periods=10, freq="2s", tz="UTC")
    df = pd.DataFrame({"[grp]DB420.DBW 1": range(10)}, index=idx)
    result = get_flag_context(df, idx[5], group_id="grp", context_sec=10)
    assert result.empty


def test_no_group_returns_all_flags():
    """group_id=None 이면 그룹 필터 없이 전체 flag 반환."""
    df = _make_df()
    ts = df.index[30]
    result_all   = get_flag_context(df, ts, group_id=None, context_sec=30)
    result_group = get_flag_context(df, ts, group_id="3_P_CAR_1", context_sec=30)
    assert len(result_all) >= len(result_group)
