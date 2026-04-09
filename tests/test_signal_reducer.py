import numpy as np
import pandas as pd
import pytest
from src.utils.signal_reducer import SignalReducer, ClusterMap


def _make_df() -> pd.DataFrame:
    """
    - 'const'    : 상수 → zero-variance, 드롭
    - 'digital'  : 0/1 이진 → 상관 분석 제외, 출력엔 포함
    - 'a', 'b'   : 완전 상관(r=1.0) → 하나만 남아야 함
    - 'c'        : 'a'와 무관 → 독립 클러스터
    - 'tiny_var' : std/range < 0.01 → 저분산 드롭
    """
    n = 100
    rng = np.random.default_rng(0)
    t = pd.date_range("2026-03-12", periods=n, freq="2s", tz="UTC")
    base = rng.normal(0, 10, n)
    return pd.DataFrame({
        "const":    np.ones(n) * 5.0,
        "digital":  np.where(np.arange(n) < 50, 1.0, 0.0),
        "a":        base,
        "b":        base * 2.0 + 1.0,
        "c":        rng.normal(0, 10, n),
        "tiny_var": np.ones(n) * 100 + rng.uniform(0, 0.001, n),
    }, index=t)


def test_zero_variance_dropped():
    df = _make_df()
    reduced, _ = SignalReducer().fit_transform(df)
    assert "const" not in reduced.columns


def test_digital_preserved_in_output():
    df = _make_df()
    reduced, _ = SignalReducer().fit_transform(df)
    assert "digital" in reduced.columns


def test_correlated_pair_reduced_to_one():
    df = _make_df()
    reduced, cluster_map = SignalReducer().fit_transform(df)
    ab_cols = [c for c in reduced.columns if c in ("a", "b")]
    assert len(ab_cols) == 1


def test_independent_signal_preserved():
    df = _make_df()
    reduced, _ = SignalReducer().fit_transform(df)
    assert "c" in reduced.columns


def test_cluster_map_type():
    df = _make_df()
    _, cluster_map = SignalReducer().fit_transform(df)
    assert isinstance(cluster_map, dict)
    for rep, members in cluster_map.items():
        assert isinstance(members, list)
        assert rep in members


def test_representative_is_highest_variance():
    df = _make_df()
    reduced, cluster_map = SignalReducer().fit_transform(df)
    # b = a*2+1 → 분산이 a의 4배 → b가 대표여야 함
    ab_rep = next((r for r, m in cluster_map.items() if "a" in m and "b" in m), None)
    assert ab_rep == "b"


def test_low_variance_dropped():
    df = _make_df()
    reduced, _ = SignalReducer().fit_transform(df)
    assert "tiny_var" not in reduced.columns


def test_empty_df_returns_empty():
    df = pd.DataFrame()
    reduced, cluster_map = SignalReducer().fit_transform(df)
    assert reduced.empty
    assert cluster_map == {}
