import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """15초 분량의 샘플 데이터 (10ms 샘플링)"""
    n = 1500
    t = pd.date_range("2026-03-20 15:49:00", periods=n, freq="10ms", tz="Asia/Seoul")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Speed_Act": np.concatenate(
                [
                    np.linspace(0, 50, 500),
                    np.full(700, 50),
                    np.linspace(50, 0, 300),
                ]
            ),
            "Position_Act": np.cumsum(rng.uniform(0.05, 0.15, n)),
            "Current_1": np.concatenate(
                [
                    rng.normal(100, 5, 500),
                    rng.normal(80, 5, 500),
                    rng.normal(180, 10, 500),
                ]
            ),
        },
        index=t,
    )
