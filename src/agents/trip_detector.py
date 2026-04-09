from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass
class TripEvent:
    timestamp: pd.Timestamp
    trigger: Literal["multi_signal_drop", "single_signal_drop", "spike"]
    top_changed_signals: list[tuple[str, float]]  # (컬럼명, 변화량)
    description: str
    window_size: int


class TripDetector:
    """Rolling rate-of-change z-score 기반 Trip 이벤트 탐지.

    신호 역할(속도/전류 등)을 몰라도 동작한다.
    아날로그 신호(unique값 > 10)만 대상으로 한다.
    """

    def __init__(
        self,
        window: int = 50,
        roc_zscore_threshold: float = 3.0,
        min_signals_for_multi: int = 2,
        top_n: int = 5,
        min_duration: int = 3,
        excluded_signals: list[str] | None = None,
        signal_overrides: dict[str, float] | None = None,
    ):
        self.window = window
        self.threshold = roc_zscore_threshold
        self.min_signals_for_multi = min_signals_for_multi
        self.top_n = top_n
        self.min_duration = min_duration
        self.excluded_signals: set[str] = set(excluded_signals or [])
        self.signal_overrides: dict[str, float] = signal_overrides or {}

    def detect(self, df: pd.DataFrame) -> list[TripEvent]:
        analog_cols = [
            c for c in df.select_dtypes(include="number").columns
            if df[c].nunique() > 10 and c not in self.excluded_signals
        ]
        if not analog_cols:
            return []

        analog_df = df[analog_cols].copy()

        # 1차 차분(변화량) 계산
        roc = analog_df.diff().abs()

        # 컬럼별 rolling z-score
        # std 하한값 = 전체 구간 std의 10% → flat 구간에서 tiny noise가 폭발하는 현상 방지
        roll_mean = roc.rolling(self.window, min_periods=self.window // 2).mean()
        roll_std  = roc.rolling(self.window, min_periods=self.window // 2).std()
        global_std_floor = roc.std() * 0.1
        roll_std_floored = roll_std.clip(lower=global_std_floor, axis=1)
        zscore_df = (roc - roll_mean) / roll_std_floored

        # 신호별 임계값 (오버라이드 적용)
        thresholds = pd.Series(self.threshold, index=analog_cols)
        for sig, thr in self.signal_overrides.items():
            if sig in thresholds.index:
                thresholds[sig] = thr

        # 임계값 초과 여부 (True/False)
        exceeded = zscore_df.gt(thresholds, axis=1)

        # 한 샘플에서 초과한 신호 수
        exceeded_count = exceeded.sum(axis=1)

        # Trip 후보: 초과 신호가 1개 이상인 시점
        candidate_mask = exceeded_count >= 1

        # 단발성 노이즈 제거: min_duration 샘플 이상 연속으로 초과해야 Trip으로 인정
        sustained = (
            candidate_mask.rolling(self.min_duration, min_periods=self.min_duration)
            .sum() >= self.min_duration
        )

        # 연속 구간 병합: 앞 샘플이 0이고 현재가 1인 rising edge만
        rising = sustained & ~sustained.shift(1, fill_value=False)
        trip_times = df.index[rising]

        events: list[TripEvent] = []
        for ts in trip_times:
            row = zscore_df.loc[ts]
            n_exceeded = int(exceeded.loc[ts].sum())

            # top_n 변화 신호: z-score 절대값 기준
            sorted_signals = (
                row.dropna()
                .abs()
                .sort_values(ascending=False)
                .head(self.top_n)
            )
            top_changed = [
                (col, float(roc.loc[ts, col])) for col in sorted_signals.index
            ]

            if n_exceeded >= self.min_signals_for_multi:
                trigger: Literal["multi_signal_drop", "single_signal_drop", "spike"] = "multi_signal_drop"
                desc = f"{n_exceeded}개 신호 동시 급변 (Trip 추정)"
            else:
                # 단일 신호: 값이 낮아지면 drop, 올라가면 spike
                col = top_changed[0][0]
                diff_val = float(analog_df[col].diff().loc[ts])
                if diff_val < 0:
                    trigger = "single_signal_drop"
                    desc = f"신호 급감: {col}"
                else:
                    trigger = "spike"
                    desc = f"신호 급등: {col}"

            events.append(
                TripEvent(
                    timestamp=ts,
                    trigger=trigger,
                    top_changed_signals=top_changed,
                    description=desc,
                    window_size=self.window,
                )
            )
        return events
