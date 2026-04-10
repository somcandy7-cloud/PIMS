from __future__ import annotations
import operator as _op
import pandas as pd
from src.utils.operation_discovery import OperationCondition

_OPS: dict[str, object] = {
    "==": _op.eq, "!=": _op.ne,
    ">":  _op.gt, ">=": _op.ge,
    "<":  _op.lt, "<=": _op.le,
}


class OperationFilter:
    """OperationCondition 목록을 AND로 결합하는 가동 구간 필터.

    Parameters
    ----------
    conditions  : 가동 조건 목록
    warmup_sec  : 각 가동 블록 시작 후 제외할 초 (기동 과도 구간)
    cooldown_sec: 각 가동 블록 종료 전 제외할 초 (관성·감속 구간)
    """

    def __init__(
        self,
        conditions: list[OperationCondition],
        warmup_sec: float = 0,
        cooldown_sec: float = 0,
    ) -> None:
        self.conditions = conditions
        self.warmup_sec = warmup_sec
        self.cooldown_sec = cooldown_sec

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.conditions:
            return df.copy()

        # ── 1. 조건 마스크 ──────────────────────────────────────────────────
        mask = pd.Series(True, index=df.index)
        for cond in self.conditions:
            if cond.column not in df.columns:
                continue  # CSV 전환 시 컬럼 없을 수 있음 — 해당 조건 스킵
            op_func = _OPS.get(cond.op)
            if op_func is None:
                raise ValueError(f"지원하지 않는 연산자: {cond.op!r}")
            mask &= op_func(df[cond.column], cond.value)

        # ── 2. warmup / cooldown 버퍼 트리밍 ───────────────────────────────
        if (self.warmup_sec > 0 or self.cooldown_sec > 0) and mask.any():
            mask = self._trim_transients(df.index, mask)

        return df.loc[mask].copy()

    def _trim_transients(
        self,
        idx: pd.Index,
        mask: pd.Series,
    ) -> pd.Series:
        """각 연속 가동 블록의 앞·뒤 과도 구간을 False로 설정한다."""
        mask = mask.copy()

        warmup_td   = pd.Timedelta(seconds=self.warmup_sec)
        cooldown_td = pd.Timedelta(seconds=self.cooldown_sec)

        # 연속 블록 식별: 값이 바뀔 때마다 블록 ID 증가
        block_id = (mask.astype(int).diff().ne(0)).cumsum()

        for _bid, group in mask.groupby(block_id, sort=False):
            if not group.iloc[0]:  # 정지 블록 스킵
                continue
            t_start = group.index[0]
            t_end   = group.index[-1]

            block_idx = group.index

            if self.warmup_sec > 0:
                trim = block_idx[block_idx < t_start + warmup_td]
                mask.loc[trim] = False

            if self.cooldown_sec > 0:
                trim = block_idx[block_idx > t_end - cooldown_td]
                mask.loc[trim] = False

        return mask
