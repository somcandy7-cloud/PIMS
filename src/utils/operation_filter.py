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
    """OperationCondition 목록을 AND로 결합하는 가동 구간 필터."""

    def __init__(self, conditions: list[OperationCondition]):
        self.conditions = conditions

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.conditions:
            return df.copy()
        mask = pd.Series(True, index=df.index)
        for cond in self.conditions:
            col_data = df[cond.column]  # KeyError 의도적 전파
            op_func = _OPS.get(cond.op)
            if op_func is None:
                raise ValueError(f"지원하지 않는 연산자: {cond.op!r}")
            mask &= op_func(col_data, cond.value)
        return df.loc[mask].copy()
