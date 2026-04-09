from __future__ import annotations

import re
from collections import defaultdict

_PREFIX_RE = re.compile(r'^\[([^\]]+)\]')


class DeviceGroupParser:
    """컬럼명 브래킷 접두사로 신호를 장치 그룹으로 분류한다.

    Examples
    --------
    '[3_CO_PLC_B]QB 42'  → group_id='3_CO_PLC_B'
    'MB    118'           → group_id=DEFAULT_GROUP ('__default__')
    """

    DEFAULT_GROUP: str = "__default__"

    @staticmethod
    def parse(columns: list[str]) -> dict[str, list[str]]:
        """컬럼 목록을 장치 그룹별로 분류한다.

        Returns
        -------
        dict[group_id, list[column_name]]
        """
        groups: dict[str, list[str]] = defaultdict(list)
        for col in columns:
            m = _PREFIX_RE.match(col)
            gid = m.group(1) if m else DeviceGroupParser.DEFAULT_GROUP
            groups[gid].append(col)
        return dict(groups)

    @staticmethod
    def group_id(column: str) -> str:
        """단일 컬럼의 그룹 ID를 반환한다."""
        m = _PREFIX_RE.match(column)
        return m.group(1) if m else DeviceGroupParser.DEFAULT_GROUP
