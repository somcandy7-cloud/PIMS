"""PLC 주소 타입 기반 신호 분류 및 필터링 모듈.

Siemens S7 PLC 주소 체계:
  MB  (Merker Byte)   : PLC 내부 플래그 메모리 — 이상 탐지 제외
  QB  (Ausgang Byte)  : PLC 출력 바이트 (제어 명령) — 이상 탐지 제외
  IB  (Eingang Byte)  : PLC 입력 바이트 (디지털 입력) — 이상 탐지 제외
  DBB (Data Block Byte): 바이트 단위 상태 플래그 묶음 — 이상 탐지 제외
  DBW (Data Block Word): 2바이트 정수 측정값 — 이상 탐지 포함
  DBD (Data Block DWord): 4바이트 실수 측정값 — 이상 탐지 포함
  분류 불명           : 보수적으로 포함

컬럼명 형식:
  [그룹ID]주소  예) [3_CO_PLC_A]MB      43
  주소만        예) MB    118
"""
from __future__ import annotations

import re

import pandas as pd

# 브래킷 접두사 제거 패턴: [3_CO_PLC_A] 등
_PREFIX_RE = re.compile(r'^\[([^\]]+)\]')

# 제외 대상 패턴 (flag 신호)
_FLAG_PATTERNS: list[re.Pattern] = [
    re.compile(r'^MB[\s_]'),       # MB (Merker Byte)
    re.compile(r'^QB[\s_]'),       # QB (Output Byte)
    re.compile(r'^IB[\s_]'),       # IB (Input Byte)
    re.compile(r'^MD[\s_]'),       # MD (Merker DWord)
    re.compile(r'^DB\d+\.DBB'),    # 임의 데이터블록 Byte (상태 플래그)
]

# 포함 대상 패턴 (measurement 신호)
_MEAS_PATTERNS: list[re.Pattern] = [
    re.compile(r'^DB\d+\.DBW'),    # 데이터블록 Word (정수 측정값)
    re.compile(r'^DB\d+\.DBD'),    # 데이터블록 DWord (실수 측정값)
]

SignalKind = str  # 'measurement' | 'flag' | 'unknown'


def classify_signal(col: str) -> SignalKind:
    """컬럼명으로 신호 종류를 반환한다.

    Returns
    -------
    'flag'        : PLC 내부 플래그·제어·디지털 입력 → 이상 탐지 제외 권장
    'measurement' : 실측 물리량 (DBW/DBD) → 이상 탐지 포함
    'unknown'     : 분류 불명 → 보수적으로 포함
    """
    addr = _PREFIX_RE.sub("", col).strip()
    for pat in _FLAG_PATTERNS:
        if pat.match(addr):
            return "flag"
    for pat in _MEAS_PATTERNS:
        if pat.match(addr):
            return "measurement"
    return "unknown"


class SignalTypeFilter:
    """PLC 주소 타입 기반으로 이상 탐지에 의미없는 플래그 신호를 제거한다.

    가동 조건 탐색(AutoOperationDiscovery)은 필터 적용 전에 수행해야 한다.
    IB/QB 신호가 가동 조건 판별에 필요할 수 있기 때문이다.
    """

    def filter(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """플래그 신호를 제거하고 (측정값 DataFrame, 통계 dict)를 반환한다.

        Parameters
        ----------
        df : 전처리 완료된 신호 DataFrame

        Returns
        -------
        filtered_df : 측정값·분류불명 신호만 남긴 DataFrame
        stats       : {'kept': int, 'excluded': int}
        """
        if df.empty:
            return df.copy(), {"kept": 0, "excluded": 0}

        keep_cols = [c for c in df.columns if classify_signal(c) != "flag"]
        excluded_n = len(df.columns) - len(keep_cols)

        if not keep_cols:
            return pd.DataFrame(index=df.index), {"kept": 0, "excluded": excluded_n}

        return df[keep_cols].copy(), {"kept": len(keep_cols), "excluded": excluded_n}
