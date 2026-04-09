import re
from pathlib import Path

import pandas as pd

_ISO_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
_ISO_TS_FULL = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


class IbaCSVLoader:
    """iba PDA Analyzer CSV 파일을 로드한다.

    실제 포맷:
      Row 0 : iba 채널 주소  (Time ; [0:0] ; [0:1] ...)
      Row 1 : PLC 태그명    (time ; MB 118 ; MB 119 ...)
      Row 2 : 단위
      Row 3~N : 메타데이터
      Row N+1 : 모든 데이터가 한 줄로 — 타임스탬프 포함 레코드들이 ';'로 연결됨
                예) ts1;v1;v2;...;vK;ts2;v1;v2;...;vK;...
    """

    SEP = ";"
    _DETECT_BYTES = 200 * 1024

    def __init__(self, encoding: str = "utf-8-sig"):
        self.encoding = encoding

    def load(self, filepath: str) -> pd.DataFrame:
        path = Path(filepath)
        enc = self._detect_encoding(path)
        col_names = self._read_header_cols(path, enc)
        data_start = self._find_data_start(path, enc)
        data_line = self._read_data_line(path, enc, data_start)
        return self._parse_records(data_line, col_names)

    # ── 인코딩 탐지 ─────────────────────────────────────────────────────────

    def _detect_encoding(self, path: Path) -> str:
        raw = path.read_bytes()[: self._DETECT_BYTES]
        for enc in (self.encoding, "cp949", "latin-1"):
            try:
                raw.decode(enc)
                return enc
            except (UnicodeDecodeError, LookupError):
                continue
        return "latin-1"

    # ── 헤더 파싱 ────────────────────────────────────────────────────────────

    def _find_data_start(self, path: Path, enc: str) -> int:
        """ISO 8601 타임스탬프로 시작하는 첫 번째 행 번호를 반환한다."""
        with open(path, encoding=enc, errors="replace") as f:
            for i, line in enumerate(f):
                if _ISO_TS_RE.match(line.split(self.SEP)[0].strip()):
                    return i
        return 23

    def _read_header_cols(self, path: Path, enc: str) -> list[str]:
        """Row 1(PLC 태그명 행)에서 컬럼명을 추출하고 중복명을 처리한다."""
        with open(path, encoding=enc, errors="replace") as f:
            for i, line in enumerate(f):
                if i == 1:
                    names = [c.strip() for c in line.rstrip("\r\n").split(self.SEP)]
                    return self._dedup_names(names)
        return []

    @staticmethod
    def _dedup_names(names: list[str]) -> list[str]:
        seen: dict[str, int] = {}
        result: list[str] = []
        for name in names:
            if name not in seen:
                seen[name] = 0
                result.append(name)
            else:
                seen[name] += 1
                result.append(f"{name}_{seen[name]}")
        return result

    # ── 데이터 행 읽기 ────────────────────────────────────────────────────────

    def _read_data_line(self, path: Path, enc: str, data_start: int) -> str:
        """data_start 행부터 EOF까지 전체 데이터를 하나의 문자열로 반환한다."""
        lines = []
        with open(path, encoding=enc, errors="replace") as f:
            for i, line in enumerate(f):
                if i >= data_start:
                    lines.append(line.rstrip("\r\n"))
        return self.SEP.join(lines)

    # ── 레코드 파싱 → DataFrame ───────────────────────────────────────────────

    def _parse_records(self, data_line: str, col_names: list[str]) -> pd.DataFrame:
        """타임스탬프를 기준으로 레코드를 분리하고 DataFrame으로 변환한다.

        iba CSV 데이터는 모든 레코드가 단일 행에 ';'로 연결되어 있다.
        각 레코드는 타임스탬프 + 고정 개수의 값으로 구성된다.
        """
        fields = data_line.split(self.SEP)
        if not fields:
            return pd.DataFrame()

        # 타임스탬프 필드 인덱스 탐색
        ts_indices = [
            i for i, f in enumerate(fields)
            if _ISO_TS_RE.match(f.strip())
        ]
        if not ts_indices:
            return pd.DataFrame()

        # 레코드 너비 결정: 가장 많이 등장하는 간격 사용
        if len(ts_indices) >= 2:
            gaps = [ts_indices[j + 1] - ts_indices[j] for j in range(len(ts_indices) - 1)]
            record_width = max(set(gaps), key=gaps.count)
        else:
            record_width = len(fields) - ts_indices[0]

        # 실제 데이터 폭이 헤더 컬럼 수보다 작을 때 컬럼을 제한한다
        effective_cols = min(record_width, len(col_names))
        use_col_names = col_names[:effective_cols]

        # 레코드 추출 (너비보다 짧은 마지막 레코드는 패딩)
        records: list[list] = []
        for idx in ts_indices:
            chunk = fields[idx: idx + effective_cols]
            if len(chunk) < effective_cols:
                chunk.extend([""] * (effective_cols - len(chunk)))
            records.append(chunk)

        df = pd.DataFrame(records, columns=use_col_names)

        # 타임스탬프 인덱스 설정
        ts_col = col_names[0]
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.set_index(ts_col)

        # 수치형 변환
        non_ts_cols = df.columns.tolist()
        df[non_ts_cols] = df[non_ts_cols].apply(pd.to_numeric, errors="coerce")

        return df
