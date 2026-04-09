"""UI 표시용 포맷 헬퍼."""
from __future__ import annotations


def display_name(tag: str, mapper) -> str:
    """태그명 → '변수명 (태그주소)' 형식. 매퍼 없거나 미매핑이면 태그명만."""
    if mapper is None:
        return tag
    var = mapper.label(tag, fallback=None)
    if var and var != tag:
        return f"{var} ({tag})"
    return tag
