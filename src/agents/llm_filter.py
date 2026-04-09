# src/agents/llm_filter.py
from __future__ import annotations
import re
from pathlib import Path

import pandas as pd

from src.agents.base_detector import AnomalyEvent
from src.agents.llm_backends import LLMBackend

_PROMPT_PATH = Path(__file__).parent.parent.parent / "config" / "llm_filter_prompt.md"
_CONTEXT_SEC = 10


def _load_prompt_template() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    return (
        "이상 후보를 분석하세요.\n"
        "timestamp: {timestamp}, score: {score:.4f}, signals: {top_signals}\n"
        "context: {context_stats}\n"
        "첫 줄에 KEEP: 이유 또는 REJECT: 이유 형식으로 응답하세요."
    )


def _build_context_stats(event: AnomalyEvent, df: pd.DataFrame) -> str:
    ts = event.timestamp
    t_start = ts - pd.Timedelta(seconds=_CONTEXT_SEC)
    t_end = ts + pd.Timedelta(seconds=_CONTEXT_SEC)

    sig_names = [s[0] for s in event.top_signals]
    available = [s for s in sig_names if s in df.columns]
    if not available:
        return "(신호 데이터 없음)"

    try:
        window = df.loc[t_start:t_end, available]
    except Exception:
        window = df[available]

    if window.empty:
        return "(윈도우 데이터 없음)"

    lines = []
    for col in available:
        s = window[col].dropna()
        if len(s) == 0:
            continue
        lines.append(
            f"  {col}: 평균={s.mean():.2f}, 최솟값={s.min():.2f}, "
            f"최댓값={s.max():.2f}, 표준편차={s.std():.2f}"
        )
    return "\n".join(lines) if lines else "(통계 계산 불가)"


def _parse_verdict(response_text: str) -> tuple[str, str]:
    first_line = response_text.strip().splitlines()[0] if response_text.strip() else ""
    m = re.match(r"^(KEEP|REJECT):\s*(.+)", first_line, re.IGNORECASE)
    if m:
        return m.group(1).upper(), m.group(2).strip()
    return "ERROR_KEEP", f"파싱 불가 응답: {first_line[:80]}"


class LLMFilter:
    """LLM 백엔드를 사용해 이상치 후보 목록에서 유효 이상치만 선별한다.

    backend=None이면 비활성화 — 모든 후보를 통과시킨다 (안전 fallback).
    프롬프트 템플릿: config/llm_filter_prompt.md
    """

    def __init__(self, backend: LLMBackend | None, max_candidates_per_run: int | None = None):
        self.backend = backend
        self.max_candidates_per_run = max_candidates_per_run

    def filter(
        self,
        candidates: list[AnomalyEvent],
        df: pd.DataFrame,
    ) -> list[AnomalyEvent]:
        """후보 목록을 LLM으로 검증하고 KEEP 판정된 이벤트만 반환한다."""
        if not candidates:
            return []
        if self.backend is None:
            return candidates

        template = _load_prompt_template()
        validated = []

        should_limit = (
            self.max_candidates_per_run is not None
            and self.max_candidates_per_run > 0
            and len(candidates) > self.max_candidates_per_run
        )
        callable_indexes: set[int]
        if should_limit:
            ranked = sorted(
                list(enumerate(candidates)),
                key=lambda x: abs(float(x[1].score)),
                reverse=True,
            )
            callable_indexes = {idx for idx, _ in ranked[: self.max_candidates_per_run]}
        else:
            callable_indexes = set(range(len(candidates)))

        for idx, event in enumerate(candidates):
            if idx in callable_indexes:
                verdict, reason = self._call_backend(event, df, template)
            else:
                verdict = "SKIP_KEEP"
                reason = (
                    f"LLM 검증 생략: 후보 {len(candidates)}건 중 "
                    f"상위 {self.max_candidates_per_run}건만 검증"
                )

            updated_meta = {
                **event.metadata,
                "llm_verdict": verdict,
                "llm_reason": reason,
                "llm_backend": self.backend.name(),
            }
            updated_event = AnomalyEvent(
                timestamp=event.timestamp,
                score=event.score,
                top_signals=event.top_signals,
                label=event.label,
                metadata=updated_meta,
            )
            if verdict in ("KEEP", "ERROR_KEEP", "SKIP_KEEP"):
                validated.append(updated_event)

        return validated

    def _call_backend(
        self,
        event: AnomalyEvent,
        df: pd.DataFrame,
        template: str,
    ) -> tuple[str, str]:
        top_signals_text = "\n".join(
            f"  - {name}: 변동성={val:.3f}" for name, val in event.top_signals
        )
        context_stats = _build_context_stats(event, df)

        s = event.score
        if s <= -0.5:
            score_level = "매우 강함 (상위 1% 수준의 이상)"
        elif s <= -0.3:
            score_level = "강함 (명확한 이상 신호)"
        elif s <= -0.15:
            score_level = "보통 (이상 가능성 있음)"
        else:
            score_level = "약함 (경계 수준)"

        try:
            prompt = template.format(
                timestamp=str(event.timestamp),
                score_level=score_level,
                top_signals=top_signals_text,
                context_sec=_CONTEXT_SEC,
                context_stats=context_stats,
            )
        except KeyError:
            prompt = (
                f"이상 후보: timestamp={event.timestamp}, score={event.score:.4f}\n"
                f"신호: {top_signals_text}\n"
                "첫 줄에 KEEP: 이유 또는 REJECT: 이유 형식으로 응답하세요."
            )

        try:
            response_text = self.backend.generate(prompt)
            verdict, reason = _parse_verdict(response_text)
            print(f"\n[LLM] {event.timestamp} score={event.score:.4f}")
            print(f"  context_stats: {context_stats[:120]!r}")
            print(f"  response: {response_text[:200]!r}")
            print(f"  → {verdict}: {reason}")
            return verdict, reason
        except Exception as exc:
            return "ERROR_KEEP", f"LLM 오류: {exc}"
