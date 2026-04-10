# src/agents/llm_filter.py
from __future__ import annotations
import re
from pathlib import Path

import pandas as pd

from src.agents.base_detector import AnomalyEvent
from src.agents.llm_backends import LLMBackend

_PROMPT_PATH = Path(__file__).parent.parent.parent / "config" / "llm_filter_prompt.md"
_DEFAULT_CONTEXT_SEC = 60

# 자동 판정 임계값 (LLM 호출 없이 코드가 직접 판정)
_AUTO_KEEP_THRESHOLD   = 0.6   # 이상도 ≥ 60% → 자동 KEEP (score ≤ -0.60)
_AUTO_REJECT_THRESHOLD = 0.25  # 이상도 < 25% → 자동 REJECT (score > -0.25)


def _anomaly_degree(score: float) -> float:
    """IF score(음수) → 이상도 0~100%. 높을수록 이상."""
    return min(100.0, round(abs(score) * 100, 1))


def _load_prompt_template() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    return (
        "이상 후보를 분석하세요.\n"
        "이상도: {anomaly_degree}%, 신호:\n{top_signals}\n통계:\n{context_stats}\n"
        "첫 줄에 '이상: 이유' 또는 '정상: 이유' 형식으로 응답하세요."
    )


def _build_context_stats(event: AnomalyEvent, df: pd.DataFrame, context_sec: int) -> str:
    ts = event.timestamp
    t_start = ts - pd.Timedelta(seconds=context_sec)
    t_end = ts + pd.Timedelta(seconds=context_sec)

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
            f"  {col}: 평균={s.mean():.3f}, 최솟값={s.min():.3f}, "
            f"최댓값={s.max():.3f}, 표준편차={s.std():.3f}"
        )
    return "\n".join(lines) if lines else "(통계 계산 불가)"


def _parse_verdict(response_text: str) -> tuple[str, str]:
    """LLM 응답 첫 줄에서 판정과 이유를 추출한다.

    지원 형식:
      이상: 이유   → KEEP
      정상: 이유   → REJECT
      KEEP: 이유   → KEEP  (레거시)
      REJECT: 이유 → REJECT (레거시)
    """
    first_line = response_text.strip().splitlines()[0] if response_text.strip() else ""
    m = re.match(r"^(이상|정상):\s*(.+)", first_line)
    if m:
        return ("KEEP" if m.group(1) == "이상" else "REJECT"), m.group(2).strip()
    m = re.match(r"^(KEEP|REJECT):\s*(.+)", first_line, re.IGNORECASE)
    if m:
        return m.group(1).upper(), m.group(2).strip()
    return "ERROR_KEEP", f"파싱 불가 응답: {first_line[:80]}"


class LLMFilter:
    """LLM 백엔드를 사용해 이상치 후보 목록에서 유효 이상치만 선별한다.

    이상도(abs(score)*100) 기준으로 자동 판정 후 애매한 구간만 LLM에 질의한다.
    - 이상도 ≥ 60%  → 자동 KEEP (LLM 불필요)
    - 이상도 < 25%  → 자동 REJECT (LLM 불필요)
    - 그 외          → LLM 판정
    """

    def __init__(
        self,
        backend: LLMBackend | None,
        max_candidates_per_run: int | None = None,
        context_sec: int = _DEFAULT_CONTEXT_SEC,
    ):
        self.backend = backend
        self.max_candidates_per_run = max_candidates_per_run
        self.context_sec = context_sec

    def filter(
        self,
        candidates: list[AnomalyEvent],
        df: pd.DataFrame,
    ) -> tuple[list[AnomalyEvent], list[AnomalyEvent]]:
        """후보 목록을 검증하고 (kept, rejected) 튜플을 반환한다."""
        if not candidates:
            return [], []

        kept: list[AnomalyEvent] = []
        rejected: list[AnomalyEvent] = []

        # 자동 판정 먼저 처리 (LLM 없이)
        llm_queue: list[tuple[int, AnomalyEvent]] = []
        auto_results: dict[int, tuple[str, str]] = {}

        for idx, event in enumerate(candidates):
            deg = _anomaly_degree(event.score)
            if deg >= _AUTO_KEEP_THRESHOLD * 100:
                auto_results[idx] = (
                    "KEEP",
                    f"이상도 {deg:.1f}% — 자동 확정 (LLM 생략)",
                )
            elif deg < _AUTO_REJECT_THRESHOLD * 100:
                auto_results[idx] = (
                    "REJECT",
                    f"이상도 {deg:.1f}% — 경계 수준 자동 기각 (LLM 생략)",
                )
            else:
                llm_queue.append((idx, event))

        # LLM 호출 대상 결정 (max_candidates_per_run 제한)
        if self.backend is not None and llm_queue:
            if (
                self.max_candidates_per_run is not None
                and self.max_candidates_per_run > 0
                and len(llm_queue) > self.max_candidates_per_run
            ):
                llm_queue.sort(key=lambda x: abs(float(x[1].score)), reverse=True)
                skipped = llm_queue[self.max_candidates_per_run:]
                llm_queue = llm_queue[: self.max_candidates_per_run]
                for idx, event in skipped:
                    deg = _anomaly_degree(event.score)
                    auto_results[idx] = (
                        "SKIP_KEEP",
                        f"LLM 검증 생략: 최대 {self.max_candidates_per_run}건 초과 (이상도 {deg:.1f}%)",
                    )

            template = _load_prompt_template()
            for idx, event in llm_queue:
                verdict, reason = self._call_backend(event, df, template)
                auto_results[idx] = (verdict, reason)
        elif llm_queue:
            # backend 없음 → 전부 통과
            for idx, event in llm_queue:
                deg = _anomaly_degree(event.score)
                auto_results[idx] = ("KEEP", f"LLM 비활성화 — 이상도 {deg:.1f}%")

        # 결과 조립
        backend_name = self.backend.name() if self.backend else "auto"
        for idx, event in enumerate(candidates):
            verdict, reason = auto_results[idx]
            updated_event = AnomalyEvent(
                timestamp=event.timestamp,
                score=event.score,
                top_signals=event.top_signals,
                label=event.label,
                metadata={
                    **event.metadata,
                    "llm_verdict": verdict,
                    "llm_reason": reason,
                    "llm_backend": backend_name,
                },
            )
            if verdict in ("KEEP", "ERROR_KEEP", "SKIP_KEEP"):
                kept.append(updated_event)
            else:
                rejected.append(updated_event)

        return kept, rejected

    def _call_backend(
        self,
        event: AnomalyEvent,
        df: pd.DataFrame,
        template: str,
    ) -> tuple[str, str]:
        deg = _anomaly_degree(event.score)
        top_signals_text = "\n".join(
            f"  - {name}: 기여도={val*100:.1f}%" for name, val in event.top_signals
        )
        context_stats = _build_context_stats(event, df, self.context_sec)

        try:
            prompt = template.format(
                timestamp=str(event.timestamp),
                anomaly_degree=deg,
                top_signals=top_signals_text,
                context_sec=self.context_sec,
                context_stats=context_stats,
            )
        except KeyError:
            prompt = (
                f"이상 후보: timestamp={event.timestamp}, 이상도={deg}%\n"
                f"신호:\n{top_signals_text}\n"
                f"통계:\n{context_stats}\n"
                "첫 줄에 '이상: 이유' 또는 '정상: 이유'로 응답하세요."
            )

        try:
            response_text = self.backend.generate(prompt)
            verdict, reason = _parse_verdict(response_text)
            print(f"\n[LLM] {event.timestamp} 이상도={deg}%")
            print(f"  context_stats: {context_stats[:120]!r}")
            print(f"  response: {response_text[:200]!r}")
            print(f"  → {verdict}: {reason}")
            return verdict, reason
        except Exception as exc:
            return "ERROR_KEEP", f"LLM 오류: {exc}"
