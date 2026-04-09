from __future__ import annotations
import json
import pandas as pd
from src.agents.base_detector import AnomalyEvent

_CROSS_SIGNALS = ["MD    348_1"]
_CONTEXT_MIN   = 5


def _score_level(score: float) -> str:
    if score <= -0.5:  return "매우 강함 (상위 1% 수준)"
    if score <= -0.3:  return "강함 (명확한 이상 신호)"
    if score <= -0.15: return "보통 (이상 가능성 있음)"
    return "약함 (경계 수준)"


def _window_stats(df: pd.DataFrame, ts: pd.Timestamp, cols: list[str]) -> dict:
    t0 = ts - pd.Timedelta(minutes=_CONTEXT_MIN)
    t1 = ts + pd.Timedelta(minutes=_CONTEXT_MIN)
    stats: dict[str, dict] = {}
    for col in [c for c in cols if c in df.columns]:
        try:
            s = df.loc[t0:t1, col].dropna()
        except Exception:
            s = df[col].dropna()
        if len(s) == 0:
            continue
        trend = float(s.iloc[-1] - s.iloc[0]) if len(s) >= 2 else 0.0
        stats[col] = {
            "mean": round(float(s.mean()), 3),
            "std":  round(float(s.std()), 3) if len(s) > 1 else 0.0,
            "min":  round(float(s.min()), 3),
            "max":  round(float(s.max()), 3),
            "trend": round(trend, 3),
        }
    return stats


class AnomalyContextFormatter:
    def __init__(
        self,
        cross_signals: list[str] | None = None,
        context_minutes: int = _CONTEXT_MIN,
    ):
        self.cross_signals = cross_signals or _CROSS_SIGNALS
        self.context_minutes = context_minutes

    def format(self, event: AnomalyEvent, df: pd.DataFrame) -> dict[str, str]:
        ts    = event.timestamp
        level = _score_level(event.score)
        sig_names = [s[0] for s in event.top_signals]
        ctx   = _window_stats(df, ts, sig_names + self.cross_signals)

        payload = {
            "timestamp": str(ts),
            "score":     round(event.score, 4),
            "score_level": level,
            "top_signals": [{"name": n, "variability": round(v, 4)} for n, v in event.top_signals],
            "context_window_minutes": self.context_minutes,
            "context_stats": ctx,
        }

        lines = [
            "[이상 탐지 리포트]",
            f"  발생 시각    : {ts}",
            f"  이상 강도    : {level}  (score={event.score:.4f})",
            f"  주요 신호    : {', '.join(sig_names)}",
            f"  분석 구간    : ±{self.context_minutes}분",
            "", "  [신호별 트렌드 통계]",
        ]
        for col, s in ctx.items():
            lines.append(
                f"  {col:<35} mean={s['mean']:.2f}  std={s['std']:.2f}"
                f"  [{s['min']:.2f}~{s['max']:.2f}]  trend={s['trend']:+.2f}"
            )

        return {
            "json_payload": json.dumps(payload, ensure_ascii=False, indent=2),
            "text_summary": "\n".join(lines),
        }
