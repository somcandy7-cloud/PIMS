from pathlib import Path

from src.agents.fault_classifier import FaultReport
from src.agents.trip_detector import TripEvent


_TEMPLATE = """\
================================================================
PIMS 장애 분석 리포트
================================================================
발생 시각  : {timestamp}
장애 유형  : {fault_type}
심각도     : {severity}

[원인 추정]
{root_cause}

[점검 항목]
{items}

[급변 신호 Top {n_signals}]
{signals}
================================================================
"""


def generate(event: TripEvent, report: FaultReport, output_path: str | None = None) -> str:
    items_str = "\n".join(f"  {i+1}. {item}" for i, item in enumerate(report.inspection_items))
    signals_str = "\n".join(
        f"  {i+1}. {col}  (변화량: {mag:.2f})"
        for i, (col, mag) in enumerate(event.top_changed_signals)
    )

    text = _TEMPLATE.format(
        timestamp=report.timestamp or str(event.timestamp),
        fault_type=report.fault_type,
        severity=report.severity,
        root_cause=report.root_cause_hypothesis,
        items=items_str,
        n_signals=len(event.top_changed_signals),
        signals=signals_str,
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(text, encoding="utf-8")

    return text
