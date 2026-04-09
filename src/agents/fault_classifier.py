from dataclasses import dataclass, field

from src.agents.trip_detector import TripEvent


@dataclass
class FaultReport:
    fault_type: str
    severity: str  # "높음" | "중간" | "낮음"
    inspection_items: list[str]
    root_cause_hypothesis: str
    timestamp: str = ""


_INSPECTION_MULTI_DROP = [
    "인버터 과전류/과전압 트립 이력 확인",
    "속도·전류 신호 동시 급변 → 구동부 전원 차단 여부 점검",
    "모터 절연 저항 측정",
    "엔코더 케이블 및 커넥터 접촉 상태 확인",
    "PLC 출력 접점 및 릴레이 동작 상태 확인",
]

_INSPECTION_SPEED_DROP = [
    "속도 피드백 신호 케이블 단선/접촉 불량 확인",
    "엔코더 또는 타코미터 기계적 결합 상태 점검",
    "인버터 속도 지령 회로 점검",
]

_INSPECTION_CURRENT_SPIKE = [
    "모터 권선 단락 또는 지락 여부 점검",
    "인버터 IGBT 모듈 상태 확인",
    "부하 측 기계적 과부하 원인(레일 걸림 등) 점검",
    "전원 전압 불평형 측정",
]

_INSPECTION_GENERIC = [
    "관련 PLC 고장 이력(Fault Code) 확인",
    "현장 설비 육안 점검 실시",
    "담당 전기 기술자에게 점검 의뢰",
]


class FaultClassifier:
    """TripEvent를 받아 룰 기반 FaultReport를 생성한다."""

    def classify(self, event: TripEvent) -> FaultReport:
        trigger = event.trigger
        top = event.top_changed_signals

        if trigger == "multi_signal_drop":
            return FaultReport(
                fault_type="복합 신호 급감 (Trip 추정)",
                severity="높음",
                inspection_items=_INSPECTION_MULTI_DROP,
                root_cause_hypothesis=(
                    f"{len(top)}개 신호 동시 급변: "
                    + ", ".join(f"{n}({v:.1f})" for n, v in top[:3])
                    + " — 구동부 전원 차단 또는 인버터 보호 동작 가능성"
                ),
                timestamp=str(event.timestamp),
            )

        if trigger == "single_signal_drop":
            col = top[0][0] if top else "unknown"
            is_speed = any(k in col.lower() for k in ["speed", "db420", "md"])
            items = _INSPECTION_SPEED_DROP if is_speed else _INSPECTION_GENERIC
            return FaultReport(
                fault_type="단일 신호 급감",
                severity="중간",
                inspection_items=items,
                root_cause_hypothesis=(
                    f"신호 '{col}' 급감 → 센서 불량 또는 단선 가능성"
                ),
                timestamp=str(event.timestamp),
            )

        # spike
        col = top[0][0] if top else "unknown"
        is_current = any(k in col.lower() for k in ["current", "db420", "db421"])
        items = _INSPECTION_CURRENT_SPIKE if is_current else _INSPECTION_GENERIC
        return FaultReport(
            fault_type="신호 스파이크",
            severity="중간",
            inspection_items=items,
            root_cause_hypothesis=(
                f"신호 '{col}' 급등 → 과부하 또는 노이즈 유입 가능성"
            ),
            timestamp=str(event.timestamp),
        )
