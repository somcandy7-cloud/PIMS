import logging
from pathlib import Path

from src.agents.fault_classifier import FaultReport


def _setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("pims.alarm")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
    return logger


def notify(report: FaultReport, log_file: str = "alarm.log") -> None:
    """Windows 토스트 알림 + 로그 파일 기록."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(log_file)

    msg = f"[{report.severity}] {report.fault_type} @ {report.timestamp}"
    logger.warning(msg)

    try:
        from plyer import notification
        notification.notify(
            title=f"PIMS 장애 알람 [{report.severity}]",
            message=f"{report.fault_type}\n{report.root_cause_hypothesis[:80]}",
            app_name="PIMS",
            timeout=10,
        )
    except Exception:
        # plyer 없거나 headless 환경 → 로그만 남김
        pass
