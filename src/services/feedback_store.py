from __future__ import annotations
import json
import sqlite3
from pathlib import Path
import pandas as pd
from src.agents.base_detector import AnomalyEvent

_DEFAULT_DB = "feedback.db"
_MIN_RETRAIN = 20


class FeedbackStore:
    """관리자 O/X 피드백 SQLite 저장소."""

    def __init__(self, db_path: str = _DEFAULT_DB):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT NOT NULL,
                    score       REAL NOT NULL,
                    top_signals TEXT,
                    admin_label TEXT NOT NULL,
                    reason      TEXT,
                    llm_verdict TEXT,
                    created_at  TEXT DEFAULT (datetime('now'))
                )
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def save(self, event: AnomalyEvent, label: str, reason: str = "") -> None:
        if label not in ("O", "X"):
            raise ValueError(f"label은 'O' 또는 'X'여야 합니다. 입력: {label!r}")
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO feedback (timestamp, score, top_signals, admin_label, reason, llm_verdict) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (str(event.timestamp), event.score,
                 json.dumps(event.top_signals, ensure_ascii=False),
                 label, reason, event.metadata.get("llm_verdict")),
            )

    def load_labeled(self) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query(
                "SELECT id, timestamp, score, top_signals, admin_label, reason, created_at "
                "FROM feedback ORDER BY timestamp",
                conn,
            )

    def retrain_scaffold(self) -> dict:
        df = self.load_labeled()
        n_normal  = int((df["admin_label"] == "X").sum())
        n_anomaly = int((df["admin_label"] == "O").sum())
        total = n_normal + n_anomaly
        ready = total >= _MIN_RETRAIN
        return {
            "total_labeled": total,
            "normal_count":  n_normal,
            "anomaly_count": n_anomaly,
            "ready":         ready,
            "message": (
                f"재학습 준비 완료 ({total}개 라벨 확보)"
                if ready
                else f"재학습까지 {_MIN_RETRAIN - total}개 더 필요"
            ),
        }
