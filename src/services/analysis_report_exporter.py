from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd

from src.agents.base_detector import AnomalyEvent


def _ts_to_kst_str(ts: pd.Timestamp, kst: str) -> str:
    if ts.tzinfo is None:
        ts = ts.tz_localize(kst)
    else:
        ts = ts.tz_convert(kst)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _top_signals_to_str(top_signals: list[tuple[str, float]], limit: int = 10) -> str:
    if not top_signals:
        return ""
    cut = top_signals[:limit]
    return " | ".join(f"{sig}:{mag:.4f}" for sig, mag in cut)


def _events_to_df(
    events: list[AnomalyEvent],
    kst: str,
    *,
    mark_rejected: bool = False,
) -> pd.DataFrame:
    """AnomalyEvent 목록을 DataFrame으로 변환한다.

    Parameters
    ----------
    mark_rejected : True이면 verdict 열에 'REJECT'를 명시하고
                   llm_reject_reason 열에 사유를 채운다.
    """
    rows: list[dict] = []
    for i, ev in enumerate(events, 1):
        verdict = ev.metadata.get("llm_verdict", "REJECT" if mark_rejected else "")
        reason  = ev.metadata.get("llm_reason", "")
        rows.append(
            {
                "rank": i,
                "timestamp_kst": _ts_to_kst_str(ev.timestamp, kst),
                "score": float(ev.score),
                "group_id": ev.metadata.get("group_id", ""),
                "top_signal_count": len(ev.top_signals),
                "top_signals": _top_signals_to_str(ev.top_signals),
                "llm_verdict": verdict,
                "llm_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def build_report_frames(
    *,
    df: pd.DataFrame,
    source_file: str,
    candidates: list[AnomalyEvent],
    events: list[AnomalyEvent],
    rejected: list[AnomalyEvent] | None = None,
    running_rows: int,
    reduced_cols: int,
    group_count: int,
    group_signal_counts: dict[str, int] | None = None,
    groups_profile: dict | None = None,
    kst: str = "Asia/Seoul",
) -> dict[str, pd.DataFrame]:
    """분석 결과를 여러 시트 DataFrame으로 빌드한다.

    Parameters
    ----------
    rejected : LLMFilter가 REJECT 판정한 이벤트 목록.
               None이면 candidates 중 llm_verdict가 없는 항목만 IFCandidates에 포함.
    """
    total_rows = len(df)
    raw_cols = len(df.columns)
    running_pct = (running_rows / max(total_rows, 1)) * 100.0
    duration_min = (df.index[-1] - df.index[0]).total_seconds() / 60 if total_rows > 1 else 0.0
    now_kst = pd.Timestamp.now(tz=kst).strftime("%Y-%m-%d %H:%M:%S")

    summary_df = pd.DataFrame(
        [
            {
                "generated_at_kst": now_kst,
                "source_file": source_file,
                "total_samples": total_rows,
                "running_rows": running_rows,
                "running_ratio_pct": round(running_pct, 2),
                "device_group_count": group_count,
                "raw_signal_count": raw_cols,
                "reduced_signal_count": reduced_cols,
                "reduction_delta": reduced_cols - raw_cols,
                "if_candidate_count": len(candidates),
                "llm_verified_count": len(events),
                "data_duration_min": round(duration_min, 2),
            }
        ]
    )

    group_rows: list[dict] = []
    group_signal_counts = group_signal_counts or {}
    groups_profile = groups_profile or {}
    all_group_ids = sorted(set(group_signal_counts.keys()) | set(groups_profile.keys()))
    for gid in all_group_ids:
        gprof = groups_profile.get(gid, {}) or {}
        gconds = gprof.get("conditions") or []
        gclusters = gprof.get("clusters") or {}
        group_rows.append(
            {
                "group_id": gid,
                "signal_count": int(group_signal_counts.get(gid, 0)),
                "condition_count": len(gconds),
                "has_cluster_cache": bool(gclusters),
                "cluster_rep_signal_count": len(gclusters),
            }
        )
    groups_df = pd.DataFrame(group_rows)

    # IFCandidates: KEEP(events) + REJECT(rejected) 통합, 없으면 원본 candidates
    if rejected is not None:
        kept_df = _events_to_df(events, kst)
        rej_df  = _events_to_df(rejected, kst, mark_rejected=True)
        if_candidates_df = (
            pd.concat([kept_df, rej_df], ignore_index=True)
            if not rej_df.empty
            else kept_df
        )
        # 점수 내림차순 정렬 (KEEP 먼저)
        if not if_candidates_df.empty:
            if_candidates_df = if_candidates_df.sort_values(
                ["llm_verdict", "score"],
                ascending=[True, True],  # KEEP < REJECT 알파벳순, score 오름차순
            ).reset_index(drop=True)
            if_candidates_df["rank"] = range(1, len(if_candidates_df) + 1)
    else:
        if_candidates_df = _events_to_df(candidates, kst)

    llm_verified_df = _events_to_df(events, kst)

    return {
        "Summary": summary_df,
        "DeviceGroups": groups_df,
        "IFCandidates": if_candidates_df,
        "LLMVerified": llm_verified_df,
    }


def frames_to_excel_bytes(frames: dict[str, pd.DataFrame]) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return buffer.getvalue()


def save_report_file(
    *,
    root: Path,
    source_file: str,
    excel_bytes: bytes,
    suffix: str | None = None,
) -> Path:
    export_dir = root / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(source_file).stem or "analysis"
    ts = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y%m%d_%H%M%S")
    extra = f"_{suffix}" if suffix else ""
    out = export_dir / f"report_{stem}_{ts}{extra}.xlsx"
    out.write_bytes(excel_bytes)
    return out
