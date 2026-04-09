"""PLC 신호 클러스터링 현황을 Excel로 내보낸다.

실행:
    python scripts/export_cluster_map.py

출력:
    outputs/cluster_map_<equipment_id>_<timestamp>.xlsx

Sheet1 cluster_map  : 신호 전체 목록
  - equipment_id / group_id / cluster_rep / signal / signal_type
  - is_representative / cluster_size / label / description

Sheet2 summary      : 그룹별 통계
  - 전체 신호 수 / 측정값 / 플래그 / 분류불명 / 클러스터 대표 수 / 축소율
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.services.signal_label_mapper import SignalLabelMapper
from src.utils.signal_type_filter import classify_signal


# ── 설정 ──────────────────────────────────────────────────────────────────────

PROFILES_PATH = ROOT / "config" / "equipment_profiles.yaml"
LABEL_FILES   = [
    ROOT / "반출데이터" / "3OVEN.xlsx",
    ROOT / "반출데이터" / "4OVEN.xlsx",
]
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── 라벨 매퍼 로드 ─────────────────────────────────────────────────────────────

existing_labels = [p for p in LABEL_FILES if p.exists()]
mapper = SignalLabelMapper(existing_labels) if existing_labels else None
print(f"라벨 매퍼: {mapper.size if mapper else 0}개 항목")


# ── profiles.yaml 파싱 ────────────────────────────────────────────────────────

with open(PROFILES_PATH, encoding="utf-8") as f:
    profiles: dict = yaml.safe_load(f) or {}

cluster_rows: list[dict] = []
summary_rows: list[dict] = []

for eq_id, eq_data in profiles.items():
    groups_data: dict = eq_data.get("groups", {})

    for group_id, gdata in groups_data.items():
        clusters: dict | None = gdata.get("clusters")
        if not clusters:
            continue

        # 그룹 통계 집계용
        all_signals_in_group: list[str] = []
        for rep, members in clusters.items():
            all_signals_in_group.extend(members)

        n_total      = len(all_signals_in_group)
        n_meas       = sum(1 for s in all_signals_in_group if classify_signal(s) == "measurement")
        n_flag       = sum(1 for s in all_signals_in_group if classify_signal(s) == "flag")
        n_unknown    = n_total - n_meas - n_flag
        n_reps       = len(clusters)
        reduction_pct = round((1 - n_reps / max(n_total, 1)) * 100, 1)

        summary_rows.append({
            "equipment_id"  : eq_id,
            "group_id"      : group_id,
            "total_signals" : n_total,
            "measurement"   : n_meas,
            "flag"          : n_flag,
            "unknown"       : n_unknown,
            "cluster_reps"  : n_reps,
            "reduction_%"   : reduction_pct,
            "confirmed"     : bool(gdata.get("confirmed", False)),
        })

        # 신호별 행 생성
        for rep, members in clusters.items():
            rep_type = classify_signal(rep)
            cluster_size = len(members)

            for sig in members:
                sig_type = classify_signal(sig)
                label, desc = "", ""
                if mapper:
                    hit = mapper.lookup(sig)
                    if hit:
                        label, desc = hit[0], hit[1]

                cluster_rows.append({
                    "equipment_id"   : eq_id,
                    "group_id"       : group_id,
                    "cluster_rep"    : rep,
                    "rep_type"       : rep_type,
                    "signal"         : sig,
                    "signal_type"    : sig_type,
                    "is_rep"         : sig == rep,
                    "cluster_size"   : cluster_size,
                    "label"          : label,
                    "description"    : desc,
                })


# ── DataFrame 생성 ─────────────────────────────────────────────────────────────

df_clusters = pd.DataFrame(cluster_rows, columns=[
    "equipment_id", "group_id", "cluster_rep", "rep_type",
    "signal", "signal_type", "is_rep", "cluster_size",
    "label", "description",
])

df_summary = pd.DataFrame(summary_rows, columns=[
    "equipment_id", "group_id", "total_signals", "measurement",
    "flag", "unknown", "cluster_reps", "reduction_%", "confirmed",
])

# 정렬
df_clusters = df_clusters.sort_values(
    ["equipment_id", "group_id", "cluster_rep", "is_rep"],
    ascending=[True, True, True, False],
).reset_index(drop=True)

df_summary = df_summary.sort_values(
    ["equipment_id", "group_id"]
).reset_index(drop=True)


# ── Excel 출력 ─────────────────────────────────────────────────────────────────

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
eq_ids = "_".join(profiles.keys())
out_path = OUTPUT_DIR / f"cluster_map_{eq_ids}_{ts}.xlsx"

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    df_clusters.to_excel(writer, sheet_name="cluster_map", index=False)
    df_summary.to_excel(writer, sheet_name="summary", index=False)

    # 열 너비 자동 조정
    for sheet_name, df in [("cluster_map", df_clusters), ("summary", df_summary)]:
        ws = writer.sheets[sheet_name]
        for col_cells in ws.columns:
            max_len = max(
                (len(str(cell.value)) if cell.value is not None else 0)
                for cell in col_cells
            )
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 60)


print(f"\n저장 완료: {out_path}")
print(f"  cluster_map : {len(df_clusters):,}행")
print(f"  summary     : {len(df_summary):,}행")
print(f"\n[summary 미리보기]")
print(df_summary.to_string(index=False))
