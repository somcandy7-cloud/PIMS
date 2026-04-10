"""전체 신호 레지스트리를 생성해 outputs/signal_registry.xlsx 에 저장한다.

신호명이 변하지 않는 한 한 번만 실행하면 된다.

실행:
    python scripts/build_signal_registry.py

출력:
    outputs/signal_registry.xlsx

시트 구성:
  Sheet1 signals   : 신호 전체 목록 (타입 + 라벨 + 클러스터)
  Sheet2 unmapped  : xlsx에서 매핑 못 찾은 신호 목록 (추가 라벨링 참고용)
  Sheet3 summary   : 그룹별 / 타입별 통계
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.services.loader import IbaCSVLoader
from src.services.signal_label_mapper import SignalLabelMapper
from src.utils.device_group_parser import DeviceGroupParser
from src.utils.signal_type_filter import classify_signal

# ── 설정 ─────────────────────────────────────────────────────────────────────

CSV_DIR      = ROOT / "반출데이터"
PROFILES     = ROOT / "config" / "equipment_profiles.yaml"
LABEL_FILES  = [ROOT / "반출데이터" / "3OVEN.xlsx",
                ROOT / "반출데이터" / "4OVEN.xlsx"]
OUTPUT_PATH  = ROOT / "outputs" / "signal_registry.xlsx"
OUTPUT_PATH.parent.mkdir(exist_ok=True)

# ── 1. 대표 CSV 한 개에서 전체 컬럼 목록 추출 ────────────────────────────────

csv_files = sorted(CSV_DIR.glob("*.csv"))
if not csv_files:
    print("ERROR: 반출데이터/ 에 CSV 파일이 없습니다.")
    sys.exit(1)

sample_csv = csv_files[0]
print(f"신호 목록 기준 CSV: {sample_csv.name}")

df_raw = IbaCSVLoader().load(str(sample_csv))
all_cols: list[str] = df_raw.columns.tolist()
print(f"전체 신호 수: {len(all_cols):,}")

# ── 2. 라벨 매퍼 ─────────────────────────────────────────────────────────────

existing = [p for p in LABEL_FILES if p.exists()]
mapper   = SignalLabelMapper(existing) if existing else None
print(f"라벨 매퍼: {mapper.size if mapper else 0}개 항목 ({[p.name for p in existing]})")

# ── 3. 클러스터 맵 구축 (profiles.yaml → signal → rep 역매핑) ────────────────

with open(PROFILES, encoding="utf-8") as f:
    profiles: dict = yaml.safe_load(f) or {}

# signal → (equipment_id, group_id, cluster_rep, cluster_size)
cluster_lookup: dict[str, tuple[str, str, str, int]] = {}

for eq_id, eq_data in profiles.items():
    for group_id, gdata in (eq_data.get("groups") or {}).items():
        clusters = gdata.get("clusters") or {}
        for rep, members in clusters.items():
            size = len(members)
            for sig in members:
                cluster_lookup[sig] = (eq_id, group_id, rep, size)

# ── 4. 신호별 레지스트리 행 생성 ─────────────────────────────────────────────

device_groups: dict[str, list[str]] = DeviceGroupParser.parse(all_cols)
# signal → group_id 역매핑
col_to_group: dict[str, str] = {
    col: gid
    for gid, cols in device_groups.items()
    for col in cols
}

rows: list[dict] = []
for col in all_cols:
    sig_type  = classify_signal(col)
    group_id  = col_to_group.get(col, "__default__")

    label, desc = "", ""
    mapped = False
    if mapper:
        hit = mapper.lookup(col)
        if hit and (hit[0] or hit[1]):
            label, desc = hit[0], hit[1]
            mapped = True

    cl_info   = cluster_lookup.get(col)
    cl_eq     = cl_info[0] if cl_info else ""
    cl_group  = cl_info[1] if cl_info else ""
    cl_rep    = cl_info[2] if cl_info else ""
    cl_size   = cl_info[3] if cl_info else 0
    is_rep    = (col == cl_rep) if cl_rep else False

    # 샘플 데이터 특성
    series    = df_raw[col].dropna()
    n_unique  = int(series.nunique())
    val_min   = round(float(series.min()), 4) if len(series) > 0 else None
    val_max   = round(float(series.max()), 4) if len(series) > 0 else None
    val_mean  = round(float(series.mean()), 4) if len(series) > 0 else None

    rows.append({
        "signal"         : col,
        "group_id"       : group_id,
        "signal_type"    : sig_type,
        "label"          : label,
        "description"    : desc,
        "mapped"         : mapped,
        "cluster_eq"     : cl_eq,
        "cluster_group"  : cl_group,
        "cluster_rep"    : cl_rep,
        "is_rep"         : is_rep,
        "cluster_size"   : cl_size,
        "n_unique_values": n_unique,
        "val_min"        : val_min,
        "val_max"        : val_max,
        "val_mean"       : val_mean,
    })

df_signals = pd.DataFrame(rows)

# ── 5. 미매핑 신호 (measurement 타입인데 라벨 없는 것) ──────────────────────

df_unmapped = df_signals[
    (df_signals["signal_type"] == "measurement") &
    (~df_signals["mapped"])
][["signal", "group_id", "cluster_rep", "n_unique_values", "val_min", "val_max"]].copy()

# ── 6. 요약 통계 ─────────────────────────────────────────────────────────────

grp_summary = (
    df_signals
    .groupby(["group_id", "signal_type"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

type_total = df_signals["signal_type"].value_counts().reset_index()
type_total.columns = ["signal_type", "count"]

mapped_total = df_signals["mapped"].value_counts()
n_mapped     = int(mapped_total.get(True, 0))
n_total      = len(df_signals)

print(f"\n=== 결과 요약 ===")
print(f"전체 신호: {n_total:,}")
for t, cnt in df_signals["signal_type"].value_counts().items():
    print(f"  {t}: {cnt:,}")
print(f"xlsx 매핑 성공: {n_mapped:,} ({n_mapped/n_total*100:.1f}%)")
print(f"measurement 중 미매핑: {len(df_unmapped):,}")

# ── 7. Excel 저장 ────────────────────────────────────────────────────────────

with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
    # Sheet1: 전체 신호
    df_signals.sort_values(
        ["group_id", "signal_type", "cluster_rep", "signal"]
    ).to_excel(writer, sheet_name="signals", index=False)

    # Sheet2: measurement 미매핑
    df_unmapped.sort_values(["group_id", "signal"]).to_excel(
        writer, sheet_name="unmapped_measurements", index=False
    )

    # Sheet3: 그룹별 통계
    grp_summary.to_excel(writer, sheet_name="summary", index=False)

    # 열 너비 조정
    for sheet_name, df in [
        ("signals", df_signals),
        ("unmapped_measurements", df_unmapped),
        ("summary", grp_summary),
    ]:
        ws = writer.sheets[sheet_name]
        for col_cells in ws.columns:
            max_len = max(
                len(str(c.value)) if c.value is not None else 0
                for c in col_cells
            )
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 60)

print(f"\n저장 완료: {OUTPUT_PATH}")
print("  Sheet1 'signals'                : 전체 신호 레지스트리")
print("  Sheet2 'unmapped_measurements'  : 라벨 없는 측정값 신호 (추가 라벨링 대상)")
print("  Sheet3 'summary'                : 그룹별 타입 통계")
