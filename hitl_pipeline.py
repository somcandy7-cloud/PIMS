# hitl_pipeline.py
"""HITL(Human-in-the-Loop) 능동 학습 파이프라인.

사용법:
  python hitl_pipeline.py --file 2603201549_oven.csv
  python hitl_pipeline.py --file 2603201549_oven.csv --no-interactive
  python hitl_pipeline.py --retrain-status
  python hitl_pipeline.py --show-profile oven
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).parent


def _load_settings() -> dict:
    with open(BASE_DIR / "config" / "settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_llm_prompt(template: str, payload: dict) -> str:
    top_signals_text = "\n".join(
        f"  - {s['name']}: 변동성={s['variability']:.4f}"
        for s in payload["top_signals"]
    )
    context_stats_text = "\n".join(
        f"  {col}: mean={v['mean']:.2f}, std={v['std']:.2f}, "
        f"범위=[{v['min']:.2f}~{v['max']:.2f}], trend={v['trend']:+.2f}"
        for col, v in payload["context_stats"].items()
    )
    return template.format(
        timestamp=payload["timestamp"],
        score_level=payload["score_level"],
        score=payload["score"],
        top_signals=top_signals_text,
        context_window_minutes=payload["context_window_minutes"],
        context_stats=context_stats_text,
    )


def _get_operation_conditions(df, equipment_id: str, hitl_cfg: dict):
    """캐시 hit → 바로 반환, miss → AutoDiscovery 후 관리자 확인."""
    from src.services.equipment_profile_store import EquipmentProfileStore
    from src.utils.operation_discovery import AutoOperationDiscovery

    disc_cfg = hitl_cfg.get("discovery", {})
    store = EquipmentProfileStore()

    cached = store.load(equipment_id)
    if cached and store.is_confirmed(equipment_id):
        print(f"  [프로파일 캐시 hit] '{equipment_id}' — 저장된 조건 {len(cached)}개 적용")
        return cached

    print(f"  [프로파일 없음] '{equipment_id}' — 가동 신호 자동 탐색 중...")
    discoverer = AutoOperationDiscovery(
        top_binary_n=int(disc_cfg.get("top_binary_n", 3)),
        top_bimodal_n=int(disc_cfg.get("top_bimodal_n", 3)),
    )
    conditions = discoverer.discover(df)

    if not conditions:
        print("  [경고] 가동 판별 후보 신호를 찾지 못했습니다. 전체 데이터로 진행합니다.")
        return []

    print(f"\n  발견된 가동 판별 후보 조건 ({len(conditions)}개):")
    for i, c in enumerate(conditions, 1):
        print(f"    {i}. [{c.source}] {c.column} {c.op} {c.value:.2f}  (신뢰도 {c.confidence:.2f})")

    ans = input("\n  이 조건들로 가동 상태를 판별하겠습니다. 확인하시겠습니까? [Y=확인 / N=건너뜀]: ").strip().upper()
    confirmed = ans == "Y"
    store.save(equipment_id, conditions, confirmed=confirmed)

    if confirmed:
        print(f"  → '{equipment_id}' 프로파일 저장 완료.")
    else:
        print("  → 미확인 상태로 저장. 이번 실행에서는 발견된 조건을 그대로 사용합니다.")

    return conditions


def _get_group_conditions(
    group_df,
    equipment_id: str,
    group_id: str,
    hitl_cfg: dict,
    interactive: bool,
    store,                 # EquipmentProfileStore 인스턴스 주입 (루프마다 재생성 방지)
):
    """그룹별 가동 조건을 캐시에서 로드하거나 자동 탐색한다."""
    from src.utils.operation_discovery import AutoOperationDiscovery

    disc_cfg = hitl_cfg.get("discovery", {})

    cached = store.load_group(equipment_id, group_id)
    if cached is not None and store.is_group_confirmed(equipment_id, group_id):
        print(f"    [캐시 hit] '{group_id}' — 저장된 조건 {len(cached)}개")
        return cached

    discoverer = AutoOperationDiscovery(
        top_binary_n=int(disc_cfg.get("top_binary_n", 1)),
        top_bimodal_n=int(disc_cfg.get("top_bimodal_n", 1)),
    )
    conditions = discoverer.discover(group_df)

    if not conditions:
        store.save_group(equipment_id, group_id, [], confirmed=False)
        return []

    if interactive:
        print(f"\n    [{group_id}] 발견된 가동 조건 ({len(conditions)}개):")
        for i, c in enumerate(conditions, 1):
            print(f"      {i}. [{c.source}] {c.column} {c.op} {c.value:.2f}  (신뢰도={c.confidence:.2f})")
        ans = input(f"\n    '{group_id}' 조건 확정? [Y/N/S(건너뜀)]: ").strip().upper()
        confirmed = (ans == "Y")
        if ans == "S":
            return []
    else:
        confirmed = False

    store.save_group(equipment_id, group_id, conditions, confirmed=confirmed)
    return conditions


def _interactive_feedback(event, context: dict, store) -> None:
    print("\n" + "=" * 70)
    print(context["text_summary"])
    print("=" * 70)

    while True:
        ans = input("\n이 탐지 결과가 실제 이상입니까? [O=이상 확정 / X=정상 패턴 / S=건너뜀]: ").strip().upper()
        if ans in ("O", "X", "S"):
            break
        print("  O, X, S 중 하나를 입력하세요.")

    if ans == "S":
        print("  → 건너뜀.")
        return

    reason = input("  판단 근거를 입력하세요 (엔터=생략): ").strip()
    store.save(event, label=ans, reason=reason)
    print(f"  → '{ans}' 피드백 저장 완료.")


def run_hitl_pipeline(csv_path: str, interactive: bool = True) -> list[dict]:
    from src.services.loader import IbaCSVLoader
    from src.utils.preprocessor import Preprocessor
    from src.agents.llm_filter import LLMFilter
    from src.agents.llm_backends import build_llm_backend
    from src.utils.context_formatter import AnomalyContextFormatter
    from src.services.feedback_store import FeedbackStore
    from src.services.equipment_profile_store import EquipmentProfileStore

    settings  = _load_settings()
    hitl_cfg  = settings.get("hitl", {})
    equipment_id = EquipmentProfileStore.extract_id(Path(csv_path).name)

    # Step 1: 로드 + 수치 전처리
    print(f"\n[Step 1] CSV 로드: {csv_path}")
    df_raw = Preprocessor().process(IbaCSVLoader().load(csv_path))
    print(f"  전체: {len(df_raw)}행 × {len(df_raw.columns)}열")

    # Step 1-B ~ 2: 그룹별 독립 파이프라인
    from src.utils.device_group_parser import DeviceGroupParser
    from src.utils.signal_reducer import SignalReducer
    from src.utils.operation_filter import OperationFilter
    from src.utils.rolling_features import RollingFeatureExtractor
    from src.agents.isolation_forest_adapter import IsolationForestAdapter

    ep_store = EquipmentProfileStore()   # 루프 밖에서 한 번만 생성
    groups = DeviceGroupParser.parse(df_raw.columns.tolist())
    reducer_cfg = hitl_cfg
    det_cfg = hitl_cfg.get("detector", {})
    roll_cfg = hitl_cfg.get("rolling", {})

    all_candidates: list = []

    print(f"\n[Step 1-B~2] 장치 그룹 {len(groups)}개 개별 분석")

    for group_id, group_cols in groups.items():
        if len(group_cols) < 5:
            # 신호가 5개 미만인 소형 그룹은 건너뜀
            continue

        group_df = df_raw[group_cols]
        print(f"\n  ── 그룹: {group_id} ({len(group_cols)}개 신호) ──")

        # 가동 조건 확보 (ep_store 주입으로 루프마다 재생성 방지)
        conditions = _get_group_conditions(
            group_df, equipment_id, group_id, hitl_cfg, interactive, ep_store
        )

        # 가동 구간 필터
        df_running = OperationFilter(conditions).filter(group_df) if conditions else group_df.copy()
        pct = len(df_running) / max(len(group_df), 1) * 100
        print(f"    가동 구간: {len(df_running)}행 ({pct:.1f}%)")

        if df_running.empty:
            continue

        # SignalReducer
        cached_clusters = ep_store.load_group_clusters(equipment_id, group_id)
        sr = SignalReducer.from_config(reducer_cfg)
        if cached_clusters is not None:
            rep_cols = [c for c in cached_clusters if c in df_running.columns]
            df_reduced = df_running[rep_cols].copy()
        else:
            df_reduced, cluster_map = sr.fit_transform(df_running)
            ep_store.save_group_clusters(equipment_id, group_id, cluster_map)
        print(f"    축소: {len(group_cols)}→{len(df_reduced.columns)}열")

        if df_reduced.empty:
            continue

        # Rolling 특징 + IF 탐지
        df_feat = RollingFeatureExtractor(
            window=int(roll_cfg.get("window_rows", 30))
        ).transform(df_reduced)

        candidates = IsolationForestAdapter(
            window_size=int(det_cfg.get("window_size", 1)),
            contamination=float(det_cfg.get("contamination", 0.02)),
            n_estimators=int(det_cfg.get("n_estimators", 100)),
            top_n=int(det_cfg.get("top_n", 5)),
        ).detect(df_feat)

        # 그룹 태깅
        for ev in candidates:
            ev.metadata["group_id"] = group_id

        print(f"    IF 후보: {len(candidates)}건")
        all_candidates.extend(candidates)

    candidates = all_candidates
    print(f"\n  전체 IF 후보: {len(candidates)}건")

    if not candidates:
        print("  이상 후보 없음.")
        return []

    # Step 2-B: LLM 1차 자동 필터
    backend = build_llm_backend(settings.get("llm", {}))
    if backend:
        print(f"\n[Step 2-B] LLM 1차 필터 ({backend.name()})...")
        events = LLMFilter(backend).filter(candidates, df_raw)
        print(f"  LLM KEEP: {len(events)}건")
    else:
        events = candidates
        print("  LLM 비활성화 — 전체 후보를 관리자 검토로 전달")

    # Step 3: 컨텍스트 포맷팅
    print(f"\n[Step 3] 컨텍스트 포맷팅 (±{hitl_cfg.get('context_minutes', 5)}분)")
    formatter = AnomalyContextFormatter(
        context_minutes=int(hitl_cfg.get("context_minutes", 5))
    )
    contexts = []
    for event in events:
        ctx = formatter.format(event, df_raw)
        ctx["event"]   = event
        ctx["payload"] = json.loads(ctx["json_payload"])
        contexts.append(ctx)

    # Step 4: 관리자 피드백 루프
    fb_store = FeedbackStore(
        db_path=str(BASE_DIR / hitl_cfg.get("feedback_db", "feedback.db"))
    )

    if interactive:
        print(f"\n[Step 4] 관리자 피드백 루프 ({len(events)}건)")
        for i, ctx in enumerate(contexts, 1):
            print(f"\n  [{i}/{len(events)}]", end="")
            _interactive_feedback(ctx["event"], ctx, fb_store)

        status = fb_store.retrain_scaffold()
        print(f"\n[재학습 상태] {status['message']}")
    else:
        for ctx in contexts:
            print(ctx["text_summary"])

    return contexts


def main() -> None:
    parser = argparse.ArgumentParser(description="PIMS HITL 파이프라인")
    parser.add_argument("--file",            help="분석할 CSV 파일 경로")
    parser.add_argument("--no-interactive",  action="store_true")
    parser.add_argument("--retrain-status",  action="store_true")
    parser.add_argument("--show-profile",    metavar="EQUIPMENT_ID")
    args = parser.parse_args()

    settings = _load_settings()
    hitl_cfg = settings.get("hitl", {})

    if args.retrain_status:
        from src.services.feedback_store import FeedbackStore
        status = FeedbackStore(
            db_path=str(BASE_DIR / hitl_cfg.get("feedback_db", "feedback.db"))
        ).retrain_scaffold()
        print(json.dumps(status, ensure_ascii=False, indent=2))

    elif args.show_profile:
        from src.services.equipment_profile_store import EquipmentProfileStore
        store = EquipmentProfileStore()
        conds = store.load(args.show_profile)
        if conds is None:
            print(f"'{args.show_profile}' 프로파일 없음.")
        else:
            confirmed = store.is_confirmed(args.show_profile)
            print(f"설비: {args.show_profile}  (confirmed={confirmed})")
            for c in conds:
                print(f"  {c.column} {c.op} {c.value:.2f}  [{c.source}, conf={c.confidence:.2f}]")

    elif args.file:
        run_hitl_pipeline(args.file, interactive=not args.no_interactive)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
