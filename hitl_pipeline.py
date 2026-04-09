from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).parent


def _load_settings() -> dict:
    with open(BASE_DIR / "config" / "settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_group_conditions(
    group_df: pd.DataFrame,
    equipment_id: str,
    group_id: str,
    hitl_cfg: dict,
    interactive: bool,
    store,
):
    from src.utils.operation_discovery import AutoOperationDiscovery

    disc_cfg = hitl_cfg.get("discovery", {})

    cached = store.load_group(equipment_id, group_id)
    if cached is not None and store.is_group_confirmed(equipment_id, group_id):
        print(f"    [cache hit] {group_id}: {len(cached)} conditions")
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
        print(f"\n    [{group_id}] discovered conditions ({len(conditions)}):")
        for idx, cond in enumerate(conditions, 1):
            print(
                f"      {idx}. [{cond.source}] {cond.column} {cond.op} {cond.value:.2f} "
                f"(conf={cond.confidence:.2f})"
            )
        ans = input(f"\n    Confirm conditions for '{group_id}'? [Y/N/S(skip)]: ").strip().upper()
        if ans == "S":
            return []
        confirmed = ans == "Y"
    else:
        confirmed = False

    store.save_group(equipment_id, group_id, conditions, confirmed=confirmed)
    return conditions


def _interactive_feedback(event, context: dict, store) -> None:
    print("\n" + "=" * 70)
    print(context["text_summary"])
    print("=" * 70)

    while True:
        ans = input("\nIs this event a real anomaly? [O=anomaly / X=normal / S=skip]: ").strip().upper()
        if ans in ("O", "X", "S"):
            break
        print("  Please type O, X, or S.")

    if ans == "S":
        print("  skipped")
        return

    reason = input("  reason (optional): ").strip()
    store.save(event, label=ans, reason=reason)
    print(f"  saved feedback: {ans}")


def run_hitl_pipeline(csv_path: str, interactive: bool = True) -> list[dict]:
    from src.services.loader import IbaCSVLoader
    from src.utils.preprocessor import Preprocessor
    from src.agents.llm_filter import LLMFilter
    from src.agents.llm_backends import build_llm_backend
    from src.utils.context_formatter import AnomalyContextFormatter
    from src.services.feedback_store import FeedbackStore
    from src.services.equipment_profile_store import EquipmentProfileStore
    from src.utils.device_group_parser import DeviceGroupParser
    from src.utils.signal_reducer import SignalReducer
    from src.utils.operation_filter import OperationFilter
    from src.utils.rolling_features import RollingFeatureExtractor
    from src.agents.isolation_forest_adapter import IsolationForestAdapter

    settings = _load_settings()
    hitl_cfg = settings.get("hitl", {})
    det_cfg = hitl_cfg.get("detector", {})
    roll_cfg = hitl_cfg.get("rolling", {})

    equipment_id = EquipmentProfileStore.extract_id(Path(csv_path).name)

    print(f"\n[Step 1] load csv: {csv_path}")
    df_raw = Preprocessor().process(IbaCSVLoader().load(csv_path))
    print(f"  rows={len(df_raw):,}, cols={len(df_raw.columns):,}")

    ep_store = EquipmentProfileStore()
    groups = DeviceGroupParser.parse(df_raw.columns.tolist())
    print(f"\n[Step 1-B~2] analyze per group: {len(groups)} groups")

    all_candidates = []
    all_feature_frames: list[pd.DataFrame] = []

    for group_id, group_cols in groups.items():
        if len(group_cols) < 5:
            continue

        group_df = df_raw[group_cols]
        print(f"\n  -> group={group_id} signals={len(group_cols)}")

        conditions = _get_group_conditions(
            group_df, equipment_id, group_id, hitl_cfg, interactive, ep_store
        )

        df_running = OperationFilter(conditions).filter(group_df) if conditions else group_df.copy()
        pct = len(df_running) / max(len(group_df), 1) * 100
        print(f"     running rows={len(df_running)} ({pct:.1f}%)")

        if df_running.empty:
            continue

        cached_clusters = ep_store.load_group_clusters(equipment_id, group_id)
        reducer = SignalReducer.from_config(hitl_cfg)
        if cached_clusters is not None:
            rep_cols = [col for col in cached_clusters if col in df_running.columns]
            df_reduced = df_running[rep_cols].copy()
        else:
            df_reduced, cluster_map = reducer.fit_transform(df_running)
            ep_store.save_group_clusters(equipment_id, group_id, cluster_map)

        print(f"     reduced signals={len(df_reduced.columns)}")

        if df_reduced.empty:
            continue

        df_feat = RollingFeatureExtractor(
            window=int(roll_cfg.get("window_rows", 30))
        ).transform(df_reduced)
        all_feature_frames.append(df_feat)

        candidates = IsolationForestAdapter(
            window_size=int(det_cfg.get("window_size", 1)),
            contamination=float(det_cfg.get("contamination", 0.02)),
            n_estimators=int(det_cfg.get("n_estimators", 100)),
            top_n=int(det_cfg.get("top_n", 5)),
        ).detect(df_feat)

        for event in candidates:
            event.metadata["group_id"] = group_id

        print(f"     IF candidates={len(candidates)}")
        all_candidates.extend(candidates)

    candidates = all_candidates
    print(f"\n  total IF candidates={len(candidates)}")

    if not candidates:
        print("  no anomalies")
        return []

    if all_feature_frames:
        llm_input_df = pd.concat(all_feature_frames, axis=1)
        llm_input_df = llm_input_df.loc[:, ~llm_input_df.columns.duplicated()]
        llm_input_df = llm_input_df.sort_index()
    else:
        llm_input_df = df_raw

    backend = build_llm_backend(settings.get("llm", {}))
    if backend:
        print(f"\n[Step 2-B] LLM filter: {backend.name()}")
        events = LLMFilter(backend).filter(candidates, llm_input_df)
        print(f"  LLM keep={len(events)}")
    else:
        events = candidates
        print("  LLM disabled: passing all IF candidates")

    formatter = AnomalyContextFormatter(
        context_minutes=int(hitl_cfg.get("context_minutes", 5))
    )
    contexts = []
    for event in events:
        ctx = formatter.format(event, llm_input_df)
        ctx["event"] = event
        ctx["payload"] = json.loads(ctx["json_payload"])
        contexts.append(ctx)

    fb_store = FeedbackStore(
        db_path=str(BASE_DIR / hitl_cfg.get("feedback_db", "feedback.db"))
    )

    if interactive:
        print(f"\n[Step 4] feedback loop: {len(events)} events")
        for i, ctx in enumerate(contexts, 1):
            print(f"\n  [{i}/{len(events)}]", end="")
            _interactive_feedback(ctx["event"], ctx, fb_store)

        status = fb_store.retrain_scaffold()
        print(f"\n[retrain status] {status['message']}")
    else:
        for ctx in contexts:
            print(ctx["text_summary"])

    return contexts


def main() -> None:
    parser = argparse.ArgumentParser(description="PIMS HITL pipeline")
    parser.add_argument("--file", help="CSV file path")
    parser.add_argument("--no-interactive", action="store_true")
    parser.add_argument("--retrain-status", action="store_true")
    parser.add_argument("--show-profile", metavar="EQUIPMENT_ID")
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
        groups = store.load_all_groups(args.show_profile)
        if not groups:
            print(f"no profile: {args.show_profile}")
        else:
            print(f"equipment={args.show_profile}, groups={len(groups)}")
            for gid, gdata in groups.items():
                conds = gdata.get("conditions") or []
                print(f"  - {gid}: conditions={len(conds)}, confirmed={bool(gdata.get('confirmed', False))}")

    elif args.file:
        run_hitl_pipeline(args.file, interactive=not args.no_interactive)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
