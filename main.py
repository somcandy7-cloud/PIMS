"""PIMS 장애 분석 헬퍼 진입점.

사용법:
  python main.py                         # 폴더 감시 데몬 실행
  python main.py --file <csv경로>        # 단일 파일 즉시 분석
  python main.py --export <csv경로>      # CSV → Excel 변환만 수행
  python main.py --export-all            # 프로젝트 내 모든 CSV → Excel 변환
"""

import argparse
import logging
import sys
from pathlib import Path

# Windows 터미널 CP949에서도 한글·특수문자 출력 가능하게
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import yaml

BASE_DIR = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "config"


def load_settings() -> dict:
    with open(CONFIG_DIR / "settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_label_mapper():
    """label_files.yaml이 있으면 SignalLabelMapper를 반환한다."""
    from src.services.signal_label_mapper import SignalLabelMapper
    cfg_path = CONFIG_DIR / "label_files.yaml"
    if not cfg_path.exists():
        return None
    with open(cfg_path, encoding="utf-8") as f:
        import yaml as _yaml
        cfg = _yaml.safe_load(f)
    paths = [BASE_DIR / p for p in (cfg.get("label_files") or [])]
    existing = [p for p in paths if p.exists()]
    return SignalLabelMapper(existing) if existing else None


def export_csv_to_excel(csv_path: str, output_path: str | None = None) -> str:
    """CSV 한 파일을 파싱해 Excel로 저장하고 저장 경로를 반환한다."""
    from src.services.excel_exporter import csv_to_excel
    mapper = _build_label_mapper()
    out = csv_to_excel(csv_path, output_path=output_path, label_mapper=mapper)
    print(f"Excel 저장 완료: {out}")
    return out


def run_pipeline(csv_path: str, output_folder: str) -> None:
    """CSV 한 파일에 대해 전체 분석 파이프라인 실행 (Excel 자동 export 포함)."""
    from src.services.loader import IbaCSVLoader
    from src.utils.preprocessor import Preprocessor
    from src.agents.trip_detector import TripDetector
    from src.agents.fault_classifier import FaultClassifier
    from src.services.notifier import notify
    from src.utils.template_reporter import generate

    settings = load_settings()
    ana = settings.get("analysis", {})

    logger = logging.getLogger("pims.pipeline")
    logger.info(f"분석 시작: {csv_path}")

    # 0. CSV → Excel 자동 변환
    output_dir = Path(output_folder)
    stem = Path(csv_path).stem
    excel_path = str(output_dir / f"{stem}.xlsx")
    try:
        export_csv_to_excel(csv_path, excel_path)
    except Exception as exc:
        logger.warning(f"Excel 변환 실패 (분석은 계속): {exc}")

    # 1. 로드 + 전처리
    df = IbaCSVLoader().load(csv_path)
    df = Preprocessor().process(df)

    if df.empty:
        logger.warning("데이터 없음, 건너뜀")
        return

    # 2. Trip 탐지
    detector = TripDetector(
        window=ana.get("trip_window", 50),
        roc_zscore_threshold=ana.get("trip_roc_zscore_threshold", 3.0),
        top_n=ana.get("top_n_signals", 5),
    )
    events = detector.detect(df)

    if not events:
        logger.info("Trip 이벤트 없음")
        return

    logger.info(f"Trip {len(events)}건 탐지")

    # 3. 분류 + 리포트 + 알람
    classifier = FaultClassifier()
    log_file = settings.get("log_file", str(output_dir / "alarm.log"))

    for i, event in enumerate(events):
        report = classifier.classify(event)
        out_path = str(output_dir / f"{stem}_trip{i+1:02d}.txt")
        text = generate(event, report, out_path)
        print(text)
        notify(report, log_file)

    logger.info(f"리포트 저장 완료: {output_folder}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PIMS 장애 분석 헬퍼")
    parser.add_argument("--file", help="단일 CSV 파일 즉시 분석 (Excel 자동 변환 포함)")
    parser.add_argument("--export", metavar="CSV", help="CSV → Excel 변환만 수행")
    parser.add_argument("--export-all", action="store_true",
                        help="프로젝트 루트의 모든 CSV를 Excel로 변환")
    args = parser.parse_args()

    settings = load_settings()
    log_file = settings.get("log_file", "alarm.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

    output_folder = settings.get("output_folder", "reports")
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    if args.export:
        export_csv_to_excel(args.export)

    elif args.export_all:
        csv_files = list(BASE_DIR.glob("*.csv"))
        if not csv_files:
            print("변환할 CSV 파일이 없습니다.")
        for csv_path in sorted(csv_files):
            export_csv_to_excel(str(csv_path))

    elif args.file:
        run_pipeline(args.file, output_folder)

    else:
        from src.services.file_watcher import FolderWatcher
        watch_folder = settings.get("watch_folder", "watch")
        watcher = FolderWatcher(watch_folder, run_pipeline, output_folder)
        print(f"폴더 감시 시작: {watch_folder}  (종료: Ctrl+C)")
        watcher.run_forever()


if __name__ == "__main__":
    main()
