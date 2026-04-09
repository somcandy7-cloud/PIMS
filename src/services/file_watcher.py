import logging
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler, FileCreatedEvent
from watchdog.observers import Observer

logger = logging.getLogger("pims.watcher")


class CsvHandler(FileSystemEventHandler):
    """신규 CSV 파일 감지 시 파이프라인을 실행한다."""

    def __init__(self, pipeline_fn, output_folder: str):
        self.pipeline_fn = pipeline_fn
        self.output_folder = output_folder

    def on_created(self, event: FileCreatedEvent) -> None:
        if not isinstance(event, FileCreatedEvent):
            return
        path = Path(event.src_path)
        if path.suffix.lower() != ".csv":
            return
        logger.info(f"신규 CSV 감지: {path}")
        try:
            self.pipeline_fn(str(path), self.output_folder)
        except Exception as exc:
            logger.error(f"파이프라인 오류 ({path.name}): {exc}", exc_info=True)


class FolderWatcher:
    def __init__(self, watch_folder: str, pipeline_fn, output_folder: str):
        self.watch_folder = watch_folder
        self.handler = CsvHandler(pipeline_fn, output_folder)
        self.observer = Observer()

    def start(self) -> None:
        Path(self.watch_folder).mkdir(parents=True, exist_ok=True)
        self.observer.schedule(self.handler, self.watch_folder, recursive=False)
        self.observer.start()
        logger.info(f"폴더 감시 시작: {self.watch_folder}")

    def stop(self) -> None:
        self.observer.stop()
        self.observer.join()
        logger.info("폴더 감시 중지")

    def run_forever(self) -> None:
        self.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
