"""Watchdog 파일 감시 테스트."""
import time
from pathlib import Path

import pandas as pd
import pytest

from src.services.file_watcher import FolderWatcher


def _wait_for(condition, timeout=3.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return False


def test_watcher_calls_pipeline_on_csv(tmp_path):
    received: list[str] = []

    def fake_pipeline(csv_path: str, output_folder: str):
        received.append(csv_path)

    watcher = FolderWatcher(str(tmp_path), fake_pipeline, str(tmp_path / "out"))
    watcher.start()
    try:
        csv_file = tmp_path / "test_data.csv"
        csv_file.write_text("a;b\n1;2\n", encoding="utf-8")
        assert _wait_for(lambda: len(received) >= 1), "파이프라인이 호출되지 않음"
        assert received[0].endswith("test_data.csv")
    finally:
        watcher.stop()


def test_watcher_ignores_non_csv(tmp_path):
    received: list[str] = []

    def fake_pipeline(csv_path: str, output_folder: str):
        received.append(csv_path)

    watcher = FolderWatcher(str(tmp_path), fake_pipeline, str(tmp_path / "out"))
    watcher.start()
    try:
        (tmp_path / "test_data.txt").write_text("hello", encoding="utf-8")
        time.sleep(0.3)
        assert len(received) == 0
    finally:
        watcher.stop()
