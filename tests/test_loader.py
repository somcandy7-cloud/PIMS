import pandas as pd
import pytest
from src.services.schema import SignalSchema


def test_signal_schema_has_required_fields():
    schema = SignalSchema(name="Speed_Act", unit="m/min", signal_type="analog")
    assert schema.name == "Speed_Act"
    assert schema.signal_type == "analog"


def test_signal_schema_default_threshold():
    schema = SignalSchema(name="Current_1", unit="A", signal_type="analog")
    assert schema.threshold.high == float("inf")
    assert schema.threshold.low == float("-inf")


def test_signal_schema_digital_type():
    schema = SignalSchema(name="QB 0", unit="", signal_type="digital")
    assert schema.signal_type == "digital"


# ── IbaCSVLoader 테스트 (Task 3에서 구현) ──────────────────────────────────

def _make_iba_csv(tmp_path, signal_names: list[str], n_rows: int = 5) -> str:
    """실제 iba PDA CSV 포맷을 모사하는 테스트용 파일 생성"""
    sep = ";"
    header0 = sep.join(["Time"] + [f"[0:{i}]" for i in range(len(signal_names))])
    header1 = sep.join(["time"] + signal_names)
    unit_row = sep.join(["sec"] + [""] * len(signal_names))
    lines = [
        header0,
        header1,
        unit_row,
        "",
        "Group_imageIndex_0;-1",
        "LicenseCustomer;TestPlant",
        "",
    ]
    base = pd.Timestamp("2026-03-12T15:49:15+09:00")
    for i in range(n_rows):
        ts = (base + pd.Timedelta(milliseconds=i * 100)).isoformat()
        vals = sep.join([ts] + [str(float(i * 10)) for _ in signal_names])
        lines.append(vals)
    path = tmp_path / "test_iba.csv"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def test_loader_returns_dataframe(tmp_path):
    from src.services.loader import IbaCSVLoader
    path = _make_iba_csv(tmp_path, ["MD 100", "DB420.DBW 2"], n_rows=5)
    df = IbaCSVLoader().load(path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5


def test_loader_sets_timestamp_index(tmp_path):
    from src.services.loader import IbaCSVLoader
    path = _make_iba_csv(tmp_path, ["MD 100"], n_rows=3)
    df = IbaCSVLoader().load(path)
    assert isinstance(df.index, pd.DatetimeIndex)


def test_loader_columns_match_signal_names(tmp_path):
    from src.services.loader import IbaCSVLoader
    signals = ["MD 100", "DB420.DBW 2", "QB 0"]
    path = _make_iba_csv(tmp_path, signals, n_rows=3)
    df = IbaCSVLoader().load(path)
    for sig in signals:
        assert sig in df.columns
