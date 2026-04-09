import pytest
from src.utils.device_group_parser import DeviceGroupParser


def test_bracketed_column_uses_prefix():
    groups = DeviceGroupParser.parse(["[3_CO_PLC_B]QB 42"])
    assert "3_CO_PLC_B" in groups
    assert "[3_CO_PLC_B]QB 42" in groups["3_CO_PLC_B"]


def test_unbracketed_column_goes_to_default():
    groups = DeviceGroupParser.parse(["MB    118", "MD    356_1"])
    assert DeviceGroupParser.DEFAULT_GROUP in groups
    assert len(groups[DeviceGroupParser.DEFAULT_GROUP]) == 2


def test_mixed_columns_split_correctly():
    cols = ["[3_CO_PLC_B]QB 42", "[3_ECS_PLC_B]MB    31", "MB    118"]
    groups = DeviceGroupParser.parse(cols)
    assert set(groups.keys()) == {"3_CO_PLC_B", "3_ECS_PLC_B", DeviceGroupParser.DEFAULT_GROUP}


def test_empty_columns_returns_empty():
    assert DeviceGroupParser.parse([]) == {}


def test_group_id_single_column():
    assert DeviceGroupParser.group_id("[3_CO_PLC_B]QB 42") == "3_CO_PLC_B"
    assert DeviceGroupParser.group_id("MB    118") == DeviceGroupParser.DEFAULT_GROUP


def test_multiple_signals_same_group():
    cols = ["[3_CO_PLC_B]QB 42", "[3_CO_PLC_B]QB 43", "[3_CO_PLC_B]MB    10"]
    groups = DeviceGroupParser.parse(cols)
    assert len(groups["3_CO_PLC_B"]) == 3
