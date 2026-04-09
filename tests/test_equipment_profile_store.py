# tests/test_equipment_profile_store.py
import pytest
from pathlib import Path
from src.utils.operation_discovery import OperationCondition
from src.services.equipment_profile_store import EquipmentProfileStore


@pytest.fixture
def store(tmp_path):
    return EquipmentProfileStore(profile_path=str(tmp_path / "profiles.yaml"))


def _conds() -> list[OperationCondition]:
    return [
        OperationCondition("ctrl", "==", 1.0, 0.9, "binary"),
        OperationCondition("speed", ">", 100_000.0, 0.8, "bimodal"),
    ]


def test_save_and_load(store):
    store.save("oven", _conds(), confirmed=True)
    loaded = store.load("oven")
    assert loaded is not None
    assert len(loaded) == 2
    assert loaded[0].column == "ctrl"


def test_load_unknown_returns_none(store):
    assert store.load("unknown_equipment") is None


def test_is_confirmed(store):
    store.save("oven", _conds(), confirmed=True)
    assert store.is_confirmed("oven") is True


def test_unconfirmed_profile(store):
    store.save("oven", _conds(), confirmed=False)
    assert store.is_confirmed("oven") is False


def test_overwrite_updates_profile(store):
    store.save("oven", _conds(), confirmed=False)
    new_conds = [OperationCondition("new_col", ">", 50.0, 0.95, "bimodal")]
    store.save("oven", new_conds, confirmed=True)
    loaded = store.load("oven")
    assert len(loaded) == 1
    assert loaded[0].column == "new_col"


def test_extract_equipment_id():
    assert EquipmentProfileStore.extract_id("2603201549_oven.csv") == "oven"
    assert EquipmentProfileStore.extract_id("2603211200_loco_A.csv") == "loco_A"
    assert EquipmentProfileStore.extract_id("no_timestamp.csv") == "no_timestamp"


def test_save_and_load_clusters(store):
    cluster_map = {
        "speed": ["speed", "speed_backup"],
        "temp":  ["temp"],
    }
    store.save_clusters("oven", cluster_map)
    loaded = store.load_clusters("oven")
    assert loaded is not None
    assert loaded["speed"] == ["speed", "speed_backup"]
    assert loaded["temp"] == ["temp"]


def test_load_clusters_unknown_returns_none(store):
    assert store.load_clusters("unknown") is None


def test_save_clusters_does_not_overwrite_conditions(store):
    store.save("oven", _conds(), confirmed=True)
    store.save_clusters("oven", {"a": ["a", "b"]})
    assert store.load("oven") is not None
    assert store.is_confirmed("oven") is True


# ── 그룹별 메서드 테스트 ────────────────────────────────────────────────────

def test_save_and_load_group(store):
    conds = [OperationCondition("[grp]QB 1", "==", 1.0, 0.9, "binary")]
    store.save_group("oven", "3_CO_PLC_B", conds, confirmed=False)
    loaded = store.load_group("oven", "3_CO_PLC_B")
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded[0].column == "[grp]QB 1"


def test_load_group_unknown_returns_none(store):
    assert store.load_group("oven", "nonexistent_group") is None


def test_is_group_confirmed(store):
    conds = [OperationCondition("c", "==", 1.0, 1.0, "binary")]
    store.save_group("oven", "g1", conds, confirmed=True)
    assert store.is_group_confirmed("oven", "g1") is True
    store.save_group("oven", "g2", conds, confirmed=False)
    assert store.is_group_confirmed("oven", "g2") is False


def test_save_group_does_not_affect_other_groups(store):
    conds = [OperationCondition("c", "==", 1.0, 1.0, "binary")]
    store.save_group("oven", "g1", conds, confirmed=True)
    store.save_group("oven", "g2", conds, confirmed=False)
    assert store.is_group_confirmed("oven", "g1") is True
    assert store.is_group_confirmed("oven", "g2") is False


def test_save_and_load_group_clusters(store):
    cm = {"rep": ["rep", "other"]}
    store.save_group_clusters("oven", "3_CO_PLC_B", cm)
    loaded = store.load_group_clusters("oven", "3_CO_PLC_B")
    assert loaded is not None
    assert loaded["rep"] == ["rep", "other"]


def test_load_group_clusters_unknown_returns_none(store):
    assert store.load_group_clusters("oven", "nonexistent") is None


def test_load_all_groups(store):
    conds = [OperationCondition("c", "==", 1.0, 1.0, "binary")]
    store.save_group("oven", "g1", conds, confirmed=True)
    store.save_group("oven", "g2", conds, confirmed=False)
    all_groups = store.load_all_groups("oven")
    assert set(all_groups.keys()) == {"g1", "g2"}


def test_group_methods_do_not_affect_top_level_conditions(store):
    """그룹 메서드가 기존 top-level save/load에 영향을 주면 안 된다."""
    store.save("oven", _conds(), confirmed=True)
    store.save_group("oven", "g1", [], confirmed=False)
    loaded = store.load("oven")
    assert loaded is not None
    assert len(loaded) == 2


def test_clear_group_clusters(store):
    """clear_group_clusters가 clusters만 제거하고 conditions는 보존한다."""
    conds = [OperationCondition("c", "==", 1.0, 1.0, "binary")]
    store.save_group("oven", "g1", conds, confirmed=True)
    store.save_group_clusters("oven", "g1", {"rep": ["rep"]})
    store.clear_group_clusters("oven", "g1")
    assert store.load_group_clusters("oven", "g1") is None
    # conditions 보존 확인
    assert store.load_group("oven", "g1") is not None
