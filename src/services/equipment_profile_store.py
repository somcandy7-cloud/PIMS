from __future__ import annotations
import re
from pathlib import Path
import yaml
from src.utils.operation_discovery import OperationCondition

_DEFAULT_PATH = Path(__file__).parent.parent.parent / "config" / "equipment_profiles.yaml"
_TS_PREFIX_RE = re.compile(r"^\d{10}_(.+)$")


class EquipmentProfileStore:
    def __init__(self, profile_path: str | Path | None = None):
        self.profile_path = Path(profile_path) if profile_path else _DEFAULT_PATH

    def load(self, equipment_id: str) -> list[OperationCondition] | None:
        data = self._read()
        profile = data.get(equipment_id)
        if profile is None:
            return None
        return [OperationCondition.from_dict(c) for c in profile["conditions"]]

    def save(self, equipment_id: str, conditions: list[OperationCondition], confirmed: bool = False) -> None:
        data = self._read()
        data[equipment_id] = {
            "confirmed": confirmed,
            "conditions": [c.to_dict() for c in conditions],
        }
        self._write(data)

    def is_confirmed(self, equipment_id: str) -> bool:
        data = self._read()
        profile = data.get(equipment_id)
        return bool(profile and profile.get("confirmed", False))

    def save_clusters(self, equipment_id: str, cluster_map: dict) -> None:
        """클러스터맵을 저장한다. 기존 conditions/confirmed는 보존한다."""
        data = self._read()
        if equipment_id not in data:
            data[equipment_id] = {"confirmed": False, "conditions": []}
        data[equipment_id]["clusters"] = cluster_map
        self._write(data)

    def load_clusters(self, equipment_id: str) -> dict | None:
        """저장된 클러스터맵을 반환한다. 없으면 None."""
        data = self._read()
        profile = data.get(equipment_id)
        if profile is None:
            return None
        return profile.get("clusters")

    # ── 그룹별 메서드 ──────────────────────────────────────────────────────────

    def save_group(
        self,
        equipment_id: str,
        group_id: str,
        conditions: list[OperationCondition],
        confirmed: bool = False,
    ) -> None:
        """그룹별 가동 조건을 저장한다. 다른 그룹에는 영향 없음."""
        data = self._read()
        eq = data.setdefault(equipment_id, {})
        groups = eq.setdefault("groups", {})
        existing = groups.get(group_id, {})
        groups[group_id] = {
            **existing,
            "confirmed": confirmed,
            "conditions": [c.to_dict() for c in conditions],
        }
        self._write(data)

    def load_group(
        self, equipment_id: str, group_id: str
    ) -> list[OperationCondition] | None:
        """그룹별 가동 조건을 반환한다. 없으면 None."""
        data = self._read()
        group = data.get(equipment_id, {}).get("groups", {}).get(group_id)
        if group is None:
            return None
        return [OperationCondition.from_dict(c) for c in group.get("conditions", [])]

    def is_group_confirmed(self, equipment_id: str, group_id: str) -> bool:
        data = self._read()
        group = data.get(equipment_id, {}).get("groups", {}).get(group_id)
        return bool(group and group.get("confirmed", False))

    def save_group_clusters(
        self, equipment_id: str, group_id: str, cluster_map: dict
    ) -> None:
        """그룹별 클러스터맵을 저장한다."""
        data = self._read()
        eq = data.setdefault(equipment_id, {})
        groups = eq.setdefault("groups", {})
        existing = groups.get(group_id, {})
        groups[group_id] = {**existing, "clusters": cluster_map}
        self._write(data)

    def load_group_clusters(
        self, equipment_id: str, group_id: str
    ) -> dict | None:
        """그룹별 클러스터맵을 반환한다. 없으면 None."""
        data = self._read()
        group = data.get(equipment_id, {}).get("groups", {}).get(group_id)
        if group is None:
            return None
        return group.get("clusters")

    def load_all_groups(self, equipment_id: str) -> dict[str, dict]:
        """장치의 모든 그룹 프로파일 {group_id: {confirmed, conditions, clusters}} 반환."""
        data = self._read()
        return data.get(equipment_id, {}).get("groups", {})

    def clear_group_clusters(self, equipment_id: str, group_id: str) -> None:
        """그룹 클러스터맵만 삭제한다. conditions/confirmed는 보존."""
        data = self._read()
        group = data.get(equipment_id, {}).get("groups", {}).get(group_id)
        if group is not None:
            group.pop("clusters", None)
            self._write(data)

    @staticmethod
    def extract_id(csv_filename: str) -> str:
        stem = Path(csv_filename).stem
        m = _TS_PREFIX_RE.match(stem)
        return m.group(1) if m else stem

    def _read(self) -> dict:
        if not self.profile_path.exists():
            return {}
        with open(self.profile_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _write(self, data: dict) -> None:
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.profile_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
