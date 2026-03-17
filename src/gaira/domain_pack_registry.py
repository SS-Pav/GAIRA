from __future__ import annotations

from pathlib import Path

import yaml

from gaira.config import get_project_root


def get_domain_pack_registry_path() -> Path:
    return get_project_root() / "config" / "domain_pack_registry.yaml"


def load_domain_pack_registry() -> dict:
    registry_path = get_domain_pack_registry_path()
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Domain-pack registry not found: {registry_path}. Create config/domain_pack_registry.yaml first."
        )

    with registry_path.open("r", encoding="utf-8") as handle:
        registry = yaml.safe_load(handle) or {}

    registry.setdefault("packs", {})
    return registry


def list_domain_packs() -> list[dict]:
    registry = load_domain_pack_registry()
    rows = []
    for pack_id, entry in registry.get("packs", {}).items():
        row = dict(entry)
        row["pack_id"] = pack_id
        rows.append(row)
    return sorted(rows, key=lambda row: row["pack_id"])


def get_domain_pack(pack_id: str) -> dict:
    registry = load_domain_pack_registry()
    packs = registry.get("packs", {})
    if pack_id not in packs:
        raise KeyError(f"No domain pack is configured for pack_id='{pack_id}'.")
    entry = dict(packs[pack_id])
    entry["pack_id"] = pack_id
    return entry


def get_pack_datasets(pack_id: str) -> list[str]:
    entry = get_domain_pack(pack_id)
    return list(entry.get("datasets", []))


def get_pack_default_embedding(pack_id: str) -> str:
    entry = get_domain_pack(pack_id)
    return str(entry.get("default_embedding", ""))


def find_packs_for_dataset(dataset_id: str) -> list[str]:
    rows = []
    for entry in list_domain_packs():
        if dataset_id in entry.get("datasets", []):
            rows.append(entry["pack_id"])
    return sorted(rows)


def list_shared_datasets() -> list[dict]:
    dataset_to_packs: dict[str, list[str]] = {}
    for entry in list_domain_packs():
        for dataset_id in entry.get("datasets", []):
            dataset_to_packs.setdefault(dataset_id, []).append(entry["pack_id"])

    rows = []
    for dataset_id, pack_ids in sorted(dataset_to_packs.items()):
        if len(pack_ids) > 1:
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "pack_ids": sorted(pack_ids),
                    "pack_count": len(pack_ids),
                }
            )
    return rows
