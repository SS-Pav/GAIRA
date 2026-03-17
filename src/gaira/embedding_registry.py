from __future__ import annotations

from pathlib import Path

import yaml

from gaira.config import get_project_root


def get_embedding_registry_path() -> Path:
    return get_project_root() / "config" / "embedding_registry.yaml"


def load_embedding_registry() -> dict:
    registry_path = get_embedding_registry_path()
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Embedding registry not found: {registry_path}. Create config/embedding_registry.yaml first."
        )

    with registry_path.open("r", encoding="utf-8") as handle:
        registry = yaml.safe_load(handle) or {}

    registry.setdefault("defaults", {})
    registry.setdefault("embeddings", {})
    return registry


def get_default_embedding_key(dataset_id: str) -> str:
    registry = load_embedding_registry()
    defaults = registry.get("defaults", {})
    if dataset_id not in defaults:
        raise KeyError(f"No default embedding is configured for dataset_id='{dataset_id}'.")
    return defaults[dataset_id]


def get_embedding_entry(embedding_key: str) -> dict:
    registry = load_embedding_registry()
    embeddings = registry.get("embeddings", {})
    if embedding_key not in embeddings:
        raise KeyError(f"No embedding entry is configured for embedding_key='{embedding_key}'.")
    return embeddings[embedding_key]


def get_default_embedding_entry(dataset_id: str) -> dict:
    embedding_key = get_default_embedding_key(dataset_id)
    entry = get_embedding_entry(embedding_key).copy()
    entry["embedding_key"] = embedding_key
    return entry


def list_embedding_entries(dataset_id: str | None = None) -> list[dict]:
    registry = load_embedding_registry()
    rows = []
    for embedding_key, entry in registry.get("embeddings", {}).items():
        if dataset_id is not None and entry.get("dataset_id") != dataset_id:
            continue
        row = dict(entry)
        row["embedding_key"] = embedding_key
        row["is_default"] = registry.get("defaults", {}).get(entry.get("dataset_id")) == embedding_key
        rows.append(row)
    return sorted(rows, key=lambda row: (row.get("dataset_id", ""), row.get("benchmark_version", ""), row["embedding_key"]))
