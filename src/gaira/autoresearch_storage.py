from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from gaira.config import get_project_root


DEFAULT_STORAGE_CONFIG_PATH = get_project_root() / "config" / "gaira_autoresearch_storage_v1.yaml"
SPRINT_SUBDIRS = ("runs", "figures", "tables", "logs", "report")


@dataclass(frozen=True)
class AutoresearchStorageConfig:
    output_root: Path
    sprint_id: str
    require_output_root_writable: bool
    allow_local_fallback: bool


@dataclass(frozen=True)
class AutoresearchSprintPaths:
    output_root: Path
    sprint_root: Path
    runs_dir: Path
    figures_dir: Path
    tables_dir: Path
    logs_dir: Path
    report_dir: Path
    manifest_path: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "output_root": str(self.output_root),
            "sprint_root": str(self.sprint_root),
            "runs_dir": str(self.runs_dir),
            "figures_dir": str(self.figures_dir),
            "tables_dir": str(self.tables_dir),
            "logs_dir": str(self.logs_dir),
            "report_dir": str(self.report_dir),
            "manifest_path": str(self.manifest_path),
        }


def load_autoresearch_storage_config(path: Path | None = None) -> AutoresearchStorageConfig:
    config_path = path or DEFAULT_STORAGE_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Autoresearch storage config missing: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    required = [
        "output_root",
        "sprint_id",
        "require_output_root_writable",
        "allow_local_fallback",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Autoresearch storage config missing required keys: {', '.join(missing)}")
    output_root = Path(str(payload["output_root"]))
    if not output_root.is_absolute():
        raise ValueError(f"Autoresearch output_root must be absolute: {output_root}")
    allow_local_fallback = bool(payload["allow_local_fallback"])
    if allow_local_fallback:
        raise ValueError("allow_local_fallback must remain false for GAIRAv3 autoresearch storage.")
    sprint_id = str(payload["sprint_id"]).strip()
    if not sprint_id:
        raise ValueError("Autoresearch storage sprint_id must be non-empty.")
    return AutoresearchStorageConfig(
        output_root=output_root,
        sprint_id=sprint_id,
        require_output_root_writable=bool(payload["require_output_root_writable"]),
        allow_local_fallback=allow_local_fallback,
    )


def ensure_autoresearch_output_root(config: AutoresearchStorageConfig) -> None:
    mount_root = Path("/Volumes/SSD_Rad")
    if not mount_root.exists():
        raise FileNotFoundError(
            "Autoresearch SSD mount is unavailable. Expected /Volumes/SSD_Rad to exist."
        )
    parent = config.output_root.parent
    if not parent.exists():
        raise FileNotFoundError(
            f"Configured autoresearch output parent does not exist: {parent}"
        )
    parent.mkdir(parents=True, exist_ok=True)
    config.output_root.mkdir(parents=True, exist_ok=True)
    if config.require_output_root_writable:
        verify_path_writable(config.output_root)


def verify_path_writable(path: Path) -> None:
    probe_dir = path / ".gaira_autoresearch_probe"
    probe_file = probe_dir / "write_test.txt"
    try:
        probe_dir.mkdir(parents=True, exist_ok=True)
        probe_file.write_text("ok\n", encoding="utf-8")
        probe_file.unlink()
        probe_dir.rmdir()
    except Exception as exc:
        raise PermissionError(f"Autoresearch output root is not writable: {path}") from exc


def resolve_autoresearch_sprint_paths(
    config_path: Path | None = None,
    *,
    sprint_id: str | None = None,
) -> AutoresearchSprintPaths:
    config = load_autoresearch_storage_config(config_path)
    ensure_autoresearch_output_root(config)
    effective_sprint_id = sprint_id.strip() if sprint_id else config.sprint_id
    if not effective_sprint_id:
        raise ValueError("Effective sprint_id must be non-empty.")
    sprint_root = config.output_root / effective_sprint_id
    return AutoresearchSprintPaths(
        output_root=config.output_root,
        sprint_root=sprint_root,
        runs_dir=sprint_root / "runs",
        figures_dir=sprint_root / "figures",
        tables_dir=sprint_root / "tables",
        logs_dir=sprint_root / "logs",
        report_dir=sprint_root / "report",
        manifest_path=sprint_root / "storage_manifest.json",
    )


def initialize_autoresearch_sprint(
    config_path: Path | None = None,
    *,
    sprint_id: str | None = None,
) -> AutoresearchSprintPaths:
    paths = resolve_autoresearch_sprint_paths(config_path, sprint_id=sprint_id)
    paths.sprint_root.mkdir(parents=True, exist_ok=True)
    for subdir in [paths.runs_dir, paths.figures_dir, paths.tables_dir, paths.logs_dir, paths.report_dir]:
        subdir.mkdir(parents=True, exist_ok=True)
    verify_path_writable(paths.sprint_root)
    write_storage_manifest(paths, config_path or DEFAULT_STORAGE_CONFIG_PATH)
    return paths


def write_storage_manifest(paths: AutoresearchSprintPaths, config_path: Path) -> None:
    payload = {
        "config_path": str(config_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **paths.as_dict(),
    }
    paths.manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
