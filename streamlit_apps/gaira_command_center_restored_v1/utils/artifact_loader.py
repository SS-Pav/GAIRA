"""Artifact manifest scanner + lookups.

The Command Center is precomputed-artifact-driven. This module discovers
*.csv / *.png / *.md under configured roots, writes a manifest YAML, and
exposes lookup helpers. It never recomputes GAIRA pipelines.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import os
import yaml


SUPPORTED_EXTS = {".csv", ".png", ".md"}


@dataclass
class ArtifactRecord:
    phase: str
    kind: str            # "tables" | "figures" | "reports" | "audit" | "registry" | "other"
    extension: str       # ".csv" / ".png" / ".md"
    name: str            # filename
    path: str            # absolute path (string for YAML serialization)
    relative_path: str   # path relative to phase root
    size_bytes: int
    modified_iso: str


def _infer_kind(parts: list[str]) -> str:
    for p in parts:
        if p in ("tables", "figures", "reports", "audit", "registry", "code_snapshot"):
            return p
    return "other"


def scan_phase_folder(phase_root: Path) -> list[ArtifactRecord]:
    """Return all supported artifacts under a phase folder. Skip ._ AppleDouble files."""
    out: list[ArtifactRecord] = []
    if not phase_root.exists():
        return out
    phase_name = phase_root.name
    for root, dirs, files in os.walk(phase_root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fn in files:
            if fn.startswith("._"):
                continue
            ext = Path(fn).suffix.lower()
            if ext not in SUPPORTED_EXTS:
                continue
            full = Path(root) / fn
            try:
                st = full.stat()
            except OSError:
                continue
            rel = str(full.relative_to(phase_root))
            kind = _infer_kind(rel.split("/"))
            out.append(ArtifactRecord(
                phase=phase_name,
                kind=kind,
                extension=ext,
                name=fn,
                path=str(full),
                relative_path=rel,
                size_bytes=st.st_size,
                modified_iso=datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            ))
    return out


def build_manifest(build_root: Path, phase_folders: list[str]) -> dict:
    """Walk every configured phase folder, return manifest dict."""
    phases: dict[str, list[dict]] = {}
    missing: list[str] = []
    for ph in phase_folders:
        root = build_root / ph
        if not root.exists():
            missing.append(ph)
            continue
        records = scan_phase_folder(root)
        phases[ph] = [asdict(r) for r in records]

    summary = {
        "phases_total": len(phase_folders),
        "phases_present": len(phases),
        "phases_missing": len(missing),
        "artifacts_total": sum(len(v) for v in phases.values()),
        "csv_total": sum(1 for v in phases.values() for r in v if r["extension"] == ".csv"),
        "png_total": sum(1 for v in phases.values() for r in v if r["extension"] == ".png"),
        "md_total": sum(1 for v in phases.values() for r in v if r["extension"] == ".md"),
    }
    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "build_root": str(build_root),
        "summary": summary,
        "missing_phases": missing,
        "phases": phases,
    }


def write_manifest(manifest: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)


def load_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        return yaml.safe_load(f)


def ensure_manifest(build_root: Path, phase_folders: list[str], manifest_path: Path,
                    rebuild: bool = False) -> dict:
    """Load manifest from disk, or build it if missing/stale/rebuild=True."""
    if not rebuild:
        existing = load_manifest(manifest_path)
        if existing is not None:
            return existing
    manifest = build_manifest(build_root, phase_folders)
    write_manifest(manifest, manifest_path)
    return manifest


def find_artifacts(manifest: dict, phase: str | None = None,
                   kind: str | None = None,
                   ext: str | None = None,
                   name_contains: str | None = None) -> list[dict]:
    """Filter manifest records. Returns a list of dicts (preserves YAML form)."""
    out = []
    phases = manifest.get("phases", {})
    if phase is not None:
        items = [(phase, phases.get(phase, []))]
    else:
        items = list(phases.items())
    for ph, records in items:
        for r in records:
            if kind is not None and r.get("kind") != kind:
                continue
            if ext is not None and r.get("extension") != ext:
                continue
            if name_contains is not None and name_contains.lower() not in r.get("name", "").lower():
                continue
            out.append(r)
    return out


def first_existing(paths: list[str | Path]) -> Path | None:
    for p in paths:
        pp = Path(p)
        if pp.exists():
            return pp
    return None
