"""GAIRA Demo v2 — migration-hardened path resolution.

v1 hardcoded two brittle absolute locations:
  - the external data volume  `/Volumes/SSD_Rad/GAIRA_DATA`
  - the legacy demo CSVs at `<repo>/streamlit_apps/gaira_demo/data`
and it degraded to placeholders *silently* if either moved.

v2 resolves every root through this module so the demo is portable across
machines and drives:

  1. Environment variables win (explicit operator intent):
       GAIRA_DATA_ROOT           → the GAIRA_DATA volume (contains raw/ + processed/)
       GAIRA_LEGACY_DEMO_DATA    → the folder with grounding_molecule_bsv.csv etc.
  2. Otherwise a list of candidate locations is probed, first-existing wins.
  3. The 5 tiny legacy CSVs are BUNDLED inside v2 (`data/legacy/`), so the
     calibration / biochemical-space / uric-acid tabs work even with no repo
     checkout and no external drive.
  4. Nothing raises on a missing root — callers get `None` and a status object
     so the UI can show an explicit real-vs-placeholder banner instead of
     failing silently.

No path in this file is tied to a specific username or mount name except as
one entry in an ordered candidate list.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# Anchors
# ─────────────────────────────────────────────────────────────────────

# gaira_core/paths.py → gaira_core → demo root → repo root
DEMO_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = DEMO_ROOT.parent
BUNDLED_LEGACY_DIR = DEMO_ROOT / "data" / "legacy"

# Autoresearch sub-path relative to a GAIRA_DATA root (stable across machines).
_AUTORESEARCH_REL = Path("processed") / "gaira_autoresearch" / "gaira_autoresearch_v1"
_ADENINE_REL = Path("raw") / "adenine_sers_control"

# Filenames the loaders expect inside the legacy demo-data dir.
_LEGACY_FILES = (
    "grounding_molecule_bsv.csv",
    "grounding_molecule_index.csv",
    "ergothioneine_dose_response.csv",
    "calibration_conditions.csv",
    "calibration_delta_bsv.csv",
)


def _env_path(var: str) -> Path | None:
    val = os.environ.get(var, "").strip()
    if not val:
        return None
    p = Path(val).expanduser()
    return p if p.exists() else None


def _first_existing(candidates) -> Path | None:
    for c in candidates:
        try:
            if c and Path(c).exists():
                return Path(c)
        except (OSError, ValueError):
            continue
    return None


# ─────────────────────────────────────────────────────────────────────
# GAIRA_DATA volume (raw/ + processed/) — external, large
# ─────────────────────────────────────────────────────────────────────

def data_root() -> Path | None:
    """Resolve the GAIRA_DATA root (holds raw/ and processed/).

    Order: $GAIRA_DATA_ROOT → common mount/home candidates → None.
    Returns None (not an error) when unavailable so the UI can degrade
    explicitly.
    """
    env = _env_path("GAIRA_DATA_ROOT")
    if env is not None:
        return env

    home = Path.home()
    candidates = [
        Path("/Volumes/SSD_Rad/GAIRA_DATA"),           # original external SSD
        Path("/Volumes/SSD_Rad2/GAIRA_DATA"),          # relabeled drive
        Path("/Volumes/GAIRA_DATA"),                    # volume named directly
        home / "GAIRA_DATA",                            # copied to home
        home / "projects" / "GAIRA_DATA",
        REPO_ROOT / "GAIRA_DATA",                       # sibling of demo, in repo
        REPO_ROOT.parent / "GAIRA_DATA",
        DEMO_ROOT / "data" / "external",                # local symlink/junction
    ]
    # Only accept a candidate that actually looks like GAIRA_DATA
    for c in candidates:
        if (c / "raw").exists() or (c / "processed").exists():
            return c
    # Fall back to a bare-existing candidate (better a real dir than None)
    return _first_existing(candidates)


def autoresearch_root() -> Path | None:
    root = data_root()
    if root is None:
        return None
    ar = root / _AUTORESEARCH_REL
    return ar if ar.exists() else None


def adenine_raw_dir() -> Path | None:
    root = data_root()
    if root is None:
        return None
    ad = root / _ADENINE_REL
    return ad if ad.exists() else None


# ─────────────────────────────────────────────────────────────────────
# Legacy demo CSVs — tiny, bundled, always available
# ─────────────────────────────────────────────────────────────────────

def _looks_like_legacy(d: Path) -> bool:
    try:
        return (d / "grounding_molecule_bsv.csv").exists()
    except OSError:
        return False


def legacy_demo_data() -> Path:
    """Resolve the legacy demo-CSV dir. Never None — falls back to the copy
    bundled inside v2 so the demo is self-contained.

    Order: $GAIRA_LEGACY_DEMO_DATA → repo streamlit_apps copy → bundled.
    """
    env = _env_path("GAIRA_LEGACY_DEMO_DATA")
    if env is not None and _looks_like_legacy(env):
        return env

    repo_copy = REPO_ROOT / "streamlit_apps" / "gaira_demo" / "data"
    if _looks_like_legacy(repo_copy):
        return repo_copy

    return BUNDLED_LEGACY_DIR


def legacy_source_kind() -> str:
    """Where the legacy CSVs are coming from (for provenance display)."""
    d = legacy_demo_data()
    if d == BUNDLED_LEGACY_DIR:
        return "bundled"
    if "streamlit_apps" in str(d):
        return "repo"
    return "env"


# ─────────────────────────────────────────────────────────────────────
# Status object — powers the app-level data-source banner
# ─────────────────────────────────────────────────────────────────────

@dataclass
class DataStatus:
    data_root: Path | None
    data_root_mounted: bool
    autoresearch_root: Path | None
    adenine_dir: Path | None
    legacy_dir: Path
    legacy_kind: str
    checks: dict = field(default_factory=dict)

    @property
    def mode(self) -> str:
        """'real' if the external volume resolved and pilots are reachable,
        'degraded' if only the bundled/legacy calibration data is available,
        else 'placeholder'."""
        if self.data_root_mounted and self.autoresearch_root is not None:
            return "real"
        if self.legacy_dir is not None:
            return "degraded"
        return "placeholder"

    @property
    def real_section_count(self) -> int:
        return sum(1 for v in self.checks.values() if v)


def get_data_status() -> DataStatus:
    dr = data_root()
    ar = autoresearch_root()
    ad = adenine_raw_dir()
    legacy = legacy_demo_data()

    def _has(p: Path | None, *rel: str) -> bool:
        if p is None:
            return False
        q = p
        for r in rel:
            q = q / r
        return q.exists()

    checks = {
        "11-Axis Biochemical Space": (legacy / "grounding_molecule_bsv.csv").exists(),
        "Ergothioneine dose":        (legacy / "ergothioneine_dose_response.csv").exists(),
        "Uric-acid validation":      (legacy / "calibration_conditions.csv").exists(),
        "Adenine detection":         ad is not None,
        "Grounding corpus map":      _has(ar, "gaira_evidence_warehouse_grounding_backbone_v1",
                                          "tables", "warehouse_source_registry.csv"),
        "Serum-liver pilot":         _has(ar, "pilot4_1_cca_hcc_lm_serum_patient_level",
                                          "tables", "patient_level_mean_spectra.csv"),
        "EV-diabetes pilot":         _has(ar, "pilot2_target_validation_v1",
                                          "tables", "sample_query_spectra.csv"),
        "SHINE pilot":               _has(ar, "pilot3_shine_single_set_day0_day2",
                                          "tables", "class_mean_bsv_day0_day2.csv"),
    }

    return DataStatus(
        data_root=dr,
        data_root_mounted=dr is not None and (dr / "processed").exists(),
        autoresearch_root=ar,
        adenine_dir=ad,
        legacy_dir=legacy,
        legacy_kind=legacy_source_kind(),
        checks=checks,
    )
