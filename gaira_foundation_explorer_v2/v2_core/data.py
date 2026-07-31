"""Data layer for Foundation Explorer V2.

Every number the app shows is loaded here from the committed theme-preservation module
(results/v5_rebuild/pure_ag_sers_theme_preservation/) — tables, figures, cards, docs — plus
the frozen atlas fingerprint. No scientific value is hardcoded in a page.
"""
from __future__ import annotations
import json
from pathlib import Path
from functools import lru_cache

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "results/v5_rebuild/pure_ag_sers_theme_preservation"
TABLES = BASE / "tables"
FIGURES = BASE / "figures"
ARTIFACTS = BASE / "artifacts"
CANON_FINGERPRINT = "09ed804a40836f4a05a91ba10900cded"


def present() -> bool:
    return (TABLES / "per_analyte_transfer_metrics.csv").exists()


@lru_cache(maxsize=None)
def _csv(name: str) -> pd.DataFrame:
    p = TABLES / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@lru_cache(maxsize=None)
def _json(rel: str) -> dict:
    p = ARTIFACTS / rel
    return json.loads(p.read_text()) if p.exists() else {}


@lru_cache(maxsize=None)
def doc(name: str) -> str:
    p = BASE / name if not name.endswith(".md") or "/" in name else BASE / name
    return p.read_text() if p.exists() else f"_Document `{name}` not found._"


def framework_doc() -> str:
    p = REPO / "GAIRA_MULTI_LEVEL_VALIDATION_FRAMEWORK.md"
    return p.read_text() if p.exists() else "_Framework doc not found._"


# ── typed loaders ──
def metrics() -> pd.DataFrame:
    return _csv("per_analyte_transfer_metrics.csv")


def component_vs_theme() -> pd.DataFrame:
    return _csv("component_vs_theme_preservation.csv")


def by_family() -> pd.DataFrame:
    return _csv("theme_preservation_by_family.csv")


def confusion() -> pd.DataFrame:
    return _csv("dominant_theme_confusion.csv")


def mss() -> pd.DataFrame:
    return _csv("mss_preservation.csv")


def perturbation() -> pd.DataFrame:
    return _csv("perturbation_sensitivity.csv")


def matrix() -> pd.DataFrame:
    return _csv("matrix_recoverability_linkage.csv")


def summary() -> dict:
    return _json("theme_preservation_summary.json")


def cards() -> dict:
    return _json("all_cards.json")


def figure(name: str) -> Path | None:
    p = FIGURES / name
    return p if p.exists() else None


@lru_cache(maxsize=1)
def fingerprint_ok() -> bool:
    """Verify the FROZEN atlas fingerprint at runtime (the app is only valid on it)."""
    try:
        import sys
        sys.path.insert(0, str(REPO / "src"))
        from gaira.engine import GAIRAEngine
        return GAIRAEngine().atlas.meta["fingerprint"] == CANON_FINGERPRINT
    except Exception:
        # engine unavailable (e.g. CI without full deps) — fall back to the recorded summary
        return summary().get("atlas_fingerprint") == CANON_FINGERPRINT
