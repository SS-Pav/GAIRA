"""Data layer for Foundation Explorer V3. Every number is loaded from the committed
representation-hierarchy module (results/v5_rebuild/representation_hierarchy_v3/) + the frozen
atlas fingerprint. No scientific value is hardcoded in a page.
"""
from __future__ import annotations
import json
from pathlib import Path
from functools import lru_cache

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "results/v5_rebuild/representation_hierarchy_v3"
TABLES = BASE / "tables"
FIGURES = BASE / "figures"
ARTIFACTS = BASE / "artifacts"
CANON_FINGERPRINT = "09ed804a40836f4a05a91ba10900cded"


def present() -> bool:
    return (TABLES / "per_analyte_hierarchy.csv").exists()


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
    p = BASE / name
    return p.read_text() if p.exists() else f"_Document `{name}` not found._"


# typed loaders
def metrics() -> pd.DataFrame: return _csv("per_analyte_hierarchy.csv")
def hierarchy_summary_tbl() -> pd.DataFrame: return _csv("representation_hierarchy_summary.csv")
def rank_by_family() -> pd.DataFrame: return _csv("rank_by_family.csv")
def topk() -> pd.DataFrame: return _csv("topk_overlap.csv")
def theme_rank() -> pd.DataFrame: return _csv("theme_rank_preservation.csv")
def delta_purine() -> pd.DataFrame: return _csv("delta_purine.csv")
def sankey_flow() -> pd.DataFrame: return _csv("sankey_dominant_flow.csv")
def matrix() -> pd.DataFrame: return _csv("matrix_robustness.csv")
def summary() -> dict: return _json("hierarchy_summary.json")
def cards() -> dict: return _json("all_cards_v3.json")


def figure(name: str) -> Path | None:
    p = FIGURES / name
    return p if p.exists() else None


@lru_cache(maxsize=1)
def fingerprint_ok() -> bool:
    try:
        import sys
        sys.path.insert(0, str(REPO / "src"))
        from gaira.engine import GAIRAEngine
        return GAIRAEngine().atlas.meta["fingerprint"] == CANON_FINGERPRINT
    except Exception:
        return summary().get("atlas_fingerprint") == CANON_FINGERPRINT


@lru_cache(maxsize=1)
def reproducible_vs_v2() -> bool:
    repro = summary().get("reproducibility_vs_v2", {})
    return bool(repro) and all(abs(v) < 1e-6 for v in repro.values())
