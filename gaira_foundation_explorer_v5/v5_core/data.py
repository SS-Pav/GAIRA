"""Data layer for Foundation Explorer V5 — reads the committed abstraction-recovery module."""
from __future__ import annotations
import json
from pathlib import Path
from functools import lru_cache
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "results/v5_rebuild/abstraction_recovery_v5"
TABLES = BASE / "tables"; FIGURES = BASE / "figures"; ARTIFACTS = BASE / "artifacts"
CANON_FINGERPRINT = "09ed804a40836f4a05a91ba10900cded"


def present() -> bool:
    return (TABLES / "per_analyte_abstraction_recovery.csv").exists()


@lru_cache(maxsize=None)
def _csv(n):
    p = TABLES / n
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@lru_cache(maxsize=None)
def _json(n):
    p = ARTIFACTS / n
    return json.loads(p.read_text()) if p.exists() else {}


@lru_cache(maxsize=None)
def doc(n):
    p = BASE / n
    return p.read_text() if p.exists() else f"_{n} not found._"


def analytes(): return _csv("per_analyte_abstraction_recovery.csv")
def overlay(): return _csv("analyte_classification_overlay.csv")
def ladder(): return _csv("recovery_by_abstraction_level.csv")
def classification(): return _csv("subclass_classification_results.csv")
def nn(): return _csv("nearest_neighbor_retrieval.csv")
def family_breakdown(): return _csv("family_abstraction_breakdown.csv")
def summary(): return _json("abstraction_summary.json")
def cards(): return _json("all_cards_v5.json")


def figure(name):
    p = FIGURES / name
    return p if p.exists() else None


@lru_cache(maxsize=1)
def fingerprint_ok():
    try:
        import sys
        sys.path.insert(0, str(REPO / "src"))
        from gaira.engine import GAIRAEngine
        return GAIRAEngine().atlas.meta["fingerprint"] == CANON_FINGERPRINT
    except Exception:
        return summary().get("atlas_fingerprint") == CANON_FINGERPRINT


@lru_cache(maxsize=1)
def reproduces_v4():
    r = summary().get("reproducibility_vs_v4_identity", {})
    return bool(r) and all(r.values())
