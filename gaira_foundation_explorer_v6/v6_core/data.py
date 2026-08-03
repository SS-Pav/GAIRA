"""Data layer for Foundation Explorer V6 — detection gate (V6) + abstraction recovery (V5)."""
from __future__ import annotations
import json
from pathlib import Path
from functools import lru_cache
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
V6 = REPO / "results/v5_rebuild/detection_gate_v6"
V5 = REPO / "results/v5_rebuild/abstraction_recovery_v5"
CANON_FINGERPRINT = "09ed804a40836f4a05a91ba10900cded"


def present() -> bool:
    return (V6 / "tables/detection_metrics.csv").exists()


@lru_cache(maxsize=None)
def _csv(base, n):
    p = (V6 if base == "v6" else V5) / "tables" / n
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@lru_cache(maxsize=None)
def _json(base, n):
    p = (V6 if base == "v6" else V5) / "artifacts" / n
    return json.loads(p.read_text()) if p.exists() else {}


@lru_cache(maxsize=None)
def doc(n):
    p = V6 / n
    return p.read_text() if p.exists() else f"_{n} not found._"


def detection(): return _csv("v6", "detection_metrics.csv")
def ladder(): return _csv("v6", "recovery_detectable_vs_all.csv")
def transfer(): return _csv("v6", "per_analyte_transfer_decision.csv")
def det_summary(): return _json("v6", "detection_summary.json")
def restricted(): return _json("v6", "restricted_hierarchy_summary.json")
def v5_cards(): return _json("v5", "all_cards_v5.json")


def figure(name):
    p = V6 / "figures" / name
    return p if p.exists() else None


@lru_cache(maxsize=1)
def fingerprint_ok():
    try:
        import sys
        sys.path.insert(0, str(REPO / "src"))
        from gaira.engine import GAIRAEngine
        return GAIRAEngine().atlas.meta["fingerprint"] == CANON_FINGERPRINT
    except Exception:
        return det_summary().get("atlas_fingerprint") == CANON_FINGERPRINT
