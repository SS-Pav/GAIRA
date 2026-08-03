"""Data layer for Foundation Explorer V4. Every value loaded from the committed V4 module."""
from __future__ import annotations
import json
from pathlib import Path
from functools import lru_cache
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "results/v5_rebuild/hierarchical_recoverability_v4"
TABLES = BASE / "tables"; FIGURES = BASE / "figures"; ARTIFACTS = BASE / "artifacts"
CANON_FINGERPRINT = "09ed804a40836f4a05a91ba10900cded"


def present() -> bool:
    return (TABLES / "per_analyte_evidence_profile.csv").exists()


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


def evidence(): return _csv("per_analyte_evidence_profile.csv")
def counts(): return _csv("recoverable_analyte_counts.csv")
def by_level(): return _csv("recoverable_analytes_by_level.csv")
def overlap(): return _csv("recoverability_overlap_matrix.csv")
def threshold(): return _csv("recoverability_threshold_sensitivity.csv")
def level_null(): return _csv("level_null_summary.csv")
def variants(): return _csv("theme_variant_comparison.csv")
def mss_rank(): return _csv("mss_specificity_ranking.csv")
def matrix_pred(): return _csv("matrix_prediction.csv")
def decision(): return _csv("metric_decision_table.csv")
def delta_purine_corr(): return _csv("delta_purine_correlations.csv")
def purine_blank(): return _csv("purine_blank_controls.csv")
def summary(): return _json("recoverability_summary.json")
def cards(): return _json("all_cards_v4.json")


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
def reproduces_v3():
    r = summary().get("reproducibility_vs_v3", {})
    return bool(r) and all(abs(v) < 1e-6 for v in r.values())
