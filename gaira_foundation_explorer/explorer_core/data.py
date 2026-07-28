"""Data layer for the GAIRA Foundation Explorer.

EVERYTHING the app shows is loaded here from the completed audit at
results/v5_rebuild/foundation_audit/ (reports, tables, figures) and the frozen atlas
artifacts. No scientific number is hardcoded in the UI — pages call these loaders.
"""
from __future__ import annotations
import json
from pathlib import Path
from functools import lru_cache
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[2]
AUDIT = REPO / "results/v5_rebuild/foundation_audit"
REPORTS = AUDIT / "reports"
TABLES = AUDIT / "tables"
FIGURES = AUDIT / "figures"
COMPONENTS = AUDIT / "components"
FROZEN = REPO / "results/v5_rebuild/foundation/artifacts"


def audit_present() -> bool:
    return AUDIT.exists() and (TABLES / "corpus_summary.json").exists()


# ── generic loaders (cached) ──────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_json(rel: str) -> dict:
    p = TABLES / rel if not rel.startswith("/") else Path(rel)
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data(show_spinner=False)
def load_csv(rel: str) -> pd.DataFrame:
    p = TABLES / rel
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_report(name: str) -> str:
    p = REPORTS / name
    return p.read_text() if p.exists() else f"_Report `{name}` not found._"


@st.cache_data(show_spinner=False)
def load_component_page(j: int) -> str:
    p = COMPONENTS / f"component_c{j:02d}.md"
    return p.read_text() if p.exists() else ""


def figure(name: str) -> Path | None:
    p = FIGURES / name
    return p if p.exists() else None


def component_figure(j: int) -> Path | None:
    return figure(f"component_c{j:02d}.png")


# ── domain-specific accessors ─────────────────────────────────────────
@st.cache_data(show_spinner=False)
def atlas_meta() -> dict:
    p = FROZEN / "manifold.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


@st.cache_data(show_spinner=False)
def headline() -> dict:
    """Top-level facts, all read from the frozen atlas + audit tables."""
    m = atlas_meta()
    corp = load_json("corpus_summary.json")
    verdict = load_json("c1_repro_verdict.json")
    sel = load_json("c1_selection_repro.json")
    stats = m.get("stats", {})
    return {
        "fingerprint": m.get("fingerprint", "—"),
        "representation": m.get("representation", "NMF"),
        "k": m.get("k", 24),
        "n_spectra": corp.get("n_spectra", stats.get("n_spectra")),
        "n_analytes": corp.get("n_analytes", stats.get("n_analytes")),
        "n_bins": corp.get("n_bins", m.get("n_bins")),
        "window_cm": corp.get("window_cm", [m.get("grid_min"), m.get("grid_max")]),
        "grid_step_cm": corp.get("grid_step_cm", 2.0),
        "explained_variance": stats.get("explained_variance"),
        "mean_stability": load_json("component_global_classification.json").get("mean_stability"),
        "ranking_identical": verdict.get("full_ranking_identical"),
        "max_diff_vs_committed": (max(verdict.get("max_abs_diff_vs_committed", {}).values())
                                  if verdict.get("max_abs_diff_vs_committed") else None),
        "raw_top": sel.get("raw_top", {}),
        "selected": {"representation": sel.get("representation"), "k": sel.get("k"),
                     "total_score": sel.get("total_score")},
        "n_sources": len(corp.get("sources_spectra", {})),
    }


@st.cache_data(show_spinner=False)
def benchmark() -> pd.DataFrame:
    df = load_csv("c1_representation_benchmark_repro.csv")
    return df


@st.cache_data(show_spinner=False)
def selection_comparison() -> dict:
    return load_json("nmf_selection_comparison.json")


@st.cache_data(show_spinner=False)
def components_table() -> pd.DataFrame:
    return load_csv("component_audit_summary.csv")


@st.cache_data(show_spinner=False)
def component_classification() -> dict:
    return load_json("component_global_classification.json")


@st.cache_data(show_spinner=False)
def mss_registry() -> dict:
    return load_json("mss_registry.json")


@st.cache_data(show_spinner=False)
def validation() -> dict:
    return load_json("validation_results.json")


@st.cache_data(show_spinner=False)
def transfer_pairs() -> pd.DataFrame:
    return load_csv("validation_transfer_pairs.csv")


@st.cache_data(show_spinner=False)
def corpus_summary() -> dict:
    return load_json("corpus_summary.json")


@st.cache_data(show_spinner=False)
def corpus_analytes() -> pd.DataFrame:
    return load_csv("corpus_analytes.csv")


@st.cache_data(show_spinner=False)
def cross_source_duplicates() -> pd.DataFrame:
    return load_csv("corpus_cross_source_duplicates.csv")


@st.cache_data(show_spinner=False)
def preprocessing_stats() -> dict:
    return load_json("preprocessing_stats.json")


@st.cache_data(show_spinner=False)
def selection_repro() -> dict:
    return load_json("c1_selection_repro.json")
