"""Safe loaders for context-graph discovery artifacts."""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import pandas as pd
import streamlit as st


def load_csv_safe(path: str | Path,
                  show_warning: bool = False) -> pd.DataFrame | None:
    p = Path(path) if path is not None else None
    if p is None or not p.exists():
        if show_warning:
            st.info(f"🗂️ Table not found: `{path}`")
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        st.warning(f"Could not read `{p.name}`: {e}")
        return None


def load_text_safe(path: str | Path) -> str | None:
    p = Path(path) if path is not None else None
    if p is None or not p.exists():
        return None
    try:
        return p.read_text(errors="ignore")
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_all_context(context_root_str: str) -> dict:
    """Load every context-graph table once. Missing files are tolerated."""
    root = Path(context_root_str)
    T = root / "tables"
    out: dict = {}
    for key, fname in [
        ("events",         "gaira_evidence_events_long.csv"),
        ("nodes",          "context_graph_nodes.csv"),
        ("edges",          "context_graph_edges.csv"),
        ("axis_transfer",  "axis_transfer_scores.csv"),
        ("mss_transfer",   "mss_transfer_classification.csv"),
        ("sample_axis",    "sample_type_axis_recurrence.csv"),
        ("cf_axis",        "condition_axis_motif_recurrence.csv"),
        ("emergent",       "emergent_behavior_metrics.csv"),
        ("findings",       "top_emergent_findings.csv"),
        ("caveats",        "caveat_recurrence.csv"),
        ("dataset_features","context_dataset_bsv_features.csv"),
        ("clusters",       "context_cluster_assignments.csv"),
        ("axis_neighborhood","axis_neighborhood_summary.csv"),
        ("ctx_dependence", "context_dependence_scores.csv"),
        ("inventory",      "context_graph_artifact_inventory.csv"),
    ]:
        out[key] = load_csv_safe(T / fname)
    out["_root"] = str(root)
    return out


def figure_path(context_root_str: str, name: str) -> Path:
    return Path(context_root_str) / "figures" / name


def report_path(context_root_str: str) -> Path:
    return (Path(context_root_str) / "reports"
            / "REPORT_context_graph_discovery_v1.md")
