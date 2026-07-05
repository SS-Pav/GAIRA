"""v2 loaders + cache builder.

Reads the v1 context-graph discovery outputs, attaches specific-condition
labels via `condition_mapper`, then derives the v2 cache tables.
Everything is cached at the Streamlit layer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils.condition_mapper import load_rules, attach_specific_conditions


# ─── primitive safe loaders ──────────────────────────────────────────────

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


# ─── v1 base context tables ──────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_v1_context(context_root_str: str) -> dict:
    root = Path(context_root_str)
    T = root / "tables"
    out: dict = {"_root": str(root)}
    for key, fname in [
        ("events",          "gaira_evidence_events_long.csv"),
        ("nodes",           "context_graph_nodes.csv"),
        ("edges",           "context_graph_edges.csv"),
        ("axis_transfer",   "axis_transfer_scores.csv"),
        ("mss_transfer",    "mss_transfer_classification.csv"),
        ("sample_axis",     "sample_type_axis_recurrence.csv"),
        ("cf_axis_v1",      "condition_axis_motif_recurrence.csv"),
        ("emergent",        "emergent_behavior_metrics.csv"),
        ("findings",        "top_emergent_findings.csv"),
        ("caveats",         "caveat_recurrence.csv"),
        ("dataset_features","context_dataset_bsv_features.csv"),
        ("clusters",        "context_cluster_assignments.csv"),
        ("axis_neighborhood","axis_neighborhood_summary.csv"),
        ("ctx_dependence", "context_dependence_scores.csv"),
        ("inventory",       "context_graph_artifact_inventory.csv"),
    ]:
        out[key] = load_csv_safe(T / fname)
    return out


# ─── v2 cache builder ────────────────────────────────────────────────────

def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def build_condition_axis_edges_specific(events: pd.DataFrame,
                                         broad: bool = False) -> pd.DataFrame:
    """Aggregate (specific_condition × bsv_axis). If broad=True use
    `condition_family_v2` instead of `specific_condition`."""
    cond_col = "condition_family_v2" if broad else "specific_condition"
    if events is None or events.empty or cond_col not in events.columns:
        return pd.DataFrame()
    rows = []
    for (cf, ax), sub in events.groupby([cond_col, "bsv_axis"]):
        if not isinstance(ax, str) or not ax.startswith("G"):
            continue
        try:
            effs = _safe_num(sub["effect_size"]).dropna()
            me = float(effs.mean()) if len(effs) else 0.0
            mae = float(effs.abs().mean()) if len(effs) else 0.0
        except Exception:
            me, mae = 0.0, 0.0
        dirs = sub["direction"].value_counts(dropna=True)
        dom = dirs.idxmax() if len(dirs) else "ambiguous"
        cons = float(dirs.max() / dirs.sum()) if len(dirs) else 0.0
        rows.append({
            ("condition_family" if broad else "specific_condition"): cf,
            "bsv_axis": ax, "n_events": len(sub),
            "n_datasets": sub["dataset"].nunique(),
            "n_sample_types": sub["sample_type"].nunique(),
            "mean_effect": round(me, 3),
            "mean_abs_effect": round(mae, 3),
            "dom_direction": dom,
            "direction_consistency": round(cons, 2),
            "weight": round(len(sub) * (mae + 0.10), 3),
            "datasets": ";".join(sorted(sub["dataset"].dropna().unique())[:5]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("weight", ascending=False).reset_index(drop=True)


def build_condition_mss_edges(events: pd.DataFrame) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame()
    e = events.dropna(subset=["mss_candidate"])
    e = e[e["mss_candidate"].astype(str) != ""]
    rows = []
    for (cond, m), sub in e.groupby(["specific_condition", "mss_candidate"]):
        try:
            effs = _safe_num(sub["effect_size"]).dropna()
            me = float(effs.abs().mean()) if len(effs) else 0.0
        except Exception:
            me = 0.0
        dirs = sub["direction"].value_counts(dropna=True)
        dom = dirs.idxmax() if len(dirs) else "stable"
        rows.append({
            "specific_condition": cond, "mss_candidate": m,
            "n_events": len(sub),
            "n_datasets": sub["dataset"].nunique(),
            "mean_abs_effect": round(me, 3),
            "dom_direction": dom,
        })
    return pd.DataFrame(rows).sort_values("n_events", ascending=False).reset_index(drop=True)


def build_sample_type_axis_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame()
    rows = []
    for (st_, ax), sub in events.groupby(["sample_type", "bsv_axis"]):
        if not isinstance(ax, str) or not ax.startswith("G"):
            continue
        try:
            effs = _safe_num(sub["effect_size"]).dropna()
            me = float(effs.mean()) if len(effs) else 0.0
        except Exception:
            me = 0.0
        dirs = sub["direction"].value_counts(dropna=True)
        dom = dirs.idxmax() if len(dirs) else "ambiguous"
        cons = float(dirs.max() / dirs.sum()) if len(dirs) else 0.0
        rows.append({
            "sample_type": st_, "bsv_axis": ax,
            "n_datasets": sub["dataset"].nunique(),
            "n_events": len(sub),
            "mean_effect": round(me, 3),
            "dom_direction": dom,
            "direction_consistency": round(cons, 2),
        })
    return pd.DataFrame(rows)


def build_sample_type_mss_summary(events: pd.DataFrame,
                                   top_n: int = 25) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame()
    e = events.dropna(subset=["mss_candidate"])
    e = e[e["mss_candidate"].astype(str) != ""]
    top_mss = e["mss_candidate"].value_counts().head(top_n).index.tolist()
    e = e[e["mss_candidate"].isin(top_mss)]
    rows = []
    for (st_, m), sub in e.groupby(["sample_type", "mss_candidate"]):
        rows.append({
            "sample_type": st_, "mss_candidate": m,
            "n_datasets": sub["dataset"].nunique(),
            "n_events": len(sub),
            "dom_direction": (sub["direction"].value_counts().idxmax()
                                if len(sub) else "stable"),
        })
    return pd.DataFrame(rows)


def build_emergent_paths(events: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Top sample_type → dataset → specific_condition → bsv_axis paths,
    ranked by evidence count × |mean effect|."""
    if events is None or events.empty:
        return pd.DataFrame()
    rows = []
    grp_keys = ["sample_type", "dataset", "specific_condition", "bsv_axis"]
    for keys, sub in events.groupby(grp_keys):
        st_, ds, cond, ax = keys
        if not isinstance(ax, str) or not ax.startswith("G"):
            continue
        try:
            effs = _safe_num(sub["effect_size"]).dropna()
            mean_abs = float(effs.abs().mean()) if len(effs) else 0.0
        except Exception:
            mean_abs = 0.0
        dirs = sub["direction"].value_counts()
        dom = dirs.idxmax() if len(dirs) else "stable"
        cons = float(dirs.max() / dirs.sum()) if len(dirs) else 0.0
        score = len(sub) * (mean_abs + 0.1) * cons
        # Top MSS for this path
        mss_top = (sub["mss_candidate"].dropna().value_counts()
                    .head(2).index.tolist())
        rows.append({
            "sample_type": st_, "dataset": ds,
            "specific_condition": cond, "bsv_axis": ax,
            "dom_direction": dom,
            "n_events": len(sub),
            "mean_abs_effect": round(mean_abs, 3),
            "consistency": round(cons, 2),
            "path_score": round(score, 3),
            "top_mss": ";".join(mss_top),
            "confidence_tier": ("STRONG" if score >= 5
                                 else "MODERATE" if score >= 1
                                 else "WEAK"),
        })
    df = pd.DataFrame(rows).sort_values("path_score", ascending=False)
    return df.head(top_n).reset_index(drop=True)


def build_context_embedding_v2(features: pd.DataFrame,
                                events: pd.DataFrame,
                                short_label_map: dict[str, str],
                                caveats: pd.DataFrame | None) -> pd.DataFrame:
    if features is None or features.empty:
        return pd.DataFrame()
    df = features.copy()
    df["short_label"] = df["dataset"].map(short_label_map).fillna(
        df["dataset"].str.replace("gaira_base_4_", "").str[:18])
    if caveats is not None and not caveats.empty:
        cav = (caveats.groupby("dataset")["n_mentions"].sum()
               .rename("caveat_burden").reset_index())
        df = df.merge(cav, on="dataset", how="left")
    if "caveat_burden" not in df.columns:
        df["caveat_burden"] = 0
    df["caveat_burden"] = df["caveat_burden"].fillna(0)
    if events is not None and not events.empty:
        ev_count = (events.groupby("dataset").size()
                    .rename("evidence_count").reset_index())
        df = df.merge(ev_count, on="dataset", how="left")
    if "evidence_count" not in df.columns:
        df["evidence_count"] = 0
    return df


# ─── Public façade ───────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_v2_context(context_root_str: str, app_dir_str: str,
                     short_label_map: dict[str, str]) -> dict:
    """Load v1 base + derive v2 enrichment + cache to disk."""
    base = load_v1_context(context_root_str)
    rules_path = Path(app_dir_str) / "config" / "condition_mapping.yaml"
    rules = load_rules(rules_path)
    events = base["events"]
    enriched = attach_specific_conditions(events, rules) if events is not None else None

    cache_dir = Path(app_dir_str) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _save(df: pd.DataFrame, name: str) -> None:
        if df is not None and not df.empty:
            df.to_csv(cache_dir / name, index=False)

    cf_specific = build_condition_axis_edges_specific(enriched, broad=False)
    cf_broad    = build_condition_axis_edges_specific(enriched, broad=True)
    mss_edges   = build_condition_mss_edges(enriched)
    st_axis     = build_sample_type_axis_summary(enriched)
    st_mss      = build_sample_type_mss_summary(enriched)
    emergent_paths = build_emergent_paths(enriched, top_n=50)
    embed_df = build_context_embedding_v2(
        base.get("dataset_features"), enriched, short_label_map, base.get("caveats"))

    _save(enriched,        "condition_specific_events.csv")
    _save(cf_specific,     "condition_axis_edges_specific.csv")
    _save(cf_broad,        "condition_axis_edges_broad.csv")
    _save(mss_edges,       "condition_mss_edges.csv")
    _save(st_axis,         "sample_type_axis_summary.csv")
    _save(st_mss,          "sample_type_mss_summary.csv")
    _save(embed_df,        "context_embedding_points_v2.csv")
    _save(emergent_paths,  "emergent_paths_ranked.csv")

    return {
        **base,
        "events_v2":        enriched,
        "cf_axis_specific": cf_specific,
        "cf_axis_broad":    cf_broad,
        "cond_mss_edges":   mss_edges,
        "st_axis_summary":  st_axis,
        "st_mss_summary":   st_mss,
        "emergent_paths":   emergent_paths,
        "embedding_v2":     embed_df,
    }


def figure_path(context_root_str: str, name: str) -> Path:
    return Path(context_root_str) / "figures" / name


def report_path(context_root_str: str) -> Path:
    return (Path(context_root_str) / "reports"
            / "REPORT_context_graph_discovery_v1.md")
