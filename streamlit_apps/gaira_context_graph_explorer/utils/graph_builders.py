"""Graph-construction helpers — bipartite condition↔axis, layered Sankey,
MSS-transfer bipartite. Pure data manipulation; no Plotly here."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd


BSV_AXES = [f"G{i:02d}" for i in range(1, 12)]


# ─── Condition × Axis (bipartite) ─────────────────────────────────────────

def build_condition_axis_edges(events: pd.DataFrame,
                               cf_axis: pd.DataFrame | None = None,
                               weight_kind: str = "evidence",
                               normalize: bool = False) -> pd.DataFrame:
    """Return edges = condition_family × bsv_axis with weight + dominant
    direction.

    weight_kind:
        'evidence'  → n_events × |mean_effect|
        'datasets'  → n_datasets only
        'effect'    → mean_effect (signed)
    """
    if cf_axis is None or cf_axis.empty:
        if events is None or events.empty:
            return pd.DataFrame(columns=["condition_family", "bsv_axis",
                                          "weight", "direction", "n_datasets",
                                          "n_events", "mean_effect"])
        rows = []
        for (cf, ax), sub in events.groupby(["condition_family", "bsv_axis"]):
            if not isinstance(ax, str) or ax not in BSV_AXES:
                continue
            try:
                effs = pd.to_numeric(sub["effect_size"], errors="coerce").dropna()
                me = float(effs.mean()) if len(effs) else 0.0
            except Exception:
                me = 0.0
            dirs = sub["direction"].value_counts()
            dom = dirs.idxmax() if len(dirs) else "ambiguous"
            rows.append({"condition_family": cf, "bsv_axis": ax,
                         "n_events": len(sub),
                         "n_datasets": sub["dataset"].nunique(),
                         "mean_effect": me, "dom_direction": dom})
        cf_axis = pd.DataFrame(rows)

    df = cf_axis.copy()
    if df.empty:
        return df
    df["mean_abs_effect"] = df["mean_effect"].abs()
    if weight_kind == "evidence":
        df["weight"] = df["n_events"] * (df["mean_abs_effect"] + 0.10)
    elif weight_kind == "datasets":
        df["weight"] = df["n_datasets"].astype(float)
    else:
        df["weight"] = df["mean_effect"]
    if normalize and df["weight"].max() > 0:
        df["weight"] = df["weight"] / df["weight"].max()
    df["direction"] = df["dom_direction"]
    return df.sort_values("weight", ascending=False).reset_index(drop=True)


# ─── Hierarchical Sankey ──────────────────────────────────────────────────

def build_hierarchical_sankey(events: pd.DataFrame,
                              show_mss: bool = True,
                              max_edges_per_layer: int = 60,
                              filter_sample_types: list[str] | None = None,
                              filter_datasets: list[str] | None = None) -> dict:
    """Build a 5-layer Sankey: sample_type → dataset → condition_family →
    BSV axis → MSS candidate.

    Returns {"node_label": [...], "node_color": [...], "src": [...],
              "tgt": [...], "value": [...], "edge_color": [...],
              "node_layer": [...] }.
    """
    e = events.copy()
    if filter_sample_types:
        e = e[e["sample_type"].isin(filter_sample_types)]
    if filter_datasets:
        e = e[e["dataset"].isin(filter_datasets)]
    if e.empty:
        return dict(node_label=[], node_color=[], src=[], tgt=[],
                    value=[], edge_color=[], node_layer=[])

    layer_label_to_idx: dict[tuple[int, str], int] = {}
    node_label, node_color, node_layer = [], [], []
    src, tgt, value, edge_color = [], [], [], []

    sample_palette = {"EV": "#79c0ff", "serum": "#bc8cff",
                       "mixed": "#aab7b8", "pure_Raman": "#7ee787",
                       "SERS": "#56d4dd", "unknown": "#6e7681"}
    bsv_palette = {"G01": "#79c0ff", "G02": "#a5d6ff", "G03": "#56d4dd",
                    "G04": "#bc8cff", "G05": "#ffa657", "G06": "#7ee787",
                    "G07": "#d2a8ff", "G08": "#ffdf5d", "G09": "#f0883e",
                    "G10": "#ff7b72", "G11": "#85e89d"}

    def _add(label: str, layer: int, color: str = "#9ecbff") -> int:
        key = (layer, label)
        if key in layer_label_to_idx:
            return layer_label_to_idx[key]
        i = len(node_label)
        layer_label_to_idx[key] = i
        node_label.append(label); node_color.append(color); node_layer.append(layer)
        return i

    def _link(s: int, t: int, v: float, direction: str = "stable") -> None:
        src.append(s); tgt.append(t); value.append(max(0.5, float(v)))
        col = ("rgba(255,123,114,0.45)" if direction == "up"
                else "rgba(121,192,255,0.45)" if direction == "down"
                else "rgba(160,160,160,0.30)")
        edge_color.append(col)

    # Layer 0 → 1: sample_type → dataset
    for (st, ds), sub in e.groupby(["sample_type", "dataset"]):
        s = _add(st, 0, sample_palette.get(str(st), "#9ecbff"))
        t = _add(ds, 1, sample_palette.get(str(st), "#9ecbff"))
        _link(s, t, len(sub))

    # Layer 1 → 2: dataset → condition_family
    for (ds, cf), sub in e.groupby(["dataset", "condition_family"]):
        s_key = (1, ds)
        if s_key not in layer_label_to_idx: continue
        s = layer_label_to_idx[s_key]
        t = _add(cf, 2, "#d2a8ff")
        _link(s, t, len(sub))

    # Layer 2 → 3: condition_family → BSV axis (top per pair by evidence)
    cf_ax_pairs = []
    for (cf, ax), sub in e.groupby(["condition_family", "bsv_axis"]):
        if ax not in BSV_AXES: continue
        try:
            effs = pd.to_numeric(sub["effect_size"], errors="coerce").dropna()
            mean_abs = float(effs.abs().mean()) if len(effs) else 0.0
        except Exception:
            mean_abs = 0.0
        dom = sub["direction"].value_counts().idxmax() if len(sub) else "stable"
        cf_ax_pairs.append((cf, ax, len(sub), mean_abs, dom))
    cf_ax_pairs.sort(key=lambda x: x[2] * (x[3] + 0.1), reverse=True)
    for cf, ax, n_ev, _, dom in cf_ax_pairs[:max_edges_per_layer]:
        s_key = (2, cf)
        if s_key not in layer_label_to_idx: continue
        s = layer_label_to_idx[s_key]
        t = _add(f"{ax} · {dom}",
                 3, bsv_palette.get(ax, "#79c0ff"))
        _link(s, t, n_ev, dom)

    # Layer 3 → 4: BSV axis (via condition_family) → MSS candidate
    if show_mss:
        ax_mss_pairs = []
        for (ax, m), sub in e.groupby(["bsv_axis", "mss_candidate"]):
            if ax not in BSV_AXES or not isinstance(m, str) or not m:
                continue
            ax_mss_pairs.append((ax, m, len(sub)))
        ax_mss_pairs.sort(key=lambda x: x[2], reverse=True)
        for ax, m, n_ev in ax_mss_pairs[:max_edges_per_layer]:
            # find the axis nodes (could be with multiple "ax · dom" labels)
            for (layer, lbl), idx in list(layer_label_to_idx.items()):
                if layer == 3 and lbl.startswith(f"{ax} ·"):
                    t = _add(m, 4, "#85e89d")
                    _link(idx, t, n_ev)
                    break

    return dict(node_label=node_label, node_color=node_color,
                node_layer=node_layer, src=src, tgt=tgt, value=value,
                edge_color=edge_color)


# ─── MSS Transfer (bipartite MSS ↔ {dataset|condition|sample_type}) ──────

def build_mss_transfer_edges(events: pd.DataFrame,
                             group_by: str = "dataset",
                             classification_filter: list[str] | None = None,
                             top_n: int = 25,
                             mss_classes: pd.DataFrame | None = None,
                             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (edges, nodes) where edges link MSS candidate to a chosen
    grouping (dataset / condition_family / sample_type) with direction +
    evidence count."""
    e = events.dropna(subset=["mss_candidate"])
    e = e[e["mss_candidate"].astype(str) != ""]
    if classification_filter and mss_classes is not None and not mss_classes.empty:
        keep = mss_classes[
            mss_classes["classification"].isin(classification_filter)
        ]["mss_candidate"].tolist()
        e = e[e["mss_candidate"].isin(keep)]

    if e.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Top N by total events
    top_mss = (e["mss_candidate"].value_counts().head(top_n).index.tolist())
    e = e[e["mss_candidate"].isin(top_mss)]

    rows = []
    for (m, g), sub in e.groupby(["mss_candidate", group_by]):
        if not g or pd.isna(g):
            continue
        dirs = sub["direction"].value_counts()
        dom = dirs.idxmax() if len(dirs) else "stable"
        try:
            effs = pd.to_numeric(sub["effect_size"], errors="coerce").dropna()
            me = float(effs.abs().mean()) if len(effs) else 0.0
        except Exception:
            me = 0.0
        rows.append({"mss_candidate": m, group_by: g,
                     "weight": len(sub), "mean_abs_effect": round(me, 3),
                     "direction": dom})
    edges = pd.DataFrame(rows)

    nodes = []
    for m in top_mss:
        sub = e[e["mss_candidate"] == m]
        nodes.append({"node_id": f"MSS::{m}", "label": m, "kind": "MSS",
                      "size": int(sub["dataset"].nunique() * 4 + 6),
                      "n_datasets": int(sub["dataset"].nunique()),
                      "n_events": int(len(sub))})
    for g in edges[group_by].unique():
        sub = e[e[group_by] == g]
        nodes.append({"node_id": f"{group_by}::{g}", "label": str(g),
                      "kind": group_by,
                      "size": int(sub["mss_candidate"].nunique() * 3 + 8),
                      "n_datasets": "",
                      "n_events": int(len(sub))})
    return edges, pd.DataFrame(nodes)
