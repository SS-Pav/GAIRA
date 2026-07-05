"""Graph-construction helpers for v2 — bipartite condition↔axis (broad &
specific), Sankey with specific conditions, MSS-transfer bipartite."""
from __future__ import annotations

import pandas as pd
import numpy as np


BSV_AXES = [f"G{i:02d}" for i in range(1, 12)]


# ─── Hierarchical Sankey (v2: 5-layer with specific_condition) ────────────

def build_hierarchical_sankey_v2(events: pd.DataFrame,
                                 show_mss: bool = True,
                                 max_edges_per_layer: int = 60,
                                 filter_sample_types: list[str] | None = None,
                                 filter_conditions: list[str] | None = None,
                                 dataset_short_labels: dict[str, str] | None = None,
                                 ) -> dict:
    """5-layer Sankey: sample_type → dataset → specific_condition →
    bsv_axis × direction → mss_candidate."""
    if events is None or events.empty:
        return dict(node_label=[], node_color=[], src=[], tgt=[],
                    value=[], edge_color=[], node_layer=[])
    e = events.copy()
    if filter_sample_types:
        e = e[e["sample_type"].isin(filter_sample_types)]
    if filter_conditions:
        e = e[e["specific_condition"].isin(filter_conditions)]
    if e.empty:
        return dict(node_label=[], node_color=[], src=[], tgt=[],
                    value=[], edge_color=[], node_layer=[])

    short = dataset_short_labels or {}
    sample_palette = {"EV": "#79c0ff", "serum": "#bc8cff",
                       "mixed": "#aab7b8", "pure_Raman": "#7ee787",
                       "SERS": "#56d4dd", "unknown": "#6e7681"}
    bsv_palette = {"G01": "#79c0ff", "G02": "#a5d6ff", "G03": "#56d4dd",
                    "G04": "#bc8cff", "G05": "#ffa657", "G06": "#7ee787",
                    "G07": "#d2a8ff", "G08": "#ffdf5d", "G09": "#f0883e",
                    "G10": "#ff7b72", "G11": "#85e89d"}

    layer_label_to_idx: dict[tuple[int, str], int] = {}
    node_label, node_color, node_layer = [], [], []
    src, tgt, value, edge_color = [], [], [], []

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
        col = ("rgba(255,123,114,0.55)" if direction == "up"
                else "rgba(121,192,255,0.55)" if direction == "down"
                else "rgba(160,160,160,0.30)")
        edge_color.append(col)

    # Layer 0 → 1
    for (st_, ds), sub in e.groupby(["sample_type", "dataset"]):
        s = _add(str(st_), 0, sample_palette.get(str(st_), "#9ecbff"))
        t = _add(short.get(str(ds), str(ds)[:24]), 1,
                 sample_palette.get(str(st_), "#9ecbff"))
        _link(s, t, len(sub))

    # Layer 1 → 2: dataset → specific_condition
    for (ds, cond), sub in e.groupby(["dataset", "specific_condition"]):
        s_key = (1, short.get(str(ds), str(ds)[:24]))
        if s_key not in layer_label_to_idx: continue
        s = layer_label_to_idx[s_key]
        t = _add(str(cond), 2, "#d2a8ff")
        _link(s, t, len(sub))

    # Layer 2 → 3: specific_condition → BSV axis × direction
    cf_ax_pairs = []
    for (cond, ax), sub in e.groupby(["specific_condition", "bsv_axis"]):
        if not isinstance(ax, str) or ax not in BSV_AXES: continue
        try:
            effs = pd.to_numeric(sub["effect_size"], errors="coerce").dropna()
            mean_abs = float(effs.abs().mean()) if len(effs) else 0.0
        except Exception:
            mean_abs = 0.0
        dom = sub["direction"].value_counts().idxmax() if len(sub) else "stable"
        cf_ax_pairs.append((cond, ax, len(sub), mean_abs, dom))
    cf_ax_pairs.sort(key=lambda x: x[2] * (x[3] + 0.1), reverse=True)
    for cond, ax, n_ev, _, dom in cf_ax_pairs[:max_edges_per_layer]:
        s_key = (2, str(cond))
        if s_key not in layer_label_to_idx: continue
        s = layer_label_to_idx[s_key]
        t = _add(f"{ax} · {dom}", 3, bsv_palette.get(ax, "#79c0ff"))
        _link(s, t, n_ev, dom)

    # Layer 3 → 4
    if show_mss:
        ax_mss_pairs = []
        for (ax, m), sub in e.groupby(["bsv_axis", "mss_candidate"]):
            if ax not in BSV_AXES or not isinstance(m, str) or not m:
                continue
            ax_mss_pairs.append((ax, m, len(sub)))
        ax_mss_pairs.sort(key=lambda x: x[2], reverse=True)
        for ax, m, n_ev in ax_mss_pairs[:max_edges_per_layer]:
            for (layer, lbl), idx in list(layer_label_to_idx.items()):
                if layer == 3 and lbl.startswith(f"{ax} ·"):
                    t = _add(m, 4, "#85e89d")
                    _link(idx, t, n_ev)
                    break

    return dict(node_label=node_label, node_color=node_color,
                node_layer=node_layer, src=src, tgt=tgt, value=value,
                edge_color=edge_color)


# ─── MSS transfer edges with sample-type subset support ───────────────────

def build_mss_transfer_edges_v2(events: pd.DataFrame,
                                 group_by: str = "specific_condition",
                                 sample_filter: list[str] | None = None,
                                 classification_filter: list[str] | None = None,
                                 mss_classes: pd.DataFrame | None = None,
                                 top_n: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    e = events.dropna(subset=["mss_candidate"])
    e = e[e["mss_candidate"].astype(str) != ""]
    if sample_filter:
        e = e[e["sample_type"].isin(sample_filter)]
    if classification_filter and mss_classes is not None and not mss_classes.empty:
        keep = mss_classes[
            mss_classes["classification"].isin(classification_filter)
        ]["mss_candidate"].tolist()
        e = e[e["mss_candidate"].isin(keep)]
    if e.empty:
        return pd.DataFrame(), pd.DataFrame()

    top_mss = (e["mss_candidate"].value_counts().head(top_n).index.tolist())
    e = e[e["mss_candidate"].isin(top_mss)]

    rows = []
    for (m, g), sub in e.groupby(["mss_candidate", group_by]):
        if not g or pd.isna(g): continue
        dirs = sub["direction"].value_counts()
        dom = dirs.idxmax() if len(dirs) else "stable"
        try:
            effs = pd.to_numeric(sub["effect_size"], errors="coerce").dropna()
            me = float(effs.abs().mean()) if len(effs) else 0.0
        except Exception:
            me = 0.0
        rows.append({"mss_candidate": m, group_by: g, "weight": len(sub),
                     "mean_abs_effect": round(me, 3), "direction": dom})
    edges = pd.DataFrame(rows)

    nodes = []
    for m in top_mss:
        sub = e[e["mss_candidate"] == m]
        nodes.append({"node_id": f"MSS::{m}", "label": m, "kind": "MSS",
                      "size": int(sub[group_by].nunique() * 4 + 8),
                      "n_datasets": int(sub["dataset"].nunique()),
                      "n_events": int(len(sub))})
    for g in edges[group_by].unique():
        sub = e[e[group_by] == g]
        nodes.append({"node_id": f"{group_by}::{g}", "label": str(g),
                      "kind": group_by,
                      "size": int(sub["mss_candidate"].nunique() * 3 + 8),
                      "n_datasets": "", "n_events": int(len(sub))})
    return edges, pd.DataFrame(nodes)
