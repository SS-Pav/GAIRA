"""
Spectral-mode trust graph — cohort-specific traversal.

Builds a 5-column graph per cohort:
  Cohort → Windows → Motifs → Themes → BSV
"""
from __future__ import annotations

import re

import numpy as np
import plotly.graph_objects as go

from gaira.spectral.window_panel import BSV_COMPONENTS, WINDOW_DEFS
from gaira.spectral.bsv_projection import CohortBSV


# Motif definitions with peak regions (reused from text-query motif mapper)
_MOTIF_DEFS = {
    "CH2/CH3":        {"windows": ["1380-1450"], "theme": "membrane_lipid", "peaks": "1380-1450 cm⁻¹"},
    "lipid band":     {"windows": ["1140-1200"], "theme": "membrane_lipid", "peaks": "1140-1200 cm⁻¹"},
    "amide III":      {"windows": ["1200-1260", "1260-1320"], "theme": "protein_backbone", "peaks": "1200-1320 cm⁻¹"},
    "amide II":       {"windows": ["1450-1520"], "theme": "protein_backbone", "peaks": "1450-1520 cm⁻¹"},
    "phenylalanine":  {"windows": ["980-1020"], "theme": "aromatic_amino_acid", "peaks": "980-1020 cm⁻¹"},
    "tyrosine":       {"windows": ["620-660", "820-860"], "theme": "aromatic_amino_acid", "peaks": "620-860 cm⁻¹"},
    "aromatic ring":  {"windows": ["1520-1600"], "theme": "aromatic_amino_acid", "peaks": "1520-1600 cm⁻¹"},
    "purine bases":   {"windows": ["660-700", "700-740", "1320-1380"], "theme": "purine_nucleotide", "peaks": "660-740, 1320-1380 cm⁻¹"},
    "pyrimidine":     {"windows": ["740-780", "780-820"], "theme": "pyrimidine_nucleotide", "peaks": "740-820 cm⁻¹"},
    "glycan/C-O":     {"windows": ["860-920", "1080-1140"], "theme": "glycan_carbohydrate", "peaks": "860-920, 1080-1140 cm⁻¹"},
    "disulfide/S-S":  {"windows": ["450-500", "500-540"], "theme": "redox_metabolite", "peaks": "450-540 cm⁻¹"},
    "PO₂ backbone":   {"windows": ["1020-1080"], "theme": "nucleic_acid_backbone", "peaks": "1020-1080 cm⁻¹"},
    "protein/C-C":    {"windows": ["920-980"], "theme": "protein_backbone", "peaks": "920-980 cm⁻¹"},
}

THEME_DISPLAY = {
    "membrane_lipid": "Lipid", "protein_backbone": "Protein",
    "aromatic_amino_acid": "Aromatic AA", "purine_nucleotide": "Purine",
    "pyrimidine_nucleotide": "Pyrimidine", "glycan_carbohydrate": "Glycan",
    "redox_metabolite": "Redox", "nucleic_acid_backbone": "Nuc. Backbone",
}

COL_X = {0: 0.0, 1: 0.20, 2: 0.42, 3: 0.64, 4: 0.88}
Y_SPACING = 0.065
BG = "#1a1a2e"

NODE_COLORS = {
    "cohort": "#ECF0F1", "window": "#3498DB", "motif": "#E8D44D",
    "theme": "#1ABC9C", "bsv": "#9B59B6",
}


def build_cohort_graph(
    cohort_name: str,
    cohort_bsv: CohortBSV,
    window_features: np.ndarray,
    cohort_mask: np.ndarray,
) -> dict:
    """Build a spectral traversal graph for one cohort.

    Columns: Cohort → Active Windows → Motifs → Themes → BSV Components
    """
    nodes = []
    edges = []

    # Mean window values for this cohort
    mean_windows = window_features[cohort_mask].mean(axis=0)

    # ── Col 0: Cohort ──────────────────────────────────────────
    nodes.append({
        "id": "cohort", "label": cohort_name.replace("_", " "),
        "column": 0, "row": 0, "node_type": "cohort",
        "color": NODE_COLORS["cohort"],
    })

    # ── Col 1: Active windows (top by intensity) ───────────────
    window_ranked = sorted(range(len(WINDOW_DEFS)), key=lambda i: -mean_windows[i])
    top_windows = window_ranked[:8]  # show top 8

    for ri, wi in enumerate(top_windows):
        wid, ws, we, comp = WINDOW_DEFS[wi]
        nid = f"win_{wi}"
        nodes.append({
            "id": nid, "label": f"{ws}-{we}",
            "column": 1, "row": ri, "node_type": "window",
            "color": NODE_COLORS["window"],
            "detail": f"{wid} cm⁻¹ → {comp}\nmean={mean_windows[wi]:.4f}",
        })
        edges.append({"from": "cohort", "to": nid, "weight": 1.0})

    # ── Col 2: Motifs (from window→motif mapping) ──────────────
    active_win_ids = {WINDOW_DEFS[wi][0] for wi in top_windows}
    motif_row = 0
    active_motifs = {}
    for mname, mdef in _MOTIF_DEFS.items():
        if any(w in active_win_ids for w in mdef["windows"]):
            mid = f"motif_{motif_row}"
            active_motifs[mname] = mid
            nodes.append({
                "id": mid, "label": mname[:18],
                "column": 2, "row": motif_row, "node_type": "motif",
                "color": NODE_COLORS["motif"],
                "peaks": mdef["peaks"],
                "detail": f"Peaks: {mdef['peaks']}\nTheme: {mdef['theme']}",
            })
            # Connect windows → motif
            for w in mdef["windows"]:
                wi_match = next((i for i, (wid, _, _, _) in enumerate(WINDOW_DEFS) if wid == w), None)
                if wi_match is not None and wi_match in top_windows:
                    edges.append({"from": f"win_{wi_match}", "to": mid, "weight": 1.0})
            motif_row += 1

    # ── Col 3: Themes ──────────────────────────────────────────
    active_themes = {}
    theme_row = 0
    for mname, mid in active_motifs.items():
        theme = _MOTIF_DEFS[mname]["theme"]
        if theme not in active_themes:
            tid = f"theme_{theme_row}"
            active_themes[theme] = tid
            nodes.append({
                "id": tid, "label": THEME_DISPLAY.get(theme, theme),
                "column": 3, "row": theme_row, "node_type": "theme",
                "color": NODE_COLORS["theme"],
            })
            theme_row += 1
        edges.append({"from": mid, "to": active_themes[theme], "weight": 1.0})

    # ── Col 4: BSV components ──────────────────────────────────
    bsv_row = 0
    for ci, comp in enumerate(BSV_COMPONENTS):
        val = cohort_bsv.mean_bsv[comp]
        if comp not in active_themes:
            continue
        bid = f"bsv_{ci}"
        nodes.append({
            "id": bid, "label": f"{THEME_DISPLAY.get(comp, comp)} ({val:.4f})",
            "column": 4, "row": bsv_row, "node_type": "bsv",
            "color": NODE_COLORS["bsv"],
        })
        edges.append({"from": active_themes[comp], "to": bid, "weight": 1.0})
        bsv_row += 1

    return {
        "nodes": nodes, "edges": edges,
        "n_windows": len(top_windows),
        "n_motifs": len(active_motifs),
        "n_themes": len(active_themes),
        "n_bsv": bsv_row,
        "column_labels": ["Cohort", "Windows", "Motifs", "Themes", "BSV"],
    }


def render_cohort_graph(graph_data: dict, height: int = 340) -> go.Figure:
    """Render a cohort traversal graph using plotly."""
    nodes = graph_data["nodes"]
    edges = graph_data["edges"]
    col_labels = graph_data.get("column_labels", [])

    # Compute positions
    col_counts: dict[int, int] = {}
    for n in nodes:
        col = n["column"]
        col_counts[col] = col_counts.get(col, 0) + 1

    pos: dict[str, tuple[float, float]] = {}
    col_row: dict[int, int] = {i: 0 for i in range(5)}
    for n in nodes:
        col = n["column"]
        total = col_counts.get(col, 1)
        row = col_row[col]
        col_row[col] = row + 1
        x = COL_X[col]
        h = (total - 1) * Y_SPACING
        y = 0.5 + h / 2 - row * Y_SPACING
        pos[n["id"]] = (x, y)

    # Edges
    ex, ey = [], []
    for e in edges:
        if e["from"] in pos and e["to"] in pos:
            x0, y0 = pos[e["from"]]
            x1, y1 = pos[e["to"]]
            ex.extend([x0, x1, None])
            ey.extend([y0, y1, None])

    traces = []
    if ex:
        traces.append(go.Scatter(
            x=ex, y=ey, mode="lines",
            line=dict(color="rgba(255,255,255,0.12)", width=1),
            hoverinfo="skip", showlegend=False,
        ))

    # Nodes
    SYMBOLS = {"cohort": "square", "window": "circle", "motif": "hexagon2",
               "theme": "pentagon", "bsv": "diamond"}
    SIZES = {"cohort": 16, "window": 8, "motif": 10, "theme": 12, "bsv": 13}

    type_groups: dict[str, dict] = {}
    for n in nodes:
        nt = n["node_type"]
        if nt not in type_groups:
            type_groups[nt] = {"x": [], "y": [], "text": [], "hover": [],
                               "size": [], "color": n["color"],
                               "symbol": SYMBOLS.get(nt, "circle")}
        g = type_groups[nt]
        x, y = pos[n["id"]]
        g["x"].append(x); g["y"].append(y)
        g["text"].append(n["label"])
        g["size"].append(SIZES.get(nt, 9))
        detail = n.get("detail", "")
        peaks = n.get("peaks", "")
        hover = f"<b>{n['label']}</b>"
        if peaks:
            hover += f"<br>Peaks: {peaks}"
        if detail:
            hover += f"<br>{detail}"
        g["hover"].append(hover)

    LEGEND = {"cohort": "Cohort", "window": "Window", "motif": "Motif",
              "theme": "Theme", "bsv": "BSV"}
    for nt, g in type_groups.items():
        traces.append(go.Scatter(
            x=g["x"], y=g["y"], mode="markers+text",
            marker=dict(color=g["color"], size=g["size"], symbol=g["symbol"],
                        line=dict(color="rgba(255,255,255,0.3)", width=0.8)),
            text=g["text"], textposition="middle right",
            textfont=dict(size=9, color="rgba(255,255,255,0.85)"),
            hovertext=g["hover"], hoverinfo="text",
            name=LEGEND.get(nt, nt), showlegend=True,
        ))

    annotations = [
        dict(x=COL_X.get(i, 0), y=1.05, text=f"<b>{lbl}</b>", showarrow=False,
             font=dict(size=10, color="rgba(255,255,255,0.5)"), xanchor="center")
        for i, lbl in enumerate(col_labels)
    ]

    fig = go.Figure(data=traces)
    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG,
        margin=dict(l=10, r=10, t=40, b=10), height=height,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.06, 1.06]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.12]),
        annotations=annotations,
        legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5,
                    font=dict(size=8, color="rgba(255,255,255,0.6)"), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#2d2d44", font_size=10, font_color="white"),
    )
    return fig
