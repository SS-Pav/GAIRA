"""
Plotly-based trust graph renderer — GAIRA-native 6-column flow (Phase 7A).

Renders:
  Query → Evidence → Motifs → Themes → BSV → Output
with fixed positions, dark theme, edge-weight opacity.
"""
from __future__ import annotations

import plotly.graph_objects as go

from gaira.retrieval.trust_graph_builder import (
    NODE_TYPE_COLORS,
    TIER_COLORS,
    TIER_DISPLAY,
)


COL_X = {0: 0.0, 1: 0.16, 2: 0.36, 3: 0.56, 4: 0.76, 5: 0.96}
BG_COLOR = "#1a1a2e"
TEXT_COLOR = "rgba(255,255,255,0.85)"
Y_SPACING = 0.065
LANE_GAP = 0.10  # visual gap between grounding and literature lanes


def _compute_positions(nodes: list[dict]) -> dict[str, tuple[float, float]]:
    # For column 1 (evidence), split into grounding and literature lanes
    grounding_nodes = [n for n in nodes if n["column"] == 1 and n.get("lane") == "grounding"]
    literature_nodes = [n for n in nodes if n["column"] == 1 and n.get("lane") == "literature"]
    other_nodes = [n for n in nodes if n["column"] != 1]

    # Count non-evidence columns
    col_counts: dict[int, int] = {}
    for n in other_nodes:
        col = n["column"]
        col_counts[col] = col_counts.get(col, 0) + 1

    positions: dict[str, tuple[float, float]] = {}

    # Position evidence column with lane gap
    total_ev = len(grounding_nodes) + len(literature_nodes)
    ev_height = (total_ev - 1) * Y_SPACING + (LANE_GAP if grounding_nodes and literature_nodes else 0)
    y_top = 0.5 + ev_height / 2

    for i, n in enumerate(grounding_nodes):
        positions[n["id"]] = (COL_X[1], y_top - i * Y_SPACING)

    lit_start = y_top - len(grounding_nodes) * Y_SPACING - LANE_GAP
    for i, n in enumerate(literature_nodes):
        positions[n["id"]] = (COL_X[1], lit_start - i * Y_SPACING)

    # Position other columns normally
    col_row: dict[int, int] = {i: 0 for i in range(6)}
    for n in other_nodes:
        col = n["column"]
        total = col_counts.get(col, 1)
        row = col_row[col]
        col_row[col] = row + 1

        x = COL_X[col]
        height = (total - 1) * Y_SPACING
        y = 0.5 + height / 2 - row * Y_SPACING
        positions[n["id"]] = (x, y)

    return positions


def render_trust_graph(graph_data: dict, height: int = 400) -> go.Figure:
    nodes = graph_data["nodes"]
    edges = graph_data["edges"]
    col_labels = graph_data.get("column_labels", [])

    pos = _compute_positions(nodes)
    node_map = {n["id"]: n for n in nodes}

    # ── Edges (batched by weight tier) ─────────────────────────────
    strong_x, strong_y = [], []
    normal_x, normal_y = [], []
    weak_x, weak_y = [], []

    for edge in edges:
        s, d = edge["from"], edge["to"]
        if s not in pos or d not in pos:
            continue
        x0, y0 = pos[s]
        x1, y1 = pos[d]
        w = edge.get("weight", 0.5)

        if w >= 0.9:
            strong_x.extend([x0, x1, None])
            strong_y.extend([y0, y1, None])
        elif w >= 0.5:
            normal_x.extend([x0, x1, None])
            normal_y.extend([y0, y1, None])
        else:
            weak_x.extend([x0, x1, None])
            weak_y.extend([y0, y1, None])

    edge_traces = []
    if strong_x:
        edge_traces.append(go.Scatter(
            x=strong_x, y=strong_y, mode="lines",
            line=dict(color="rgba(255,255,255,0.18)", width=1.5),
            hoverinfo="skip", showlegend=False,
        ))
    if normal_x:
        edge_traces.append(go.Scatter(
            x=normal_x, y=normal_y, mode="lines",
            line=dict(color="rgba(255,255,255,0.10)", width=1.0),
            hoverinfo="skip", showlegend=False,
        ))
    if weak_x:
        edge_traces.append(go.Scatter(
            x=weak_x, y=weak_y, mode="lines",
            line=dict(color="rgba(255,255,255,0.04)", width=0.6),
            hoverinfo="skip", showlegend=False,
        ))

    # ── Nodes (grouped by type for legend) ─────────────────────────
    SYMBOLS = {
        "query": "square", "grounded_evidence": "circle",
        "domain_context": "circle", "benchmark_summary": "circle",
        "spectral_query": "circle", "meta_summary": "circle",
        "motif": "hexagon2", "theme": "pentagon",
        "bsv": "diamond", "output": "star",
    }
    LEGEND_NAMES = {
        "query": "Query", "grounded_evidence": "Grounded",
        "domain_context": "Context", "benchmark_summary": "Benchmark",
        "spectral_query": "Spectral", "meta_summary": "Meta",
        "motif": "Motif", "theme": "Theme",
        "bsv": "BSV Component", "output": "Output",
    }
    SIZES = {
        "query": 16, "motif": 10, "theme": 12,
        "bsv": 13, "output": 16,
    }

    type_groups: dict[str, dict] = {}
    for n in nodes:
        nt = n["node_type"]
        if nt not in type_groups:
            type_groups[nt] = {
                "x": [], "y": [], "text": [], "hover": [],
                "size": [], "color": n["color"],
                "symbol": SYMBOLS.get(nt, "circle"),
            }
        g = type_groups[nt]
        x, y = pos[n["id"]]
        g["x"].append(x)
        g["y"].append(y)
        g["text"].append(n["label"])
        g["size"].append(SIZES.get(nt, 9))

        # Compact hover
        meta = n.get("meta", {})
        if nt == "query":
            hover = f"<b>Query</b><br>{n.get('detail', '')[:80]}"
        elif nt in TIER_COLORS:
            explainer = meta.get("explainer", "")
            sec_title = meta.get("title", "")
            src_path = meta.get("source_path", meta.get("source", ""))
            tier_label = meta.get("tier", nt)
            score = meta.get("score", "")
            hover = (f"<b>{n['label']}</b><br>"
                     f"<i>{explainer}</i><br>"
                     f"Section: {sec_title}<br>"
                     f"File: {src_path}<br>"
                     f"{tier_label} · score {score}")
        elif nt == "motif":
            peaks = n.get("peaks", "")
            peak_line = f"<br>Peaks: {peaks}" if peaks else ""
            hover = f"<b>{n['label']}</b>{peak_line}<br>{n.get('detail', '')}"
        elif nt == "theme":
            hover = f"<b>{n['label']}</b><br>{n.get('detail', '')}"
        elif nt == "bsv":
            hover = f"<b>{n['label']}</b><br>{n.get('detail', '')}"
        else:
            hover = f"<b>{n['label']}</b><br>{n.get('detail', '')[:60]}"
        g["hover"].append(hover)

    node_traces = []
    for nt, g in type_groups.items():
        text_pos = "middle right"
        if nt in ("query", "output"):
            text_pos = "bottom center"

        node_traces.append(go.Scatter(
            x=g["x"], y=g["y"],
            mode="markers+text",
            marker=dict(
                color=g["color"], size=g["size"],
                symbol=g["symbol"],
                line=dict(color="rgba(255,255,255,0.3)", width=0.8),
            ),
            text=g["text"],
            textposition=text_pos,
            textfont=dict(size=9, color=TEXT_COLOR),
            hovertext=g["hover"],
            hoverinfo="text",
            name=LEGEND_NAMES.get(nt, nt),
            showlegend=True,
        ))

    # ── Column headers ─────────────────────────────────────────────
    annotations = []
    for ci, label in enumerate(col_labels):
        annotations.append(dict(
            x=COL_X.get(ci, 0), y=1.05,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=11, color="rgba(255,255,255,0.5)"),
            xanchor="center",
        ))

    # Lane labels for evidence column
    grounding_nodes = [n for n in nodes if n.get("lane") == "grounding"]
    literature_nodes = [n for n in nodes if n.get("lane") == "literature"]
    if grounding_nodes:
        gy = max(pos[n["id"]][1] for n in grounding_nodes) + Y_SPACING * 0.6
        annotations.append(dict(
            x=COL_X[1] - 0.06, y=gy,
            text="<i>Grounding<br>Components</i>", showarrow=False,
            font=dict(size=7, color="rgba(46,204,113,0.5)"),
            xanchor="center",
        ))
    if literature_nodes:
        ly = max(pos[n["id"]][1] for n in literature_nodes) + Y_SPACING * 0.6
        annotations.append(dict(
            x=COL_X[1] - 0.06, y=ly,
            text="<i>Context /<br>Summary</i>", showarrow=False,
            font=dict(size=7, color="rgba(243,156,18,0.5)"),
            xanchor="center",
        ))

    # ── Figure ─────────────────────────────────────────────────────
    fig = go.Figure(data=edge_traces + node_traces)
    fig.update_layout(
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        margin=dict(l=10, r=10, t=45, b=10),
        height=height,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-0.06, 1.06]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-0.15, 1.12]),
        annotations=annotations,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.08,
            xanchor="center", x=0.5,
            font=dict(size=8, color="rgba(255,255,255,0.6)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor="#2d2d44", font_size=10, font_color="white",
            bordercolor="rgba(255,255,255,0.15)",
        ),
    )
    return fig
