"""Plotly rendering helpers — bipartite, layered network, Sankey."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ─── shared layout style ──────────────────────────────────────────────────

DARK_LAYOUT = dict(
    template="plotly_dark",
    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
    margin=dict(l=10, r=10, t=44, b=10),
    hoverlabel=dict(bgcolor="#161b22", font_color="#c9d1d9",
                    bordercolor="#30363d"),
)


# ─── Bipartite condition × axis ───────────────────────────────────────────

def render_bipartite_condition_axis(edges_df: pd.DataFrame,
                                    bsv_palette: dict[str, str],
                                    direction_palette: dict[str, str],
                                    weight_threshold: float = 0.0,
                                    height: int = 620,
                                    title: str = "") -> go.Figure:
    """Bipartite layout: condition_family on left, BSV axes on right."""
    if edges_df is None or edges_df.empty:
        return go.Figure()
    df = edges_df[edges_df["weight"] >= weight_threshold].copy()
    if df.empty:
        return go.Figure()

    # Position nodes
    cf_list = sorted(df["condition_family"].unique())
    ax_order = [f"G{i:02d}" for i in range(1, 12)]
    ax_list = [a for a in ax_order if a in df["bsv_axis"].unique()]

    cf_y = {cf: i for i, cf in enumerate(cf_list)}
    ax_y = {ax: i for i, ax in enumerate(ax_list)}

    fig = go.Figure()

    # Edges
    max_w = float(df["weight"].max() or 1)
    for _, e in df.iterrows():
        cf = e["condition_family"]; ax = e["bsv_axis"]
        if cf not in cf_y or ax not in ax_y:
            continue
        y0 = cf_y[cf] * (max(len(ax_list), 1) / max(len(cf_list), 1))
        y1 = ax_y[ax]
        color = direction_palette.get(str(e.get("direction", "stable")),
                                      "#aab7b8")
        width = 0.6 + 4.5 * (float(e["weight"]) / max_w)
        opacity = 0.30 + 0.55 * (float(e["weight"]) / max_w)
        rgba = color
        if rgba.startswith("#") and len(rgba) == 7:
            rgba = (f"rgba({int(rgba[1:3],16)},{int(rgba[3:5],16)},"
                    f"{int(rgba[5:7],16)},{opacity:.2f})")
        sub_label = (f"{cf} → {ax}<br>weight={e['weight']:.2f}"
                      f"<br>dir={e.get('direction','')}"
                      f"<br>n_datasets={int(e.get('n_datasets', 0))}"
                      f"<br>n_events={int(e.get('n_events', 0))}"
                      f"<br>mean_effect={e.get('mean_effect', 0):.2f}")
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[y0, y1], mode="lines",
            line=dict(color=rgba, width=width),
            hovertemplate=sub_label + "<extra></extra>",
            showlegend=False, name=f"{cf}-{ax}",
        ))

    # Left nodes (condition families)
    for cf, y in cf_y.items():
        y_norm = y * (max(len(ax_list), 1) / max(len(cf_list), 1))
        fig.add_trace(go.Scatter(
            x=[0], y=[y_norm], mode="markers+text",
            marker=dict(size=18, color="#d2a8ff",
                        line=dict(color="#0d1117", width=1)),
            text=[cf], textposition="middle left",
            textfont=dict(color="#c9d1d9", size=11),
            hovertemplate=f"<b>{cf}</b><extra></extra>",
            showlegend=False, name=f"cf::{cf}",
        ))

    # Right nodes (axes)
    for ax, y in ax_y.items():
        color = bsv_palette.get(ax, "#79c0ff")
        fig.add_trace(go.Scatter(
            x=[1], y=[y], mode="markers+text",
            marker=dict(size=22, color=color,
                        line=dict(color="#0d1117", width=1)),
            text=[f"<b>{ax}</b>"], textposition="middle right",
            textfont=dict(color=color, size=11),
            hovertemplate=f"<b>{ax}</b><extra></extra>",
            showlegend=False, name=f"ax::{ax}",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#c9d1d9")),
        height=height,
        xaxis=dict(visible=False, range=[-0.35, 1.35]),
        yaxis=dict(visible=False, autorange="reversed"),
        **DARK_LAYOUT,
    )
    return fig


# ─── Hierarchical Sankey ──────────────────────────────────────────────────

def render_hierarchical_sankey(sk: dict, title: str = "",
                               height: int = 660) -> go.Figure:
    if not sk or not sk.get("node_label"):
        return go.Figure()
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=sk["node_label"],
            color=sk["node_color"],
            pad=14, thickness=14,
            line=dict(color="#0d1117", width=0.5),
        ),
        link=dict(
            source=sk["src"], target=sk["tgt"],
            value=sk["value"], color=sk["edge_color"],
        ),
    ))
    fig.update_layout(title=dict(text=title, font=dict(size=13, color="#c9d1d9")),
                       height=height, **DARK_LAYOUT)
    return fig


# ─── Bipartite MSS transfer ──────────────────────────────────────────────

def render_bipartite_mss(edges_df: pd.DataFrame, nodes_df: pd.DataFrame,
                         group_by: str, direction_palette: dict[str, str],
                         height: int = 660,
                         title: str = "") -> go.Figure:
    if edges_df is None or edges_df.empty:
        return go.Figure()

    mss_list = (nodes_df[nodes_df["kind"] == "MSS"]["label"]
                .tolist() if nodes_df is not None else
                sorted(edges_df["mss_candidate"].unique().tolist()))
    grp_list = (nodes_df[nodes_df["kind"] == group_by]["label"]
                 .tolist() if nodes_df is not None else
                 sorted(edges_df[group_by].unique().tolist()))

    mss_y = {m: i for i, m in enumerate(mss_list)}
    grp_y = {g: i for i, g in enumerate(grp_list)}

    fig = go.Figure()
    max_w = float(edges_df["weight"].max() or 1)
    for _, e in edges_df.iterrows():
        m = e["mss_candidate"]; g = e[group_by]
        if m not in mss_y or g not in grp_y:
            continue
        y0 = mss_y[m] * (max(len(grp_list), 1) / max(len(mss_list), 1))
        y1 = grp_y[g]
        color = direction_palette.get(str(e.get("direction", "stable")),
                                      "#aab7b8")
        width = 0.6 + 4.0 * (float(e["weight"]) / max_w)
        opacity = 0.25 + 0.55 * (float(e["weight"]) / max_w)
        rgba = color
        if rgba.startswith("#") and len(rgba) == 7:
            rgba = (f"rgba({int(rgba[1:3],16)},{int(rgba[3:5],16)},"
                    f"{int(rgba[5:7],16)},{opacity:.2f})")
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[y0, y1], mode="lines",
            line=dict(color=rgba, width=width),
            hovertemplate=(f"<b>{m} ↔ {g}</b><br>"
                           f"events={int(e['weight'])} · "
                           f"mean_abs_effect={e.get('mean_abs_effect', 0):.2f} · "
                           f"dir={e.get('direction','')}<extra></extra>"),
            showlegend=False, name=f"{m}-{g}",
        ))

    # MSS nodes (left)
    for m, y in mss_y.items():
        y_n = y * (max(len(grp_list), 1) / max(len(mss_list), 1))
        sz = (nodes_df[nodes_df.label == m]["size"].iloc[0]
              if nodes_df is not None and (nodes_df.label == m).any() else 12)
        fig.add_trace(go.Scatter(
            x=[0], y=[y_n], mode="markers+text",
            marker=dict(size=sz, color="#85e89d",
                        line=dict(color="#0d1117", width=1)),
            text=[m], textposition="middle left",
            textfont=dict(color="#c9d1d9", size=11),
            hovertemplate=f"<b>{m}</b><extra></extra>",
            showlegend=False, name=f"mss::{m}",
        ))

    # Group nodes (right)
    for g, y in grp_y.items():
        sz = (nodes_df[nodes_df.label == g]["size"].iloc[0]
              if nodes_df is not None and (nodes_df.label == g).any() else 12)
        fig.add_trace(go.Scatter(
            x=[1], y=[y], mode="markers+text",
            marker=dict(size=sz, color="#bc8cff",
                        line=dict(color="#0d1117", width=1)),
            text=[g[:36]], textposition="middle right",
            textfont=dict(color="#c9d1d9", size=10),
            hovertemplate=f"<b>{g}</b><extra></extra>",
            showlegend=False, name=f"grp::{g}",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#c9d1d9")),
        height=height,
        xaxis=dict(visible=False, range=[-0.35, 1.45]),
        yaxis=dict(visible=False, autorange="reversed"),
        **DARK_LAYOUT,
    )
    return fig
