"""Plotly rendering helpers for v2."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


DARK_LAYOUT = dict(
    template="plotly_dark",
    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
    margin=dict(l=10, r=10, t=44, b=10),
    hoverlabel=dict(bgcolor="#161b22", font_color="#c9d1d9",
                    bordercolor="#30363d"),
)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    if not hex_color or not hex_color.startswith("#") or len(hex_color) != 7:
        return f"rgba(160,160,160,{alpha:.2f})"
    return (f"rgba({int(hex_color[1:3],16)},{int(hex_color[3:5],16)},"
            f"{int(hex_color[5:7],16)},{alpha:.2f})")


# ─── Bipartite condition × axis ───────────────────────────────────────────

def render_bipartite_condition_axis(edges_df: pd.DataFrame,
                                     left_col: str,
                                     bsv_palette: dict[str, str],
                                     direction_palette: dict[str, str],
                                     weight_threshold: float = 0.0,
                                     top_n: int | None = None,
                                     height: int = 640,
                                     title: str = "") -> go.Figure:
    """Bipartite layout: condition (specific or broad) on left, BSV axes right."""
    if edges_df is None or edges_df.empty:
        return go.Figure()
    df = edges_df[edges_df["weight"] >= weight_threshold].copy()
    if top_n:
        df = df.head(top_n)
    if df.empty:
        return go.Figure()

    cf_list = sorted(df[left_col].unique())
    ax_order = [f"G{i:02d}" for i in range(1, 12)]
    ax_list = [a for a in ax_order if a in df["bsv_axis"].unique()]

    cf_y = {cf: i for i, cf in enumerate(cf_list)}
    ax_y = {ax: i for i, ax in enumerate(ax_list)}
    n_left = max(len(cf_list), 1)
    n_right = max(len(ax_list), 1)

    fig = go.Figure()
    max_w = float(df["weight"].max() or 1)
    for _, e in df.iterrows():
        cf = e[left_col]; ax = e["bsv_axis"]
        if cf not in cf_y or ax not in ax_y:
            continue
        y0 = cf_y[cf] * (n_right / n_left)
        y1 = ax_y[ax]
        color = direction_palette.get(str(e.get("dom_direction",
                                                 e.get("direction", "stable"))),
                                      "#aab7b8")
        width = 0.6 + 4.5 * (float(e["weight"]) / max_w)
        opacity = 0.30 + 0.55 * (float(e["weight"]) / max_w)
        rgba = _hex_to_rgba(color, opacity)
        hover = (f"<b>{cf} → {ax}</b><br>"
                 f"weight={e['weight']:.2f}<br>"
                 f"dir={e.get('dom_direction', '')}<br>"
                 f"n_datasets={int(e.get('n_datasets', 0))}<br>"
                 f"n_events={int(e.get('n_events', 0))}<br>"
                 f"mean_effect={e.get('mean_effect', 0):.2f}")
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[y0, y1], mode="lines",
            line=dict(color=rgba, width=width),
            hovertemplate=hover + "<extra></extra>",
            showlegend=False, name=f"{cf}-{ax}",
        ))

    for cf, y in cf_y.items():
        y_n = y * (n_right / n_left)
        fig.add_trace(go.Scatter(
            x=[0], y=[y_n], mode="markers+text",
            marker=dict(size=18, color="#d2a8ff",
                        line=dict(color="#0d1117", width=1)),
            text=[cf[:36]], textposition="middle left",
            textfont=dict(color="#c9d1d9", size=10),
            hovertemplate=f"<b>{cf}</b><extra></extra>",
            showlegend=False, name=f"cf::{cf}",
        ))
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
        xaxis=dict(visible=False, range=[-0.55, 1.35]),
        yaxis=dict(visible=False, autorange="reversed"),
        **DARK_LAYOUT,
    )
    return fig


# ─── Hierarchical Sankey ──────────────────────────────────────────────────

def render_hierarchical_sankey(sk: dict, title: str = "",
                                height: int = 700) -> go.Figure:
    if not sk or not sk.get("node_label"):
        return go.Figure()
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=sk["node_label"], color=sk["node_color"],
            pad=14, thickness=14,
            line=dict(color="#0d1117", width=0.5),
        ),
        link=dict(source=sk["src"], target=sk["tgt"],
                   value=sk["value"], color=sk["edge_color"]),
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
    mss_list = (nodes_df[nodes_df["kind"] == "MSS"]["label"].tolist()
                 if nodes_df is not None and not nodes_df.empty else
                 sorted(edges_df["mss_candidate"].unique().tolist()))
    grp_list = (nodes_df[nodes_df["kind"] == group_by]["label"].tolist()
                 if nodes_df is not None and not nodes_df.empty else
                 sorted(edges_df[group_by].unique().tolist()))
    mss_y = {m: i for i, m in enumerate(mss_list)}
    grp_y = {g: i for i, g in enumerate(grp_list)}
    n_left = max(len(mss_list), 1); n_right = max(len(grp_list), 1)

    fig = go.Figure()
    max_w = float(edges_df["weight"].max() or 1)
    for _, e in edges_df.iterrows():
        m = e["mss_candidate"]; g = e[group_by]
        if m not in mss_y or g not in grp_y: continue
        y0 = mss_y[m] * (n_right / n_left)
        y1 = grp_y[g]
        color = direction_palette.get(str(e.get("direction", "stable")),
                                      "#aab7b8")
        width = 0.6 + 4.0 * (float(e["weight"]) / max_w)
        opacity = 0.25 + 0.55 * (float(e["weight"]) / max_w)
        rgba = _hex_to_rgba(color, opacity)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[y0, y1], mode="lines",
            line=dict(color=rgba, width=width),
            hovertemplate=(f"<b>{m} ↔ {g}</b><br>"
                            f"events={int(e['weight'])} · "
                            f"mean_abs_effect={e.get('mean_abs_effect', 0):.2f} · "
                            f"dir={e.get('direction','')}<extra></extra>"),
            showlegend=False, name=f"{m}-{g}",
        ))

    for m, y in mss_y.items():
        y_n = y * (n_right / n_left)
        fig.add_trace(go.Scatter(
            x=[0], y=[y_n], mode="markers+text",
            marker=dict(size=14, color="#85e89d",
                        line=dict(color="#0d1117", width=1)),
            text=[m], textposition="middle left",
            textfont=dict(color="#c9d1d9", size=10),
            hovertemplate=f"<b>{m}</b><extra></extra>",
            showlegend=False))
    for g, y in grp_y.items():
        fig.add_trace(go.Scatter(
            x=[1], y=[y], mode="markers+text",
            marker=dict(size=14, color="#bc8cff",
                        line=dict(color="#0d1117", width=1)),
            text=[str(g)[:36]], textposition="middle right",
            textfont=dict(color="#c9d1d9", size=10),
            hovertemplate=f"<b>{g}</b><extra></extra>",
            showlegend=False))

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#c9d1d9")),
        height=height,
        xaxis=dict(visible=False, range=[-0.5, 1.5]),
        yaxis=dict(visible=False, autorange="reversed"),
        **DARK_LAYOUT,
    )
    return fig


# ─── Heatmap helper ──────────────────────────────────────────────────────

def render_heatmap(values: np.ndarray, x_labels: list[str], y_labels: list[str],
                    title: str = "", colorscale: str = "Blues",
                    zmin: float | None = None, zmax: float | None = None,
                    show_text: bool = True, height: int = 460) -> go.Figure:
    text = ([[f"{v:.2f}" if abs(v) >= 0.10 else "" for v in row]
             for row in values] if show_text else None)
    fig = go.Figure(go.Heatmap(
        z=values, x=x_labels, y=y_labels,
        colorscale=colorscale, zmin=zmin, zmax=zmax,
        text=text, texttemplate=("%{text}" if show_text else None),
        textfont=dict(color="#0d1117", size=10),
        colorbar=dict(tickfont=dict(color="#c9d1d9"),
                      title_font=dict(color="#c9d1d9")),
        hovertemplate="x=%{x}<br>y=%{y}<br>z=%{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color="#c9d1d9")),
        height=height,
        xaxis=dict(side="bottom", tickangle=-30,
                    tickfont=dict(size=10, color="#c9d1d9")),
        yaxis=dict(autorange="reversed",
                    tickfont=dict(size=10, color="#c9d1d9")),
        **DARK_LAYOUT,
    )
    return fig
