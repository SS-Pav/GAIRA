"""Tab 7 — Context embeddings (UMAP / PCA) with short labels and hulls."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from components import ui_blocks as ui


def _convex_hull_xy(points: np.ndarray):
    if len(points) < 3:
        return None
    try:
        from scipy.spatial import ConvexHull
        h = ConvexHull(points)
        idx = list(h.vertices) + [h.vertices[0]]
        return points[idx]
    except Exception:
        return None


def _ellipse_xy(points: np.ndarray, n_sigma: float = 2.0,
                 n_samples: int = 80):
    if len(points) < 3:
        return None
    try:
        mean = points.mean(axis=0)
        cov = np.cov(points.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-8, None)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
        a = n_sigma * np.sqrt(eigvals[0])
        b = n_sigma * np.sqrt(eigvals[1])
        theta = np.linspace(0, 2 * np.pi, n_samples)
        circle = np.column_stack([a * np.cos(theta), b * np.sin(theta)])
        return (circle @ eigvecs.T) + mean
    except Exception:
        return None


def render(ctx: dict, app_cfg: dict) -> None:
    st.markdown("# Context embeddings")
    st.caption("Each point is one dataset, embedded in BSV-effect space "
                "(11-dim feature → PCA / UMAP).")

    feats = ctx.get("embedding_v2")
    if feats is None or feats.empty:
        ui.warning_card("Embedding table missing.")
        return

    st_palette = app_cfg.get("sample_type_colors", {})

    cols = st.columns([1, 1.2, 1.2, 1])
    with cols[0]:
        view = st.radio("View", options=["UMAP", "PCA"], index=0,
                          horizontal=True, key="cge7_view")
    with cols[1]:
        color_by = st.selectbox(
            "Colour by",
            options=["sample_type", "condition_family",
                       "specific_condition", "caveat burden"],
            index=0, key="cge7_color")
    with cols[2]:
        size_by = st.selectbox(
            "Point size by",
            options=["evidence count", "caveat burden", "fixed"],
            index=0, key="cge7_size")
    with cols[3]:
        show_labels = st.checkbox("Show labels", value=True,
                                    key="cge7_labels")

    cols_b = st.columns(2)
    with cols_b[0]:
        hull_st = st.checkbox("Hull · sample_type", value=True,
                                key="cge7_hull_st")
    with cols_b[1]:
        ellipse_st = st.checkbox("Ellipse · sample_type", value=False,
                                    key="cge7_ell_st")

    df = feats.copy()
    # Pull condition info from events if available
    ev = ctx.get("events_v2")
    if ev is not None and not ev.empty:
        # Map dataset → most-common specific_condition
        first_cond = (ev.groupby("dataset")["specific_condition"]
                      .agg(lambda x: x.value_counts().idxmax()
                           if len(x.value_counts()) else "")
                      .rename("specific_condition_first").reset_index())
        df = df.merge(first_cond, on="dataset", how="left")

    if view == "UMAP":
        xcol, ycol = "umap_1", "umap_2"
    else:
        xcol, ycol = "pca_1", "pca_2"

    # color mapping
    if color_by == "caveat burden":
        # numeric continuous
        col_vals = df["caveat_burden"]
    else:
        if color_by == "specific_condition":
            col_field = "specific_condition_first"
        else:
            col_field = color_by
        groups = sorted(df[col_field].dropna().unique())
        # use a stable palette
        if color_by == "sample_type":
            cmap = {g: st_palette.get(g, "#9ecbff") for g in groups}
        else:
            base = ["#79c0ff", "#bc8cff", "#ffa657", "#7ee787", "#56d4dd",
                     "#ff7b72", "#ffdf5d", "#d2a8ff", "#a5d6ff", "#85e89d",
                     "#aab7b8", "#f0883e", "#ff9492"]
            cmap = {g: base[i % len(base)] for i, g in enumerate(groups)}
        df["__color"] = df[col_field].map(cmap).fillna("#9ecbff")

    if size_by == "evidence count":
        df["__size"] = (np.log1p(df["evidence_count"].clip(lower=0)) * 6 + 8)
    elif size_by == "caveat burden":
        df["__size"] = df["caveat_burden"].clip(0, 60) / 4 + 10
    else:
        df["__size"] = 14

    fig = go.Figure()

    # Hulls / ellipses (sample_type)
    if hull_st or ellipse_st:
        for st_lbl in df["sample_type"].dropna().unique():
            sub = df[df["sample_type"] == st_lbl]
            if len(sub) < 3:
                continue
            color = st_palette.get(st_lbl, "#9ecbff")
            pts = sub[[xcol, ycol]].values
            poly = (_convex_hull_xy(pts) if hull_st
                     else _ellipse_xy(pts, n_sigma=2.0))
            if poly is None:
                continue
            fig.add_trace(go.Scatter(
                x=poly[:, 0], y=poly[:, 1],
                fill="toself",
                fillcolor=f"rgba(125,160,200,0.07)",
                line=dict(color=color, width=1, dash="dot"),
                mode="lines", hoverinfo="skip", showlegend=False,
                name=f"hull_{st_lbl}",
            ))

    # Points
    if color_by == "caveat burden":
        fig.add_trace(go.Scatter(
            x=df[xcol], y=df[ycol], mode="markers+text" if show_labels else "markers",
            marker=dict(color=df["caveat_burden"], colorscale="OrRd",
                          size=df["__size"], showscale=True,
                          colorbar=dict(title="caveat<br>burden",
                                          tickfont=dict(color="#c9d1d9"),
                                          title_font=dict(color="#c9d1d9")),
                          line=dict(color="#0d1117", width=0.5)),
            text=df["short_label"] if show_labels else None,
            textposition="top center",
            textfont=dict(size=9, color="#c9d1d9"),
            hovertext=[(f"<b>{r['short_label']}</b><br>"
                        f"<span style='color:#8b949e;'>{r['dataset']}</span><br>"
                        f"sample={r['sample_type']} · cf={r['condition_family']}<br>"
                        f"specific={r.get('specific_condition_first', '')}<br>"
                        f"events={int(r['evidence_count'])} · "
                        f"caveat={int(r['caveat_burden'])}")
                       for _, r in df.iterrows()],
            hoverinfo="text", showlegend=False))
    else:
        col_field = ("specific_condition_first" if color_by == "specific_condition"
                      else color_by)
        for label, sub in df.groupby(col_field):
            color = sub["__color"].iloc[0] if "__color" in sub.columns else "#79c0ff"
            fig.add_trace(go.Scatter(
                x=sub[xcol], y=sub[ycol],
                mode="markers+text" if show_labels else "markers",
                name=str(label),
                marker=dict(color=color, size=sub["__size"],
                              line=dict(color="#0d1117", width=0.5)),
                text=sub["short_label"] if show_labels else None,
                textposition="top center",
                textfont=dict(size=9, color="#c9d1d9"),
                hovertext=[(f"<b>{r['short_label']}</b><br>"
                            f"<span style='color:#8b949e;'>{r['dataset']}</span><br>"
                            f"sample={r['sample_type']} · cf={r['condition_family']}<br>"
                            f"specific={r.get('specific_condition_first', '')}<br>"
                            f"events={int(r['evidence_count'])} · "
                            f"caveat={int(r['caveat_burden'])}")
                           for _, r in sub.iterrows()],
                hoverinfo="text"))

    fig.update_layout(template="plotly_dark", height=620,
                       plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                       margin=dict(l=10, r=10, t=44, b=10),
                       title=dict(
                           text=f"{view} · coloured by {color_by} · "
                                f"size by {size_by}",
                           font=dict(size=12, color="#c9d1d9")),
                       xaxis=dict(title=xcol, gridcolor="#21262d"),
                       yaxis=dict(title=ycol, gridcolor="#21262d"),
                       legend=dict(font=dict(size=10, color="#c9d1d9"),
                                    bgcolor="rgba(13,17,23,0.6)"),
                       hoverlabel=dict(bgcolor="#161b22", font_color="#c9d1d9",
                                        bordercolor="#30363d"))
    st.plotly_chart(fig, use_container_width=True)

    # nearest-neighbour table
    if len(df) >= 2 and "umap_1" in df.columns:
        from scipy.spatial.distance import cdist
        xs = df[[xcol, ycol]].values
        D = cdist(xs, xs)
        np.fill_diagonal(D, np.inf)
        nn_idx = D.argmin(axis=1)
        nn_dist = D.min(axis=1)
        nn = pd.DataFrame({
            "dataset": df["short_label"].values,
            "sample_type": df["sample_type"].values,
            "nearest neighbour": df["short_label"].values[nn_idx],
            "neighbour sample_type": df["sample_type"].values[nn_idx],
            "distance": np.round(nn_dist, 3),
        }).sort_values("distance")
        ui.section_header("Nearest-neighbour table",
                           "Closest dataset in this embedding space")
        st.dataframe(nn, use_container_width=True, hide_index=True)

    ui.interpretation(
        "Reading the embedding",
        "<ul style='margin: 4px 0 4px 18px;'>"
        "<li><strong>Distance</strong> reflects similarity in GAIRA "
        "BSV/MSS effect space — <em>not</em> raw spectral similarity.</li>"
        "<li><strong>EV and serum</strong> form distinct neighbourhoods "
        "(median UMAP distance ≈ 1.5 in the source corpus).</li>"
        "<li><strong>Mixed / calibration</strong> datasets sit between "
        "biological contexts because their effect vectors aggregate cohorts.</li>"
        "</ul>")
