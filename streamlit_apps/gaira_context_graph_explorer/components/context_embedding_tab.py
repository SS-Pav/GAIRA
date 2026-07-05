"""Tab 5 — Dataset embedding in BSV-effect space (PCA / UMAP)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from components import ui_blocks as ui
from utils.load_context_data import figure_path


def _scatter(features: pd.DataFrame, view: str, color_by: str,
             show_labels: bool, point_size_by: str,
             palette_st: dict, palette_cf: dict,
             caveats: pd.DataFrame | None) -> go.Figure:
    if view == "UMAP":
        xcol, ycol = "umap_1", "umap_2"
    else:
        xcol, ycol = "pca_1", "pca_2"

    df = features.copy()

    # Caveat burden = total caveat mentions per dataset (joined later)
    if caveats is not None and not caveats.empty:
        cav = (caveats.groupby("dataset")["n_mentions"].sum()
               .rename("caveat_burden").reset_index())
        df = df.merge(cav, on="dataset", how="left")
    if "caveat_burden" not in df.columns:
        df["caveat_burden"] = 0
    df["caveat_burden"] = df["caveat_burden"].fillna(0)

    if color_by == "sample_type":
        cmap = palette_st
        df["__color"] = df["sample_type"].map(cmap).fillna("#9ecbff")
        groups = df.groupby("sample_type")
    elif color_by == "condition_family":
        cmap = palette_cf
        # ad-hoc colour assignment if not in palette
        cf_list = sorted(df["condition_family"].dropna().unique())
        autocols = ["#79c0ff", "#bc8cff", "#ffa657", "#7ee787", "#56d4dd",
                    "#ff7b72", "#ffdf5d", "#d2a8ff", "#a5d6ff", "#85e89d"]
        cmap = {cf: autocols[i % len(autocols)] for i, cf in enumerate(cf_list)}
        df["__color"] = df["condition_family"].map(cmap).fillna("#9ecbff")
        groups = df.groupby("condition_family")
    elif color_by == "caveat burden":
        # continuous colour
        df["__color"] = df["caveat_burden"]
        groups = [(None, df)]
    else:
        df["__color"] = "#79c0ff"
        groups = [(None, df)]

    # size mapping
    if point_size_by == "evidence count":
        # use mean of |feature| across the 11 axes as proxy for evidence
        try:
            ax_cols = [c for c in df.columns if c.startswith("G") and len(c) == 3]
            if ax_cols:
                df["__size"] = (df[ax_cols].abs().sum(axis=1) * 4 + 10)
            else:
                df["__size"] = 14
        except Exception:
            df["__size"] = 14
    elif point_size_by == "caveat burden":
        df["__size"] = (df["caveat_burden"].clip(0, 60) / 4 + 10)
    else:
        df["__size"] = 14

    fig = go.Figure()
    if color_by == "caveat burden":
        sub = df
        fig.add_trace(go.Scatter(
            x=sub[xcol], y=sub[ycol], mode="markers" + ("+text" if show_labels else ""),
            marker=dict(color=sub["__color"], colorscale="OrRd",
                        size=sub["__size"], showscale=True,
                        colorbar=dict(title="caveat burden",
                                       tickfont=dict(color="#c9d1d9"),
                                       title_font=dict(color="#c9d1d9")),
                        line=dict(color="#0d1117", width=0.5)),
            text=sub["dataset"].str[:24] if show_labels else None,
            textposition="top center",
            textfont=dict(size=8, color="#c9d1d9"),
            hovertext=[(f"<b>{r['dataset']}</b><br>"
                         f"sample={r['sample_type']} · cf={r['condition_family']}"
                         f"<br>caveat_burden={r.get('caveat_burden', 0)}")
                        for _, r in sub.iterrows()],
            hoverinfo="text",
            showlegend=False, name=""))
    else:
        for label, sub in groups:
            color = (sub["__color"].iloc[0] if isinstance(label, str)
                      else "#79c0ff")
            fig.add_trace(go.Scatter(
                x=sub[xcol], y=sub[ycol],
                mode="markers" + ("+text" if show_labels else ""),
                name=str(label) if label is not None else "",
                marker=dict(color=color, size=sub["__size"],
                            line=dict(color="#0d1117", width=0.5)),
                text=sub["dataset"].str[:24] if show_labels else None,
                textposition="top center",
                textfont=dict(size=8, color="#c9d1d9"),
                hovertext=[(f"<b>{r['dataset']}</b><br>"
                             f"sample={r['sample_type']} · cf={r['condition_family']}"
                             f"<br>caveat_burden={r.get('caveat_burden', 0)}")
                            for _, r in sub.iterrows()],
                hoverinfo="text"))

    fig.update_layout(
        template="plotly_dark", height=600,
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        margin=dict(l=10, r=10, t=44, b=10),
        title=dict(text=f"{view} · coloured by {color_by} · size by {point_size_by}",
                   font=dict(size=12, color="#c9d1d9")),
        xaxis=dict(title=xcol, gridcolor="#21262d"),
        yaxis=dict(title=ycol, gridcolor="#21262d"),
        legend=dict(font=dict(size=10, color="#c9d1d9"),
                    bgcolor="rgba(13,17,23,0.6)"),
        hoverlabel=dict(bgcolor="#161b22", font_color="#c9d1d9",
                        bordercolor="#30363d"))
    return fig


def render(ctx: dict, app_cfg: dict) -> None:
    st.markdown("# Context embedding · clusters")
    st.caption("Each point is one dataset, embedded in BSV-effect space "
               "(11-dim feature vector → PCA / UMAP).")

    feats = ctx.get("dataset_features")
    clusters = ctx.get("clusters")
    caveats = ctx.get("caveats")

    if feats is None or feats.empty:
        ui.warning_card("context_dataset_bsv_features.csv missing.")
        return

    cols = st.columns([1, 1, 1, 1])
    with cols[0]:
        view = st.radio("View", options=["UMAP", "PCA"], index=0,
                          horizontal=True, key="cge5_view")
    with cols[1]:
        color_by = st.selectbox(
            "Colour by",
            options=["sample_type", "condition_family", "caveat burden"],
            index=0, key="cge5_color_by")
    with cols[2]:
        point_size_by = st.selectbox(
            "Point size by",
            options=["fixed", "evidence count", "caveat burden"],
            index=1, key="cge5_size_by")
    with cols[3]:
        show_labels = st.checkbox("Show dataset labels", value=True,
                                    key="cge5_labels")

    fig = _scatter(feats, view=view, color_by=color_by,
                    show_labels=show_labels, point_size_by=point_size_by,
                    palette_st=app_cfg.get("sample_type_colors", {}),
                    palette_cf={},
                    caveats=caveats)
    st.plotly_chart(fig, use_container_width=True)

    # Optional pre-rendered HTML
    pre = figure_path(ctx["_root"], "context_embedding_dataset.html")
    if pre.exists():
        with st.expander("Pre-rendered Plotly HTML (from discovery driver)",
                          expanded=False):
            st.components.v1.html(pre.read_text(), height=620, scrolling=True)

    ui.interpretation(
        "How to read this embedding",
        "<ul style='margin: 4px 0 4px 18px;'>"
        "<li><strong>EV and serum</strong> form distinct neighbourhoods — "
        "the median UMAP distance between them is roughly 1.5 units in this corpus.</li>"
        "<li><strong>mixed / calibration</strong> datasets sit between biological "
        "contexts because their effect vectors aggregate cohorts.</li>"
        "<li>Distance reflects similarity in <em>BSV / MSS effect space</em>, "
        "<em>not</em> raw spectral similarity. Two pilots can sit close because "
        "their biochemistry transfers, even if their substrates differ.</li>"
        "</ul>")
