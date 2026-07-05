"""Tab 2 — Motif · MSS · BSV (v2 redesign · pass 3).

Pass-3 fixes per user feedback:
  * Family map = real motif UMAP coloured by BSV family (no curated centroids).
  * MSS map = annotated cluster structure with class/family/cluster colour modes
    and a dominant-class side summary table.
  * MSS-within-family drilldown dropdown is INLINE (not in sidebar) and covers
    all G01–G11.
  * Axis overlap = confusion-matrix-style heatmap (Blues, row-normalised),
    with the network demoted to an expander.
  * BSV saliency unchanged.

Section order:
  A · Representation hierarchy
  B · Motif / family UMAP                  (real embedding · coloured by G01–G11)
  C · MSS analyte cluster map              (annotated clusters · summary table)
  D · MSS-within-family drilldown          (inline dropdown G01–G11)
  E · BSV saliency map                     (band ⇒ axis)
  F · Shared-band ambiguity map            (traffic-light)
  G · Axis overlap confusion-style matrix  (Blues; network optional)
  H · Hierarchical clustering support      (expander)
  I · Hybrid evidence flow
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.spatial import ConvexHull, QhullError

from components import ui_blocks as ui
from utils.embedding_loader import (
    load_embedding, load_cluster_breakdown, load_mss_signatures,
    load_analyte_to_group, load_bsv_registry,
    top_anchors_for_class, build_bsv_band_saliency,
    BSV_AXES_ORDER, attach_bsv_family,
    family_name_lookup, family_short_lookup,
)
from utils.figure_loader import load_image_safe
from utils.plotly_cluster_utils import (
    BSV_FAMILY_COLORS, color_map,
)
from utils.bsv_saliency_utils import (
    top_bands_for_axis, axis_overlap_edges, edge_interpretation,
    traffic_light_overlay, traffic_light_colors, axis_node_weights,
    axis_overlap_matrix, family_band_frequencies,
)
from utils.layout_constants import (
    AXIS_POSITIONS, AXIS_INFO, MAJOR_CLASS_LABELS, CANONICAL_BANDS,
)


# ─── caches ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _cached_embedding(build_root_str: str, mode: str):
    return load_embedding(Path(build_root_str), mode)


@st.cache_data(show_spinner=False)
def _cached_breakdown(build_root_str: str, mode: str):
    return load_cluster_breakdown(Path(build_root_str), mode)


@st.cache_data(show_spinner=False)
def _cached_signatures(build_root_str: str):
    return load_mss_signatures(Path(build_root_str))


@st.cache_data(show_spinner=False)
def _cached_amap(build_root_str: str):
    return load_analyte_to_group(Path(build_root_str))


@st.cache_data(show_spinner=False)
def _cached_bsv_registry(build_root_str: str):
    return load_bsv_registry(Path(build_root_str))


@st.cache_data(show_spinner=False)
def _cached_saliency(build_root_str: str, band_min: int, band_max: int,
                     band_bin: int):
    sig = load_mss_signatures(Path(build_root_str))
    amap = load_analyte_to_group(Path(build_root_str))
    return build_bsv_band_saliency(sig, amap, band_min, band_max, band_bin)


@st.cache_data(show_spinner=False)
def _cached_family_embedding(build_root_str: str, mode: str):
    emb = load_embedding(Path(build_root_str), mode)
    amap = load_analyte_to_group(Path(build_root_str))
    return attach_bsv_family(emb, amap)


# ────────────────────────────────────────────────────────────────────────
# Shared helpers
# ────────────────────────────────────────────────────────────────────────

def _convex_hull_pts(points: np.ndarray) -> np.ndarray | None:
    if len(points) < 3:
        return None
    try:
        hull = ConvexHull(points)
        idx = list(hull.vertices) + [hull.vertices[0]]
        return points[idx]
    except (QhullError, ValueError):
        return None


def _stagger_label(cx: float, cy: float, placed: list[tuple[float, float]],
                   x_thresh: float = 0.7, y_thresh: float = 0.5,
                   nudge: float = 0.6) -> tuple[float, float]:
    """Nudge label to avoid colliding with already-placed labels."""
    out_x, out_y = cx, cy
    while True:
        collision = False
        for px, py in placed:
            if abs(px - out_x) < x_thresh and abs(py - out_y) < y_thresh:
                out_y += nudge
                collision = True
                break
        if not collision:
            break
    placed.append((out_x, out_y))
    return out_x, out_y


# ────────────────────────────────────────────────────────────────────────
# A · Representation hierarchy
# ────────────────────────────────────────────────────────────────────────

def render_representation_hierarchy() -> None:
    ui.section_header(
        "A · Representation hierarchy",
        "Five resolutions GAIRA builds from one Raman/SERS spectrum.")

    steps = [
        ("Spectral primitives", "#79c0ff",
         "peaks · FWHM · prominence · co-band patterns · envelope"),
        ("Motif family evidence", "#bc8cff",
         "24 spectral grammar units fitted from grounding spectra"),
        ("11 BSV biochemical axes", "#ffa657",
         "G01 purine_nucleotide … G11 metabolic_small_molecule"),
        ("MSS analyte candidates", "#7ee787",
         "257 analyte-level signatures · anchor + competitor logic"),
        ("ΔBSV / classifier / report", "#56d4dd",
         "reference-relative deltas · output-tier policy"),
    ]
    fig = go.Figure()
    n = len(steps)
    box_w, box_h = 0.16, 0.42
    for i, (label, color, hover) in enumerate(steps):
        x = (i + 0.5) / n
        y = 0.5
        fig.add_shape(type="rect",
                      x0=x - box_w / 2, y0=y - box_h / 2,
                      x1=x + box_w / 2, y1=y + box_h / 2,
                      fillcolor="#161b22", line=dict(color=color, width=2),
                      layer="below")
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="text",
            text=[f"<b>{label}</b>"],
            textfont=dict(color="#f0f6fc", size=12),
            hovertext=[hover], hoverinfo="text", showlegend=False,
        ))
        if i < n - 1:
            x_next = (i + 1.5) / n
            fig.add_annotation(
                x=x_next - box_w / 2, y=y, ax=x + box_w / 2, ay=y,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.0, arrowwidth=1.4,
                arrowcolor="#6e7681",
            )
    fig.update_layout(
        template="plotly_dark", height=180,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        xaxis=dict(visible=False, range=[-0.02, 1.02]),
        yaxis=dict(visible=False, range=[0, 1]),
        hoverlabel=dict(bgcolor="#161b22", font_color="#c9d1d9",
                        bordercolor="#30363d"),
    )
    st.plotly_chart(fig, use_container_width=True)

    ui.interpretation(
        "Thesis",
        "<strong>GAIRA first learns biochemical family structure, then "
        "resolves analyte-level evidence within those families.</strong> "
        "Below: the real motif/family UMAP, then MSS analyte structure, "
        "then per-family drilldown.",
    )


# ────────────────────────────────────────────────────────────────────────
# B · Motif / family UMAP — real observed embedding (PASS-3 REPLACEMENT)
# ────────────────────────────────────────────────────────────────────────

def render_normal_motif_family_umap(build_root: Path, controls: dict) -> None:
    ui.section_header(
        "B · Motif / family UMAP",
        "The observed motif embedding, coloured by BSV family (G01–G11).")

    df = _cached_family_embedding(str(build_root), "MOTIF")
    if df is None or "primary_group" not in df.columns:
        ui.warning_card("Motif embedding or family map missing.")
        return

    sig_df = _cached_signatures(str(build_root))
    bsv_reg = _cached_bsv_registry(str(build_root))
    short = family_short_lookup(bsv_reg)

    fig = go.Figure()

    # One trace per BSV family — keeps the legend tidy at exactly 11 entries.
    for fam in BSV_AXES_ORDER:
        sub = df[df["primary_group"] == fam]
        if sub.empty:
            continue
        color = BSV_FAMILY_COLORS.get(fam, "#79c0ff")
        anchors = (sub["broad_class"].apply(
            lambda c: top_anchors_for_class(sig_df, c, k=3))
            if sig_df is not None else pd.Series([""] * len(sub), index=sub.index))
        custom = np.column_stack([
            sub["analyte_id"].astype(str).values,
            sub["broad_class"].astype(str).values,
            np.full(len(sub), fam, dtype=object),
            anchors.values,
        ])
        fig.add_trace(go.Scattergl(
            x=sub["umap_1"], y=sub["umap_2"], mode="markers",
            name=f"{fam} · {short.get(fam, '')}",
            marker=dict(color=color, size=8, opacity=controls.get("family_opacity", 0.85),
                        line=dict(width=0.4, color="#0d1117")),
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "biochemical class: %{customdata[1]}<br>"
                "BSV family: %{customdata[2]}<br>"
                "top anchors: %{customdata[3]}<extra></extra>"),
        ))

        # Optional convex hull (default OFF)
        if controls.get("family_show_hulls", False) and len(sub) >= 4:
            pts = sub[["umap_1", "umap_2"]].values
            hull = _convex_hull_pts(pts)
            if hull is not None:
                fig.add_trace(go.Scatter(
                    x=hull[:, 0], y=hull[:, 1],
                    fill="toself",
                    fillcolor="rgba(125,160,200,0.06)",
                    line=dict(color=color, width=1, dash="dot"),
                    mode="lines", hoverinfo="skip", showlegend=False,
                    name=f"hull_{fam}",
                ))

    # Centroid label per family — semi-transparent box, manually staggered
    if controls.get("family_show_labels", True):
        placed: list[tuple[float, float]] = []
        # Sort by member count so big families get placed first
        order = sorted(
            BSV_AXES_ORDER,
            key=lambda f: -int((df["primary_group"] == f).sum()))
        for fam in order:
            sub = df[df["primary_group"] == fam]
            if len(sub) < 1:
                continue
            cx = float(sub["umap_1"].mean())
            cy = float(sub["umap_2"].mean())
            cx, cy = _stagger_label(cx, cy, placed,
                                    x_thresh=0.9, y_thresh=0.5,
                                    nudge=0.45)
            color = BSV_FAMILY_COLORS.get(fam, "#79c0ff")
            fig.add_annotation(
                x=cx, y=cy, text=f"<b>{fam}</b> · {short.get(fam, '')}",
                font=dict(color=color, size=11),
                bgcolor="rgba(13,17,23,0.85)",
                bordercolor=color, borderwidth=1, borderpad=3,
                showarrow=False, align="center",
            )

    fig.update_layout(
        template="plotly_dark", height=620,
        title=dict(
            text="Motif / family UMAP — observed embedding coloured by BSV family",
            font=dict(size=13, color="#c9d1d9")),
        margin=dict(l=10, r=10, t=46, b=10),
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        legend=dict(font=dict(size=10, color="#c9d1d9"),
                    bgcolor="rgba(13,17,23,0.6)",
                    bordercolor="#30363d", borderwidth=1,
                    itemsizing="constant"),
        xaxis=dict(title="UMAP-1", gridcolor="#21262d", zeroline=False),
        yaxis=dict(title="UMAP-2", gridcolor="#21262d", zeroline=False),
        hoverlabel=dict(bgcolor="#161b22", font_color="#c9d1d9",
                        bordercolor="#30363d"),
    )
    st.plotly_chart(fig, use_container_width=True)

    ui.interpretation(
        "How to read this map",
        "This is the <strong>observed motif/MSS embedding</strong>. Families "
        "overlap where Raman evidence is chemically shared — overlap is "
        "expected, not failure. Each colour is one BSV axis (G01–G11); the "
        "label sits at the family centroid.",
    )


# ────────────────────────────────────────────────────────────────────────
# C · MSS analyte cluster map (PASS-3 IMPROVED)
# ────────────────────────────────────────────────────────────────────────

def render_mss_analyte_cluster_map(build_root: Path, controls: dict) -> None:
    ui.section_header(
        "C · MSS analyte cluster map",
        "Real MSS UMAP. Cluster annotations come from the precomputed "
        "agglomerative cluster_id; colour mode is selectable.")

    df = _cached_family_embedding(str(build_root), "MSS")
    if df is None:
        ui.warning_card("MSS embedding missing.")
        return

    sig_df = _cached_signatures(str(build_root))
    bsv_reg = _cached_bsv_registry(str(build_root))
    short = family_short_lookup(bsv_reg)

    color_mode = controls.get("mss_color_mode", "biochemical class")

    fig = go.Figure()

    if color_mode == "BSV family":
        groups = df.groupby(df["primary_group"].fillna("(unmapped)"))
        cmap = {fam: BSV_FAMILY_COLORS.get(fam, "#79c0ff")
                for fam in BSV_AXES_ORDER}
        cmap["(unmapped)"] = "#6e7681"
        legend_label = lambda lbl: (f"{lbl} · {short.get(lbl, '')}"
                                    if lbl in BSV_FAMILY_COLORS else str(lbl))
    elif color_mode == "cluster id (precomputed)":
        groups = df.groupby(df["cluster_id"].astype(str))
        cluster_labels = sorted(df["cluster_id"].astype(str).unique())
        cmap = color_map(cluster_labels)
        legend_label = lambda lbl: f"cluster {lbl}"
    else:  # biochemical class
        groups = df.groupby(df["broad_class"].astype(str))
        cmap = color_map(df["broad_class"].astype(str).tolist())
        legend_label = lambda lbl: str(lbl)

    for label, sub in groups:
        anchors = (sub["broad_class"].apply(
            lambda c: top_anchors_for_class(sig_df, c, k=3))
            if sig_df is not None else pd.Series([""] * len(sub), index=sub.index))
        custom = np.column_stack([
            sub["analyte_id"].astype(str).values,
            sub["broad_class"].astype(str).values,
            sub.get("primary_group", pd.Series([""] * len(sub),
                                               index=sub.index)).fillna("").astype(str).values,
            sub["cluster_id"].astype(str).values,
            anchors.values,
        ])
        fig.add_trace(go.Scattergl(
            x=sub["umap_1"], y=sub["umap_2"], mode="markers",
            name=str(legend_label(label)),
            marker=dict(color=cmap.get(str(label), "#79c0ff"),
                        size=8, opacity=controls.get("mss_opacity", 0.85),
                        line=dict(width=0.4, color="#0d1117")),
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "class: %{customdata[1]}<br>"
                "BSV family: %{customdata[2]} · cluster: %{customdata[3]}<br>"
                "anchors: %{customdata[4]}<extra></extra>"),
        ))

    # Cluster annotations — label up to 10 clusters with ≥ 5 members,
    # using the dominant biochemical class as the label text.
    if controls.get("mss_show_cluster_labels", True):
        sizes = df.groupby("cluster_id").size().sort_values(ascending=False)
        placed: list[tuple[float, float]] = []
        labelled = 0
        for cid, n in sizes.items():
            if labelled >= 10:
                break
            if n < 5:
                continue
            sub = df[df["cluster_id"] == cid]
            cx = float(sub["umap_1"].mean())
            cy = float(sub["umap_2"].mean())
            cx, cy = _stagger_label(cx, cy, placed,
                                    x_thresh=1.0, y_thresh=0.55, nudge=0.55)
            dom = Counter(sub["broad_class"].astype(str)).most_common(1)[0][0]
            fig.add_annotation(
                x=cx, y=cy, text=f"<b>{dom}</b>",
                font=dict(color="#f0f6fc", size=11),
                bgcolor="rgba(13,17,23,0.85)",
                bordercolor="#30363d", borderwidth=1, borderpad=3,
                showarrow=False, align="center",
            )
            labelled += 1

    fig.update_layout(
        template="plotly_dark", height=620,
        title=dict(text=f"MSS analyte UMAP · 236 analytes · coloured by {color_mode}",
                   font=dict(size=13, color="#c9d1d9")),
        margin=dict(l=10, r=10, t=46, b=80),
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        showlegend=controls.get("mss_show_legend", False),
        legend=dict(orientation="h", yanchor="top", y=-0.18,
                    xanchor="center", x=0.5,
                    font=dict(size=9, color="#c9d1d9"),
                    bgcolor="rgba(13,17,23,0.6)"),
        xaxis=dict(title="UMAP-1", gridcolor="#21262d", zeroline=False),
        yaxis=dict(title="UMAP-2", gridcolor="#21262d", zeroline=False),
        hoverlabel=dict(bgcolor="#161b22", font_color="#c9d1d9",
                        bordercolor="#30363d"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cluster summary table — derived from the precomputed cluster_breakdown
    bdown = _cached_breakdown(str(build_root), "MSS")
    if bdown is not None and not bdown.empty:
        ui.section_header(
            "MSS cluster summary",
            "cluster_id · n · dominant class · representative analytes · likely BSV family")
        # Map dominant_broad_class → most-common BSV family (mode of primary_group
        # over analytes whose broad_class equals dominant_broad_class)
        amap = _cached_amap(str(build_root))
        class_to_family = {}
        if amap is not None:
            class_to_family = (amap.groupby("broad_class")["primary_group"]
                               .agg(lambda x: x.value_counts().idxmax())
                               .to_dict())
        rows = []
        for _, r in bdown.iterrows():
            members = str(r.get("sample_members", "")).split(";")[:5]
            dom_class = r["dominant_broad_class"]
            fam = class_to_family.get(dom_class, "")
            fam_label = (f"{fam} · {short.get(fam, '')}"
                         if fam in BSV_AXES_ORDER else "")
            rows.append({
                "cluster_id": int(r["cluster_id"]),
                "n": int(r["n_members"]),
                "dominant class": dom_class,
                "representative analytes": ", ".join(
                    m.strip() for m in members if m.strip()),
                "likely BSV family": fam_label,
            })
        st.dataframe(pd.DataFrame(rows),
                     use_container_width=True, hide_index=True)

    ui.interpretation(
        "How to read this map",
        "Switch the colour mode in the sidebar to see the same UMAP under "
        "<em>biochemical class</em>, <em>BSV family</em>, or <em>cluster id</em>. "
        "Labelled clusters use their dominant biochemical class. Overlap "
        "between clusters reflects shared vibrational chemistry, not noise.",
    )


# ────────────────────────────────────────────────────────────────────────
# D · MSS-within-family drilldown (INLINE dropdown, all 11 families)
# ────────────────────────────────────────────────────────────────────────

def render_mss_within_family_drilldown(build_root: Path) -> None:
    ui.section_header(
        "D · MSS-within-family drilldown",
        "Pick any of the 11 families to see its analytes, anchor bands, and "
        "competitors. Inline dropdown.")

    df = _cached_family_embedding(str(build_root), "MSS")
    if df is None or "primary_group" not in df.columns:
        ui.warning_card("Drilldown needs the MSS embedding + family map.")
        return

    sig_df = _cached_signatures(str(build_root))
    bsv_reg = _cached_bsv_registry(str(build_root))
    short = family_short_lookup(bsv_reg)

    # Inline dropdown — covers ALL G01..G11 even if some have 0 analytes
    options = [f"{ax} · {short.get(ax, '')}" for ax in BSV_AXES_ORDER]
    sel_label = st.selectbox(
        "BSV family", options=options, index=0,
        key="v2p3_drilldown_family_inline",
        help="Drilldown is independent of the sidebar; pick any of the 11 BSV families.")
    sel_family = sel_label.split(" ·")[0].strip()

    sub = df[df["primary_group"] == sel_family].copy()
    color = BSV_FAMILY_COLORS.get(sel_family, "#79c0ff")
    n_in_family = len(sub)
    st.caption(f"Family **{sel_family} · {short.get(sel_family, '')}** — "
               f"{n_in_family} analytes mapped")

    if n_in_family == 0:
        st.info(f"No analytes are mapped to {sel_family} in the current "
                "embedding. (Family is still real — it just has no member "
                "analytes in this corpus.)")
        return

    # 1 · Subset MSS UMAP — selected family highlighted; others grey
    fig = go.Figure()
    bg = df[df["primary_group"] != sel_family]
    fig.add_trace(go.Scattergl(
        x=bg["umap_1"], y=bg["umap_2"], mode="markers",
        marker=dict(color="#30363d", size=4, opacity=0.35),
        hoverinfo="skip", showlegend=False, name="other families",
    ))
    anchors = (sub["broad_class"].apply(
        lambda c: top_anchors_for_class(sig_df, c, k=3))
        if sig_df is not None else pd.Series([""] * n_in_family, index=sub.index))
    custom = np.column_stack([
        sub["analyte_id"].astype(str).values,
        sub["broad_class"].astype(str).values,
        anchors.values,
    ])
    fig.add_trace(go.Scattergl(
        x=sub["umap_1"], y=sub["umap_2"], mode="markers",
        marker=dict(color=color, size=10, opacity=0.95,
                    line=dict(width=0.6, color="#0d1117")),
        customdata=custom,
        hovertemplate=("<b>%{customdata[0]}</b><br>"
                       "class: %{customdata[1]}<br>"
                       "anchors: %{customdata[2]}<extra></extra>"),
        showlegend=False, name=sel_family,
    ))
    fig.update_layout(
        template="plotly_dark", height=380,
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        xaxis=dict(title="UMAP-1", gridcolor="#21262d", zeroline=False),
        yaxis=dict(title="UMAP-2", gridcolor="#21262d", zeroline=False),
        hoverlabel=dict(bgcolor="#161b22", font_color="#c9d1d9",
                        bordercolor="#30363d"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 2 · Analyte table
    fam_classes = list(sub["broad_class"].astype(str).unique())
    sig_lookup = (sig_df.set_index("analyte_class")
                  if sig_df is not None and not sig_df.empty else None)
    rows = []
    for _, r in sub.iterrows():
        cls = r.get("broad_class", "")
        anchor_str = (top_anchors_for_class(sig_df, cls, k=4)
                      if sig_df is not None else "")
        support_str = ""
        competitors = ""
        reliability = ""
        if sig_lookup is not None and cls in sig_lookup.index:
            sig_row = sig_lookup.loc[cls]
            if isinstance(sig_row, pd.DataFrame):
                sig_row = sig_row.iloc[0]
            support = str(sig_row.get("raman_support_features", "")
                          or sig_row.get("sers_support_features", ""))
            import re
            sup_bands = re.findall(r"(\d+)cm-1", support)[:5]
            support_str = ", ".join(f"{b}cm⁻¹" for b in sup_bands)
            comp = str(sig_row.get("competitor_signatures", ""))
            competitors = ", ".join([c.split("::")[-1]
                                     for c in comp.split(",")[:3]
                                     if c.strip() and c.strip() != "nan"])
            stability = sig_row.get("replicate_stability", "")
            reliability = (f"replicate stability {float(stability):.2f}"
                           if stability not in ("", None) and not pd.isna(stability)
                           else "")
        rows.append({
            "analyte": r["analyte_id"],
            "biochemical class": cls,
            "top anchor bands": anchor_str,
            "support bands": support_str,
            "competitors": competitors,
            "reliability": reliability,
        })
    st.dataframe(pd.DataFrame(rows),
                 use_container_width=True, hide_index=True)

    # 3 · Optional band-frequency expander (NOT shown by default)
    with st.expander("Band frequency within this family (advanced)",
                     expanded=False):
        st.caption(
            "Counts how many MSS-signature rows in this family list each "
            "band as anchor or support. Low counts reflect diverse "
            "analyte-specific anchors; this is informational, not a "
            "ranking.")
        bands_df = family_band_frequencies(sig_df, fam_classes, top_k=12)
        if bands_df.empty:
            st.info("(no MSS-signature rows for this family.)")
        else:
            bar = go.Figure(go.Bar(
                x=bands_df["band_cm"].astype(str), y=bands_df["count"],
                marker_color=color,
                hovertemplate="band %{x} cm⁻¹<br># MSS rows: %{y}"
                              "<extra></extra>",
            ))
            bar.update_layout(
                template="plotly_dark", height=240,
                margin=dict(l=10, r=10, t=36, b=10),
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                title=dict(
                    text=f"Bands appearing as anchor/support in "
                         f"{sel_family} family signatures",
                    font=dict(size=11, color="#c9d1d9")),
                xaxis=dict(title="Raman shift (cm⁻¹)", gridcolor="#21262d"),
                yaxis=dict(title="# MSS-signature rows hitting this band",
                           gridcolor="#21262d"),
            )
            st.plotly_chart(bar, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────
# E · BSV saliency map (unchanged — keeps the cleaned defaults)
# ────────────────────────────────────────────────────────────────────────

def render_bsv_saliency_map(build_root: Path, controls: dict) -> None:
    ui.section_header(
        "E · BSV saliency · band ⇒ axis mapping",
        "Rows are BSV axes; bright columns are bands where that axis draws evidence.")

    band_min, band_max = 400, 1800
    band_bin = controls.get("saliency_bin", 10)
    mat, bin_centers, contributors = _cached_saliency(
        str(build_root), band_min, band_max, band_bin)
    if mat.sum() == 0:
        ui.warning_card("BSV saliency could not be built.")
        return

    bsv_reg = _cached_bsv_registry(str(build_root))
    long_family = family_name_lookup(bsv_reg)
    y_labels = [long_family.get(ax, ax) for ax in BSV_AXES_ORDER]

    contrib_text = np.empty(mat.shape, dtype=object)
    for i, ax in enumerate(BSV_AXES_ORDER):
        per_bin = contributors.get(ax, {})
        for j, bc in enumerate(bin_centers):
            classes = per_bin.get(bc, [])
            top = (", ".join([c for c, _ in Counter(classes).most_common(3)])
                   if classes else "(none)")
            contrib_text[i, j] = top
    customdata = np.dstack([
        np.broadcast_to(np.array(bin_centers), mat.shape),
        contrib_text,
    ])

    fig = go.Figure(go.Heatmap(
        z=mat, x=bin_centers, y=y_labels,
        colorscale="Viridis", zmin=0, zmax=1,
        colorbar=dict(title="band weight (per-axis max-norm)",
                      tickfont=dict(color="#c9d1d9"),
                      title_font=dict(color="#c9d1d9")),
        customdata=customdata,
        hovertemplate=("<b>%{y}</b><br>band centre: %{customdata[0]} cm⁻¹<br>"
                       "weight: %{z:.2f}<br>"
                       "top contributors: %{customdata[1]}<extra></extra>"),
    ))

    if controls.get("show_canonical_labels", False):
        for i, (band, label) in enumerate(CANONICAL_BANDS[:8]):
            ypos = 1.04 + (i % 2) * 0.07
            fig.add_vline(x=band, line=dict(color="#ff7b72", width=1, dash="dot"),
                          opacity=0.45)
            fig.add_annotation(
                x=band, y=ypos, xref="x", yref="paper",
                text=f"{band}<br><span style='font-size:9px;'>{label}</span>",
                showarrow=False, font=dict(size=9, color="#ff7b72"),
                align="center")
    else:
        for band, _ in CANONICAL_BANDS[:8]:
            fig.add_vline(x=band, line=dict(color="#ff7b72", width=1, dash="dot"),
                          opacity=0.20)

    fig.update_layout(
        template="plotly_dark",
        height=520 if not controls.get("show_canonical_labels", False) else 600,
        margin=dict(l=10, r=10,
                    t=80 if controls.get("show_canonical_labels", False) else 30,
                    b=10),
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        xaxis=dict(title="Raman shift (cm⁻¹)", range=[band_min, band_max],
                   gridcolor="#21262d"),
        yaxis=dict(title=""),
        hoverlabel=dict(bgcolor="#161b22", font_color="#c9d1d9",
                        bordercolor="#30363d"),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Inspect axis details (top bands table)", expanded=False):
        sel_axis = st.selectbox(
            "Pick axis", options=BSV_AXES_ORDER, index=0,
            key="v2p3_saliency_axis_inline")
        top_df = top_bands_for_axis(mat, bin_centers, contributors, sel_axis, k=10)
        long_label = long_family.get(sel_axis, sel_axis)
        st.caption(f"Top 10 bands driving {long_label}")
        if top_df.empty:
            st.info("No bands above zero for this axis.")
        else:
            st.dataframe(top_df, use_container_width=True, hide_index=True)


# ────────────────────────────────────────────────────────────────────────
# F · Shared-band ambiguity map (unchanged)
# ────────────────────────────────────────────────────────────────────────

def render_shared_band_ambiguity_map(build_root: Path, controls: dict) -> None:
    ui.section_header(
        "F · Shared bands & ambiguity",
        "Why a single Raman peak rarely means a single molecule.")

    band_min, band_max = 400, 1800
    band_bin = controls.get("saliency_bin", 10)
    mat, bin_centers, _ = _cached_saliency(
        str(build_root), band_min, band_max, band_bin)
    if mat.sum() == 0:
        ui.warning_card("BSV saliency unavailable.")
        return
    threshold = controls.get("ambiguity_threshold", 0.30)
    counts = traffic_light_overlay(mat, threshold=threshold)
    colors = traffic_light_colors(counts)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bin_centers, y=counts, marker_color=colors,
        name="axes contributing",
        hovertemplate="band %{x} cm⁻¹<br>axes ≥ thr: %{y}<extra></extra>",
        showlegend=False,
    ))
    annotations = [
        (725, "purine"),
        (1080, "glycan/phosphate"),
        (1450, "CH bending"),
        (1655, "amide / lipid"),
    ]
    for band, label in annotations:
        fig.add_vline(x=band, line=dict(color="#c9d1d9", width=1, dash="dot"),
                      opacity=0.35)
        fig.add_annotation(
            x=band, y=max(counts) + 0.5, text=label,
            showarrow=False, font=dict(size=10, color="#c9d1d9"))
    for cnt_label, color in [("1 axis · clean(-ish)", "#7ee787"),
                              ("2 axes · shared", "#ffa657"),
                              ("≥3 axes · collision", "#ff7b72")]:
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color=color,
                              name=cnt_label, showlegend=True))
    fig.update_layout(
        template="plotly_dark", height=320,
        margin=dict(l=10, r=10, t=44, b=10),
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        title=dict(text="Per-band BSV-axis count · bars ≥2 mark where MSS competitor logic earns its keep",
                   font=dict(size=12, color="#c9d1d9")),
        xaxis=dict(title="Raman shift (cm⁻¹)", range=[band_min, band_max],
                   gridcolor="#21262d"),
        yaxis=dict(title="# BSV axes ≥ threshold", gridcolor="#21262d"),
        legend=dict(font=dict(size=10, color="#c9d1d9"),
                    bgcolor="rgba(13,17,23,0.6)"),
        barmode="overlay",
    )
    st.plotly_chart(fig, use_container_width=True)

    ui.interpretation(
        "Reading the colours",
        "<ul style='margin: 4px 0 4px 18px;'>"
        "<li><span style='color:#7ee787;'>green (1 axis)</span> — band is "
        "axis-specific within the current BSV taxonomy. <em>Axis-specific is "
        "not the same as molecule-specific.</em></li>"
        "<li><span style='color:#ffa657;'>orange (2 axes)</span> — moderate "
        "overlap; co-fire evidence usually disambiguates.</li>"
        "<li><span style='color:#ff7b72;'>red (≥3 axes)</span> — collision-prone; "
        "co-features and MSS competitor logic are necessary.</li>"
        "</ul>",
    )


# ────────────────────────────────────────────────────────────────────────
# G · Axis overlap CONFUSION-style matrix (PASS-3 REWORK)
# ────────────────────────────────────────────────────────────────────────

def render_axis_overlap_confusion_style(build_root: Path,
                                        controls: dict) -> None:
    ui.section_header(
        "G · Axis overlap · confusion-style matrix",
        "Rows = source axis · columns = overlapping axis. Diagonal "
        "is axis-specific evidence; off-diagonal is shared spectral evidence.")

    mat, bin_centers, _ = _cached_saliency(
        str(build_root), 400, 1800, controls.get("saliency_bin", 10))
    if mat.sum() == 0:
        ui.warning_card("BSV saliency unavailable.")
        return
    threshold = controls.get("network_threshold", 0.30)

    # Build co-occurrence count matrix: M_count[i,j] = # bands with both
    # axis i and axis j ≥ threshold (symmetric; diagonal = bands axis owns).
    binary = (mat >= threshold).astype(int)
    co = binary @ binary.T  # shape (11, 11)
    own = np.diag(co).astype(float)  # bands each axis owns

    overlap_mode = controls.get("overlap_mode", "row-normalised (fraction of axis A)")
    if overlap_mode.startswith("row-normalised"):
        # Each row divided by axis i's own total ⇒ diagonal becomes 1.0,
        # off-diagonal = fraction of axis i's bands also owned by axis j.
        denom = np.where(own > 0, own, 1.0)[:, None]
        z = co / denom
        annot_thresh = 0.15
        annot_text_fn = lambda v: f"{v:.2f}" if v >= annot_thresh else ""
        zmin, zmax = 0.0, 1.0
        cb_title = "fraction of axis A bands shared"
    else:
        z = co.astype(float)
        annot_thresh = 2.0
        annot_text_fn = lambda v: f"{int(v)}" if v >= annot_thresh else ""
        zmin = 0.0
        zmax = float(z.max() or 1)
        cb_title = "# shared bands"

    bsv_reg = _cached_bsv_registry(str(build_root))
    short = family_short_lookup(bsv_reg)
    short_labels = [ax for ax in BSV_AXES_ORDER]            # G01, G02, ...
    full_labels  = [f"{ax} · {short.get(ax, '')}"
                    for ax in BSV_AXES_ORDER]               # full hover

    # Hover: full names from full_labels; cell value formatted appropriately
    customdata = np.empty(z.shape, dtype=object)
    for i in range(len(BSV_AXES_ORDER)):
        for j in range(len(BSV_AXES_ORDER)):
            count_v = int(co[i, j])
            customdata[i, j] = (
                f"{full_labels[i]} ⟶ {full_labels[j]}<br>"
                f"shared bands: {count_v}<br>"
                f"axis A own bands ≥ thr: {int(own[i])}<br>"
                f"axis B own bands ≥ thr: {int(own[j])}"
            )

    text_grid = np.array([[annot_text_fn(v) for v in row] for row in z],
                         dtype=object)

    fig = go.Figure(go.Heatmap(
        z=z, x=short_labels, y=short_labels,
        colorscale="Blues", zmin=zmin, zmax=zmax,
        text=text_grid, texttemplate="%{text}",
        textfont=dict(color="#0d1117", size=11),
        colorbar=dict(title=cb_title,
                      tickfont=dict(color="#c9d1d9"),
                      title_font=dict(color="#c9d1d9")),
        customdata=customdata,
        hovertemplate="%{customdata}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark", height=560,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        xaxis=dict(side="bottom", tickangle=0,
                   tickfont=dict(size=11, color="#c9d1d9"),
                   title="overlapping axis (B)"),
        yaxis=dict(autorange="reversed",
                   tickfont=dict(size=11, color="#c9d1d9"),
                   title="source axis (A)"),
        hoverlabel=dict(bgcolor="#161b22", font_color="#c9d1d9",
                        bordercolor="#30363d"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Side table — top axis-pair overlaps with curated interpretation
    edges = axis_overlap_edges(mat, bin_centers, threshold=threshold)
    if len(edges):
        ui.section_header("Top axis-pair overlaps",
                          "axis pair · shared bands · interpretation · risk")
        view = edges.head(10).copy()
        view["interpretation"] = view.apply(
            lambda r: edge_interpretation(r["axis_a"], r["axis_b"]) or "(uncurated)",
            axis=1)
        view["risk"] = view["n_shared_bands"].apply(
            lambda n: ("HIGH" if n >= 8 else "MODERATE" if n >= 4 else "LOW"))
        view = view[["axis_a", "axis_b", "n_shared_bands", "shared_bands",
                     "interpretation", "risk"]].rename(columns={
            "axis_a": "axis A", "axis_b": "axis B",
            "n_shared_bands": "# shared bands",
            "shared_bands": "sample shared bands (cm⁻¹)",
        })
        st.dataframe(view, use_container_width=True, hide_index=True)

    ui.interpretation(
        "How to read this matrix",
        "<ul style='margin: 4px 0 4px 18px;'>"
        "<li><strong>Diagonal</strong> = axis-specific evidence (each axis owns 100% of its own bands; this is by definition).</li>"
        "<li><strong>Off-diagonal</strong> = shared spectral evidence between two axes — i.e., ambiguity / collision.</li>"
        "<li><strong>Strong off-diagonal pairs</strong> (G01 ↔ G02, G04 ↔ G05, G06 ↔ G07, G06 ↔ G08, G08 ↔ G09) are exactly where MSS competitor logic carries the disambiguation.</li>"
        "</ul>",
    )

    # Optional network view (kept behind expander)
    with st.expander("Show network view (manual chemistry-grouped layout)",
                     expanded=False):
        _render_axis_overlap_network(build_root, controls)


def _render_axis_overlap_network(build_root: Path, controls: dict) -> None:
    mat, bin_centers, _ = _cached_saliency(
        str(build_root), 400, 1800, controls.get("saliency_bin", 10))
    if mat.sum() == 0:
        return
    threshold = controls.get("network_threshold", 0.30)
    edges = axis_overlap_edges(mat, bin_centers, threshold=threshold)
    node_weights = axis_node_weights(mat, threshold=threshold)
    bsv_reg = _cached_bsv_registry(str(build_root))
    short = family_short_lookup(bsv_reg)
    pos = AXIS_POSITIONS

    fig = go.Figure()
    if len(edges):
        view = edges.head(8)
        max_edge = float(view["n_shared_bands"].max() or 1)
        for _, e in view.iterrows():
            a = e["axis_a"]; b = e["axis_b"]
            n_shared = int(e["n_shared_bands"])
            x0, y0 = pos[a]; x1, y1 = pos[b]
            width = 0.8 + 4.0 * (n_shared / max_edge)
            opacity = 0.20 + 0.45 * (n_shared / max_edge)
            interp = edge_interpretation(a, b)
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines",
                line=dict(color=f"rgba(255, 166, 87, {opacity:.2f})", width=width),
                hovertemplate=(f"<b>{a} ↔ {b}</b><br>"
                               f"shared bands: {n_shared}"
                               + (f"<br>note: {interp}" if interp else "")
                               + "<extra></extra>"),
                showlegend=False,
            ))
    max_node = float(max(node_weights.max(), 1))
    for ax in BSV_AXES_ORDER:
        idx = BSV_AXES_ORDER.index(ax)
        x, y = pos[ax]
        sz = 30 + 36 * (node_weights[idx] / max_node)
        color = BSV_FAMILY_COLORS.get(ax, "#79c0ff")
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=sz, color=color, opacity=0.95,
                        line=dict(color="#0d1117", width=1.5)),
            text=[f"<b>{ax}</b>"],
            textposition="middle center",
            textfont=dict(color="#0d1117", size=11),
            hovertemplate=(f"<b>{ax} · {short.get(ax, '')}</b><br>"
                           f"contributing bands: {int(node_weights[idx])}<extra></extra>"),
            showlegend=False,
        ))
    fig.update_layout(
        template="plotly_dark", height=440,
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        xaxis=dict(visible=False, range=[-2.4, 2.6]),
        yaxis=dict(visible=False, range=[-2.0, 1.8],
                   scaleanchor="x", scaleratio=1),
        hoverlabel=dict(bgcolor="#161b22", font_color="#c9d1d9",
                        bordercolor="#30363d"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Short G-id labels inside nodes · full family names in hover. "
               "Only top-8 strongest edges drawn.")


# ────────────────────────────────────────────────────────────────────────
# H · Hierarchical clustering support (expander, unchanged structure)
# ────────────────────────────────────────────────────────────────────────

def render_dendrogram_support(build_root: Path, controls: dict) -> None:
    rep_dir = build_root / "gaira_representation_cluster_analysis_v1" / "figures"
    cols = st.columns(2)
    with cols[0]:
        load_image_safe(rep_dir / "fig_motif_dendrogram_v1.png",
                        caption="Motif dendrogram",
                        fallback_msg="Motif dendrogram missing.")
    with cols[1]:
        load_image_safe(rep_dir / "fig_mss_dendrogram_v1.png",
                        caption="MSS dendrogram",
                        fallback_msg="MSS dendrogram missing.")

    ui.interpretation(
        "Numbered callouts (read both dendrograms together)",
        "<ol style='margin: 4px 0 4px 22px;'>"
        "<li><strong>Root split</strong> — lipid / sterol vs non-lipid; "
        "the strongest single divider in pure-Raman evidence.</li>"
        "<li><strong>Tier-2 split</strong> — nucleic acids vs protein / amino-acid "
        "space.</li>"
        "<li><strong>Tier-3 splits</strong> — within-family decomposition.</li>"
        "<li><strong>Leaves</strong> — 236 analytes; singletons live at the "
        "fringe of their family.</li>"
        "</ol>",
    )

    bdown = _cached_breakdown(str(build_root), controls.get("dendro_mode", "MOTIF"))
    if bdown is not None:
        ui.section_header(
            f"{controls.get('dendro_mode', 'MOTIF')} cluster summary",
            "cluster · dominant family · representative analytes · interpretation · caveat")
        df = bdown.copy()
        df = df[["cluster_id", "n_members", "dominant_broad_class",
                 "purity", "entropy_bits", "sample_members"]].rename(columns={
            "n_members": "n analytes",
            "dominant_broad_class": "dominant family / class",
            "entropy_bits": "entropy (bits)",
            "sample_members": "representative analytes",
        })
        def _caveat(p):
            if p >= 0.5: return "high purity"
            if p >= 0.3: return "moderate — within-family chemistry mixed"
            return "low purity — shared chemistry across families"
        df["caveat"] = df["purity"].apply(_caveat)
        df["purity"] = df["purity"].apply(lambda x: f"{float(x):.2f}")
        df["entropy (bits)"] = df["entropy (bits)"].apply(lambda x: f"{float(x):.2f}")
        st.dataframe(df, use_container_width=True, hide_index=True)


# ────────────────────────────────────────────────────────────────────────
# I · Hybrid evidence flow
# ────────────────────────────────────────────────────────────────────────

def render_hybrid_bsv_flow_diagram(build_root: Path) -> None:
    ui.section_header(
        "I · Hybrid BSV evidence flow",
        "Motif and MSS streams converge into the per-axis BSV magnitude.")

    nodes = [
        ("query", 0.04, 0.5, "Query<br>spectrum", "#79c0ff",
         "Preprocessed Raman/SERS spectrum."),
        ("prep", 0.18, 0.5, "Pre-<br>processing", "#a5d6ff",
         "AsLS · SG · L2 · interp."),
        ("motif", 0.36, 0.78, "Motif<br>layer", "#bc8cff",
         "24 learned motifs · band-family firing"),
        ("mss", 0.36, 0.22, "MSS<br>layer", "#7ee787",
         "257 analyte signatures · anchor + competitor"),
        ("hybrid", 0.58, 0.5, "Hybrid<br>BSV", "#ffa657",
         "0.25·motif + 0.75·MSS · per-axis magnitude"),
        ("confamb", 0.78, 0.5, "Confidence<br>+ ambiguity", "#56d4dd",
         "ROBUST / MODERATE / SENSITIVE tiers."),
        ("axes", 0.94, 0.5, "11 BSV axes<br>(G01–G11)", "#d2a8ff",
         "Family-state vector with caveats."),
    ]
    edges = [
        ("query", "prep"), ("prep", "motif"), ("prep", "mss"),
        ("motif", "hybrid"), ("mss", "hybrid"),
        ("hybrid", "confamb"), ("confamb", "axes"),
    ]
    pos = {n[0]: (n[1], n[2]) for n in nodes}

    fig = go.Figure()
    for src, tgt in edges:
        x0, y0 = pos[src]; x1, y1 = pos[tgt]
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.0, arrowwidth=1.4,
            arrowcolor="#6e7681",
        )
    box_w, box_h = 0.10, 0.20
    for nid, x, y, label, color, hover in nodes:
        fig.add_shape(type="rect",
                      x0=x - box_w / 2, y0=y - box_h / 2,
                      x1=x + box_w / 2, y1=y + box_h / 2,
                      fillcolor="#161b22",
                      line=dict(color=color, width=2),
                      layer="below")
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="text",
            text=[label],
            textfont=dict(color="#f0f6fc", size=11),
            hovertext=[hover], hoverinfo="text",
            showlegend=False,
        ))
    fig.add_annotation(x=0.47, y=0.66, text="0.25·motif", showarrow=False,
                       font=dict(color="#bc8cff", size=10, family="monospace"))
    fig.add_annotation(x=0.47, y=0.34, text="0.75·MSS", showarrow=False,
                       font=dict(color="#7ee787", size=10, family="monospace"))
    fig.update_layout(
        template="plotly_dark", height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        xaxis=dict(visible=False, range=[-0.02, 1.02]),
        yaxis=dict(visible=False, range=[0, 1]),
        hoverlabel=dict(bgcolor="#161b22", font_color="#c9d1d9",
                        bordercolor="#30363d"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────
# Tab 3 link
# ────────────────────────────────────────────────────────────────────────

def render_tab3_link() -> None:
    ui.section_header("Detailed molecule explorer",
                      "Per-analyte spectrum + BSV radar + MSS table.")
    ui.card(
        title="🔜 Open in Tab 3 — Grounding tests",
        subtitle="Per-analyte view with anchor / support / anti-evidence overlays",
        body_md=(
            "<div style='font-size:0.88rem;'>"
            "Molecule-level explorer lives in Tab 3."
            "</div>"
        ),
        disabled=True,
    )


# ────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ────────────────────────────────────────────────────────────────────────

def render_sidebar_controls() -> dict:
    with st.sidebar:
        st.markdown("---")
        st.caption("**Tab 2 v2 · Family UMAP**")
        family_show_labels = st.checkbox(
            "Show family centroid labels", value=True,
            key="v2p3_family_show_labels")
        family_show_hulls = st.checkbox(
            "Show convex hulls per family", value=False,
            key="v2p3_family_show_hulls")
        family_opacity = st.slider("Family marker opacity", 0.30, 1.00, 0.85,
                                   0.05, key="v2p3_family_opacity")

        st.caption("**MSS analyte map**")
        mss_color_mode = st.radio(
            "MSS map · colour by",
            options=["biochemical class", "BSV family",
                     "cluster id (precomputed)"],
            index=0, key="v2p3_mss_color_mode")
        mss_show_cluster_labels = st.checkbox(
            "Annotate MSS clusters with dominant class", value=True,
            key="v2p3_mss_show_cluster_labels")
        mss_show_legend = st.checkbox(
            "Show MSS legend", value=False, key="v2p3_mss_show_legend")
        mss_opacity = st.slider("MSS marker opacity", 0.30, 1.00, 0.85, 0.05,
                                key="v2p3_mss_opacity")

        st.caption("**Saliency + ambiguity**")
        saliency_bin = st.select_slider(
            "Saliency band bin (cm⁻¹)", options=[5, 10, 20, 25, 50],
            value=10, key="v2p3_saliency_bin")
        ambiguity_threshold = st.slider(
            "Ambiguity / overlap threshold", 0.10, 0.60, 0.30, 0.05,
            key="v2p3_ambiguity_threshold")

        st.caption("**Axis overlap matrix**")
        overlap_mode = st.radio(
            "Overlap mode",
            options=["row-normalised (fraction of axis A)",
                     "absolute shared band count"],
            index=0, key="v2p3_overlap_mode")

        with st.expander("Advanced controls", expanded=False):
            show_canonical_labels = st.checkbox(
                "Show canonical band labels on saliency", value=False,
                key="v2p3_show_canonical_labels")
            dendro_mode = st.radio(
                "Dendrogram cluster summary mode",
                options=["MOTIF", "MSS"], index=0, horizontal=True,
                key="v2p3_dendro_mode")

    return {
        "family_show_labels": family_show_labels,
        "family_show_hulls": family_show_hulls,
        "family_opacity": family_opacity,
        "mss_color_mode": mss_color_mode,
        "mss_show_cluster_labels": mss_show_cluster_labels,
        "mss_show_legend": mss_show_legend,
        "mss_opacity": mss_opacity,
        "saliency_bin": saliency_bin,
        "ambiguity_threshold": ambiguity_threshold,
        "network_threshold": ambiguity_threshold,
        "overlap_mode": overlap_mode,
        "show_canonical_labels": show_canonical_labels,
        "dendro_mode": dendro_mode,
    }


# ────────────────────────────────────────────────────────────────────────
# Public entry
# ────────────────────────────────────────────────────────────────────────

def render(app_cfg: dict, evidence_layers_cfg: dict, manifest: dict,
           manifest_path: Path) -> None:
    build_root = Path(app_cfg["paths"]["build_root"]) if "paths" in app_cfg \
        else Path(manifest.get("build_root", "/Volumes/SSD_Rad/GAIRA_BUILD"))

    st.markdown("# Motif · MSS · BSV — v2 · pass 3")
    st.caption("Real motif/family UMAP first; MSS analyte structure with "
               "annotated clusters; per-family drilldown; confusion-style "
               "overlap matrix.")

    controls = render_sidebar_controls()

    ui.divider()
    render_representation_hierarchy()

    ui.divider()
    render_normal_motif_family_umap(build_root, controls)

    ui.divider()
    with st.expander("C · MSS analyte cluster map", expanded=True):
        render_mss_analyte_cluster_map(build_root, controls)

    ui.divider()
    with st.expander("D · MSS-within-family drilldown", expanded=True):
        render_mss_within_family_drilldown(build_root)

    ui.divider()
    with st.expander("E · BSV saliency · band ⇒ axis mapping", expanded=True):
        render_bsv_saliency_map(build_root, controls)

    ui.divider()
    with st.expander("F · Shared bands & ambiguity", expanded=True):
        render_shared_band_ambiguity_map(build_root, controls)

    ui.divider()
    with st.expander("G · Axis overlap · confusion-style matrix",
                     expanded=True):
        render_axis_overlap_confusion_style(build_root, controls)

    ui.divider()
    with st.expander("H · Hierarchical clustering support", expanded=False):
        render_dendrogram_support(build_root, controls)

    ui.divider()
    with st.expander("I · Hybrid BSV evidence flow", expanded=False):
        render_hybrid_bsv_flow_diagram(build_root)

    ui.divider()
    render_tab3_link()
