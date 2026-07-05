"""Tab 2 — Motif / MSS / BSV Construction (v1 stable).

The stable readable Tab 2 from the v1 build. Interactive Plotly UMAPs +
side-by-side comparison + dendrograms + BSV saliency heatmap + hybrid
flow diagram. No family-first hull overlay (that experiment lives in v2).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import ConvexHull, QhullError

from components import ui_blocks as ui
from utils.embedding_loader import (
    load_embedding, load_cluster_breakdown, load_mss_signatures,
    load_analyte_to_group, load_bsv_registry,
    top_anchors_for_class, build_bsv_band_saliency,
    shared_band_overlay, BSV_AXES_ORDER,
)
from utils.figure_loader import load_image_safe


_PALETTE = [
    "#79c0ff", "#ff7b72", "#d2a8ff", "#a5d6ff", "#ffa657",
    "#7ee787", "#f0883e", "#bc8cff", "#56d4dd", "#ffdf5d",
    "#ff9492", "#a371f7", "#39c5cf", "#fab8c4", "#85e89d",
    "#f97583", "#b392f0", "#9ecbff", "#ffea7f", "#f0d4a3",
    "#5dade2", "#e59866", "#cb6ce6", "#7dcea0", "#f5b041",
    "#af7ac5", "#48c9b0", "#ec7063", "#5499c7", "#aab7b8",
]


def _color_map(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    uniq = sorted({str(v) for v in values if v is not None})
    for i, v in enumerate(uniq):
        out[v] = _PALETTE[i % len(_PALETTE)]
    return out


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


# ─── concept overview ─────────────────────────────────────────────────────

def _concept_overview() -> None:
    ui.section_header("Concept overview",
                      "GAIRA builds three nested resolutions of biochemical evidence.")
    cols = st.columns(3)
    levels = [
        ("Level 1 · Spectral primitives",
         "peaks, FWHM, asymmetry, local prominence, "
         "co-band patterns, quartile envelope, orphan-companion count"),
        ("Level 2 · MSS (per-analyte signatures)",
         "anchor + support + anti-evidence + competitor sets, "
         "with HIGH / MODERATE / LOW reliability tiers"),
        ("Level 3 · 11-axis BSV (family aggregation)",
         "G01–G11 biochemistry-family magnitudes + ambiguity routing + "
         "confidence × magnitude scoring"),
    ]
    for col, (title, body) in zip(cols, levels):
        with col:
            ui.card(title, f"<div style='font-size:0.88rem;'>{body}</div>")


def _mss_evolution() -> None:
    ui.section_header("MSS evolution",
                      "From 30 broad-class compressed MSS to 257 analyte-level signatures.")
    cols = st.columns(2)
    with cols[0]:
        ui.card(
            title="v4.1 — broad-class MSS",
            subtitle="initial build",
            body_md=(
                "<ul style='font-size:0.88rem; margin:4px 0 4px 18px;'>"
                "<li>30 broad-class signatures</li>"
                "<li>over-compressed — masked within-family chemistry</li>"
                "<li>weaker analyte specificity (top-3 saturated at ~76%)</li>"
                "</ul>"
            ),
        )
    with cols[1]:
        ui.card(
            title="v4.2 — analyte-level MSS",
            subtitle="repair-loop output",
            body_md=(
                "<ul style='font-size:0.88rem; margin:4px 0 4px 18px;'>"
                "<li>257 analyte-level signatures</li>"
                "<li>enriched primitives + co-band patterns + envelope</li>"
                "<li>broad-equivalent top-3 76.5% → <strong>94.8%</strong> "
                "(SERS broad-equiv 58% → <strong>92%</strong>)</li>"
                "</ul>"
            ),
        )


# ─── interactive UMAP cluster view ────────────────────────────────────────

def _convex_hull_xy(points: np.ndarray):
    if len(points) < 3:
        return None
    try:
        hull = ConvexHull(points)
        idx = list(hull.vertices) + [hull.vertices[0]]
        return points[idx]
    except (QhullError, ValueError):
        return None


def _add_cluster_hulls(fig: go.Figure, df: pd.DataFrame, opacity: float) -> None:
    for cid, sub in df.groupby("cluster_id"):
        if len(sub) < 3:
            continue
        pts = sub[["umap_1", "umap_2"]].values
        hull = _convex_hull_xy(pts)
        if hull is None:
            continue
        fig.add_trace(go.Scatter(
            x=hull[:, 0], y=hull[:, 1],
            fill="toself",
            fillcolor=f"rgba(125,160,200,{opacity * 0.18})",
            line=dict(color=f"rgba(180,200,230,{opacity * 0.55})", width=1),
            mode="lines", hoverinfo="skip", showlegend=False,
            name=f"hull_{cid}",
        ))


def _add_cluster_labels(fig: go.Figure, df: pd.DataFrame) -> None:
    for cid, sub in df.groupby("cluster_id"):
        if len(sub) < 2:
            continue
        cx = float(sub["umap_1"].mean())
        cy = float(sub["umap_2"].mean())
        dom = Counter(sub["broad_class"]).most_common(1)[0][0]
        fig.add_annotation(
            x=cx, y=cy, text=f"<b>{dom}</b>",
            font=dict(color="#f0f6fc", size=11),
            bgcolor="rgba(13,17,23,0.78)",
            bordercolor="#30363d", borderwidth=1, borderpad=3,
            showarrow=False, align="center",
        )


def _build_umap_figure(df: pd.DataFrame, sig_df,
                       *, color_by: str, show_hulls: bool, show_labels: bool,
                       opacity: float, title: str = "") -> go.Figure:
    color_col = "broad_class" if color_by == "class" else "cluster_id"
    df = df.copy()
    df[color_col] = df[color_col].astype(str)
    cmap = _color_map(df[color_col].tolist())

    fig = go.Figure()
    if show_hulls:
        _add_cluster_hulls(fig, df, opacity)

    for label, sub in df.groupby(color_col):
        anchors = (sub["broad_class"].apply(
            lambda c: top_anchors_for_class(sig_df, c, k=3))
            if sig_df is not None else pd.Series([""] * len(sub), index=sub.index))
        custom = np.column_stack([
            sub["analyte_id"].astype(str).values,
            sub["broad_class"].astype(str).values,
            sub["regime"].astype(str).values,
            sub["support_tier"].astype(str).values,
            sub["n_spectra"].astype(str).values,
            sub["cluster_id"].astype(str).values,
            anchors.values,
        ])
        hover = (
            "<b>%{customdata[0]}</b><br>"
            "class: %{customdata[1]}<br>"
            "regime: %{customdata[2]} · support: %{customdata[3]} · "
            "n_spectra: %{customdata[4]}<br>"
            "cluster: %{customdata[5]}<br>"
            "top anchors: %{customdata[6]}<extra></extra>"
        )
        fig.add_trace(go.Scattergl(
            x=sub["umap_1"], y=sub["umap_2"], mode="markers",
            name=str(label),
            marker=dict(color=cmap.get(str(label), "#79c0ff"),
                        size=8, opacity=opacity,
                        line=dict(width=0.5, color="#0d1117")),
            customdata=custom, hovertemplate=hover,
        ))

    if show_labels:
        _add_cluster_labels(fig, df)

    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, font=dict(size=14, color="#c9d1d9")),
        height=620, margin=dict(l=10, r=10, t=42, b=10),
        legend=dict(font=dict(size=10, color="#c9d1d9"),
                    bgcolor="rgba(13,17,23,0.6)",
                    bordercolor="#30363d", borderwidth=1,
                    itemsizing="constant"),
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        xaxis=dict(title="UMAP-1", gridcolor="#21262d", zeroline=False),
        yaxis=dict(title="UMAP-2", gridcolor="#21262d", zeroline=False),
        hoverlabel=dict(bgcolor="#161b22", font_color="#c9d1d9",
                        bordercolor="#30363d"),
    )
    return fig


def render_umap_cluster_view(build_root: Path, controls: dict) -> None:
    ui.section_header(
        "Interactive UMAP — analyte representation",
        "Each point is one analyte; clusters are precomputed agglomerative groupings.")

    mode = controls["umap_mode"]
    df = _cached_embedding(str(build_root), mode)
    if df is None:
        ui.warning_card(f"Embedding not found for mode `{mode}`.")
        return

    sig_df = _cached_signatures(str(build_root))
    fig = _build_umap_figure(
        df, sig_df,
        color_by=controls["color_by"],
        show_hulls=controls["show_hulls"],
        show_labels=controls["show_labels"],
        opacity=controls["opacity"],
        title=f"{mode} analyte UMAP — coloured by "
              f"{'biochemical class' if controls['color_by'] == 'class' else 'cluster id'}",
    )
    st.plotly_chart(fig, use_container_width=True)

    breakdown = _cached_breakdown(str(build_root), mode)
    if breakdown is None:
        return
    ui.section_header(
        f"{mode} clusters — biochemical interpretation",
        "Dominant class + purity + member count + sample analytes.")
    cols = st.columns(2)
    for i, (_, row) in enumerate(breakdown.iterrows()):
        col = cols[i % 2]
        with col:
            members = str(row.get("sample_members", "")).split(";")[:5]
            stats = {
                "members": str(int(row["n_members"])),
                "purity": f"{float(row['purity']):.2f}",
                "entropy": f"{float(row['entropy_bits']):.2f} bits",
                "Raman / SERS": f"{int(row['n_raman'])} / {int(row['n_sers'])}",
                "sample analytes": ", ".join(m.strip() for m in members if m.strip()),
            }
            ui.cluster_card(
                title=f"Cluster {int(row['cluster_id'])} · "
                      f"<em>{row['dominant_broad_class']}</em>",
                stats=stats,
            )

    ui.interpretation(
        "What this UMAP says",
        "Each cluster groups analytes whose <strong>spectral evidence "
        "vector</strong> is similar. Purity below 1.0 means the cluster "
        "contains chemistry from more than one biochemical family — that is "
        "expected and informative: it tells you where motifs <em>genuinely "
        "share bands</em> across families.",
    )


# ─── side-by-side dual UMAP comparison ────────────────────────────────────

def render_dual_umap_comparison(build_root: Path, controls: dict) -> None:
    ui.section_header("Side-by-side · MSS vs Motif",
                      "Same 236 analytes, two representation strategies.")
    mss = _cached_embedding(str(build_root), "MSS")
    mot = _cached_embedding(str(build_root), "MOTIF")
    if mss is None or mot is None:
        ui.warning_card("Need both MSS and Motif embeddings for the dual view.")
        return

    sig_df = _cached_signatures(str(build_root))
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("MSS analyte space (per-analyte signatures)",
                        "Motif analyte space (band-family motifs)"),
        horizontal_spacing=0.06,
    )
    cmap = _color_map(sorted(set(mss["broad_class"]).union(set(mot["broad_class"]))))

    def _add_panel(panel_df, col_idx, show_legend):
        for label, sub in panel_df.groupby("broad_class"):
            anchors = (sub["broad_class"].apply(
                lambda c: top_anchors_for_class(sig_df, c, k=3))
                if sig_df is not None else pd.Series([""] * len(sub), index=sub.index))
            custom = np.column_stack([
                sub["analyte_id"].astype(str).values,
                sub["broad_class"].astype(str).values,
                anchors.values,
            ])
            fig.add_trace(go.Scattergl(
                x=sub["umap_1"], y=sub["umap_2"], mode="markers",
                name=str(label),
                legendgroup=str(label),
                showlegend=show_legend,
                marker=dict(color=cmap.get(str(label), "#79c0ff"),
                            size=7, opacity=controls["opacity"],
                            line=dict(width=0.4, color="#0d1117")),
                customdata=custom,
                hovertemplate=("<b>%{customdata[0]}</b><br>"
                               "class: %{customdata[1]}<br>"
                               "anchors: %{customdata[2]}<extra></extra>"),
            ), row=1, col=col_idx)

    _add_panel(mss, 1, True)
    _add_panel(mot, 2, False)

    fig.update_layout(
        template="plotly_dark", height=540,
        margin=dict(l=10, r=10, t=46, b=10),
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        legend=dict(font=dict(size=9, color="#c9d1d9"),
                    bgcolor="rgba(13,17,23,0.6)"),
        hoverlabel=dict(bgcolor="#161b22", font_color="#c9d1d9",
                        bordercolor="#30363d"),
    )
    for c in (1, 2):
        fig.update_xaxes(title="UMAP-1", row=1, col=c, gridcolor="#21262d",
                         zeroline=False)
        fig.update_yaxes(title="UMAP-2", row=1, col=c, gridcolor="#21262d",
                         zeroline=False)
    st.plotly_chart(fig, use_container_width=True)

    ui.interpretation(
        "Reading the two panels together",
        "<strong>MSS compresses motif space into analyte-level structure.</strong> "
        "The Motif view shows where bands cluster across the 24 learned spectral "
        "motifs; the MSS view shows where the per-analyte evidence vector lands "
        "after anchor + support + anti-evidence + competitor scoring.",
    )


# ─── dendrograms ──────────────────────────────────────────────────────────

def render_dendrogram_panel(build_root: Path, controls: dict) -> None:
    ui.section_header(
        "Hierarchical structure (dendrograms)",
        "Pre-rendered hierarchical clustering with biochemical-family annotations.")
    rep_dir = build_root / "gaira_representation_cluster_analysis_v1" / "figures"
    cols = st.columns(2)
    with cols[0]:
        load_image_safe(rep_dir / "fig_mss_dendrogram_v1.png",
                        caption="MSS analyte dendrogram",
                        fallback_msg="MSS dendrogram artifact missing.")
    with cols[1]:
        load_image_safe(rep_dir / "fig_motif_dendrogram_v1.png",
                        caption="Motif analyte dendrogram",
                        fallback_msg="Motif dendrogram artifact missing.")

    if controls.get("show_hierarchy_levels", True):
        ui.interpretation(
            "Major splits — what the hierarchy is grouping by",
            "<ul style='margin: 4px 0 4px 18px;'>"
            "<li><strong>First split:</strong> lipid / sterol vs non-lipid chemistry.</li>"
            "<li><strong>Second tier:</strong> nucleic acids split from protein / "
            "amino-acid space; G04↔G05 phosphate/glycan collision shows up here.</li>"
            "<li><strong>Third tier:</strong> within-family decomposition — sugars "
            "from glycan polymers; aromatic from aliphatic AAs; sterol sub-classes.</li>"
            "<li><strong>Leaves:</strong> 236 analytes; singletons fringe their family.</li>"
            "</ul>",
        )


# ─── BSV saliency map ────────────────────────────────────────────────────

def render_bsv_saliency_map(build_root: Path, controls: dict) -> None:
    ui.section_header(
        "BSV saliency — band ⇒ axis mapping",
        "Where on the spectrum each BSV axis draws its evidence from.")

    band_min, band_max = 400, 1800
    band_bin = controls.get("saliency_bin", 10)
    mat, bin_centers, contributors = _cached_saliency(
        str(build_root), band_min, band_max, band_bin)
    if mat.sum() == 0:
        ui.warning_card("BSV saliency could not be built — registry/map missing.")
        return

    bsv_reg = _cached_bsv_registry(str(build_root))
    if bsv_reg is not None:
        axis_labels = {row["group_id"]: f"{row['group_id']} · {row['group_name']}"
                       for _, row in bsv_reg.iterrows()}
    else:
        axis_labels = {ax: ax for ax in BSV_AXES_ORDER}
    y_labels = [axis_labels.get(ax, ax) for ax in BSV_AXES_ORDER]

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

    canonical = [(725, "purine ring"), (785, "pyrimidine"), (1003, "Phe"),
                 (1080, "phosphate/glycan"), (1340, "glycan/purine"),
                 (1450, "amide-III/lipid"), (1655, "amide-I/C=C"),
                 (1745, "ester C=O")]
    for band, label in canonical:
        fig.add_vline(x=band, line=dict(color="#ff7b72", width=1, dash="dot"),
                      opacity=0.5,
                      annotation_text=f"{band} {label}",
                      annotation_position="top",
                      annotation_font=dict(size=9, color="#ff7b72"))

    fig.update_layout(
        template="plotly_dark", height=540,
        margin=dict(l=10, r=10, t=80, b=10),
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        xaxis=dict(title="Raman shift (cm⁻¹)", range=[band_min, band_max],
                   gridcolor="#21262d"),
        yaxis=dict(title="BSV axis"),
        hoverlabel=dict(bgcolor="#161b22", font_color="#c9d1d9",
                        bordercolor="#30363d"),
    )
    st.plotly_chart(fig, use_container_width=True)

    if controls.get("show_overlaps", True):
        overlap = shared_band_overlay(mat, threshold=controls.get("overlap_threshold", 0.3))
        ovl_fig = go.Figure(go.Bar(
            x=bin_centers, y=overlap,
            marker_color="#ffa657",
            hovertemplate="band %{x} cm⁻¹<br># axes contributing ≥thr: %{y}<extra></extra>",
        ))
        ovl_fig.update_layout(
            template="plotly_dark", height=180,
            margin=dict(l=10, r=10, t=30, b=10),
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            xaxis=dict(title="Raman shift (cm⁻¹)", range=[band_min, band_max],
                       gridcolor="#21262d"),
            yaxis=dict(title="# axes ≥ threshold"),
            title=dict(text="<b>Shared bands</b> — bars ≥ 2 mark collision-prone regions",
                       font=dict(size=12, color="#c9d1d9")),
        )
        st.plotly_chart(ovl_fig, use_container_width=True)

    ui.interpretation(
        "Why this matters",
        "<ul style='margin: 4px 0 4px 18px;'>"
        "<li><strong>G01 ↔ G02 overlap</strong> in 700-740 cm⁻¹ — purine ring shared.</li>"
        "<li><strong>G04 ↔ G05 confusion</strong> at 1080 cm⁻¹ — phosphate ↔ glycan.</li>"
        "<li><strong>G06 ↔ G08 lipid-protein overlap</strong> at 1655 cm⁻¹ — amide-I ↔ C=C.</li>"
        "<li>The orange bar plot above flags every band where ≥ 2 BSV axes draw "
        "evidence — those are the rows where MSS competitor logic earns its keep.</li>"
        "</ul>",
    )


# ─── hybrid BSV flow diagram ─────────────────────────────────────────────

def render_hybrid_bsv_flow_diagram(build_root: Path) -> None:
    ui.section_header(
        "Hybrid BSV — evidence flow",
        "How motif evidence + MSS evidence combine into a per-axis BSV magnitude.")

    nodes = [
        ("raw", 0.04, 0.5, "Raw spectrum", "#79c0ff", "Preprocessed Raman/SERS spectrum."),
        ("prim", 0.20, 0.5, "Spectral primitives", "#a5d6ff", "peaks · FWHM · prominence · co-band."),
        ("motif", 0.42, 0.78, "Motif evidence", "#bc8cff", "24 learned motifs · band-family firing"),
        ("mss", 0.42, 0.22, "MSS evidence", "#7ee787", "257 analyte signatures · anchor + competitor"),
        ("hybrid", 0.66, 0.5, "Hybrid BSV<br>0.25·motif + 0.75·MSS", "#ffa657",
         "Per-axis magnitude · max-aggregation · confidence × magnitude"),
        ("axes", 0.92, 0.5, "11 BSV axes<br>(G01–G11)", "#d2a8ff",
         "Family-state vector · ambiguity routing · output-tier policy"),
    ]
    edges = [
        ("raw", "prim"), ("prim", "motif"), ("prim", "mss"),
        ("motif", "hybrid"), ("mss", "hybrid"),
        ("hybrid", "axes"),
    ]
    fig = go.Figure()
    pos = {n[0]: (n[1], n[2]) for n in nodes}
    for src, tgt in edges:
        x0, y0 = pos[src]; x1, y1 = pos[tgt]
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.4,
            arrowcolor="#6e7681",
        )
    box_w, box_h = 0.13, 0.16
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
            textfont=dict(color="#f0f6fc", size=12),
            hovertext=[hover], hoverinfo="text",
            showlegend=False,
        ))
    fig.add_annotation(x=0.535, y=0.66, text="0.25·motif", showarrow=False,
                       font=dict(color="#bc8cff", size=10, family="monospace"))
    fig.add_annotation(x=0.535, y=0.34, text="0.75·MSS", showarrow=False,
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

    hybrid_dir = build_root / "gaira_base_4_hybrid_bsv_build_v1" / "figures"
    cols = st.columns(2)
    with cols[0]:
        load_image_safe(hybrid_dir / "fig_hybrid_family_confusion_heatmap_v1.png",
                        caption="11-axis confusion heatmap",
                        fallback_msg="Confusion heatmap missing.")
    with cols[1]:
        load_image_safe(hybrid_dir / "fig_hybrid_confidence_vs_accuracy_v1.png",
                        caption="Confidence calibration",
                        fallback_msg="Confidence calibration missing.")


# ─── Tab 3 link ──────────────────────────────────────────────────────────

def render_tab3_link() -> None:
    ui.section_header("Detailed molecule explorer",
                      "Per-analyte spectrum trace + BSV radar + MSS table.")
    ui.card(
        title="🔜 Open in Tab 3 — Grounding tests",
        subtitle="Per-analyte view with anchor / support / anti-evidence overlays",
        body_md=(
            "<div style='font-size:0.88rem;'>"
            "The molecule explorer has its own tab. Pick a dataset, molecule, "
            "and regime — Tab 3 will render the spectrum trace, anchor/support/"
            "anti-evidence overlays, BSV radar, MSS row, and the per-analyte "
            "calibration check."
            "</div>"
        ),
        disabled=True,
    )


# ─── sidebar controls ───────────────────────────────────────────────────

def render_sidebar_controls() -> dict:
    with st.sidebar:
        st.markdown("---")
        st.caption("**Tab 2 · UMAP controls**")
        umap_mode = st.radio("Embedding", options=["MSS", "MOTIF"], index=0,
                             horizontal=True, key="tab2_umap_mode")
        color_by = st.radio("Colour by", options=["class", "cluster"], index=0,
                            horizontal=True, key="tab2_color_by")
        show_hulls = st.checkbox("Cluster hulls", value=True, key="tab2_show_hulls")
        show_labels = st.checkbox("Cluster labels", value=True, key="tab2_show_labels")
        opacity = st.slider("Marker opacity", 0.30, 1.00, 0.80, 0.05,
                            key="tab2_opacity")

        st.markdown("---")
        st.caption("**Tab 2 · BSV saliency controls**")
        saliency_bin = st.select_slider(
            "Band bin (cm⁻¹)", options=[5, 10, 20, 25, 50], value=10,
            key="tab2_saliency_bin")
        show_overlaps = st.checkbox("Show shared-band overlay", value=True,
                                    key="tab2_show_overlaps")
        overlap_threshold = st.slider("Overlap threshold", 0.10, 0.60, 0.30, 0.05,
                                      key="tab2_overlap_threshold")

        st.markdown("---")
        st.caption("**Tab 2 · Dendrogram controls**")
        show_hierarchy_levels = st.checkbox(
            "Show hierarchical-level interpretation",
            value=True, key="tab2_show_hierarchy_levels")

    return {
        "umap_mode": umap_mode,
        "color_by": color_by,
        "show_hulls": show_hulls,
        "show_labels": show_labels,
        "opacity": opacity,
        "saliency_bin": saliency_bin,
        "show_overlaps": show_overlaps,
        "overlap_threshold": overlap_threshold,
        "show_hierarchy_levels": show_hierarchy_levels,
    }


# ─── public entry ────────────────────────────────────────────────────────

def render(app_cfg: dict, evidence_layers_cfg: dict, manifest: dict,
           manifest_path: Path) -> None:
    build_root = Path(app_cfg["paths"]["build_root"]) if "paths" in app_cfg \
        else Path(manifest.get("build_root", "/Volumes/SSD_Rad/GAIRA_BUILD"))

    st.markdown("# Motif · MSS · BSV — construction")
    st.caption(
        "How GAIRA builds the biochemical representation: "
        "spectral primitives → analyte-level MSS → 11-axis BSV.")

    controls = render_sidebar_controls()

    ui.divider()
    _concept_overview()
    ui.divider()
    _mss_evolution()

    ui.divider()
    with st.expander("Interactive UMAP (MSS / Motif toggle)", expanded=True):
        render_umap_cluster_view(build_root, controls)

    ui.divider()
    with st.expander("Side-by-side · MSS vs Motif", expanded=True):
        render_dual_umap_comparison(build_root, controls)

    ui.divider()
    with st.expander("Hierarchical dendrograms (annotated)", expanded=False):
        render_dendrogram_panel(build_root, controls)

    ui.divider()
    with st.expander("BSV saliency map (band ⇒ axis)", expanded=True):
        render_bsv_saliency_map(build_root, controls)

    ui.divider()
    with st.expander("Hybrid BSV flow", expanded=False):
        render_hybrid_bsv_flow_diagram(build_root)

    ui.divider()
    render_tab3_link()
