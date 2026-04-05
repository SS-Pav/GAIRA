#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from gaira.demo.ev_graph_builder import build_graph_tables
from gaira.demo.ev_latent_map import prepare_latent_map_tables


st.set_page_config(
    page_title="GAIRAM v1 EV Explorer",
    page_icon="🧬",
    layout="wide",
)


CUSTOM_CSS = """
<style>
.main > div {
  padding-top: 1.1rem;
}
[data-testid="stAppViewContainer"] {
  background: radial-gradient(circle at top left, #1a2733 0%, #111922 46%, #0e151d 100%);
}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #101822 0%, #0b1118 100%);
}
.hero {
  padding: 1.45rem 1.65rem;
  border-radius: 20px;
  background: linear-gradient(135deg, #edf3f7 0%, #f5eee6 100%);
  border: 1px solid rgba(57, 73, 88, 0.16);
  box-shadow: 0 16px 28px rgba(15, 26, 35, 0.22);
  margin-bottom: 1rem;
}
.card {
  padding: 1rem 1.05rem;
  border-radius: 16px;
  border: 1px solid rgba(57, 73, 88, 0.14);
  background: #f9fbfd;
  box-shadow: 0 8px 22px rgba(15, 26, 35, 0.10);
  margin-bottom: 0.9rem;
}
.card, .card p, .card div, .card span, .card li, .card strong, .card b {
  color: #1d2a36 !important;
}
.eyebrow {
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-size: 0.72rem;
  color: #836a52 !important;
  margin-bottom: 0.35rem;
}
.big-metric {
  font-size: 1.7rem;
  font-weight: 680;
  color: #1d2a36 !important;
}
.subtle {
  color: #556371 !important;
  font-size: 0.93rem;
}
.badge {
  display: inline-block;
  padding: 0.18rem 0.55rem;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 640;
  margin-right: 0.35rem;
  margin-bottom: 0.25rem;
}
.badge-strong { background: #dff4e8; color: #245742 !important; }
.badge-moderate { background: #fff1d7; color: #875a00 !important; }
.badge-weak { background: #f8e2e2; color: #8a3d3d !important; }
.badge-mixed { background: #e9eef3; color: #51606e !important; }
.badge-info { background: #e6eef9; color: #365d8c !important; }
.legend-dot {
  display: inline-block;
  width: 0.82rem;
  height: 0.82rem;
  border-radius: 50%;
  margin-right: 0.42rem;
}
.segmented-help {
  color: #d9e3ec;
  font-size: 0.95rem;
  margin-bottom: 0.4rem;
}
</style>
"""


def strength_badge(strength: str) -> str:
    label = strength.replace("_", " ").title()
    css = {
        "strong": "badge-strong",
        "moderate": "badge-moderate",
        "weak": "badge-weak",
    }.get(strength, "badge-mixed")
    return f'<span class="badge {css}">{label}</span>'


@st.cache_data(show_spinner=False)
def load_demo_state() -> dict[str, pd.DataFrame]:
    try:
        graph_state = build_graph_tables(write_outputs=True)
        latent_state = prepare_latent_map_tables(graph_state, write_outputs=True)
    except PermissionError:
        graph_state = build_graph_tables(write_outputs=False)
        latent_state = prepare_latent_map_tables(graph_state, write_outputs=False)
    graph_state.update(latent_state)
    return graph_state


def cluster_story_presets(cluster_df: pd.DataFrame) -> dict[str, str]:
    ranked = cluster_df.sort_values(["theme_support_strength_rank", "cluster_size"], ascending=[True, False])

    def first_match(df: pd.DataFrame) -> str:
        return str(df.iloc[0]["cluster_id"]) if not df.empty else str(ranked.iloc[0]["cluster_id"])

    return {
        "Strongest control-like EV cluster": first_match(ranked[ranked["dominant_harmonized_anchor"] == "ev_control_or_baseline"]),
        "Strongest disease/stress EV cluster": first_match(ranked[ranked["dominant_harmonized_anchor"] == "ev_disease_or_stress"]),
        "Best cross-dataset mixed cluster": first_match(ranked[ranked["cross_dataset_mixed"] == True]),
        "Strongest non-protein example": first_match(ranked[ranked["top_biochemical_theme"] != "protein_peptide_associated"]),
        "Most ambiguous cluster": str(cluster_df.sort_values(["theme_entropy", "cluster_size"], ascending=[False, False]).iloc[0]["cluster_id"]),
    }


def build_cluster_option_labels(cluster_df: pd.DataFrame) -> dict[str, str]:
    labels = {}
    for row in cluster_df.to_dict(orient="records"):
        cluster_id = str(row["cluster_id"])
        labels[cluster_id] = (
            f"{cluster_id} · {row['dominant_harmonized_anchor'].replace('_', ' ')} · "
            f"{row['top_biochemical_theme'].replace('_', ' ')} · n={int(row['cluster_size'])}"
        )
    return labels


def theme_color_map(cluster_df: pd.DataFrame) -> dict[str, str]:
    palette = [
        "#2f7f6f",
        "#c06c50",
        "#6c5f96",
        "#a47b29",
        "#3f7498",
        "#a14860",
        "#607f43",
        "#7f6a5e",
    ]
    themes = sorted(cluster_df["top_biochemical_theme"].dropna().astype(str).unique().tolist())
    return {theme: palette[idx % len(palette)] for idx, theme in enumerate(themes)}


def anchor_color_map(cluster_df: pd.DataFrame) -> dict[str, str]:
    return {
        "ev_control_or_baseline": "#54728a",
        "ev_disease_or_stress": "#bc6a50",
        "ev_cell_line_model": "#7f63a5",
    } | {
        anchor: "#6f7d89"
        for anchor in sorted(cluster_df["dominant_harmonized_anchor"].dropna().astype(str).unique().tolist())
        if anchor not in {"ev_control_or_baseline", "ev_disease_or_stress", "ev_cell_line_model"}
    }


def filter_clusters(
    cluster_df: pd.DataFrame,
    *,
    anchor_filter: str,
    theme_filter: str,
    strength_filter: str,
    cross_dataset_only: bool,
    min_cluster_size: int,
) -> pd.DataFrame:
    filtered = cluster_df.copy()
    if anchor_filter != "All":
        filtered = filtered[filtered["dominant_harmonized_anchor"] == anchor_filter]
    if theme_filter != "All":
        filtered = filtered[filtered["top_biochemical_theme"] == theme_filter]
    if strength_filter != "All":
        filtered = filtered[filtered["theme_support_strength"] == strength_filter]
    if cross_dataset_only:
        filtered = filtered[filtered["cross_dataset_mixed"] == True]
    filtered = filtered[filtered["cluster_size"] >= min_cluster_size]
    return filtered.sort_values(["theme_support_strength_rank", "cluster_size"], ascending=[True, False]).reset_index(drop=True)


def nearby_clusters(neighbor_edges: pd.DataFrame, selected_cluster: str) -> pd.DataFrame:
    if neighbor_edges.empty:
        return pd.DataFrame(columns=["neighbor_cluster_id", "similarity"])
    sub = neighbor_edges[
        (neighbor_edges["source_cluster_id"] == selected_cluster)
        | (neighbor_edges["target_cluster_id"] == selected_cluster)
    ].copy()
    if sub.empty:
        return pd.DataFrame(columns=["neighbor_cluster_id", "similarity"])
    if "similarity" not in sub.columns:
        sub["similarity"] = pd.to_numeric(sub.get("weight"), errors="coerce")
    sub["neighbor_cluster_id"] = sub.apply(
        lambda row: row["target_cluster_id"] if row["source_cluster_id"] == selected_cluster else row["source_cluster_id"],
        axis=1,
    )
    return sub[["neighbor_cluster_id", "similarity"]].sort_values("similarity", ascending=False).reset_index(drop=True)


def metric_card(title: str, value: str, subtitle: str = "") -> None:
    st.markdown(
        f"<div class='card'><div class='eyebrow'>{title}</div>"
        f"<div class='big-metric'>{value}</div>"
        f"<div class='subtle'>{subtitle}</div></div>",
        unsafe_allow_html=True,
    )


def build_map_hover_frame(df: pd.DataFrame) -> pd.DataFrame:
    hover_df = df.copy()
    hover_df["pretty_anchor"] = hover_df["dominant_harmonized_anchor"].str.replace("_", " ")
    hover_df["pretty_theme"] = hover_df["top_biochemical_theme"].str.replace("_", " ")
    hover_df["pretty_secondary"] = hover_df["secondary_biochemical_theme"].fillna("none").str.replace("_", " ")
    return hover_df


def build_structure_map(centroid_df: pd.DataFrame, selected_cluster: str) -> go.Figure:
    plot_df = build_map_hover_frame(centroid_df)
    plot_df["cluster_type"] = plot_df["cross_dataset_mixed"].map({True: "cross-dataset mixed", False: "mostly dataset-pure"})
    fig = px.scatter(
        plot_df,
        x="dim1",
        y="dim2",
        size="cluster_size",
        size_max=40,
        color="cluster_type",
        color_discrete_map={
            "cross-dataset mixed": "#d87a3a",
            "mostly dataset-pure": "#62717f",
        },
        symbol="cross_dataset_mixed",
        symbol_map={True: "diamond", False: "circle"},
        hover_data={
            "cluster_id": True,
            "cluster_size": True,
            "cross_dataset_mixed": True,
            "dataset_purity": ":.2f",
            "pretty_anchor": True,
            "pretty_theme": True,
            "theme_support_strength": True,
            "dim1": False,
            "dim2": False,
        },
    )
    selected = plot_df[plot_df["cluster_id"] == selected_cluster]
    if not selected.empty:
        fig.add_trace(
            go.Scatter(
                x=selected["dim1"],
                y=selected["dim2"],
                mode="markers",
                name="selected cluster",
                hoverinfo="skip",
                marker=dict(
                    size=(selected["cluster_size"].pow(0.5) * 1.2).clip(lower=22, upper=48),
                    color="rgba(0,0,0,0)",
                    symbol="circle-open",
                    line=dict(color="#f2c14e", width=5),
                ),
                showlegend=False,
            )
        )
    fig.update_layout(
        height=620,
        margin=dict(l=8, r=8, t=10, b=8),
        plot_bgcolor="#fbfcfd",
        paper_bgcolor="#fbfcfd",
        legend_title_text="Cluster type",
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
    )
    return fig


def build_painted_map(
    centroid_df: pd.DataFrame,
    selected_cluster: str,
    *,
    paint_mode: str,
    theme_colors: dict[str, str],
    anchor_colors: dict[str, str],
) -> go.Figure:
    plot_df = build_map_hover_frame(centroid_df)
    if paint_mode == "EV state context":
        color_col = "pretty_anchor"
        palette = {key.replace("_", " "): value for key, value in anchor_colors.items()}
    elif paint_mode == "Biochemical theme":
        color_col = "pretty_theme"
        palette = {key.replace("_", " "): value for key, value in theme_colors.items()}
    elif paint_mode == "Cross-dataset mixing":
        color_col = "cross_dataset_label"
        plot_df["cross_dataset_label"] = plot_df["cross_dataset_mixed"].map({True: "cross-dataset mixed", False: "mostly dataset-pure"})
        palette = {"cross-dataset mixed": "#d87a3a", "mostly dataset-pure": "#62717f"}
    else:
        color_col = "support_label"
        plot_df["support_label"] = plot_df["theme_support_strength"].str.replace("_", " ")
        palette = {"strong": "#2f7f6f", "moderate": "#c08c2b", "weak": "#a85050"}

    fig = px.scatter(
        plot_df,
        x="dim1",
        y="dim2",
        size="cluster_size",
        size_max=40,
        color=color_col,
        color_discrete_map=palette,
        symbol="cross_dataset_mixed",
        symbol_map={True: "diamond", False: "circle"},
        hover_data={
            "cluster_id": True,
            "cluster_size": True,
            "pretty_anchor": True,
            "pretty_theme": True,
            "pretty_secondary": True,
            "theme_support_strength": True,
            "cross_dataset_mixed": True,
            "dim1": False,
            "dim2": False,
        },
    )
    selected = plot_df[plot_df["cluster_id"] == selected_cluster]
    if not selected.empty:
        fig.add_trace(
            go.Scatter(
                x=selected["dim1"],
                y=selected["dim2"],
                mode="markers",
                name="selected cluster",
                hoverinfo="skip",
                marker=dict(
                    size=(selected["cluster_size"].pow(0.5) * 1.2).clip(lower=22, upper=48),
                    color="rgba(0,0,0,0)",
                    symbol="circle-open",
                    line=dict(color="#f2c14e", width=5),
                ),
                showlegend=False,
            )
        )
    fig.update_layout(
        height=620,
        margin=dict(l=8, r=8, t=10, b=8),
        plot_bgcolor="#fbfcfd",
        paper_bgcolor="#fbfcfd",
        legend_title_text=paint_mode,
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
    )
    return fig


def build_sampled_point_map(sampled_df: pd.DataFrame, selected_cluster: str, color_by: str) -> go.Figure:
    plot_df = sampled_df.copy()
    if color_by == "cluster_id":
        plot_df["plot_color"] = plot_df["cluster_id"]
        palette = px.colors.qualitative.Alphabet
    elif color_by == "dominant_harmonized_anchor":
        plot_df["plot_color"] = plot_df["dominant_harmonized_anchor"].str.replace("_", " ")
        palette = px.colors.qualitative.Set2
    else:
        plot_df["plot_color"] = plot_df["top_biochemical_theme"].str.replace("_", " ")
        palette = px.colors.qualitative.Safe

    fig = px.scatter(
        plot_df,
        x="dim1",
        y="dim2",
        color="plot_color",
        opacity=0.62,
        hover_data={
            "cluster_id": True,
            "dataset_id": True,
            "dominant_harmonized_anchor": True,
            "top_biochemical_theme": True,
            "label_optional": True,
            "dim1": False,
            "dim2": False,
        },
        color_discrete_sequence=palette,
    )
    selected = plot_df[plot_df["cluster_id"] == selected_cluster]
    if not selected.empty:
        fig.add_trace(
            go.Scatter(
                x=selected["dim1"],
                y=selected["dim2"],
                mode="markers",
                name="selected cluster points",
                marker=dict(color="#f2c14e", size=7, line=dict(color="#ffffff", width=0.8)),
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        height=620,
        margin=dict(l=8, r=8, t=10, b=8),
        plot_bgcolor="#fbfcfd",
        paper_bgcolor="#fbfcfd",
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        legend_title_text=color_by.replace("_", " "),
    )
    return fig


def build_pyvis_html(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    visible_clusters: set[str],
    selected_cluster: str,
    *,
    include_dataset_nodes: bool,
    include_grounding_nodes: bool,
    include_cluster_edges: bool,
    min_edge_weight: float,
    focus_selected: bool,
) -> str:
    cluster_node_ids = {f"cluster::{cluster_id}" for cluster_id in visible_clusters}
    allowed_nodes = set(cluster_node_ids)

    for row in edges_df.to_dict(orient="records"):
        edge_type = str(row["edge_type"])
        if float(row["weight"]) < min_edge_weight and edge_type != "cluster_has_anchor":
            continue
        if edge_type == "cluster_contains_dataset" and not include_dataset_nodes:
            continue
        if edge_type == "cluster_retrieves_grounding" and not include_grounding_nodes:
            continue
        if edge_type == "cluster_adjacent_cluster" and not include_cluster_edges:
            continue
        source = str(row["source"])
        target = str(row["target"])
        if source in cluster_node_ids or target in cluster_node_ids:
            allowed_nodes.add(source)
            allowed_nodes.add(target)

    if focus_selected:
        selected_node = f"cluster::{selected_cluster}"
        selected_edges = edges_df[(edges_df["source"] == selected_node) | (edges_df["target"] == selected_node)]
        for row in selected_edges.to_dict(orient="records"):
            edge_type = str(row["edge_type"])
            if edge_type == "cluster_contains_dataset" and not include_dataset_nodes:
                continue
            if edge_type == "cluster_retrieves_grounding" and not include_grounding_nodes:
                continue
            if edge_type == "cluster_adjacent_cluster" and not include_cluster_edges:
                continue
            allowed_nodes.add(str(row["source"]))
            allowed_nodes.add(str(row["target"]))

    net = Network(height="760px", width="100%", directed=False, bgcolor="#fbfcfd", font_color="#22313f")
    options = {
        "nodes": {
            "borderWidth": 1.4,
            "borderWidthSelected": 4,
            "font": {"face": "Helvetica", "size": 14, "color": "#22313f"},
            "shadow": {"enabled": True, "color": "rgba(34,49,63,0.08)", "size": 10},
        },
        "edges": {
            "color": {"color": "rgba(112, 126, 140, 0.36)", "highlight": "#d87a3a"},
            "smooth": {"type": "dynamic"},
            "selectionWidth": 2.5,
        },
        "interaction": {"hover": True, "tooltipDelay": 120, "navigationButtons": True},
        "physics": {
            "stabilization": {"iterations": 250},
            "barnesHut": {
                "gravitationalConstant": -4200,
                "springLength": 145,
                "springConstant": 0.03,
                "damping": 0.15,
            },
        },
    }
    try:
        net.set_options(json.dumps(options))
    except Exception as exc:
        st.warning(f"Graph rendered with default pyvis options because custom graph styling failed: {exc}")

    node_lookup = {row["id"]: row for row in nodes_df.to_dict(orient="records")}
    for node_id in sorted(allowed_nodes):
        row = node_lookup.get(node_id)
        if row is None:
            continue
        border_color = "#f2c14e" if node_id == f"cluster::{selected_cluster}" else "#ffffff"
        net.add_node(
            node_id,
            label=str(row["label"]),
            title=str(row["title"]),
            color={"background": str(row["color"]), "border": border_color},
            size=float(row["size"]),
            shape="dot" if row["node_type"] != "grounding_ref" else "ellipse",
        )

    for row in edges_df.to_dict(orient="records"):
        edge_type = str(row["edge_type"])
        if float(row["weight"]) < min_edge_weight and edge_type != "cluster_has_anchor":
            continue
        if edge_type == "cluster_contains_dataset" and not include_dataset_nodes:
            continue
        if edge_type == "cluster_retrieves_grounding" and not include_grounding_nodes:
            continue
        if edge_type == "cluster_adjacent_cluster" and not include_cluster_edges:
            continue
        if str(row["source"]) not in allowed_nodes or str(row["target"]) not in allowed_nodes:
            continue
        net.add_edge(
            str(row["source"]),
            str(row["target"]),
            value=float(row["weight"]),
            width=1.0 + 4.0 * float(row["weight"]),
            title=f"{edge_type} · {float(row['weight']):.2f}",
            color="rgba(112, 126, 140, 0.34)",
        )
    return net.generate_html(notebook=False)


def render_cluster_explainer(
    selected_row: pd.Series,
    theme_rows: pd.DataFrame,
    hit_rows: pd.DataFrame,
    dataset_rows: pd.DataFrame,
    neighbor_rows: pd.DataFrame,
    preset_titles: list[str],
) -> None:
    col_a, col_b = st.columns([1.02, 1.08])
    with col_a:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='eyebrow'>Cluster Identity</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='big-metric'>{selected_row['cluster_id']}</div>", unsafe_allow_html=True)
        st.markdown(
            strength_badge(str(selected_row["theme_support_strength"]))
            + ("<span class='badge badge-info'>Cross-dataset mixed</span>" if bool(selected_row["cross_dataset_mixed"]) else ""),
            unsafe_allow_html=True,
        )
        if preset_titles:
            st.markdown(
                "".join(f"<span class='badge badge-mixed'>{title}</span>" for title in preset_titles),
                unsafe_allow_html=True,
            )
        st.markdown(f"**Cluster size**: {int(selected_row['cluster_size'])}")
        st.markdown(f"**Dataset purity**: {float(selected_row['dataset_purity']):.2f}")
        st.markdown(f"**Datasets represented**  \n{selected_row['datasets_represented']}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='eyebrow'>EV State Context</div>", unsafe_allow_html=True)
        st.markdown(f"**Dominant harmonized anchor**: {selected_row['dominant_harmonized_anchor'].replace('_', ' ')}")
        st.markdown(
            "These anchors are layered on after clustering to summarize broad EV state context such as control/baseline, disease/stress, or cell-line model."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='eyebrow'>Biochemical Interpretation</div>", unsafe_allow_html=True)
        st.markdown(f"**Top biochemical theme**: {selected_row['top_biochemical_theme'].replace('_', ' ')}")
        st.markdown(f"**Secondary biochemical theme**: {selected_row['secondary_biochemical_theme'].replace('_', ' ')}")
        st.markdown(f"**Interpretation summary**  \n{selected_row['interpretation_summary']}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='eyebrow'>Why This Cluster Got Its Meaning</div>", unsafe_allow_html=True)
        theme_plot = theme_rows.copy()
        theme_plot["pretty_theme"] = theme_plot["grounding_theme"].str.replace("_", " ")
        fig = px.bar(
            theme_plot,
            x="theme_share",
            y="pretty_theme",
            orientation="h",
            color="pretty_theme",
            labels={"theme_share": "Theme score share", "pretty_theme": "Theme"},
        )
        fig.update_layout(height=260, margin=dict(l=8, r=8, t=8, b=8), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            hit_rows[["dataset_id", "label_optional", "grounding_theme", "similarity", "query_source"]].rename(
                columns={"label_optional": "grounding_ref"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown(f"**Nearest grounding examples**  \n{selected_row['nearest_grounding_examples']}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='eyebrow'>Context + Caveats</div>", unsafe_allow_html=True)
        st.markdown(f"**Uncertainty**  \n{selected_row['uncertainty_notes']}")
        st.markdown(f"**Caveat**  \n{selected_row['caveat_notes']}")
        if not neighbor_rows.empty:
            st.markdown("**Nearest neighboring clusters**")
            st.dataframe(neighbor_rows.head(6), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='eyebrow'>Dataset Composition</div>", unsafe_allow_html=True)
        st.dataframe(
            dataset_rows[["dataset_id", "count", "share"]].rename(columns={"share": "cluster_share"}),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">GAIRAM v1</div>
          <h1 style="margin:0 0 0.35rem 0; color:#1d2a36;">EV Latent Structure Explorer</h1>
          <div class="subtle">
            The app is intentionally layered. First, inspect the EV structure learned from embeddings. Then paint biological state or biochemical meaning onto those clusters. Finally, inspect why a cluster received that interpretation.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        state = load_demo_state()
    except Exception as exc:
        st.error(f"Could not load EV explorer artifacts: {exc}")
        st.stop()

    clusters = state["ev_cluster_interpretation"].copy()
    themes = state["ev_cluster_theme_scores"].copy()
    hits = state["ev_cluster_grounding_hits"].copy()
    dataset_comp = state["dataset_composition"].copy()
    nodes = state["nodes"].copy()
    edges = state["edges"].copy()
    centroids = state["cluster_centroids"].copy()
    sampled_ev = state["sampled_ev"].copy()
    neighbor_edges = state["neighbor_edges"].copy()

    story_presets = cluster_story_presets(clusters)
    theme_colors = theme_color_map(clusters)
    anchor_colors = anchor_color_map(clusters)

    st.markdown("<div class='segmented-help'>Choose the layer you want to inspect.</div>", unsafe_allow_html=True)
    active_view = st.radio(
        "View",
        ["Structure", "Painted Meaning", "Cluster Explainer", "Advanced Graph", "Explainers"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.sidebar.title("Controls")
    st.sidebar.caption("Selection is shared across views.")

    anchor_filter = st.sidebar.selectbox(
        "Filter by anchor",
        ["All"] + sorted(clusters["dominant_harmonized_anchor"].dropna().astype(str).unique().tolist()),
    )
    theme_filter = st.sidebar.selectbox(
        "Filter by biochemical theme",
        ["All"] + sorted(clusters["top_biochemical_theme"].dropna().astype(str).unique().tolist()),
    )
    strength_filter = st.sidebar.selectbox("Filter by support strength", ["All", "strong", "moderate", "weak"])
    cross_dataset_only = st.sidebar.checkbox("Show only cross-dataset mixed clusters", value=False)
    min_cluster_size = st.sidebar.slider(
        "Minimum cluster size",
        min_value=100,
        max_value=int(clusters["cluster_size"].max()),
        value=800,
        step=50,
    )
    preset_name = st.sidebar.selectbox("Story mode", ["None"] + list(story_presets.keys()))

    filtered_clusters = filter_clusters(
        clusters,
        anchor_filter=anchor_filter,
        theme_filter=theme_filter,
        strength_filter=strength_filter,
        cross_dataset_only=cross_dataset_only,
        min_cluster_size=min_cluster_size,
    )
    if filtered_clusters.empty:
        st.warning("No EV clusters match the current filters.")
        st.stop()

    cluster_labels = build_cluster_option_labels(filtered_clusters)
    default_cluster = story_presets.get(preset_name, str(filtered_clusters.iloc[0]["cluster_id"]))
    options = filtered_clusters["cluster_id"].tolist()
    selected_cluster = st.sidebar.selectbox(
        "Selected cluster",
        options,
        index=options.index(default_cluster) if default_cluster in options else 0,
        format_func=lambda cluster_id: cluster_labels.get(cluster_id, cluster_id),
    )

    # View-specific controls
    if active_view == "Painted Meaning":
        paint_mode = st.sidebar.selectbox(
            "Paint clusters by",
            ["EV state context", "Biochemical theme", "Cross-dataset mixing", "Support strength"],
        )
        sampled_color_by = "top_biochemical_theme"
    elif active_view == "Structure":
        paint_mode = "Biochemical theme"
        sampled_color_by = st.sidebar.selectbox(
            "Optional sampled point color",
            ["cluster_id", "top_biochemical_theme", "dominant_harmonized_anchor"],
        )
    elif active_view == "Advanced Graph":
        paint_mode = "Biochemical theme"
        sampled_color_by = "top_biochemical_theme"
        include_dataset_nodes = st.sidebar.checkbox("Show dataset nodes", value=False)
        include_grounding_nodes = st.sidebar.checkbox("Show grounding reference nodes", value=False)
        include_cluster_edges = st.sidebar.checkbox("Show cluster-cluster edges", value=False)
        focus_selected = st.sidebar.checkbox("Focus selected cluster neighborhood", value=True)
        min_edge_weight = st.sidebar.slider("Minimum edge weight", 0.0, 1.0, 0.12, 0.02)
    else:
        paint_mode = "Biochemical theme"
        sampled_color_by = "top_biochemical_theme"

    selected_row = clusters[clusters["cluster_id"] == selected_cluster].iloc[0]
    selected_theme_rows = themes[themes["cluster_id"] == selected_cluster].sort_values("weighted_score", ascending=False).head(6).copy()
    selected_hit_rows = (
        hits[hits["cluster_id"] == selected_cluster]
        .sort_values("weighted_score", ascending=False)
        .drop_duplicates("sample_key")
        .head(8)
        .copy()
    )
    selected_dataset_rows = dataset_comp[dataset_comp["cluster_id"] == selected_cluster].copy()
    selected_neighbors = nearby_clusters(neighbor_edges, selected_cluster)
    preset_titles = [title for title, cluster_id in story_presets.items() if cluster_id == selected_cluster]

    filtered_centroids = centroids[centroids["cluster_id"].isin(filtered_clusters["cluster_id"])].copy()
    filtered_sampled = sampled_ev[sampled_ev["cluster_id"].isin(filtered_clusters["cluster_id"])].copy()

    top_row_a, top_row_b, top_row_c = st.columns(3)
    with top_row_a:
        metric_card("Visible clusters", str(len(filtered_clusters)), "after current filters")
    with top_row_b:
        metric_card("Cross-dataset mixed", str(int(filtered_clusters["cross_dataset_mixed"].sum())), "clusters currently visible")
    with top_row_c:
        metric_card("Selected cluster", selected_cluster, selected_row["dominant_harmonized_anchor"].replace("_", " "))

    if active_view == "Structure":
        st.subheader("Structure")
        st.caption("This view shows unsupervised EV cluster structure learned from embeddings. Biological state context and biochemical themes are layered on afterward and do not define the coordinates.")
        structure_fig = build_structure_map(filtered_centroids, selected_cluster)
        st.plotly_chart(structure_fig, use_container_width=True)

        lower_left, lower_right = st.columns([1.05, 1.15])
        with lower_left:
            render_cluster_explainer(
                selected_row,
                selected_theme_rows,
                selected_hit_rows,
                selected_dataset_rows,
                selected_neighbors,
                preset_titles,
            )
        with lower_right:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='eyebrow'>Optional Sampled Point View</div>", unsafe_allow_html=True)
            st.markdown(
                "This is secondary to the centroid map. Use it to see how sampled EV points sit inside the cluster structure."
            )
            sampled_fig = build_sampled_point_map(filtered_sampled, selected_cluster, sampled_color_by)
            st.plotly_chart(sampled_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    elif active_view == "Painted Meaning":
        st.subheader("Painted Meaning")
        st.caption("The coordinates stay fixed. Only the interpretation layer changes. This makes the separation between learned structure and assigned meaning explicit.")
        painted_fig = build_painted_map(
            filtered_centroids,
            selected_cluster,
            paint_mode=paint_mode,
            theme_colors=theme_colors,
            anchor_colors=anchor_colors,
        )
        st.plotly_chart(painted_fig, use_container_width=True)

        info_col, explain_col = st.columns([1, 1.15])
        with info_col:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='eyebrow'>Selected Cluster</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='big-metric'>{selected_cluster}</div>", unsafe_allow_html=True)
            st.markdown(
                strength_badge(str(selected_row["theme_support_strength"]))
                + ("<span class='badge badge-info'>Cross-dataset mixed</span>" if bool(selected_row["cross_dataset_mixed"]) else ""),
                unsafe_allow_html=True,
            )
            st.markdown(f"**State context**: {selected_row['dominant_harmonized_anchor'].replace('_', ' ')}")
            st.markdown(f"**Biochemical theme**: {selected_row['top_biochemical_theme'].replace('_', ' ')}")
            st.markdown(f"**Secondary theme**: {selected_row['secondary_biochemical_theme'].replace('_', ' ')}")
            st.markdown("</div>", unsafe_allow_html=True)
        with explain_col:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='eyebrow'>How To Read This Layer</div>", unsafe_allow_html=True)
            st.markdown(
                "- `EV state context` paints broad anchor labels such as control/baseline, disease/stress, or cell-line model.\n"
                "- `Biochemical theme` paints the top grounding-derived theme for each cluster.\n"
                "- `Cross-dataset mixing` shows which clusters bridge datasets within EV.\n"
                "- `Support strength` shows how concentrated the grounding evidence is."
            )
            st.markdown("</div>", unsafe_allow_html=True)

    elif active_view == "Cluster Explainer":
        st.subheader("Cluster Explainer")
        st.caption("This is the interpretation layer. It explains why the selected EV cluster received its state-context and biochemical assignment.")
        render_cluster_explainer(
            selected_row,
            selected_theme_rows,
            selected_hit_rows,
            selected_dataset_rows,
            selected_neighbors,
            preset_titles,
        )

    elif active_view == "Advanced Graph":
        st.subheader("Advanced Graph")
        st.caption("Advanced relationship inspection view. Use this after understanding the latent map. It shows relationships among clusters, anchors, themes, grounding references, and optional dataset nodes.")
        try:
            graph_html = build_pyvis_html(
                nodes,
                edges,
                visible_clusters=set(filtered_clusters["cluster_id"].tolist()),
                selected_cluster=selected_cluster,
                include_dataset_nodes=include_dataset_nodes,
                include_grounding_nodes=include_grounding_nodes,
                include_cluster_edges=include_cluster_edges,
                min_edge_weight=min_edge_weight,
                focus_selected=focus_selected,
            )
            components.html(graph_html, height=780, scrolling=False)
        except Exception as exc:
            st.warning(f"The advanced graph could not be rendered, but the rest of the explorer remains available: {exc}")

    else:
        st.subheader("Explainers")
        with st.expander("What is latent space?", expanded=True):
            st.write("Latent space is the learned embedding geometry. Clusters emerge from the embedding model before biological or biochemical meaning is painted onto them.")
        with st.expander("What is a cluster?"):
            st.write("A cluster is a neighborhood of EV spectra in latent space. Some remain dataset-pure; the most interesting ones mix across datasets while keeping a coherent broad-theme interpretation.")
        with st.expander("What is cross-dataset mixing?"):
            st.write("Cross-dataset mixing means more than one EV dataset contributes to a cluster without one dataset dominating it. That matters because it suggests shared latent biology rather than only dataset identity.")
        with st.expander("How grounding paints biochemical meaning"):
            st.write("Grounding is layered on after clustering. Each EV cluster retrieves broad grounding references in latent space, and those references are aggregated into broad biochemical themes.")
        with st.expander("What is a harmonized anchor?"):
            st.write("A harmonized anchor is a cautious state-context label such as control/baseline, disease/stress, or cell-line model. It is applied after clustering to summarize broad EV state context.")
        with st.expander("Why this is not diagnosis"):
            st.write("The app supports broad biochemical interpretation at cluster level. It does not assign molecule certainty and it does not diagnose individual samples.")

        thumb_cols = st.columns(3)
        thumb_paths = {
            "EV clusters": Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_grounding_analysis_v7/umap_ev_cluster_id.png"),
            "Biochemical theme": Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_grounding_analysis_v7/umap_ev_biochemical_theme.png"),
            "Cross-dataset mixed": Path("/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_grounding_analysis_v7/umap_ev_cross_dataset_mixed.png"),
        }
        for col, (label, path) in zip(thumb_cols, thumb_paths.items(), strict=False):
            with col:
                if path.exists():
                    st.image(str(path), caption=label, use_container_width=True)
                else:
                    st.info(f"Missing preview: {label}")


if __name__ == "__main__":
    main()
