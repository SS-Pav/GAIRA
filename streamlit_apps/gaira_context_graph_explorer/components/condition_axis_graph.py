"""Tab 2 — Condition family × BSV axis bipartite network."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components import ui_blocks as ui
from utils.graph_builders import build_condition_axis_edges
from utils.plotly_graph_utils import render_bipartite_condition_axis


def render(ctx: dict, app_cfg: dict) -> None:
    st.markdown("# Condition family · BSV axis network")
    st.caption("Bipartite layout: condition families on the left, BSV axes "
               "on the right. Edge weight = recurrence × |effect|; "
               "colour = dominant direction.")

    events = ctx.get("events")
    cf_axis = ctx.get("cf_axis")
    if events is None and cf_axis is None:
        ui.warning_card("No condition-axis evidence available.")
        return

    # ── controls ──
    cols = st.columns([1.5, 1.5, 1.5, 1])
    with cols[0]:
        weight_kind = st.selectbox(
            "Edge weight",
            options=["evidence (n × |effect|)", "datasets only",
                     "signed effect"],
            index=0, key="cge2_weight_kind")
    with cols[1]:
        normalize = st.checkbox("Normalise weights to [0,1]",
                                value=False, key="cge2_normalize")
    with cols[2]:
        all_cfs = (sorted(events["condition_family"].dropna().unique())
                   if events is not None
                   else sorted(cf_axis["condition_family"].dropna().unique()))
        cf_filter = st.multiselect(
            "Condition families (default = all)",
            options=all_cfs, default=all_cfs, key="cge2_cf_filter")
    with cols[3]:
        dir_filter = st.selectbox(
            "Direction filter",
            options=["all", "up", "down", "stable", "ambiguous"],
            index=0, key="cge2_dir_filter")

    weight_kind_internal = ("evidence" if weight_kind.startswith("evidence")
                             else "datasets" if weight_kind.startswith("datasets")
                             else "effect")

    edges = build_condition_axis_edges(events, cf_axis,
                                        weight_kind=weight_kind_internal,
                                        normalize=normalize)
    if edges.empty:
        ui.warning_card("No edges to render.")
        return

    edges = edges[edges["condition_family"].isin(cf_filter)]
    if dir_filter != "all":
        edges = edges[edges["direction"] == dir_filter]

    if edges.empty:
        ui.warning_card("No edges remain after filters.")
        return

    # Threshold slider on the actual filtered weights
    w_min = float(edges["weight"].min())
    w_max = float(edges["weight"].max())
    threshold = st.slider(
        "Edge weight threshold (only edges ≥ value shown)",
        min_value=float(0 if w_min < 1 else w_min),
        max_value=float(w_max),
        value=float(max(w_min, w_max * 0.05)),
        step=float(max(0.01, (w_max - w_min) / 50)),
        key="cge2_threshold")

    fig = render_bipartite_condition_axis(
        edges,
        bsv_palette=app_cfg.get("bsv_family_colors", {}),
        direction_palette=app_cfg.get("direction_colors", {}),
        weight_threshold=threshold,
        height=620,
        title=f"{len(edges[edges.weight>=threshold])} edges shown · "
              f"weight kind = {weight_kind}")
    st.plotly_chart(fig, use_container_width=True)

    ui.section_header("Top condition × axis links")
    view = edges[edges["weight"] >= threshold].head(30).copy()
    if not view.empty:
        view = view[["condition_family", "bsv_axis", "direction",
                     "n_datasets", "n_events", "mean_effect", "weight"]]
        st.dataframe(view, use_container_width=True, hide_index=True)

    ui.interpretation(
        "How to read this graph",
        "Each edge connects a <strong>condition family</strong> on the left "
        "to a <strong>BSV axis</strong> on the right. Thick edges mean the "
        "axis recurs across many datasets in that condition with non-trivial "
        "effect sizes. Red = dominant up · blue = dominant down · grey = "
        "stable / ambiguous. Liver-cancer typically shows a multi-axis "
        "biochemical program; toxicity and diabetes show more selective "
        "programs.")
