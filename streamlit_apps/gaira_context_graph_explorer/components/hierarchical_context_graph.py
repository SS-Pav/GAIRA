"""Tab 3 — Hierarchical Sankey: sample_type → dataset → condition_family →
BSV axis → MSS candidate."""
from __future__ import annotations

import streamlit as st

from components import ui_blocks as ui
from utils.graph_builders import build_hierarchical_sankey
from utils.plotly_graph_utils import render_hierarchical_sankey


def render(ctx: dict, app_cfg: dict) -> None:
    st.markdown("# Hierarchical context graph")
    st.caption("Sankey: sample type → dataset → condition family → BSV axis → "
               "MSS candidate. The full GAIRA reasoning hierarchy on one canvas.")

    events = ctx.get("events")
    if events is None or events.empty:
        ui.warning_card("Evidence-events table missing.")
        return

    cols = st.columns([1.5, 1.5, 1, 1])
    with cols[0]:
        all_st = sorted(events["sample_type"].dropna().unique())
        st_filter = st.multiselect(
            "Sample types", options=all_st, default=all_st,
            key="cge3_st_filter")
    with cols[1]:
        # Datasets visible after sample-type filter
        ds_pool = events[events["sample_type"].isin(st_filter)]["dataset"].dropna().unique()
        ds_filter = st.multiselect(
            "Datasets (default = all)",
            options=sorted(ds_pool), default=sorted(ds_pool),
            key="cge3_ds_filter")
    with cols[2]:
        max_edges = st.slider("Max edges per layer", 10, 200, 60, 10,
                              key="cge3_max_edges")
    with cols[3]:
        show_mss = st.checkbox("Include MSS layer", value=True,
                                key="cge3_show_mss")

    sk = build_hierarchical_sankey(events,
                                    show_mss=show_mss,
                                    max_edges_per_layer=max_edges,
                                    filter_sample_types=st_filter,
                                    filter_datasets=ds_filter)
    if not sk["node_label"]:
        ui.warning_card("No edges remain after filters.")
        return
    fig = render_hierarchical_sankey(
        sk, title=("sample_type → dataset → condition_family → "
                   "BSV axis → MSS candidate"),
        height=720)
    st.plotly_chart(fig, use_container_width=True)

    ui.interpretation(
        "Reading the Sankey",
        "Width is evidence count. Edge colour reflects dominant direction "
        "(red = up, blue = down, grey = stable). Toggle the MSS layer off "
        "for a cleaner sample → axis flow. Datasets that bridge multiple "
        "condition families (e.g. cross-pilot synthesis) appear with "
        "fan-outs into several columns.")

    ui.section_header("Layer summary",
                      "How many distinct nodes appear in each layer of the Sankey.")
    layer_names = ["sample_type", "dataset", "condition_family",
                   "BSV axis × direction", "MSS candidate"]
    counts = [sk["node_layer"].count(i) for i in range(5)]
    cols = st.columns(5)
    for col, name, c in zip(cols, layer_names, counts):
        with col:
            ui.card(name, f"<div class='cge-metric-value'>{c}</div>")
