"""Tab 4 — MSS candidate transfer across pilots."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components import ui_blocks as ui
from utils.graph_builders import build_mss_transfer_edges
from utils.plotly_graph_utils import render_bipartite_mss


def render(ctx: dict, app_cfg: dict) -> None:
    st.markdown("# MSS transfer graph")
    st.caption("Which MSS candidates recur across pilots, and where. "
               "Bipartite layout — MSS on the left, chosen grouping on the right.")

    events = ctx.get("events")
    msst = ctx.get("mss_transfer")
    if events is None or events.empty:
        ui.warning_card("Evidence-events table missing.")
        return

    cols = st.columns([1.2, 1.5, 1.2, 1])
    with cols[0]:
        group_by = st.radio(
            "Group MSS by",
            options=["dataset", "condition_family", "sample_type"],
            index=0, horizontal=True, key="cge4_group_by")
    with cols[1]:
        avail_classes = (sorted(msst["classification"].dropna().unique())
                          if msst is not None and not msst.empty
                          else ["TRANSFERABLE"])
        cls_filter = st.multiselect(
            "Classification filter",
            options=avail_classes, default=avail_classes,
            key="cge4_cls_filter")
    with cols[2]:
        top_n = st.slider("Top N MSS candidates", 5, 50, 25, 5,
                          key="cge4_top_n")
    with cols[3]:
        height = st.slider("Figure height (px)", 400, 900, 660, 20,
                            key="cge4_height")

    edges, nodes = build_mss_transfer_edges(
        events, group_by=group_by, classification_filter=cls_filter,
        top_n=top_n, mss_classes=msst)

    if edges.empty:
        ui.warning_card("No MSS edges remain after filters.")
        return

    fig = render_bipartite_mss(
        edges, nodes, group_by=group_by,
        direction_palette=app_cfg.get("direction_colors", {}),
        height=height,
        title=f"MSS candidate ↔ {group_by} · top {top_n} by event count")
    st.plotly_chart(fig, use_container_width=True)

    # Required side table
    if msst is not None and not msst.empty:
        ui.section_header("MSS classification table",
                          "candidate · class · datasets · sample types · "
                          "direction consistency")
        view = msst[msst["classification"].isin(cls_filter)].copy()
        view = view[["mss_candidate", "classification", "n_datasets",
                     "n_sample_types", "n_events", "dom_direction",
                     "direction_consistency", "datasets"]]
        st.dataframe(view, use_container_width=True, hide_index=True)

    ui.interpretation(
        "Important framing",
        "These are <strong>recurrent MSS motif candidates</strong>, not "
        "definitive molecule calls in complex biofluids. Cross-pilot "
        "transfer here means the analyte's MSS evidence vector recurs "
        "across multiple pilots with consistent direction; molecular "
        "identity in serum / EV samples still requires substrate-aware "
        "validation.")
