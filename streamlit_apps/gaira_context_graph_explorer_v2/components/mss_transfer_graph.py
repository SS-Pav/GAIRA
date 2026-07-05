"""Tab 5 — MSS transfer with EV / serum / all subtabs."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components import ui_blocks as ui
from utils.graph_builders import build_mss_transfer_edges_v2
from utils.plotly_graph_utils import render_bipartite_mss


def _render_one_view(ctx: dict, app_cfg: dict, sample_filter: list[str],
                      header: str) -> None:
    ev = ctx.get("events_v2")
    msst = ctx.get("mss_transfer")

    cols = st.columns([1.4, 1.4, 1, 1])
    with cols[0]:
        view_target = st.radio("View target",
                                 options=["specific_condition", "dataset",
                                           "sample_type"],
                                 index=0, horizontal=True,
                                 key=f"cge5_target_{header}")
    with cols[1]:
        avail_classes = (sorted(msst["classification"].dropna().unique())
                          if msst is not None and not msst.empty
                          else ["TRANSFERABLE"])
        cls_filter = st.multiselect("Classification",
                                       options=avail_classes,
                                       default=avail_classes,
                                       key=f"cge5_cls_{header}")
    with cols[2]:
        top_n = st.slider("Top N MSS", 5, 50, 25, 5,
                            key=f"cge5_topn_{header}")
    with cols[3]:
        height = st.slider("Height", 400, 900, 660, 20,
                            key=f"cge5_h_{header}")

    edges, nodes = build_mss_transfer_edges_v2(
        ev, group_by=view_target,
        sample_filter=sample_filter,
        classification_filter=cls_filter,
        mss_classes=msst, top_n=top_n)
    if edges.empty:
        ui.warning_card("No MSS edges remain after filters.")
        return

    fig = render_bipartite_mss(
        edges, nodes, group_by=view_target,
        direction_palette=app_cfg.get("direction_colors", {}),
        height=height,
        title=f"MSS ↔ {view_target} · sample={','.join(sample_filter) or 'all'}"
              f" · top {top_n}")
    st.plotly_chart(fig, use_container_width=True)


def render(ctx: dict, app_cfg: dict) -> None:
    st.markdown("# MSS transfer · candidate layer")
    st.caption("Why MSS transfer currently appears mostly EV — and what "
                "happens when we look at serum on its own.")

    sub_all, sub_ev, sub_serum = st.tabs(["All sample types", "EV-only",
                                             "Serum-only"])
    with sub_all:
        _render_one_view(ctx, app_cfg, sample_filter=[], header="all")
    with sub_ev:
        _render_one_view(ctx, app_cfg, sample_filter=["EV"], header="ev")
    with sub_serum:
        _render_one_view(ctx, app_cfg, sample_filter=["serum"], header="serum")

    # MSS classification table
    msst = ctx.get("mss_transfer")
    if msst is not None and not msst.empty:
        ui.section_header("MSS classification table",
                           "candidate · class · datasets · sample types · "
                           "direction consistency")
        st.dataframe(msst, use_container_width=True, hide_index=True)

    ui.interpretation(
        "Why does MSS transfer show EV more than serum?",
        "<ul style='margin: 4px 0 4px 18px;'>"
        "<li>EV pilots reuse the same canonical analyte panel (UA, lactate, "
        "ergothioneine, palmitic acid, ...), so cross-pilot recurrence is "
        "naturally higher.</li>"
        "<li>Serum pilots span heterogeneous SERS substrates with different "
        "MSS specificity — many serum candidates land in "
        "<em>SAMPLE_TYPE_SPECIFIC</em> or <em>CANDIDATE_ONLY</em> "
        "(substrate-locked) classes.</li>"
        "<li>If the serum graph is sparse, this indicates MSS-level "
        "<strong>molecular specificity</strong> is less transferable in "
        "complex serum SERS, while <strong>BSV-level themes</strong> may "
        "remain robust (Tab 6 quantifies this).</li>"
        "</ul>")
