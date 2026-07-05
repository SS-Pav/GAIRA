"""Tab 4 — Hierarchical Sankey + Top-paths table."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components import ui_blocks as ui
from utils.graph_builders import build_hierarchical_sankey_v2
from utils.plotly_graph_utils import render_hierarchical_sankey


def render(ctx: dict, app_cfg: dict) -> None:
    st.markdown("# Hierarchical context flow")
    st.caption("Sankey: sample type → dataset → specific condition → "
                "BSV axis × direction → MSS candidate. Top emergent paths "
                "are tabulated below.")

    ev = ctx.get("events_v2")
    paths = ctx.get("emergent_paths")
    short = app_cfg.get("dataset_short_labels", {})

    if ev is None or ev.empty:
        ui.warning_card("Events table missing.")
        return

    cols = st.columns([1.2, 1.5, 1, 1, 1])
    with cols[0]:
        st_pick = st.multiselect(
            "Sample types",
            options=sorted(ev["sample_type"].dropna().unique()),
            default=sorted(ev["sample_type"].dropna().unique()),
            key="cge4_st")
    with cols[1]:
        cond_pool = sorted(ev[ev["sample_type"].isin(st_pick)]
                            ["specific_condition"].dropna().unique())
        cond_pick = st.multiselect(
            "Conditions (default = all)",
            options=cond_pool, default=cond_pool, key="cge4_cond")
    with cols[2]:
        max_edges = st.slider("Max edges per layer", 10, 200, 60, 10,
                                key="cge4_max")
    with cols[3]:
        show_mss = st.checkbox("Include MSS layer", value=True,
                                 key="cge4_mss")
    with cols[4]:
        only_top = st.checkbox("Show only top paths", value=False,
                                 key="cge4_only_top")

    if only_top and paths is not None and not paths.empty:
        top_conds = set(paths["specific_condition"].head(15).tolist())
        cond_pick = [c for c in cond_pick if c in top_conds]
        if not cond_pick:
            cond_pick = list(top_conds)

    sk = build_hierarchical_sankey_v2(
        ev, show_mss=show_mss, max_edges_per_layer=max_edges,
        filter_sample_types=st_pick, filter_conditions=cond_pick,
        dataset_short_labels=short)
    if not sk["node_label"]:
        ui.warning_card("No edges remain after filters.")
        return
    fig = render_hierarchical_sankey(
        sk, title=("sample_type → dataset → specific condition → "
                    "BSV axis × direction → MSS candidate"),
        height=720)
    st.plotly_chart(fig, use_container_width=True)

    # Top emergent paths table
    if paths is not None and not paths.empty:
        ui.section_header("Top emergent paths",
                           "sample · dataset · specific condition · axis · "
                           "direction · evidence · confidence")
        view = paths.copy()
        view["dataset"] = view["dataset"].map(
            lambda d: short.get(d, d[:24]))
        cols_show = ["sample_type", "dataset", "specific_condition",
                      "bsv_axis", "dom_direction", "n_events",
                      "mean_abs_effect", "consistency", "path_score",
                      "top_mss", "confidence_tier"]
        cols_show = [c for c in cols_show if c in view.columns]
        st.dataframe(view[cols_show].head(30),
                      use_container_width=True, hide_index=True)

    ui.interpretation(
        "How to read this",
        "A <strong>path</strong> represents evidence moving from biological "
        "context to biochemical axis to candidate motif. Edge thickness "
        "means recurrence (event count), <em>not</em> molecular abundance. "
        "Paths in the <strong>Top emergent paths</strong> table are ranked "
        "by <code>n_events × |mean effect| × direction-consistency</code>; "
        "STRONG paths (≥5) are the most defensible cross-cohort.")
