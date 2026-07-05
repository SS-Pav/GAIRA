"""Tab 2 — Condition → Axis programs with broad / specific toggle."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components import ui_blocks as ui
from utils.plotly_graph_utils import render_bipartite_condition_axis


def render(ctx: dict, app_cfg: dict) -> None:
    st.markdown("# Condition → Axis programs")
    st.caption("Bipartite layout · condition labels on the left · BSV axes on the right.")

    cf_specific = ctx.get("cf_axis_specific")
    cf_broad = ctx.get("cf_axis_broad")
    if (cf_specific is None or cf_specific.empty) and (cf_broad is None or cf_broad.empty):
        ui.warning_card("Condition × axis edge tables missing.")
        return

    # ── controls ──
    cols = st.columns([1.3, 1.3, 1.3, 1])
    with cols[0]:
        granularity = st.radio("Condition granularity",
                                options=["specific", "broad family"], index=0,
                                horizontal=True, key="cge2_gran")
    with cols[1]:
        sample_type_pick = st.radio("Sample type",
                                      options=["all", "EV", "serum", "mixed"],
                                      index=0, horizontal=True,
                                      key="cge2_st")
    with cols[2]:
        dir_filter = st.selectbox("Direction filter",
                                    options=["all", "up", "down", "stable",
                                              "mixed", "ambiguous"],
                                    index=0, key="cge2_dir")
    with cols[3]:
        top_n = st.slider("Top N edges", 5, 100, 40, 5, key="cge2_top_n")

    if granularity == "specific":
        edges = cf_specific.copy() if cf_specific is not None else pd.DataFrame()
        left_col = "specific_condition"
    else:
        edges = cf_broad.copy() if cf_broad is not None else pd.DataFrame()
        left_col = "condition_family"

    if edges.empty:
        ui.warning_card("Edges unavailable for this granularity.")
        return

    # Sample-type filter: derive via events_v2
    if sample_type_pick != "all":
        ev = ctx.get("events_v2")
        if ev is not None and not ev.empty:
            valid_conds = (ev[ev["sample_type"] == sample_type_pick]
                            [left_col].dropna().unique().tolist())
            edges = edges[edges[left_col].isin(valid_conds)]

    if dir_filter != "all":
        edges = edges[edges["dom_direction"] == dir_filter]

    if edges.empty:
        ui.warning_card("No edges remain after filters.")
        return

    edges = edges.sort_values("weight", ascending=False).head(top_n)

    # threshold slider
    w_min = float(edges["weight"].min())
    w_max = float(edges["weight"].max())
    threshold = st.slider("Edge weight threshold (only edges ≥ value shown)",
                            min_value=float(0),
                            max_value=float(max(w_max, 1.0)),
                            value=float(max(w_min, w_max * 0.05)),
                            step=float(max(0.01, (w_max - w_min) / 50)),
                            key="cge2_threshold")

    fig = render_bipartite_condition_axis(
        edges, left_col=left_col,
        bsv_palette=app_cfg.get("bsv_family_colors", {}),
        direction_palette=app_cfg.get("direction_colors", {}),
        weight_threshold=threshold,
        height=720,
        title=(f"{len(edges[edges.weight>=threshold])} edges shown · "
                f"{granularity} conditions · sample={sample_type_pick} · "
                f"dir={dir_filter}"))
    st.plotly_chart(fig, use_container_width=True)

    ui.section_header("Top condition × axis links")
    view = edges[edges["weight"] >= threshold].head(40).copy()
    if not view.empty:
        cols_show = [left_col, "bsv_axis", "dom_direction",
                      "n_datasets", "n_sample_types", "n_events",
                      "mean_effect", "mean_abs_effect", "weight",
                      "datasets"]
        cols_show = [c for c in cols_show if c in view.columns]
        st.dataframe(view[cols_show], use_container_width=True, hide_index=True)

    # Auto-generated interpretation bullets
    bullets = []
    if not view.empty:
        # Group top by condition and surface the dominant pattern
        for cond, sub in view.groupby(left_col):
            axs = sub.head(3)
            ax_str = " · ".join(
                f"{r['bsv_axis']}({r['dom_direction']})"
                for _, r in axs.iterrows())
            bullets.append(f"<li><strong>{cond}</strong> — strongest "
                            f"axes: {ax_str}</li>")
        bullets = bullets[:8]
    if bullets:
        ui.interpretation(
            "Auto-derived condition programs",
            "<ul style='margin: 4px 0 4px 18px;'>" + "".join(bullets) + "</ul>")

    ui.interpretation(
        "How to read this graph",
        "Edges connect a <strong>condition label</strong> on the left to "
        "a <strong>BSV axis</strong> on the right. Edge thickness ≈ "
        "recurrence × |effect|. Colour = dominant direction (red up · "
        "blue down · grey stable). Switch between <em>specific</em> and "
        "<em>broad family</em> granularity to zoom in on cohorts vs "
        "macro-themes.")
