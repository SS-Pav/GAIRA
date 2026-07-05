"""Tab 6 — EV vs Serum comparison · heatmaps + paired plot + contrast table."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from components import ui_blocks as ui
from utils.plotly_graph_utils import render_heatmap


BSV_AXES = [f"G{i:02d}" for i in range(1, 12)]


def _pivot_st_axis(summary: pd.DataFrame, value_col: str,
                    aggfunc: str = "first") -> pd.DataFrame:
    if summary is None or summary.empty:
        return pd.DataFrame()
    return (summary.pivot_table(index="sample_type", columns="bsv_axis",
                                  values=value_col, aggfunc=aggfunc)
              .reindex(columns=BSV_AXES))


def render(ctx: dict, app_cfg: dict) -> None:
    st.markdown("# EV vs Serum comparison")
    st.caption("How the EV and serum context spaces differ at axis and MSS "
                "level. BSV themes may transfer better than MSS in serum.")

    st_axis = ctx.get("st_axis_summary")
    st_mss = ctx.get("st_mss_summary")
    ev = ctx.get("events_v2")
    axis_names = app_cfg.get("bsv_family_names", {})

    if st_axis is None or st_axis.empty:
        ui.warning_card("Sample-type axis summary missing.")
        return

    # Panel 1 — recurrence heatmap
    rec = _pivot_st_axis(st_axis, "n_datasets")
    if not rec.empty:
        ui.section_header("1 · Axis recurrence heatmap",
                           "n_datasets per (sample_type × axis)")
        fig = render_heatmap(
            rec.fillna(0).values,
            x_labels=[f"{c}<br>{axis_names.get(c,'')[:14]}"
                       for c in rec.columns],
            y_labels=rec.index.tolist(),
            colorscale="Blues",
            title="Number of datasets per axis (richer = denser blue)",
            height=320)
        st.plotly_chart(fig, use_container_width=True)

    # Panel 2 — direction consistency
    cons = _pivot_st_axis(st_axis, "direction_consistency")
    if not cons.empty:
        ui.section_header("2 · Direction-consistency heatmap",
                           "fraction of events agreeing on dominant direction")
        fig = render_heatmap(
            cons.fillna(0).values,
            x_labels=[f"{c}<br>{axis_names.get(c,'')[:14]}"
                       for c in cons.columns],
            y_labels=cons.index.tolist(),
            colorscale="Greens", zmin=0, zmax=1,
            title="Direction agreement (1.0 = unanimous)",
            height=320)
        st.plotly_chart(fig, use_container_width=True)

    # Panel 3 — MSS recurrence heatmap (top candidates × sample types)
    if st_mss is not None and not st_mss.empty:
        ui.section_header("3 · MSS recurrence heatmap",
                           "Top recurrent MSS candidates × sample type")
        pivot = (st_mss.pivot_table(index="mss_candidate",
                                       columns="sample_type",
                                       values="n_events", aggfunc="sum")
                   .fillna(0))
        # Order candidates by total events
        pivot = pivot.reindex(pivot.sum(axis=1).sort_values(
            ascending=False).index)
        fig = render_heatmap(
            pivot.values, x_labels=pivot.columns.tolist(),
            y_labels=pivot.index.tolist(),
            colorscale="Oranges",
            title="MSS event count (rows = candidates, cols = sample types)",
            height=max(300, 18 * len(pivot)),
            show_text=False)
        st.plotly_chart(fig, use_container_width=True)

    # Panel 4 — contrast table
    ui.section_header("4 · EV vs serum contrast (per axis)",
                       "axis · EV recurrence · serum recurrence · "
                       "EV direction · serum direction · context dependence")
    rows = []
    for ax in BSV_AXES:
        ev_row = st_axis[(st_axis.bsv_axis == ax) &
                          (st_axis.sample_type == "EV")]
        ser_row = st_axis[(st_axis.bsv_axis == ax) &
                           (st_axis.sample_type == "serum")]
        ev_n = int(ev_row["n_datasets"].iloc[0]) if not ev_row.empty else 0
        ser_n = int(ser_row["n_datasets"].iloc[0]) if not ser_row.empty else 0
        ev_d = (ev_row["dom_direction"].iloc[0]
                 if not ev_row.empty else "")
        ser_d = (ser_row["dom_direction"].iloc[0]
                  if not ser_row.empty else "")
        ev_e = (float(ev_row["mean_effect"].iloc[0])
                 if not ev_row.empty else 0.0)
        ser_e = (float(ser_row["mean_effect"].iloc[0])
                  if not ser_row.empty else 0.0)
        same_dir = "yes" if ev_d == ser_d and ev_d else "no"
        rows.append({
            "axis": ax, "name": axis_names.get(ax, ""),
            "EV n_datasets": ev_n, "serum n_datasets": ser_n,
            "EV direction": ev_d, "serum direction": ser_d,
            "EV mean effect": round(ev_e, 3),
            "serum mean effect": round(ser_e, 3),
            "directions agree": same_dir,
        })
    contrast = pd.DataFrame(rows)
    st.dataframe(contrast, use_container_width=True, hide_index=True)

    # Panel 5 — paired EV vs serum scatter (mean effect)
    ui.section_header("5 · Paired EV vs serum effect — points are BSV axes",
                       "Quadrants → directional agreement / disagreement")
    palette = app_cfg.get("bsv_family_colors", {})
    fig = go.Figure()
    for _, r in contrast.iterrows():
        ax = r["axis"]
        color = palette.get(ax, "#79c0ff")
        fig.add_trace(go.Scatter(
            x=[r["EV mean effect"]], y=[r["serum mean effect"]],
            mode="markers+text", text=[ax],
            textposition="top center",
            textfont=dict(color=color, size=11),
            marker=dict(size=14, color=color,
                          line=dict(color="#0d1117", width=0.5)),
            hovertemplate=(f"<b>{ax} · {r['name']}</b><br>"
                            f"EV mean effect: {r['EV mean effect']:.3f}<br>"
                            f"serum mean effect: {r['serum mean effect']:.3f}<br>"
                            f"directions agree: {r['directions agree']}"
                            "<extra></extra>"),
            showlegend=False))
    # quadrant guides
    fig.add_hline(y=0, line=dict(color="#444", width=0.5, dash="dot"))
    fig.add_vline(x=0, line=dict(color="#444", width=0.5, dash="dot"))
    fig.update_layout(template="plotly_dark", height=480,
                       plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                       margin=dict(l=10, r=10, t=44, b=10),
                       title=dict(text="EV mean effect vs serum mean effect",
                                    font=dict(size=12, color="#c9d1d9")),
                       xaxis=dict(title="EV mean effect", gridcolor="#21262d"),
                       yaxis=dict(title="serum mean effect", gridcolor="#21262d"))
    st.plotly_chart(fig, use_container_width=True)

    ui.interpretation(
        "Reading the comparison",
        "<ul style='margin: 4px 0 4px 18px;'>"
        "<li><strong>Recurrence heatmap</strong> shows where each axis is "
        "<em>active</em>. Wider blue = the axis fires across more datasets "
        "of that sample type.</li>"
        "<li><strong>Direction-consistency heatmap</strong> shows whether "
        "those activations agree on a direction. Low values mean substrate / "
        "cohort heterogeneity is washing the signal out.</li>"
        "<li><strong>Paired EV-vs-serum scatter</strong>: axes that sit "
        "in the upper-right or lower-left quadrants <em>transfer</em> "
        "directionally; opposite quadrants flag context-dependent biology.</li>"
        "<li>EV tends to be more selective (a few axes dominate); "
        "serum is broader and more substrate-caveated.</li>"
        "</ul>")
