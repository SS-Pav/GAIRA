"""Tab 3 — Specific Condition Explorer.

For a chosen specific condition, show:
1. BSV radar / bar profile (mean signed effect by axis)
2. MSS candidate panel (top recurrent for the cohort)
3. Evidence graph (cohort → axes → MSS candidates) — implemented as
   sub-Sankey
4. Source dataset table
5. For hepatotoxicity: trajectory across D2_C0 → C40 (sourced live from
   the SHINE pilot's BSV-dose-response table)
6. For small EV: probe-axis comparison
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from components import ui_blocks as ui
from utils.load_context_data import load_csv_safe
from utils.graph_builders import build_hierarchical_sankey_v2
from utils.plotly_graph_utils import render_hierarchical_sankey


BSV_AXES = [f"G{i:02d}" for i in range(1, 12)]


def _radar_or_bar(events_for_cohort: pd.DataFrame, axis_names: dict[str, str],
                   palette: dict[str, str], view: str) -> go.Figure:
    rows = []
    for ax in BSV_AXES:
        sub = events_for_cohort[events_for_cohort["bsv_axis"] == ax]
        try:
            effs = pd.to_numeric(sub["effect_size"], errors="coerce").dropna()
            me = float(effs.mean()) if len(effs) else 0.0
        except Exception:
            me = 0.0
        n = len(sub)
        rows.append({"axis": ax, "name": axis_names.get(ax, ax),
                      "mean_effect": me, "n_events": n})
    df = pd.DataFrame(rows)
    fig = go.Figure()
    if view == "Bar":
        colors = [palette.get(ax, "#79c0ff") for ax in df["axis"]]
        fig.add_trace(go.Bar(x=df["axis"], y=df["mean_effect"],
                              marker_color=colors,
                              hovertemplate=("<b>%{x}</b><br>"
                                              "mean signed effect: %{y:.3f}"
                                              "<extra></extra>")))
        fig.update_layout(template="plotly_dark", height=380,
                           plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                           xaxis=dict(title="BSV axis", gridcolor="#21262d"),
                           yaxis=dict(title="mean signed effect",
                                       gridcolor="#21262d", zerolinecolor="#444"),
                           margin=dict(l=10, r=10, t=44, b=10))
    else:  # Radar
        # Pad/wrap for radar
        thetas = df["axis"].tolist() + [df["axis"].iloc[0]]
        rs = df["mean_effect"].tolist() + [df["mean_effect"].iloc[0]]
        fig.add_trace(go.Scatterpolar(
            r=rs, theta=thetas, mode="lines+markers",
            line=dict(color="#79c0ff", width=2),
            marker=dict(size=8, color="#79c0ff"),
            fill="toself", fillcolor="rgba(121,192,255,0.20)"))
        fig.update_layout(template="plotly_dark", height=420,
                           plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                           polar=dict(bgcolor="#0d1117",
                                       angularaxis=dict(tickfont=dict(color="#c9d1d9", size=11)),
                                       radialaxis=dict(gridcolor="#30363d",
                                                        tickfont=dict(color="#8b949e"))),
                           margin=dict(l=20, r=20, t=44, b=20))
    return fig, df


def _mss_panel(events_for_cohort: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    e = events_for_cohort.dropna(subset=["mss_candidate"])
    e = e[e["mss_candidate"].astype(str) != ""]
    if e.empty:
        return pd.DataFrame()
    rows = []
    for m, sub in e.groupby("mss_candidate"):
        try:
            effs = pd.to_numeric(sub["effect_size"], errors="coerce").dropna()
            me = float(effs.abs().mean()) if len(effs) else 0.0
        except Exception:
            me = 0.0
        rows.append({
            "mss_candidate": m,
            "n_events": len(sub),
            "n_datasets": sub["dataset"].nunique(),
            "dom_direction": (sub["direction"].value_counts().idxmax()
                                if len(sub) else "stable"),
            "mean_abs_effect": round(me, 3),
        })
    return (pd.DataFrame(rows).sort_values("n_events", ascending=False)
             .head(top_n).reset_index(drop=True))


def _shine_trajectory(build_root: str) -> pd.DataFrame | None:
    p = (Path(build_root) /
         "gaira_base_4_shine_ev_gaira_pilot_v1" /
         "tables" / "shine_cohort_bsv_means_v1.csv")
    df = load_csv_safe(p)
    if df is None or df.empty:
        return None
    # Long form already: set_id, day, dose_mM, axis, mean_clr, ...
    return df


def _smallev_axis_compare(build_root: str) -> pd.DataFrame | None:
    p = (Path(build_root) /
         "gaira_base_4_small_ev_shared_structure_pass_v2" /
         "tables" / "axis_rank_comparison_v2.csv")
    return load_csv_safe(p)


def render(ctx: dict, app_cfg: dict) -> None:
    st.markdown("# Specific Condition Explorer")
    st.caption("Pick a condition and see its BSV / MSS profile, evidence "
                "graph, source datasets, and (where available) trajectory.")

    ev = ctx.get("events_v2")
    if ev is None or ev.empty or "specific_condition" not in ev.columns:
        ui.warning_card("Events table missing.")
        return

    palette = app_cfg.get("bsv_family_colors", {})
    axis_names = app_cfg.get("bsv_family_names", {})
    short = app_cfg.get("dataset_short_labels", {})

    # ── controls ──
    cols = st.columns([1, 2, 2, 1])
    with cols[0]:
        sample_pick = st.radio("Sample", options=["all", "EV", "serum",
                                                     "mixed"],
                                 index=0, horizontal=True, key="cge3_sample")
    with cols[1]:
        ev_filtered = ev if sample_pick == "all" else ev[ev["sample_type"] == sample_pick]
        cond_options = sorted(ev_filtered["specific_condition"].dropna().unique())
        cond_pick = st.selectbox("Condition", options=cond_options,
                                   index=0 if cond_options else None,
                                   key="cge3_cond")
    with cols[2]:
        ref_pick = st.selectbox("Compare to (optional)",
                                  options=["(none)"] + cond_options,
                                  index=0, key="cge3_ref")
    with cols[3]:
        view = st.radio("Profile", options=["Bar", "Radar"], index=0,
                          horizontal=True, key="cge3_view")

    if not cond_pick:
        ui.warning_card("No conditions available.")
        return

    sub = ev[ev["specific_condition"] == cond_pick]
    if sub.empty:
        ui.warning_card(f"No events for {cond_pick}.")
        return

    # ── 1: BSV profile ──
    ui.section_header(f"1 · BSV profile · {cond_pick}",
                       f"mean signed effect across G01–G11 · "
                       f"{len(sub)} events · {sub['dataset'].nunique()} datasets")
    fig, prof_df = _radar_or_bar(sub, axis_names, palette, view)
    st.plotly_chart(fig, use_container_width=True)

    # Optional reference overlay (Bar only — radar would overcrowd)
    if ref_pick and ref_pick != "(none)" and view == "Bar":
        ref_sub = ev[ev["specific_condition"] == ref_pick]
        if not ref_sub.empty:
            _, ref_df = _radar_or_bar(ref_sub, axis_names, palette, "Bar")
            cmp = prof_df.merge(ref_df, on="axis", suffixes=("", "_ref"))
            cmp_fig = go.Figure()
            cmp_fig.add_trace(go.Bar(x=cmp["axis"], y=cmp["mean_effect"],
                                       marker_color="#79c0ff",
                                       name=cond_pick))
            cmp_fig.add_trace(go.Bar(x=cmp["axis"], y=cmp["mean_effect_ref"],
                                       marker_color="#ff7b72",
                                       name=ref_pick))
            cmp_fig.update_layout(barmode="group", template="plotly_dark",
                                   height=380, plot_bgcolor="#0d1117",
                                   paper_bgcolor="#0d1117",
                                   margin=dict(l=10, r=10, t=44, b=10),
                                   title=dict(
                                       text=f"{cond_pick}  vs  {ref_pick}",
                                       font=dict(size=12, color="#c9d1d9")),
                                   xaxis=dict(title="BSV axis"),
                                   yaxis=dict(title="mean signed effect"))
            st.plotly_chart(cmp_fig, use_container_width=True)

    # ── 2: MSS candidate panel ──
    ui.section_header(f"2 · Top recurrent MSS candidates · {cond_pick}")
    mss_df = _mss_panel(sub)
    if mss_df.empty:
        st.caption("(no MSS candidate evidence for this cohort)")
    else:
        st.dataframe(mss_df, use_container_width=True, hide_index=True)
        st.caption("MSS hits are candidate-level evidence, not molecular IDs in "
                    "complex biofluids.")

    # ── 3: Evidence sub-graph (cohort → axes → MSS) ──
    ui.section_header(f"3 · Evidence sub-graph · {cond_pick}")
    sk = build_hierarchical_sankey_v2(
        sub, show_mss=True, max_edges_per_layer=40,
        dataset_short_labels=short)
    if sk["node_label"]:
        sk_fig = render_hierarchical_sankey(
            sk, title=("sample_type → dataset → condition → axis · dir → MSS"),
            height=520)
        st.plotly_chart(sk_fig, use_container_width=True)

    # ── 4: Source dataset table ──
    ui.section_header(f"4 · Source datasets · {cond_pick}")
    rows = []
    for ds, dsub in sub.groupby("dataset"):
        rows.append({
            "dataset (short)": short.get(ds, ds[:36]),
            "sample_type": dsub["sample_type"].iloc[0],
            "n_events": len(dsub),
            "top axes": ", ".join(dsub["bsv_axis"].value_counts().head(3).index),
            "top MSS": ", ".join(dsub["mss_candidate"].dropna()
                                  .value_counts().head(3).index.astype(str)),
            "comparison_types": ", ".join(
                dsub["comparison_type"].dropna().unique()[:3]),
            "full dataset": ds,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── 5: Hepatotoxicity trajectory (if applicable) ──
    ui.divider()
    if cond_pick.startswith("hepatotoxicity"):
        ui.section_header("5 · Hepatotoxicity dose trajectory · Day-2 highlighted",
                           "Live read of shine_cohort_bsv_means_v1.csv")
        traj = _shine_trajectory(app_cfg["paths"]["build_root"])
        if traj is None or traj.empty:
            st.caption("(SHINE cohort BSV means table not found)")
        else:
            # Plot mean_clr vs dose, line per axis, panel per (set_id, day)
            axis_pick = st.multiselect(
                "Axes to overlay", options=BSV_AXES,
                default=["G01", "G05", "G08", "G09", "G11"],
                key="cge3_traj_axes")
            day_pick = st.radio(
                "Day", options=sorted(traj["day"].dropna().unique()),
                index=(2 if "D2" in traj["day"].unique() else 0),
                horizontal=True, key="cge3_traj_day")
            sub_traj = traj[(traj["axis"].isin(axis_pick))
                             & (traj["day"] == day_pick)]
            if sub_traj.empty:
                st.caption("(no trajectory rows for this filter)")
            else:
                fig = go.Figure()
                for ax_id, line in sub_traj.groupby("axis"):
                    line = line.sort_values("dose_mM")
                    color = palette.get(ax_id, "#79c0ff")
                    for set_id, set_line in line.groupby("set_id"):
                        fig.add_trace(go.Scatter(
                            x=set_line["dose_mM"], y=set_line["mean_clr"],
                            mode="lines+markers",
                            line=dict(color=color, width=2),
                            marker=dict(size=8),
                            name=f"{ax_id} · {set_id}",
                            hovertemplate=(f"<b>{ax_id}</b><br>"
                                            f"set {set_id} · dose %{{x}}mM<br>"
                                            f"mean_CLR=%{{y:.3f}}<extra></extra>")))
                fig.update_layout(template="plotly_dark", height=400,
                                   plot_bgcolor="#0d1117",
                                   paper_bgcolor="#0d1117",
                                   margin=dict(l=10, r=10, t=44, b=10),
                                   title=dict(
                                       text=f"Dose response · {day_pick}",
                                       font=dict(size=12, color="#c9d1d9")),
                                   xaxis=dict(title="APAP dose (mM)",
                                               gridcolor="#21262d"),
                                   yaxis=dict(title="mean CLR(BSV)",
                                               gridcolor="#21262d"))
                st.plotly_chart(fig, use_container_width=True)

    # ── 6: Small EV probe-axis compare (if applicable) ──
    if cond_pick.startswith("smallEV"):
        ui.section_header("6 · Small-EV axis-rank comparison",
                           "Probe1 vs Probe2 ranked-axis table")
        cmp = _smallev_axis_compare(app_cfg["paths"]["build_root"])
        if cmp is None or cmp.empty:
            st.caption("(small-EV axis rank comparison table not found)")
        else:
            st.dataframe(cmp, use_container_width=True, hide_index=True)

    ui.interpretation(
        "Reading this view",
        f"<strong>{cond_pick}</strong> aggregates {len(sub)} evidence events "
        f"from {sub['dataset'].nunique()} datasets. The BSV profile shows "
        "mean signed effect per axis; the MSS panel lists candidates that "
        "recur within this cohort. Cross-cohort comparisons (e.g. "
        "<em>HCC vs LM</em> or <em>OWD vs NWD</em>) become apparent by "
        "switching the <strong>Compare to</strong> dropdown.")
