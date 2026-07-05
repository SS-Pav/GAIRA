"""Tab 1 — Overview with specific-condition coverage cards."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components import ui_blocks as ui
from utils.load_context_data import load_text_safe, report_path


def render(ctx: dict, app_cfg: dict) -> None:
    st.markdown(f"# {app_cfg['app']['title']}")
    st.caption(app_cfg["app"]["subtitle"])

    ev = ctx.get("events_v2")
    axt = ctx.get("axis_transfer")
    msst = ctx.get("mss_transfer")
    caveats = ctx.get("caveats")

    # Headline cards
    n_events = len(ev) if ev is not None else 0
    n_datasets = ev["dataset"].nunique() if ev is not None else 0
    n_st = ev["sample_type"].nunique() if ev is not None else 0
    n_specific = (ev["specific_condition"].nunique()
                   if ev is not None and "specific_condition" in ev.columns
                   else 0)
    n_strong_axes = (axt[axt["axis_transfer_score"] >= 1.0].shape[0]
                      if axt is not None and not axt.empty else 0)
    n_transferable = (msst[msst["classification"] == "TRANSFERABLE"].shape[0]
                      if msst is not None and not msst.empty else 0)
    n_caveats = (caveats["caveat_id"].nunique()
                  if caveats is not None and not caveats.empty else 0)
    n_caveat_ds = (caveats["dataset"].nunique()
                    if caveats is not None and not caveats.empty else 0)

    serum_conds = (ev[ev["sample_type"] == "serum"]["specific_condition"].nunique()
                    if ev is not None else 0)
    ev_conds = (ev[ev["sample_type"] == "EV"]["specific_condition"].nunique()
                 if ev is not None else 0)

    ui.divider()
    ui.section_header("Headline numbers",
                       "v2 separates broad condition families from specific cohort labels.")
    ui.render_metric_cards({
        "evidence events": f"{n_events:,}",
        "datasets": str(n_datasets),
        "sample types": str(n_st),
        "specific conditions": str(n_specific),
        "EV conditions": str(ev_conds),
        "serum conditions": str(serum_conds),
        "strong axes": str(n_strong_axes),
        "transferable MSS": str(n_transferable),
        "caveat categories": str(n_caveats),
        "caveat-burdened datasets": str(n_caveat_ds),
    }, cols_per_row=5)

    ui.divider()
    ui.card(
        title="What this app shows",
        subtitle="Visualisation-only. No spectrum is rescored.",
        body_md=(
            "<div style='font-size:0.90rem;'>"
            "v2 separates broad condition families from specific cohort labels "
            "so we can inspect <em>HCC</em>, <em>CCA</em>, <em>LM</em>, "
            "<em>COVID</em>, <em>OWD/NWD</em>, <em>cell-line mixtures</em>, and "
            "<em>APAP Day-2 concentration response</em> on their own — not "
            "averaged into a single condition family.</div>"
        ),
    )

    ui.divider()
    ui.section_header("Specific conditions detected")
    if ev is None or ev.empty or "specific_condition" not in ev.columns:
        ui.warning_card("Events table missing — can't summarise conditions.")
        return

    rows = []
    cav_per_ds = (caveats.groupby("dataset")["n_mentions"].sum().to_dict()
                   if caveats is not None and not caveats.empty else {})
    for cond, sub in ev.groupby("specific_condition"):
        st_dist = sub["sample_type"].value_counts().to_dict()
        st_top = ", ".join(f"{k}={v}" for k, v in st_dist.items())
        ds_count = sub["dataset"].nunique()
        ds_list = ", ".join(sorted(sub["dataset"].dropna().unique())[:3])

        # Top 3 axes by event count for this cohort
        axes = (sub["bsv_axis"].value_counts().head(3).index.tolist())
        top_axes = ", ".join(axes) if axes else ""

        # Top MSS candidates
        mss = (sub["mss_candidate"].dropna().value_counts().head(3)
                .index.tolist())
        top_mss = ", ".join([m for m in mss if m]) if mss else ""

        cav = sum(cav_per_ds.get(ds, 0)
                   for ds in sub["dataset"].dropna().unique())
        rows.append({
            "specific_condition": cond,
            "sample types": st_top,
            "n datasets": int(ds_count),
            "n events": len(sub),
            "top axes": top_axes,
            "top MSS": top_mss,
            "caveat mentions": int(cav),
            "datasets (first 3)": ds_list,
        })
    df = (pd.DataFrame(rows).sort_values("n events", ascending=False)
           .reset_index(drop=True))
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Discovery report preview
    rp = report_path(ctx["_root"])
    txt = load_text_safe(rp)
    if txt:
        with st.expander("Full discovery report (REPORT_context_graph_discovery_v1.md)",
                          expanded=False):
            st.markdown(txt)
