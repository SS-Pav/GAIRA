"""Tab 1 — Overview / headline metrics."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from components import ui_blocks as ui
from utils.load_context_data import load_text_safe, report_path


def render(ctx: dict, app_cfg: dict) -> None:
    st.markdown(f"# {app_cfg['app']['title']}")
    st.caption(app_cfg["app"]["subtitle"])

    ev = ctx.get("events")
    nodes = ctx.get("nodes")
    edges = ctx.get("edges")
    axt = ctx.get("axis_transfer")
    msst = ctx.get("mss_transfer")
    caveats = ctx.get("caveats")
    findings = ctx.get("findings")

    n_events = len(ev) if ev is not None else 0
    n_datasets = ev["dataset"].nunique() if ev is not None else 0
    n_st = ev["sample_type"].nunique() if ev is not None else 0
    n_strong_axes = (axt[axt["axis_transfer_score"] >= 1.0].shape[0]
                     if axt is not None else 0)
    n_transferable = (msst[msst["classification"] == "TRANSFERABLE"].shape[0]
                       if msst is not None else 0)
    n_caveats = (caveats["caveat_id"].nunique()
                  if caveats is not None and not caveats.empty else 0)

    ui.divider()
    ui.section_header("Headline numbers",
                      "Aggregated from the context-graph discovery v1 outputs.")
    ui.render_metric_cards({
        "evidence events": f"{n_events:,}",
        "datasets": str(n_datasets),
        "sample types": str(n_st),
        "strong axes": str(n_strong_axes),
        "transferable MSS": str(n_transferable),
        "caveat categories": str(n_caveats),
    }, cols_per_row=6)

    ui.divider()
    ui.section_header("What is this app?",
                      "How the context graph relates to GAIRA pilots.")
    ui.card(
        title="Context graph = recurring biochemical structure across pilots",
        subtitle="Visualisation-only — no spectrum is rescored",
        body_md=(
            "<div style='font-size:0.90rem;'>"
            "Each node and edge is derived from <em>completed</em> GAIRA "
            "pilot, calibration, and grounding outputs. The five tabs slice "
            "the same evidence from different angles:"
            "<ul style='margin:8px 0 4px 18px;'>"
            "<li><strong>Tab 2</strong> — biochemical programs by condition family</li>"
            "<li><strong>Tab 3</strong> — full sample → dataset → condition → axis → MSS hierarchy</li>"
            "<li><strong>Tab 4</strong> — MSS candidate transfer across pilots</li>"
            "<li><strong>Tab 5</strong> — dataset embeddings in BSV-effect space</li>"
            "</ul></div>"
        ),
    )

    ui.divider()
    ui.section_header("Top transferable BSV axes",
                      "Datasets × consistency × mean |effect|")
    if axt is not None and not axt.empty:
        view = axt.head(8).copy()
        view = view[["bsv_axis", "axis_name", "n_datasets",
                     "n_sample_types", "dom_direction",
                     "direction_consistency", "mean_abs_effect",
                     "axis_transfer_score"]].rename(columns={
            "bsv_axis": "axis", "axis_name": "name",
            "n_datasets": "n datasets", "n_sample_types": "n sample types",
            "dom_direction": "dominant direction",
            "direction_consistency": "consistency",
            "mean_abs_effect": "mean |effect|",
            "axis_transfer_score": "transfer score",
        })
        st.dataframe(view, use_container_width=True, hide_index=True)
    else:
        ui.warning_card("axis_transfer_scores.csv not found")

    ui.divider()
    ui.section_header("Top transferable MSS candidates")
    if msst is not None and not msst.empty:
        view = (msst[msst["classification"] == "TRANSFERABLE"]
                .head(20).copy())
        view = view[["mss_candidate", "n_datasets", "n_sample_types",
                     "n_events", "dom_direction", "direction_consistency",
                     "datasets"]]
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.caption("These are recurrent MSS motif candidates, not "
                   "definitive molecule calls in complex biofluids.")
    else:
        ui.warning_card("mss_transfer_classification.csv not found")

    ui.divider()
    ui.section_header("Main caveats (extracted from MD reports)")
    if caveats is not None and not caveats.empty:
        cv = (caveats.groupby("caveat_id")["dataset"].nunique()
              .sort_values(ascending=False).reset_index()
              .rename(columns={"dataset": "n datasets",
                                "caveat_id": "caveat"}))
        st.dataframe(cv, use_container_width=True, hide_index=True)
    else:
        ui.warning_card("caveat_recurrence.csv not found")

    # Optional report preview
    rp = report_path(ctx["_root"])
    txt = load_text_safe(rp)
    if txt:
        with st.expander("Full discovery report (REPORT_context_graph_discovery_v1.md)",
                          expanded=False):
            st.markdown(txt)
