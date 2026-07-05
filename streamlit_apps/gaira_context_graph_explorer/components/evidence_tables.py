"""Optional shared helper — surface a raw evidence-events table view.

Currently used only as a sub-section if a user opens the "raw events"
expander from the overview. Kept thin so future tabs can call into it.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st


def render_events_subset(events: pd.DataFrame, max_rows: int = 200) -> None:
    if events is None or events.empty:
        st.info("No evidence events available.")
        return
    cols = st.columns([1.2, 1.2, 1, 1])
    with cols[0]:
        st_filter = st.multiselect(
            "Sample type", options=sorted(events["sample_type"].dropna().unique()),
            default=[], key="cge_events_st")
    with cols[1]:
        cf_filter = st.multiselect(
            "Condition family",
            options=sorted(events["condition_family"].dropna().unique()),
            default=[], key="cge_events_cf")
    with cols[2]:
        ax_filter = st.multiselect(
            "BSV axis",
            options=[f"G{i:02d}" for i in range(1, 12)],
            default=[], key="cge_events_ax")
    with cols[3]:
        dir_filter = st.selectbox(
            "Direction", options=["all", "up", "down", "stable", "ambiguous"],
            index=0, key="cge_events_dir")

    df = events.copy()
    if st_filter: df = df[df["sample_type"].isin(st_filter)]
    if cf_filter: df = df[df["condition_family"].isin(cf_filter)]
    if ax_filter: df = df[df["bsv_axis"].isin(ax_filter)]
    if dir_filter != "all":
        df = df[df["direction"] == dir_filter]

    st.caption(f"{len(df)} events match filters · showing first {min(len(df), max_rows)}")
    cols_show = ["dataset", "sample_type", "condition_family",
                  "comparison_type", "bsv_axis", "direction",
                  "effect_size", "metric_type", "mss_candidate",
                  "confidence_tier", "source_file"]
    cols_show = [c for c in cols_show if c in df.columns]
    st.dataframe(df[cols_show].head(max_rows), use_container_width=True,
                  hide_index=True)
