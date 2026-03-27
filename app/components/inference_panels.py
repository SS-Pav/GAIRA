from __future__ import annotations

import streamlit as st

from app.services.inference import (
    format_evidence_hits,
    format_theme_table,
    synthesize_result_summary,
)
from app.services.plots import plot_theme_bars


def _severity_label(value: float) -> str:
    if value >= 0.66:
        return "High"
    if value >= 0.33:
        return "Medium"
    return "Low"


def _render_evidence_list(title: str, rows: list[dict[str, str]], empty_text: str) -> None:
    st.markdown(f"**{title}**")
    if not rows:
        st.caption(empty_text)
        return
    for row in rows[:5]:
        label = row.get("label", "") or "unlabeled"
        source = row.get("source", "") or "unknown source"
        family = row.get("family", "") or "shared"
        detail = str(row.get("detail", "")).strip()
        short_detail = detail[:180] + ("..." if len(detail) > 180 else "")
        st.markdown(
            f"<div style='border:1px solid #d7dde6;border-radius:10px;padding:0.6rem 0.8rem;margin-bottom:0.45rem;'>"
            f"<div style='font-weight:600;'>{label}</div>"
            f"<div style='font-size:0.85rem;color:#475569;'>{source} • {family}</div>"
            f"<div style='font-size:0.88rem;margin-top:0.2rem;color:#334155;'>{short_detail or 'Matched from current evidence stack.'}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_result_panel(result: dict, *, title: str) -> None:
    st.markdown(f"### {title}")
    st.info(synthesize_result_summary(result))
    st.caption(f"Routing family: {result.get('query_routing_family') or 'legacy'}")

    positive_df = format_theme_table(result, "positive")
    caution_df = format_theme_table(result, "caution")

    chart_col, caution_col = st.columns([1.15, 0.95], gap="large")
    with chart_col:
        st.markdown("**Biochemical themes**")
        if not positive_df.empty:
            st.pyplot(
                plot_theme_bars(
                    positive_df["display_name"].head(5).tolist(),
                    positive_df["score"].head(5).tolist(),
                    title="Dominant biochemical themes",
                    color="#0f766e",
                ),
                width="stretch",
            )
        else:
            st.caption("No positive themes returned.")

    with caution_col:
        st.markdown("**Cautions**")
        if caution_df.empty:
            st.caption("No caution themes returned.")
        else:
            for _, row in caution_df.head(5).iterrows():
                severity = _severity_label(float(row["score"]))
                st.markdown(
                    f"<div style='border:1px solid #e5c7c7;border-radius:10px;padding:0.55rem 0.75rem;margin-bottom:0.4rem;background:#fff8f8;'>"
                    f"<div style='font-weight:600;'>{row['display_name']}</div>"
                    f"<div style='font-size:0.88rem;color:#7f1d1d;'>Severity: {severity} • score {float(row['score']):.2f}</div>"
                    f"<div style='font-size:0.82rem;color:#57534e;'>confidence {float(row['confidence']):.2f} • specificity {float(row['specificity_index']):.2f}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    evidence_col, support_col = st.columns(2, gap="large")
    with evidence_col:
        _render_evidence_list(
            "Top direct evidence",
            format_evidence_hits(result, "tier1_grounding_hits", "source_label"),
            "No tier-1 hits.",
        )
    with support_col:
        support_rows = format_evidence_hits(result, "tier2_support_hits", "source_label")
        support_rows.extend(format_evidence_hits(result, "domain_context_hits", "document_id")[:3])
        _render_evidence_list(
            "Support and context",
            support_rows,
            "No support or context hits.",
        )

    with st.expander("Advanced evidence details", expanded=False):
        st.markdown("**Theme table**")
        st.dataframe(positive_df.head(8), hide_index=True, width="stretch")
        st.markdown("**Caution table**")
        st.dataframe(caution_df.head(8), hide_index=True, width="stretch")
        st.markdown("**Tier-1 grounding**")
        st.dataframe(format_evidence_hits(result, "tier1_grounding_hits", "source_label"), hide_index=True, width="stretch")
        st.markdown("**Tier-2 support**")
        st.dataframe(format_evidence_hits(result, "tier2_support_hits", "source_label"), hide_index=True, width="stretch")
        st.markdown("**Context**")
        st.dataframe(format_evidence_hits(result, "domain_context_hits", "document_id"), hide_index=True, width="stretch")


def render_what_changed(comparison: dict) -> None:
    st.markdown("### What changed")
    st.caption(
        f"Routed family: {comparison['routing_family']} • matched support hits: {comparison['family_matched_support_hits']} • matched context hits: {comparison['family_matched_context_hits']}"
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Theme shifts**")
        if comparison["changed_themes"]:
            st.dataframe(comparison["changed_themes"], width="stretch", hide_index=True)
        else:
            st.caption("No material positive-theme deltas.")
    with right:
        st.markdown("**Caution shifts**")
        if comparison.get("changed_cautions"):
            st.dataframe(comparison["changed_cautions"], width="stretch", hide_index=True)
        else:
            st.caption("No material caution deltas.")

    st.markdown("**Support changes**")
    st.write(f"Added: {', '.join(comparison['added_support']) or 'None'}")
    st.write(f"Removed: {', '.join(comparison['removed_support']) or 'None'}")
