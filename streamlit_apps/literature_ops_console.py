"""GAIRA Literature Operations Console — refreshed for Structured Evidence v2.

Preserves the existing tab structure (Overview, Candidate Queue, Blocked Rescue,
Evidence Contribution, Advanced) but reads from v2 CSV outputs instead of the v1
DuckDB warehouse.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from v2_loaders import (
    load_blocked_df,
    load_candidate_queue_df,
    load_live_evidence_registry_df,
    load_ontology_summary,
    load_overview_metrics,
)


st.set_page_config(page_title="GAIRA Literature Ops Console", layout="wide")
st.title("GAIRA Literature Operations Console")
st.caption(
    "Registry-first scientific operations console — refreshed for Structured Evidence v2. "
    "Candidate triage, blocked rescue, and evidence contribution review."
)


STATE_ORDER = ["OA READY", "OA FETCH", "BLOCKED", "NEEDS RESOLUTION", "LOW VALUE"]
DEFAULT_STATES = ["OA READY", "OA FETCH", "BLOCKED"]
STATE_COLORS = {
    "OA READY": "#1b7f3a",
    "OA FETCH": "#1565c0",
    "BLOCKED": "#b3261e",
    "LOW VALUE": "#6b7280",
    "NEEDS RESOLUTION": "#9e6b14",
    "INGESTED": "#4b5563",
    "UNKNOWN": "#999999",
}


def _badge(label: str) -> str:
    color = STATE_COLORS.get(label, "#4b5563")
    return (
        f"<span style='display:inline-block;padding:0.2rem 0.55rem;border-radius:999px;"
        f"background:{color};color:white;font-size:0.8rem;font-weight:700'>{label}</span>"
    )


def _safe_json_list(value) -> list[str]:
    try:
        loaded = json.loads(str(value or "[]"))
    except Exception:
        return []
    return [str(item) for item in loaded if item]


# =====================================================================
# LOAD DATA
# =====================================================================

@st.cache_data(ttl=60)
def _load_all():
    return {
        "overview": load_overview_metrics(),
        "queue": load_candidate_queue_df(),
        "blocked": load_blocked_df(),
        "contribution": load_live_evidence_registry_df(),
        "ontology": load_ontology_summary(),
    }


state = _load_all()
overview = state["overview"]
queue_df = state["queue"]
blocked = state["blocked"]
contribution = state["contribution"]
ontology = state["ontology"]


# =====================================================================
# TABS
# =====================================================================

overview_tab, queue_tab, blocked_tab, contribution_tab, advanced_tab = st.tabs(
    ["Overview", "Candidate Queue", "Blocked Rescue", "Evidence Contribution", "Advanced / Debug"]
)

# =====================================================================
# OVERVIEW TAB
# =====================================================================

with overview_tab:
    st.subheader("Pipeline State")
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.metric("Total candidates", overview["total_candidates"])
    p2.metric("Ready now", overview["oa_ready"])
    p3.metric("OA fetchable", overview["oa_fetch"])
    p4.metric("Blocked high-value", overview["blocked"])
    p5.metric("Needs resolution", overview.get("needs_resolution", 0))
    p6.metric("Ingested sources", overview["ingested"])

    st.subheader("Evidence Stats")
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Total structured rows", overview["total_structured_rows"])
    e2.metric("Unique meanings", overview["total_meanings"])
    e3.metric("Assignment patterns", overview["motifs_count"])
    e4.metric("Condition links", overview["condition_links"])
    e5.metric("Spectral regions", overview.get("spectral_regions", 0))

    # Modality warning
    modality_missing = overview.get("modality_missing_rows", 0)
    if modality_missing > 0:
        st.warning(
            f"**Modality gap detected:** {modality_missing} evidence rows have no `modality` field "
            f"(Raman vs SERS). Modality is currently inferred from paper titles but is not persisted "
            f"in the evidence schema. This should be fixed before further expansion.",
            icon="⚠️",
        )

    st.subheader("Last Ingestion Run")
    last = overview["last_run"]
    l1, l2, l3, l4 = st.columns(4)
    l1.metric("Run ID", last["source_kind"] or "unknown")
    l2.metric("Papers processed", last["papers_processed"])
    l3.metric("Rows added", last["rows_added"])
    l4.metric("Motifs affected", last["motifs_affected"])

    st.subheader("Blocked Inventory")
    b1, b2 = st.columns(2)
    b1.metric("High-quality blocked papers", overview.get("blocked_high_quality", 0))
    b2.metric("Low value / archived", overview.get("low_value", 0))

    with st.expander("Definitions", expanded=False):
        st.markdown(
            """
- **Ready now**: paper has OA-verified accessible content and is ready for extraction.
- **OA fetchable**: OA confirmed but fulltext not yet staged locally.
- **Blocked**: paper requires manual rescue or institutional access.
- **Needs resolution**: OA status unknown; needs additional MCP check.
- **Ingested**: paper already contributed structured evidence rows.
- **Modality gap**: evidence rows lack explicit Raman vs SERS tagging.
            """
        )


# =====================================================================
# CANDIDATE QUEUE TAB
# =====================================================================

with queue_tab:
    st.subheader("Candidate Queue")
    st.caption("Current lane assignments from Phase F reclassification + Phase G ingestion status.")

    if queue_df.empty:
        st.info("No candidate data loaded.")
    else:
        # Filter to non-ingested by default
        show_ingested = st.checkbox("Show ingested papers", value=False)
        visible = queue_df if show_ingested else queue_df[queue_df["operational_state"] != "INGESTED"]

        selected_states = st.pills("Visible states", STATE_ORDER, default=DEFAULT_STATES, selection_mode="multi")
        if not show_ingested:
            visible = visible[visible["operational_state"].isin(selected_states)]

        search_text = st.text_input("Quick filter", placeholder="title / paper_id / DOI / condition")
        if search_text:
            mask = pd.Series(False, index=visible.index)
            for col in ["title", "paper_id", "doi", "condition_guess", "biosample_guess"]:
                if col in visible.columns:
                    mask = mask | visible[col].fillna("").str.contains(search_text, case=False, regex=False)
            visible = visible[mask]

        visible = visible.reset_index(drop=True)
        st.caption(f"{len(visible)} visible papers")

        # Header
        hdr = st.columns([1.0, 4.2, 1.6, 1.6, 1.4, 1.0])
        hdr[0].markdown("**paper_id**")
        hdr[1].markdown("**title**")
        hdr[2].markdown("**state**")
        hdr[3].markdown("**biosample**")
        hdr[4].markdown("**condition**")
        hdr[5].markdown("**priority**")
        st.divider()

        for _, row in visible.iterrows():
            cols = st.columns([1.0, 4.2, 1.6, 1.6, 1.4, 1.0])
            cols[0].markdown(f"`{row['paper_id']}`")
            title = row.get("title", "")
            cols[1].markdown(f"**{title[:80]}**" if title else "—")
            state_label = row.get("operational_state", "UNKNOWN")
            cols[2].markdown(_badge(state_label), unsafe_allow_html=True)
            cols[3].markdown(f"`{row.get('biosample_guess', '-')}`")
            cols[4].markdown(f"`{row.get('condition_guess', '-')}`")
            score = row.get("priority_score", "")
            cols[5].markdown(f"`{float(score):.2f}`" if score else "—")

            doi = row.get("doi", "")
            with st.expander(f"Details · {row['paper_id']}", expanded=False):
                dcols = st.columns(2)
                with dcols[0]:
                    st.markdown(f"**DOI:** [{doi}](https://doi.org/{doi})" if doi else "**DOI:** n/a")
                    st.markdown(f"**Biosample:** `{row.get('biosample_guess', 'unknown')}`")
                    st.markdown(f"**Condition:** `{row.get('condition_guess', 'unknown')}`")
                with dcols[1]:
                    st.markdown(f"**Priority score:** `{score}`")
                    st.markdown(f"**Current lane:** `{row.get('current_lane', 'unknown')}`")
                    st.markdown(f"**Modality (inferred):** `{row.get('modality', 'unknown')}`")
            st.divider()


# =====================================================================
# BLOCKED RESCUE TAB
# =====================================================================

with blocked_tab:
    st.subheader("Blocked High-Quality Registry")
    st.caption("Only high-value blocked papers (priority >= 0.75). For manual institutional rescue.")

    if blocked.empty:
        st.info("No blocked papers in the high-quality registry.")
    else:
        # Tier filter
        if "priority_tier" in blocked.columns:
            tiers = sorted(blocked["priority_tier"].unique().tolist())
            selected_tiers = st.pills("Priority tier", tiers, default=tiers, selection_mode="multi")
            blocked_visible = blocked[blocked["priority_tier"].isin(selected_tiers)]
        else:
            blocked_visible = blocked

        st.caption(f"{len(blocked_visible)} blocked papers")

        hdr = st.columns([1.0, 4.2, 1.6, 1.4, 1.2, 1.2])
        hdr[0].markdown("**paper_id**")
        hdr[1].markdown("**title**")
        hdr[2].markdown("**publisher**")
        hdr[3].markdown("**expected yield**")
        hdr[4].markdown("**modality**")
        hdr[5].markdown("**tier**")
        st.divider()

        for _, row in blocked_visible.iterrows():
            pid = row.get("paper_id", "")
            cols = st.columns([1.0, 4.2, 1.6, 1.4, 1.2, 1.2])
            cols[0].markdown(f"`{pid}`")
            cols[1].markdown(f"**{row.get('title', '')[:75]}**")
            cols[2].markdown(f"`{row.get('blocker_type', row.get('publisher_blocker', '-'))}`")
            cols[3].markdown(f"`{row.get('expected_yield', row.get('expected_evidence_contribution', '-'))}`")
            cols[4].markdown(f"`{row.get('modality', '-')}`")
            cols[5].markdown(f"`{row.get('priority_tier', '-')}`")

            doi = row.get("doi", "")
            with st.expander(f"Rescue · {pid}", expanded=False):
                st.markdown(f"**DOI:** [{doi}](https://doi.org/{doi})" if doi else "**DOI:** n/a")
                st.markdown(f"**Article URL:** {row.get('canonical_article_url', row.get('canonical_url', 'n/a'))}")
                st.markdown(f"**Biosample:** `{row.get('biosample', '-')}`")
                st.markdown(f"**Condition:** `{row.get('condition', '-')}`")
                st.markdown(f"**Rescue status:** `{row.get('rescue_status', 'awaiting_manual_download')}`")
                st.markdown(f"**Upload target:** `{row.get('local_upload_target_path', row.get('upload_target_path', '-'))}`")
                st.markdown(f"**Priority score:** `{row.get('priority_score', '-')}`")
            st.divider()


# =====================================================================
# EVIDENCE CONTRIBUTION TAB
# =====================================================================

with contribution_tab:
    st.subheader("Evidence Contribution")
    st.caption("One row per ingested source. Shows what each paper contributes to the structured evidence layer.")

    if contribution.empty:
        st.info("No evidence-contributing sources found.")
    else:
        contribution = contribution.sort_values(
            ["impact_score", "structured_evidence_rows"],
            ascending=[False, False],
        ).reset_index(drop=True)

        # Summary bar
        total_rows = int(contribution["structured_evidence_rows"].sum())
        total_meanings = int(contribution["meanings_touched"].sum())
        total_sources = len(contribution)
        sm1, sm2, sm3 = st.columns(3)
        sm1.metric("Total evidence rows", total_rows)
        sm2.metric("Total sources", total_sources)
        sm3.metric("Sum meanings touched", total_meanings)

        # Modality warning
        if "modality" in contribution.columns:
            mod_dist = contribution.groupby("modality")["structured_evidence_rows"].sum().to_dict()
            mod_parts = [f"{k}: {v}" for k, v in sorted(mod_dist.items(), key=lambda x: -x[1])]
            st.info(f"Modality distribution (inferred from titles): {' | '.join(mod_parts)}")

        hdr = st.columns([1.8, 3.2, 0.8, 0.8, 0.8, 0.8, 1.2, 1.0, 0.8])
        hdr[0].markdown("**source_id**")
        hdr[1].markdown("**paper / title**")
        hdr[2].markdown("**rows**")
        hdr[3].markdown("**meanings**")
        hdr[4].markdown("**motifs**")
        hdr[5].markdown("**cond.**")
        hdr[6].markdown("**extraction**")
        hdr[7].markdown("**modality**")
        hdr[8].markdown("**impact**")
        st.divider()

        for _, row in contribution.iterrows():
            cols = st.columns([1.8, 3.2, 0.8, 0.8, 0.8, 0.8, 1.2, 1.0, 0.8])
            cols[0].markdown(f"`{row['source_id']}`")
            cols[1].markdown(f"**{row['title'][:60]}**  \n`{row['paper_id']}`")
            cols[2].metric("", int(row["structured_evidence_rows"]), label_visibility="collapsed")
            cols[3].metric("", int(row["meanings_touched"]), label_visibility="collapsed")
            cols[4].metric("", int(row["motifs_affected"]), label_visibility="collapsed")
            cols[5].metric("", int(row["condition_motif_links"]), label_visibility="collapsed")
            cols[6].markdown(f"`{row.get('extraction_modes', '-')}`")
            mod = row.get("modality", "—")
            cols[7].markdown(f"`{mod}`" if mod else "—")
            cols[8].markdown(f"`{row['impact_score']:.1f}`")

            with st.expander(f"Details · {row['source_id']}", expanded=False):
                dcols = st.columns(2)
                with dcols[0]:
                    st.markdown(f"**DOI:** `{row.get('doi', '')}`")
                    st.markdown(f"**Biosample:** `{row.get('biosample', '-')}`")
                    st.markdown(f"**Condition:** `{row.get('condition', '-')}`")
                    st.markdown(f"**Modality (inferred):** `{mod}`")
                    st.markdown(f"**Value class:** `{row.get('assigned_value_class', '-')}`")
                with dcols[1]:
                    st.markdown(f"**New meanings introduced:** `{int(row['new_meanings_introduced'])}`")
                    st.markdown(f"**Regions covered:** `{row.get('regions_covered', '-')}`")
                    st.markdown(f"**Region list:** `{row.get('region_list', '-')}`")
                    st.markdown(f"**Ingestion run:** `{row.get('ingestion_run_id', '-')}`")
            st.divider()


# =====================================================================
# ADVANCED TAB
# =====================================================================

with advanced_tab:
    subtabs = st.tabs(["Coverage Summary", "Ontology / Meanings", "Condition Links", "Raw Data"])

    with subtabs[0]:
        st.subheader("Coverage Summary")

        # L1 category coverage
        st.markdown("**L1 Category Coverage**")
        l1 = ontology.get("l1_counts", {})
        if l1:
            l1_df = pd.DataFrame({"category": list(l1.keys()), "evidence_rows": list(l1.values())}).sort_values("evidence_rows", ascending=False)
            st.bar_chart(l1_df.set_index("category"))
        else:
            st.info("No L1 data available.")

        # Region coverage
        st.markdown("**Spectral Region Coverage**")
        rc = ontology.get("region_counts", {})
        if rc:
            rc_df = pd.DataFrame({"region": list(rc.keys()), "evidence_rows": list(rc.values())}).sort_values("evidence_rows", ascending=False)
            st.bar_chart(rc_df.set_index("region"))
        else:
            st.info("No region data available.")

        # Condition coverage
        st.markdown("**Condition Coverage**")
        cc = ontology.get("condition_counts", {})
        if cc:
            cc_df = pd.DataFrame({"condition": list(cc.keys()), "evidence_rows": list(cc.values())}).sort_values("evidence_rows", ascending=False)
            st.bar_chart(cc_df.set_index("condition"))
        else:
            st.info("No condition data available.")

    with subtabs[1]:
        st.subheader("L2 Meaning Categories")
        l2 = ontology.get("l2_counts", {})
        if l2:
            l2_df = pd.DataFrame({"l2_category": list(l2.keys()), "evidence_rows": list(l2.values())}).sort_values("evidence_rows", ascending=False)
            st.dataframe(l2_df, use_container_width=True, hide_index=True)
        else:
            st.info("No L2 data available.")

    with subtabs[2]:
        st.subheader("Condition Links")
        cond_motif_files = list(Path("/Volumes/SSD_Rad/GAIRA_DATA/structured_evidence_v2/processed/extraction_runs").glob("*_condition_motif_links.csv"))
        if cond_motif_files:
            dfs = []
            for f in cond_motif_files:
                try:
                    dfs.append(pd.read_csv(f))
                except Exception:
                    pass
            if dfs:
                cond_df = pd.concat(dfs, ignore_index=True)
                st.dataframe(cond_df.sort_values("evidence_count", ascending=False) if "evidence_count" in cond_df.columns else cond_df, use_container_width=True, hide_index=True)
            else:
                st.info("No condition link data.")
        else:
            st.info("No condition link files found.")

    with subtabs[3]:
        st.subheader("Raw Data Tables")
        st.caption("Debug view of underlying data files.")

        data_choice = st.selectbox("Select table", [
            "Ingested Source Registry",
            "Candidate Partition Registry",
            "Blocked High-Quality Registry",
            "Updated Neighborhoods",
            "Updated Assignment Patterns",
            "Modality Audit",
        ])

        file_map = {
            "Ingested Source Registry": "/Volumes/SSD_Rad/GAIRA_DATA/structured_evidence_v2/registry/ingested_source_registry.csv",
            "Candidate Partition Registry": "/Volumes/SSD_Rad/GAIRA_DATA/structured_evidence_v2/registry/candidate_partition_registry.csv",
            "Blocked High-Quality Registry": "/Volumes/SSD_Rad/GAIRA_DATA/structured_evidence_v2/registry/blocked_high_quality_registry.csv",
            "Updated Neighborhoods": "/Volumes/SSD_Rad/GAIRA_DATA/structured_evidence_v2/processed/updated_neighborhoods.csv",
            "Updated Assignment Patterns": "/Volumes/SSD_Rad/GAIRA_DATA/structured_evidence_v2/processed/updated_assignment_patterns.csv",
            "Modality Audit": "/Volumes/SSD_Rad/GAIRA_DATA/structured_evidence_v2/reports/modality_audit.csv",
        }

        fpath = Path(file_map[data_choice])
        if fpath.exists():
            try:
                raw_df = pd.read_csv(fpath, dtype=str).fillna("")
                st.dataframe(raw_df, use_container_width=True, hide_index=True)
                st.caption(f"{len(raw_df)} rows · {fpath}")
            except Exception as e:
                st.error(f"Failed to load: {e}")
        else:
            st.warning(f"File not found: {fpath}")
