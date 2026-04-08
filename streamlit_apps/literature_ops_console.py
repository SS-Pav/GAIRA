from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gaira.evidence_v1.literature_ops_console import (  # noqa: E402
    add_search_candidates,
    build_ingest_impact_summary,
    load_activity_log_df,
    load_asset_inventory_df,
    load_blocked_df,
    load_live_evidence_registry_df,
    load_overview_metrics,
    load_paper_detail_df,
    load_registry_df,
    manual_upload_asset,
    run_discovery_search,
    set_candidate_disposition,
    trigger_oa_fetch_assets,
    trigger_oa_ingestion,
    trigger_ready_ingestion,
)


st.set_page_config(page_title="GAIRA Literature Ops Console", layout="wide")
st.title("GAIRA Literature Operations Console")
st.caption("Registry-first scientific operations console for candidate triage, blocked rescue, and evidence contribution review.")


STATE_ORDER = ["OA READY", "OA FETCH", "BLOCKED", "LOW VALUE"]
DEFAULT_STATES = ["OA READY", "OA FETCH", "BLOCKED"]
STATE_COLORS = {
    "OA READY": "#1b7f3a",
    "OA FETCH": "#1565c0",
    "BLOCKED": "#b3261e",
    "LOW VALUE": "#6b7280",
    "INGESTED": "#4b5563",
}


def _safe_json_list(value: str | None) -> list[str]:
    try:
        loaded = json.loads(value or "[]")
    except Exception:
        return []
    return [str(item) for item in loaded if item]


def _badge(label: str) -> str:
    color = STATE_COLORS.get(label, "#4b5563")
    return (
        f"<span style='display:inline-block;padding:0.2rem 0.55rem;border-radius:999px;"
        f"background:{color};color:white;font-size:0.8rem;font-weight:700'>{label}</span>"
    )


def _show_notice() -> None:
    if "lit_ops_notice" not in st.session_state:
        return
    notice = st.session_state.pop("lit_ops_notice")
    level = notice.get("level", "info")
    getattr(st, level if hasattr(st, level) else "info")(notice.get("message", ""))


def _show_latest_ingest_panel() -> None:
    if "latest_ingest_result" not in st.session_state:
        return
    latest = st.session_state["latest_ingest_result"]
    st.subheader("Latest Ingest Result")
    cols1 = st.columns(4)
    cols1[0].metric("Rows added", latest["structured_evidence_rows"])
    cols1[1].metric("Rows strengthened", latest["strengthened_support_rows"])
    cols1[2].metric("Meanings touched", latest["meanings_touched"])
    cols1[3].metric("Impact score", latest["impact_score"])
    cols2 = st.columns(4)
    cols2[0].metric("New meanings", latest["new_meanings_introduced"])
    cols2[1].metric("Motifs affected", latest["motifs_affected"])
    cols2[2].metric("Condition links added", latest["condition_links_affected"])
    cols2[3].metric("Unresolved / confounder", f"{latest['unresolved_rows']} / {latest['confounder_rows']}")
    action_cols = st.columns(3)
    if action_cols[0].button("Keep", key="keep_ingest_result"):
        del st.session_state["latest_ingest_result"]
        st.rerun()
    action_cols[1].button("Undo", key="undo_ingest_result", disabled=True)
    action_cols[1].caption(latest["rollback_note"])
    if action_cols[2].button("View in Evidence Contribution", key="open_contribution_result"):
        st.session_state["contribution_focus"] = latest["source_id"]
        del st.session_state["latest_ingest_result"]
        st.rerun()
    st.divider()


def _reload_state() -> dict[str, object]:
    return {
        "overview": load_overview_metrics(),
        "registry": load_registry_df(),
        "blocked": load_blocked_df(),
        "contribution": load_live_evidence_registry_df(),
        "assets": load_asset_inventory_df(),
        "activity": load_activity_log_df(),
    }


def _run_ingest(row: pd.Series) -> None:
    paper_id = row["paper_id"]
    result = trigger_oa_ingestion([paper_id]) if bool(row["oa_high_confidence"]) else trigger_ready_ingestion([paper_id])
    if result["returncode"] == 0:
        st.session_state["latest_ingest_result"] = result["impact_summaries"].get(paper_id, build_ingest_impact_summary(paper_id))
        outcome = result["outcomes"].get(paper_id, "completed")
        st.session_state["lit_ops_notice"] = {"level": "success", "message": f"Ingest completed for {paper_id}: {outcome}."}
        st.rerun()
    st.error("Controlled ingestion failed.")
    st.code(result["stdout"] or result["stderr"] or result["command"])


def _render_candidate_row(row: pd.Series) -> None:
    paper_id = row["paper_id"]
    state = row["operational_state"]
    cols = st.columns([1.0, 4.8, 2.2, 1.8, 1.6])
    cols[0].markdown(f"`{paper_id}`")
    cols[1].markdown(f"**{row['title']}**  \n{row['year'] or 'n/a'} · {row['journal'] or 'Unknown journal'}")
    cols[2].markdown(_badge(state), unsafe_allow_html=True)
    cols[3].markdown(f"`{row['yield_signal'] or '-'} `")

    if state == "OA READY":
        if cols[4].button("Ingest", key=f"ingest_{paper_id}", type="primary"):
            _run_ingest(row)
    elif state == "OA FETCH":
        if cols[4].button("Fetch OA", key=f"fetch_{paper_id}", type="primary"):
            result = trigger_oa_fetch_assets(paper_id)
            st.session_state["lit_ops_notice"] = {
                "level": "success",
                "message": f"OA fetch finished for {paper_id}: readiness={result['readiness_status']} next={result['next_action']}",
            }
            st.rerun()
    elif state == "BLOCKED":
        if cols[4].button("Rescue", key=f"rescue_{paper_id}", type="primary"):
            st.session_state["lit_ops_notice"] = {
                "level": "info",
                "message": f"{paper_id} is blocked. Use the inline rescue panel below.",
            }
            st.rerun()
    else:
        label = "Restore" if "operator_discarded_via_streamlit" in str(row.get("queue_notes", "")) else "Discard"
        if cols[4].button(label, key=f"discard_{paper_id}"):
            set_candidate_disposition(paper_id, keep_for_later=(label == "Restore"))
            st.session_state["lit_ops_notice"] = {
                "level": "success",
                "message": f"{paper_id} updated: {label.lower()}.",
            }
            st.rerun()

    doi = row["doi"] or ""
    doi_link = f"https://doi.org/{doi}" if doi else ""
    supp_urls = _safe_json_list(row.get("supplementary_files_json"))
    with st.expander(f"Details · {paper_id}", expanded=False):
        meta = st.columns(2)
        with meta[0]:
            st.markdown(f"**DOI:** {f'[{doi}]({doi_link})' if doi_link else 'not available'}")
            st.markdown(
                f"**Article URL:** {f'[open article]({row['canonical_article_url']})' if row['canonical_article_url'] else 'not available'}"
            )
            st.markdown(f"**SI URL:** {f'[open supplementary]({supp_urls[0]})' if supp_urls else 'not available'}")
            st.markdown(f"**Local manuscript:** `{row['manuscript_local_path'] or 'none'}`")
        with meta[1]:
            st.markdown(f"**Asset truth status:** `{row['readiness_state'] or 'not_ready'}`")
            st.markdown(f"**Detected signals:** `{row['yield_signal'] or 'none'}`")
            st.markdown(f"**Tables present:** `{int(row.get('table_text_block_count', 0) or 0)}`")
            st.markdown(f"**Assignment phrases / body text present:** `{int(row.get('body_text_chars', 0) or 0) > 0}`")
            st.markdown(f"**SI hints:** `{bool(row.get('si_hint', False))}`")

        if row["manuscript_local_path"]:
            manuscript_path = Path(row["manuscript_local_path"])
            if manuscript_path.exists():
                with manuscript_path.open("rb") as handle:
                    st.download_button(
                        "Open/download manuscript",
                        data=handle.read(),
                        file_name=manuscript_path.name,
                        key=f"download_man_{paper_id}",
                    )

        if state in {"BLOCKED", "OA FETCH"}:
            st.markdown("**Inline rescue / upload**")
            up_cols = st.columns(3)
            manuscript_upload = up_cols[0].file_uploader("Upload manuscript", key=f"candidate_man_{paper_id}")
            si_upload = up_cols[1].file_uploader("Upload SI", key=f"candidate_si_{paper_id}")
            figure_upload = up_cols[2].file_uploader("Upload figures", key=f"candidate_fig_{paper_id}")
            notes = st.text_input("Upload notes", key=f"candidate_notes_{paper_id}")
            if st.button("Validate + refresh", key=f"candidate_validate_{paper_id}"):
                results = []
                if manuscript_upload is not None:
                    results.append(manual_upload_asset(paper_id, "manuscript_pdf", manuscript_upload.name, manuscript_upload.getvalue(), notes))
                if si_upload is not None:
                    results.append(manual_upload_asset(paper_id, "supplementary", si_upload.name, si_upload.getvalue(), notes))
                if figure_upload is not None:
                    results.append(manual_upload_asset(paper_id, "figures", figure_upload.name, figure_upload.getvalue(), notes))
                if not results:
                    st.warning("Upload at least one asset first.")
                else:
                    final = results[-1]
                    st.session_state["lit_ops_notice"] = {
                        "level": "success",
                        "message": f"{paper_id} updated: readiness={final['readiness_status']} next={final['next_action']}",
                    }
                    st.rerun()

    st.divider()


def _render_blocked_row(row: pd.Series) -> None:
    paper_id = row["paper_id"]
    cols = st.columns([1.0, 4.8, 1.8, 1.6, 1.3])
    cols[0].markdown(f"`{paper_id}`")
    cols[1].markdown(f"**{row['title']}**")
    cols[2].markdown(f"`{row['blocker_type']}`")
    cols[3].markdown(f"`{row['expected_yield']}`")
    cols[4].markdown(_badge("BLOCKED"), unsafe_allow_html=True)
    doi = row["doi"] or ""
    doi_link = f"https://doi.org/{doi}" if doi else ""
    supp_urls = _safe_json_list(row.get("supplementary_files_json"))
    with st.expander(f"Rescue · {paper_id}", expanded=False):
        st.markdown(f"**DOI:** {f'[{doi}]({doi_link})' if doi_link else 'not available'}")
        st.markdown(
            f"**Article page:** {f'[open article]({row['canonical_article_url']})' if row['canonical_article_url'] else 'not available'}"
        )
        st.markdown(f"**SI URL:** {f'[open supplementary]({supp_urls[0]})' if supp_urls else 'not available'}")
        st.markdown(f"**Blocker type:** `{row['blocker_type']}`")
        st.markdown(f"**Blocker detail:** {row['blocker_detail']}")
        st.markdown(f"**Recommended action:** {row['suggested_manual_action']}")
        st.markdown(f"**Preferred rescue asset:** `{row['preferred_asset_type_to_rescue_first']}`")
        st.markdown(f"**Upload target:** `{row['local_upload_target_path']}`")
        uploads = st.columns(3)
        manuscript_upload = uploads[0].file_uploader("Upload manuscript", key=f"blocked_man_{paper_id}")
        si_upload = uploads[1].file_uploader("Upload SI", key=f"blocked_si_{paper_id}")
        figure_upload = uploads[2].file_uploader("Upload figures", key=f"blocked_fig_{paper_id}")
        notes = st.text_input("Rescue notes", key=f"blocked_notes_{paper_id}")
        if st.button("Validate rescue upload", key=f"blocked_validate_{paper_id}"):
            results = []
            if manuscript_upload is not None:
                results.append(manual_upload_asset(paper_id, "manuscript_pdf", manuscript_upload.name, manuscript_upload.getvalue(), notes))
            if si_upload is not None:
                results.append(manual_upload_asset(paper_id, "supplementary", si_upload.name, si_upload.getvalue(), notes))
            if figure_upload is not None:
                results.append(manual_upload_asset(paper_id, "figures", figure_upload.name, figure_upload.getvalue(), notes))
            if not results:
                st.warning("Upload manuscript, supplementary, or figure assets first.")
            else:
                final = results[-1]
                st.session_state["lit_ops_notice"] = {
                    "level": "success",
                    "message": f"Blocked rescue updated for {paper_id}: readiness={final['readiness_status']} next={final['next_action']}",
                }
                st.rerun()
    st.divider()


def _render_contribution_row(row: pd.Series) -> None:
    cols = st.columns([1.9, 3.8, 1.0, 1.0, 1.0, 1.0, 1.2, 1.4])
    cols[0].markdown(f"`{row['source_id']}`")
    cols[1].markdown(f"**{row['title']}**  \n`{row['paper_id'] or 'n/a'}`")
    cols[2].metric("Rows", int(row["structured_evidence_rows"]))
    cols[3].metric("Strengthened", int(row["strengthened_support_rows"]))
    cols[4].metric("Meanings", int(row["meanings_touched"]))
    cols[5].metric("Motifs", int(row["motifs_affected"]))
    cols[6].metric("Cond. links", int(row["condition_motif_links"]) + int(row["condition_neighborhood_links"]))
    cols[7].metric("Impact", float(row["impact_score"]))
    with st.expander(f"Contribution details · {row['source_id']}", expanded=False):
        st.markdown(f"**Extraction modes:** `{row['extraction_modes'] or 'unknown'}`")
        st.markdown(f"**New meanings introduced:** `{int(row['new_meanings_introduced'])}`")
        st.markdown(f"**Ingestion run id:** `{row['ingestion_run_id']}`")
        detail = load_paper_detail_df(row["source_id"])
        evidence_df = detail["evidence"]
        meanings = sorted({v for v in evidence_df["normalized_subfamily"].fillna("").tolist() if v}) if not evidence_df.empty else []
        extraction_breakdown = (
            evidence_df.groupby("extraction_method").size().reset_index(name="rows")
            if not evidence_df.empty
            else pd.DataFrame(columns=["extraction_method", "rows"])
        )
        motif_df = detail["motifs"]
        condition_df = detail["conditions"]
        st.markdown(f"**Meanings added/touched:** `{', '.join(meanings) if meanings else 'none'}`")
        st.markdown(f"**Motifs affected:** `{', '.join(motif_df['pattern_label'].fillna('').tolist()) if not motif_df.empty else 'none'}`")
        st.markdown(
            f"**Condition links:** `{', '.join(sorted(condition_df['normalized_condition_label'].astype(str).unique().tolist())) if not condition_df.empty else 'none'}`"
        )
        st.markdown("**Extraction breakdown**")
        st.dataframe(extraction_breakdown, use_container_width=True, hide_index=True)
        exp_cols = st.columns(2)
        with exp_cols[0]:
            st.markdown("**Motif details**")
            st.dataframe(motif_df, use_container_width=True, hide_index=True)
            st.markdown("**Condition details**")
            st.dataframe(condition_df, use_container_width=True, hide_index=True)
        with exp_cols[1]:
            st.markdown("**Evidence rows**")
            st.dataframe(evidence_df, use_container_width=True, hide_index=True)
            st.markdown("**Neighborhood details**")
            st.dataframe(detail["neighborhoods"], use_container_width=True, hide_index=True)
    st.divider()


_show_notice()
_show_latest_ingest_panel()

state = _reload_state()
overview = state["overview"]
registry = state["registry"]
blocked = state["blocked"]
contribution = state["contribution"]
assets = state["assets"]
activity = state["activity"]

overview_tab, queue_tab, blocked_tab, contribution_tab, advanced_tab = st.tabs(
    ["Overview", "Candidate Queue", "Blocked Rescue", "Evidence Contribution", "Advanced / Debug"]
)

with overview_tab:
    st.subheader("Pipeline State")
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Total candidate papers", overview["total_candidates"])
    p2.metric("OA-ready (ingestable)", overview["oa_ready"])
    p3.metric("OA-fetchable", overview["oa_fetch"])
    p4.metric("Blocked", overview["blocked"])
    p5.metric("Ingested", overview["ingested"])

    st.subheader("Evidence Stats")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Total structured rows", overview["total_structured_rows"])
    e2.metric("Total meanings", overview["total_meanings"])
    e3.metric("Motifs count", overview["motifs_count"])
    e4.metric("Condition links", overview["condition_links"])

    st.subheader("Last Ingestion Run")
    last = overview["last_run"]
    l1, l2, l3, l4 = st.columns(4)
    l1.metric("Run kind", last["source_kind"])
    l2.metric("Papers processed", last["papers_processed"])
    l3.metric("Contributing papers", last["contributing_papers"])
    l4.metric("Rows added", last["rows_added"])
    st.caption(f"Motifs affected across latest run sources: {last['motifs_affected']}")

    st.subheader("Bottlenecks")
    b1, b2, b3 = st.columns(3)
    b1.metric("Needs figure follow-up", overview["needs_figure_followup"])
    b2.metric("Needs SI", overview["needs_si"])
    b3.metric("Blocked high-value", overview["blocked_high_value"])

    with st.expander("Definitions", expanded=True):
        st.markdown(
            """
            - **OA-ready (ingestable)**: paper has truth-validated local manuscript/SI assets and is ready for controlled structured extraction.
            - **OA-fetchable**: open access looks likely, but usable local assets have not been validated yet.
            - **Blocked**: paper requires manual rescue or nontrivial access handling before ingestion.
            - **Ingested**: paper already contributed structured evidence rows to the warehouse.
            - **Ingest** means full structured extraction into the evidence warehouse, not just download.
            """
        )

with queue_tab:
    st.subheader("Candidate Queue")
    st.caption("Primary workbench. Only non-ingested papers are shown by default.")
    queue_df = registry[registry["operational_state"] != "INGESTED"].copy().reset_index(drop=True)
    selected_states = st.pills("Visible states", STATE_ORDER, default=DEFAULT_STATES, selection_mode="multi")
    queue_df = queue_df[queue_df["operational_state"].isin(selected_states)].copy().reset_index(drop=True)
    search_text = st.text_input("Quick filter", placeholder="title / DOI / journal")
    if search_text:
        mask = (
            queue_df["title"].fillna("").str.contains(search_text, case=False, regex=False)
            | queue_df["doi"].fillna("").str.contains(search_text, case=False, regex=False)
            | queue_df["journal"].fillna("").str.contains(search_text, case=False, regex=False)
        )
        queue_df = queue_df[mask].reset_index(drop=True)
    st.caption(f"{len(queue_df)} visible papers")
    hdr = st.columns([1.0, 4.8, 2.2, 1.8, 1.6])
    hdr[0].markdown("**paper_id**")
    hdr[1].markdown("**title**")
    hdr[2].markdown("**state**")
    hdr[3].markdown("**yield_signal**")
    hdr[4].markdown("**quick_action**")
    st.divider()
    for _, row in queue_df.iterrows():
        _render_candidate_row(row)

with blocked_tab:
    st.subheader("Blocked Rescue")
    st.caption("Only currently blocked papers are shown here.")
    if blocked.empty:
        st.info("No blocked papers.")
    else:
        hdr = st.columns([1.0, 4.8, 1.8, 1.6, 1.3])
        hdr[0].markdown("**paper_id**")
        hdr[1].markdown("**title**")
        hdr[2].markdown("**blocker_type**")
        hdr[3].markdown("**expected_yield**")
        hdr[4].markdown("**state**")
        st.divider()
        for _, row in blocked.iterrows():
            _render_blocked_row(row)

with contribution_tab:
    st.subheader("Evidence Contribution")
    st.caption("One row per source. This view shows what each ingested paper actually contributes to the warehouse.")
    if contribution.empty:
        st.info("No evidence-contributing sources found.")
    else:
        focused = st.session_state.pop("contribution_focus", None)
        if focused:
            st.info(f"Focused source: {focused}")
        contribution = contribution.sort_values(
            ["impact_score", "structured_evidence_rows", "source_id"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        hdr = st.columns([1.9, 3.8, 1.0, 1.0, 1.0, 1.0, 1.2, 1.4])
        hdr[0].markdown("**source_id**")
        hdr[1].markdown("**paper / title**")
        hdr[2].markdown("**rows**")
        hdr[3].markdown("**strengthened**")
        hdr[4].markdown("**meanings**")
        hdr[5].markdown("**motifs**")
        hdr[6].markdown("**cond. links**")
        hdr[7].markdown("**impact**")
        st.divider()
        for _, row in contribution.iterrows():
            _render_contribution_row(row)

with advanced_tab:
    subtabs = st.tabs(["Search", "Asset inventory", "Activity"])
    with subtabs[0]:
        st.caption("Secondary/diagnostic discovery view. Duplicate detection stays DOI-first with title-similarity fallback.")
        query = st.text_input("Structured search query", placeholder="serum Raman lung cancer case control")
        if st.button("Run search", key="run_search", type="primary") and query.strip():
            st.session_state["lit_ops_search_results"] = run_discovery_search(query.strip())
        search_df = st.session_state.get("lit_ops_search_results", pd.DataFrame())
        if not search_df.empty:
            st.dataframe(
                search_df[["paper_id", "title", "doi", "source", "year", "duplicate_status", "recommended_action"]],
                use_container_width=True,
                hide_index=True,
            )
            addable = search_df[search_df["duplicate_status"].isin(["NEW", "DUPLICATE_PROBABLE"])].copy()
            for row in addable.itertuples():
                with st.expander(f"{row.duplicate_status} · {row.title}", expanded=False):
                    st.markdown(f"**DOI:** {row.doi or 'not available'}")
                    st.markdown(f"**Recommended action:** {row.recommended_action}")
                    cols = st.columns(2)
                    if cols[0].button("Add", key=f"add_{row.paper_id}"):
                        add_search_candidates([row._candidate_payload], queue_selected=False)
                        st.session_state["lit_ops_notice"] = {"level": "success", "message": f"Added {row.paper_id} to candidate registry."}
                        st.rerun()
                    if cols[1].button("Add + queue", key=f"add_queue_{row.paper_id}"):
                        add_search_candidates([row._candidate_payload], queue_selected=True)
                        st.session_state["lit_ops_notice"] = {"level": "success", "message": f"Added and queued {row.paper_id}."}
                        st.rerun()
    with subtabs[1]:
        st.caption("Advanced/debug inventory of local assets and truth-validation outcomes.")
        st.dataframe(assets, use_container_width=True, hide_index=True)
    with subtabs[2]:
        st.caption("Operational diagnostics from current DB signals.")
        st.dataframe(activity, use_container_width=True, hide_index=True)
