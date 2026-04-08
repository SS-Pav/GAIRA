from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

from gaira.evidence_v1.constants import (
    DB_PATH,
    OA_PHASE1_RERUN_TABLES_ROOT,
    OA_TEXT_FIRST_TABLES_ROOT,
    OA_TEXT_FOLLOWUP_TABLES_ROOT,
)
from gaira.evidence_v1.literature_acquisition_pipeline import (
    CandidateRecord,
    _crossref_search,
    _dedupe_records,
    _enrich_candidate,
    _europepmc_search,
    _load_local_corpus_map,
    _mark_existing_processing,
    _match_local_corpus,
    _pubmed_search,
    _title_key,
    _triage,
)
from gaira.evidence_v1.literature_asset_truth_oa import _queue_partition_for_row, _validate_local_asset
from gaira.evidence_v1.literature_asset_truth_oa import _resolve_candidate_oa_asset
from gaira.evidence_v1.literature_asset_resolver import _checksum


ROOT = Path(__file__).resolve().parents[3]
RUN_OA_INGEST_SCRIPT = ROOT / "scripts" / "evidence_v1" / "run_oa_ready_controlled_ingest.py"
RUN_READY_INGEST_SCRIPT = ROOT / "scripts" / "evidence_v1" / "run_ready_paper_controlled_ingest.py"
READY_STATUSES = {"manuscript_binary_ready", "supplementary_ready", "source_data_ready", "multi_asset_ready"}
LOW_YIELD_READINESS = {"review_only_low_yield", "platform_heavy_not_assignment_ready"}
DISCARD_MARKER = "operator_discarded_via_streamlit"


def _connect(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH), read_only=read_only)


def _df(sql: str, params: list | None = None, read_only: bool = True) -> pd.DataFrame:
    con = _connect(read_only=read_only)
    try:
        return con.sql(sql, params=params).df()
    finally:
        con.close()


def _safe_json_list(value: str | None) -> list[str]:
    try:
        loaded = json.loads(value or "[]")
    except Exception:
        return []
    return [str(item) for item in loaded if item]


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open() as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def _yield_signal_for_row(row: pd.Series) -> str:
    signals: list[str] = []
    if int(row.get("table_text_block_count", 0) or 0) > 0:
        signals.append("table")
    if int(row.get("figure_caption_count", 0) or 0) > 0:
        signals.append("caption")
    if int(row.get("body_text_chars", 0) or 0) > 0:
        signals.append("body")
    return "|".join(signals)


def _latest_ingestion_kind(con: duckdb.DuckDBPyConnection) -> str:
    for source_kind in (
        "oa_phase1_rerun_v1",
        "oa_text_followup_upgrade_v1",
        "oa_text_first_expansion_v1",
        "oa_ready_controlled_ingest_v1",
        "ready_paper_controlled_ingest_v1",
        "remaining_paper_controlled_ingest_v1",
    ):
        count = con.sql(
            "SELECT COUNT(*) FROM registry.evidence_sources WHERE source_kind = ?",
            params=[source_kind],
        ).fetchone()[0]
        if count:
            return source_kind
    return ""


def load_overview_metrics() -> dict[str, object]:
    con = _connect(read_only=True)
    try:
        total_candidates = int(con.sql("SELECT COUNT(*) FROM literature.candidate_papers").fetchone()[0])
        registry = load_registry_df()
        total_local_manuscripts = int(
            con.sql(
                """
                SELECT COUNT(DISTINCT manuscript_local_path)
                FROM literature.paper_asset_resolution
                WHERE COALESCE(manuscript_local_path, '') <> ''
                """
            ).fetchone()[0]
        )
        total_supp_assets = int(
            con.sql(
                """
                SELECT SUM(CASE WHEN json_array_length(COALESCE(supplementary_local_paths_json, '[]')) > 0
                                THEN json_array_length(supplementary_local_paths_json) ELSE 0 END)
                FROM literature.paper_asset_resolution
                """
            ).fetchone()[0]
            or 0
        )
        evidence_contributing = int(
            con.sql(
                """
                SELECT COUNT(DISTINCT source_id)
                FROM evidence.peak_assignment_evidence
                WHERE source_id LIKE 'src_%_manuscript'
                """
            ).fetchone()[0]
        )
        total_structured_rows = int(
            con.sql(
                """
                SELECT COUNT(*)
                FROM evidence.peak_assignment_evidence
                WHERE source_id LIKE 'src_%_manuscript'
                """
            ).fetchone()[0]
        )
        total_meanings = int(
            con.sql(
                """
                SELECT COUNT(DISTINCT om.normalized_subfamily)
                FROM evidence.peak_assignment_evidence pae
                JOIN ontology.evidence_ontology_mappings om USING (assignment_record_id)
                WHERE pae.source_id LIKE 'src_%_manuscript'
                  AND COALESCE(om.normalized_subfamily, '') <> ''
                """
            ).fetchone()[0]
        )
        motifs_count = int(con.sql("SELECT COUNT(*) FROM evidence.assignment_patterns").fetchone()[0])
        condition_links = int(
            (con.sql("SELECT COUNT(*) FROM evidence.condition_to_motif_links").fetchone()[0] or 0)
            + (con.sql("SELECT COUNT(*) FROM evidence.condition_to_neighborhood_links").fetchone()[0] or 0)
        )
        state_counts = registry["operational_state"].value_counts().to_dict() if not registry.empty else {}
        latest_kind = _latest_ingestion_kind(con)
        live = load_live_evidence_registry_df()
        latest_live = live[live["ingestion_run_id"].fillna("").eq(latest_kind)] if latest_kind and not live.empty else pd.DataFrame()
        last_run = {
            "source_kind": latest_kind or "unknown",
            "papers_processed": int(len(latest_live)),
            "contributing_papers": int(len(latest_live)),
            "rows_added": int(latest_live["structured_evidence_rows"].sum()) if not latest_live.empty else 0,
            "motifs_affected": int(latest_live["motifs_affected"].sum()) if not latest_live.empty else 0,
        }
        phase1_rows = _read_csv_rows(OA_PHASE1_RERUN_TABLES_ROOT / "oa_rerun_paper_summary.csv")
        followup_rows = _read_csv_rows(OA_TEXT_FOLLOWUP_TABLES_ROOT / "oa_followup_figure_priority_updated.csv")
        needs_figure_followup = len(
            {
                row["paper_id"]
                for row in phase1_rows
                if row.get("needs_followup") == "needs_figure"
            }
            | {
                row["paper_id"]
                for row in followup_rows
                if row.get("figure_followup_priority_updated") in {"medium", "high"}
            }
        )
        needs_si = len({row["paper_id"] for row in phase1_rows if row.get("needs_followup") == "needs_si"})
        blocked_high_value = int(
            con.sql(
                """
                SELECT COUNT(*)
                FROM literature.blocked_assets
                WHERE COALESCE(priority_score, 0) >= 0.7
                  AND COALESCE(resolved_manually, FALSE) = FALSE
                """
            ).fetchone()[0]
        )
        return {
            "total_candidates": total_candidates,
            "total_local_manuscripts": total_local_manuscripts,
            "total_supp_assets": int(total_supp_assets),
            "evidence_contributing": evidence_contributing,
            "oa_ready": int(state_counts.get("OA READY", 0)),
            "oa_fetch": int(state_counts.get("OA FETCH", 0)),
            "blocked": int(state_counts.get("BLOCKED", 0)),
            "low_value": int(state_counts.get("LOW VALUE", 0)),
            "ingested": int(state_counts.get("INGESTED", 0)),
            "total_structured_rows": total_structured_rows,
            "total_meanings": total_meanings,
            "motifs_count": motifs_count,
            "condition_links": condition_links,
            "last_run": last_run,
            "needs_figure_followup": needs_figure_followup,
            "needs_si": needs_si,
            "blocked_high_value": blocked_high_value,
        }
    finally:
        con.close()


def load_registry_df() -> pd.DataFrame:
    df = _df(
        """
        WITH asset_links AS (
          SELECT
            paper_id,
            MAX(COALESCE(remote_url, '')) AS canonical_or_asset_url,
            MAX(COALESCE(supplementary_files_json, '[]')) AS supplementary_files_json,
            MAX(COALESCE(source_data_links_json, '[]')) AS source_data_links_json
          FROM literature.paper_assets
          GROUP BY 1
        ),
        blocked AS (
          SELECT paper_id,
                 blocker_type,
                 resolved_manually,
                 manual_upload_pending
          FROM literature.blocked_assets
        ),
        yield_guess AS (
          SELECT paper_id, MAX(yield_priority_class) AS yield_priority_class
          FROM (
            SELECT paper_id, yield_priority_class
            FROM read_csv_auto('/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/gaira_ready_paper_controlled_ingest_v1/tables/paper_yield_assessment.csv', ignore_errors=true)
            UNION ALL
            SELECT paper_id, yield_priority_class
            FROM read_csv_auto('/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/gaira_oa_ready_controlled_ingest_v1/tables/oa_paper_yield_assessment.csv', ignore_errors=true)
          )
          GROUP BY 1
        ),
        oa_harvest AS (
          SELECT
            paper_id,
            MAX(COALESCE(figure_caption_count, 0)) AS figure_caption_count,
            MAX(COALESCE(table_text_block_count, 0)) AS table_text_block_count,
            MAX(COALESCE(body_text_chars, 0)) AS body_text_chars,
            MAX(COALESCE(supplement_link_count, 0)) AS supplement_link_count
          FROM read_csv_auto('/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/gaira_oa_text_first_expansion_v1/tables/oa_text_harvest_inventory.csv', ignore_errors=true)
          GROUP BY 1
        )
        SELECT
          c.paper_id,
          c.title,
          c.year,
          c.journal,
          c.doi,
          c.source,
          COALESCE(c.open_access_candidate, FALSE) AS open_access_candidate,
          t.final_score AS triage_score,
          t.triage_decision,
          q.queue_status,
          q.selected_for_ingestion,
          q.asset_ready,
          COALESCE(q.notes, '') AS queue_notes,
          COALESCE(r.readiness_status, '') AS readiness_state,
          COALESCE(r.canonical_article_url, COALESCE(a.canonical_or_asset_url, '')) AS canonical_article_url,
          COALESCE(a.supplementary_files_json, '[]') AS supplementary_files_json,
          COALESCE(a.source_data_links_json, '[]') AS source_data_links_json,
          COALESCE(r.manuscript_local_path, '') AS manuscript_local_path,
          COALESCE(r.supplementary_local_paths_json, '[]') AS supplementary_local_paths_json,
          c.already_in_local_corpus,
          c.already_processed_in_evidence,
          COALESCE(b.blocker_type, '') AS blocked_status,
          COALESCE(p.queue_partition = 'oa_high_confidence', FALSE) AS oa_high_confidence,
          COALESCE(y.yield_priority_class, '') AS yield_class,
          COALESCE(h.figure_caption_count, 0) AS figure_caption_count,
          COALESCE(h.table_text_block_count, 0) AS table_text_block_count,
          COALESCE(h.body_text_chars, 0) AS body_text_chars,
          COALESCE(h.supplement_link_count, 0) AS supplement_link_count,
          c.query_source,
          c.notes,
          COALESCE((
            SELECT COUNT(*)
            FROM evidence.peak_assignment_evidence pae
            JOIN registry.evidence_sources es USING (source_id)
            WHERE es.source_id LIKE 'src_%_manuscript'
              AND (
                es.source_name = c.title
                OR es.source_id LIKE '%' || regexp_replace(c.paper_id, '^paper_', '') || '%'
              )
          ), 0) AS structured_evidence_rows,
          c.disease_keywords_detected,
          c.sample_keywords_detected
        FROM literature.candidate_papers c
        LEFT JOIN literature.paper_triage t USING (paper_id)
        LEFT JOIN literature.processing_queue q USING (paper_id)
        LEFT JOIN literature.paper_asset_resolution r USING (paper_id)
        LEFT JOIN asset_links a USING (paper_id)
        LEFT JOIN literature.blocked_assets b USING (paper_id)
        LEFT JOIN literature.queue_partition p USING (paper_id)
        LEFT JOIN yield_guess y USING (paper_id)
        LEFT JOIN oa_harvest h USING (paper_id)
        ORDER BY t.final_score DESC NULLS LAST, c.paper_id
        """
    )
    return _augment_registry_operational_state(df)


def _augment_registry_operational_state(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["active_blocked"] = (
        out["blocked_status"].fillna("").astype(str).ne("")
        & (~out["already_processed_in_evidence"].fillna(False))
    )
    asset_states = []
    evidence_states = []
    operational_states = []
    next_actions = []
    for row in out.itertuples():
        readiness = str(getattr(row, "readiness_state", "") or "")
        processed = bool(getattr(row, "already_processed_in_evidence", False)) or int(getattr(row, "structured_evidence_rows", 0) or 0) > 0
        active_blocked = bool(getattr(row, "active_blocked", False)) or readiness == "blocked_manual_rescue"
        yield_class = str(getattr(row, "yield_class", "") or "")
        triage_decision = str(getattr(row, "triage_decision", "") or "")
        open_access_candidate = bool(getattr(row, "open_access_candidate", False))
        already_in_local_corpus = bool(getattr(row, "already_in_local_corpus", False))
        queue_notes = str(getattr(row, "queue_notes", "") or "")
        asset_ready_flag = bool(getattr(row, "asset_ready", False))
        discarded = DISCARD_MARKER in queue_notes
        if active_blocked:
            asset_state = "blocked"
        elif readiness in READY_STATUSES:
            asset_state = "ready_to_ingest"
        else:
            asset_state = "no_assets"
        if processed:
            evidence_state = "ingested"
        elif (
            readiness in LOW_YIELD_READINESS
            or yield_class in {"low_yield_context", "review_or_context_heavy"}
            or triage_decision == "skipped_low_value"
        ):
            evidence_state = "low_yield_context_only"
        elif bool(getattr(row, "selected_for_ingestion", False)) and asset_ready_flag:
            evidence_state = "ingesting"
        else:
            evidence_state = "not_ingested"
        if evidence_state == "ingested":
            operational_state = "INGESTED"
            next_action = "View impact"
        elif evidence_state == "low_yield_context_only" or discarded:
            operational_state = "LOW VALUE"
            next_action = "Restore" if discarded else "Discard"
        elif asset_state == "ready_to_ingest":
            operational_state = "OA READY"
            next_action = "Ingest"
        elif open_access_candidate and not active_blocked:
            operational_state = "OA FETCH"
            next_action = "Fetch OA"
        else:
            operational_state = "BLOCKED"
            next_action = "Rescue"
        asset_states.append(asset_state)
        evidence_states.append(evidence_state)
        operational_states.append(operational_state)
        next_actions.append(next_action)
    out["asset_state"] = asset_states
    out["evidence_state"] = evidence_states
    out["operational_state"] = operational_states
    out["next_action"] = next_actions
    out["yield_signal"] = out.apply(_yield_signal_for_row, axis=1)
    out["si_hint"] = out["supplement_link_count"].fillna(0).astype(int) > 0
    return out


def filter_registry_df(
    df: pd.DataFrame,
    *,
    processed_only: bool,
    unprocessed_only: bool,
    oa_ready_only: bool,
    blocked_only: bool,
    selected_only: bool,
    disease_keyword: str,
    sample_keyword: str,
    source_filter: str,
    triage_filter: str,
    text_query: str,
) -> pd.DataFrame:
    out = df.copy()
    if processed_only:
        out = out[out["already_processed_in_evidence"] == True]
    if unprocessed_only:
        out = out[out["already_processed_in_evidence"] == False]
    if oa_ready_only:
        out = out[out["oa_high_confidence"] == True]
    if blocked_only:
        out = out[out["blocked_status"].astype(str) != ""]
    if selected_only:
        out = out[out["selected_for_ingestion"] == True]
    if disease_keyword:
        out = out[out["disease_keywords_detected"].fillna("").str.contains(disease_keyword, case=False, regex=False)]
    if sample_keyword:
        out = out[out["sample_keywords_detected"].fillna("").str.contains(sample_keyword, case=False, regex=False)]
    if source_filter != "All":
        out = out[out["source"] == source_filter]
    if triage_filter != "All":
        out = out[out["triage_decision"] == triage_filter]
    if text_query:
        mask = (
            out["title"].fillna("").str.contains(text_query, case=False, regex=False)
            | out["doi"].fillna("").str.contains(text_query, case=False, regex=False)
            | out["journal"].fillna("").str.contains(text_query, case=False, regex=False)
        )
        out = out[mask]
    return out.reset_index(drop=True)


def set_candidate_disposition(paper_id: str, keep_for_later: bool) -> None:
    con = _connect(read_only=False)
    try:
        current = con.sql(
            "SELECT COALESCE(notes, ''), COALESCE(queue_status, 'pending') FROM literature.processing_queue WHERE paper_id = ?",
            params=[paper_id],
        ).fetchone()
        if current:
            notes, queue_status = current
        else:
            notes, queue_status = "", "pending"
        note_parts = [part for part in str(notes).split("; ") if part]
        note_parts = [part for part in note_parts if part != DISCARD_MARKER]
        if keep_for_later:
            new_status = "pending"
            note_parts.append("kept_for_later_via_streamlit")
        else:
            new_status = "skipped_low_value"
            note_parts.append(DISCARD_MARKER)
        if current:
            con.execute(
                """
                UPDATE literature.processing_queue
                SET queue_status = ?, selected_for_ingestion = FALSE, notes = ?
                WHERE paper_id = ?
                """,
                [new_status, "; ".join(note_parts), paper_id],
            )
        else:
            con.execute(
                """
                INSERT INTO literature.processing_queue
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [f"queue_{paper_id}", paper_id, new_status, 0, False, False, False, 0, "streamlit_candidate_disposition", "; ".join(note_parts)],
            )
        con.commit()
    finally:
        con.close()


def _candidate_record_from_db(connection: duckdb.DuckDBPyConnection, paper_id: str) -> CandidateRecord:
    row = connection.sql(
        """
        SELECT
          c.paper_id,
          c.title,
          COALESCE(c.authors, '[]') AS authors,
          c.year,
          COALESCE(c.journal, '') AS journal,
          COALESCE(c.doi, '') AS doi,
          COALESCE(c.source, '') AS source,
          COALESCE(c.source_list_json, '[]') AS source_list_json,
          COALESCE(c.abstract, '') AS abstract,
          COALESCE(c.query_source, '[]') AS query_source,
          COALESCE(c.disease_keywords_detected, '[]') AS disease_keywords_detected,
          COALESCE(c.sample_keywords_detected, '[]') AS sample_keywords_detected,
          COALESCE(c.spectral_keywords_detected, '[]') AS spectral_keywords_detected,
          COALESCE(c.title_key, '') AS title_key,
          COALESCE(c.already_in_local_corpus, FALSE) AS already_in_local_corpus,
          COALESCE(c.already_processed_in_evidence, FALSE) AS already_processed_in_evidence,
          COALESCE(c.open_access_candidate, FALSE) AS open_access_candidate,
          COALESCE(c.notes, '') AS notes,
          COALESCE(MAX(pa.remote_url), '') AS remote_url,
          COALESCE(MAX(pa.supplementary_files_json), '[]') AS supplementary_files_json,
          COALESCE(MAX(pa.source_data_links_json), '[]') AS source_data_links_json,
          COALESCE(MAX(par.canonical_article_url), '') AS canonical_article_url,
          COALESCE(MAX(par.manuscript_local_path), '') AS manuscript_local_path
        FROM literature.candidate_papers c
        LEFT JOIN literature.paper_assets pa USING (paper_id)
        LEFT JOIN literature.paper_asset_resolution par USING (paper_id)
        WHERE c.paper_id = ?
        GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18
        """,
        params=[paper_id],
    ).fetchone()
    if not row:
        raise ValueError(f"Unknown paper_id: {paper_id}")
    source_set = set(_safe_json_list(row[7])) or ({row[6]} if row[6] else set())
    remote_urls = [url for url in [row[18], row[21]] if url and str(url).startswith("http")]
    return CandidateRecord(
        paper_id=row[0],
        title=row[1],
        authors=_safe_json_list(row[2]),
        year=row[3],
        journal=row[4],
        doi=row[5],
        sources=source_set,
        abstracts=[row[8]] if row[8] else [],
        query_sources=set(_safe_json_list(row[9])),
        title_key=row[13] or _title_key(row[1]),
        disease_keywords=set(_safe_json_list(row[10])),
        sample_keywords=set(_safe_json_list(row[11])),
        spectral_keywords=set(_safe_json_list(row[12])),
        open_access_candidate=bool(row[16]),
        local_manuscript_path=row[22] or "",
        supplementary_files=_safe_json_list(row[19]),
        remote_manuscript_urls=remote_urls,
        source_data_links=_safe_json_list(row[20]),
        figures_detected=False,
        tables_detected=False,
        already_in_local_corpus=bool(row[14]),
        already_processed_in_evidence=bool(row[15]),
        notes=[row[17]] if row[17] else [],
    )


def trigger_oa_fetch_assets(paper_id: str) -> dict[str, object]:
    con = _connect(read_only=False)
    try:
        candidate = _candidate_record_from_db(con, paper_id)
        final_score = float(
            con.sql("SELECT COALESCE(final_score, 0.0) FROM literature.paper_triage WHERE paper_id = ?", params=[paper_id]).fetchone()[0]
        )
        assets, resolution_row = _resolve_candidate_oa_asset(candidate, paper_id)
        validations: list[dict] = []
        manuscript_truth = "not_ready_unknown"
        supp_truth = "not_ready_unknown"
        source_truth = "not_ready_unknown"
        idx = 2000
        if resolution_row["manuscript_local_path"]:
            manuscript_validation = _validate_local_asset(
                paper_id,
                candidate.title,
                "manuscript_pdf",
                resolution_row["manuscript_local_path"],
                resolution_row["canonical_article_url"],
                "pdf",
                idx,
            )
            validations.append(manuscript_validation.__dict__)
            manuscript_truth = manuscript_validation.validation_status
            idx += 1
        supp_validations = []
        for path in _safe_json_list(resolution_row["supplementary_local_paths_json"]):
            item = _validate_local_asset(
                paper_id,
                candidate.title,
                "supplementary",
                path,
                resolution_row["canonical_article_url"],
                Path(path).suffix.lstrip(".") or "pdf",
                idx,
            )
            supp_validations.append(item)
            validations.append(item.__dict__)
            idx += 1
        if supp_validations:
            supp_truth = supp_validations[0].validation_status
        source_validations = []
        for path in _safe_json_list(resolution_row["source_data_local_paths_json"]):
            item = _validate_local_asset(
                paper_id,
                candidate.title,
                "source_data",
                path,
                resolution_row["canonical_article_url"],
                Path(path).suffix.lstrip(".") or "csv",
                idx,
            )
            source_validations.append(item)
            validations.append(item.__dict__)
            idx += 1
        if source_validations:
            source_truth = source_validations[0].validation_status

        if manuscript_truth == "binary_pdf_ready" and (supp_truth == "supplementary_ready" or source_truth == "source_data_ready"):
            readiness = "multi_asset_ready"
        elif manuscript_truth == "binary_pdf_ready":
            readiness = "manuscript_binary_ready"
        elif supp_truth == "supplementary_ready":
            readiness = "supplementary_ready"
        elif source_truth == "source_data_ready":
            readiness = "source_data_ready"
        elif manuscript_truth in LOW_YIELD_READINESS:
            readiness = manuscript_truth
        elif manuscript_truth == "html_stub_not_ready":
            readiness = "blocked_manual_rescue"
        else:
            readiness = "not_ready"

        con.execute("DELETE FROM literature.paper_asset_resolution WHERE paper_id = ?", [paper_id])
        con.execute(
            "INSERT INTO literature.paper_asset_resolution VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                paper_id,
                resolution_row["canonical_article_url"],
                resolution_row["doi_resolved"],
                manuscript_truth,
                supp_truth,
                source_truth,
                resolution_row["manuscript_local_path"],
                resolution_row["supplementary_local_paths_json"],
                resolution_row["source_data_local_paths_json"],
                readiness,
                "streamlit_row_level_oa_fetch",
            ],
        )
        con.execute("DELETE FROM literature.asset_truth_validation WHERE paper_id = ?", [paper_id])
        if validations:
            con.executemany(
                "INSERT INTO literature.asset_truth_validation VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        row["asset_validation_id"],
                        row["paper_id"],
                        row["asset_type"],
                        row["local_path"],
                        row["source_url"],
                        row["expected_file_type"],
                        row["detected_file_type"],
                        row["file_exists"],
                        row["header_valid"],
                        row["parseable_text"],
                        row["page_count"],
                        row["text_char_count"],
                        row["validation_status"],
                        row["readiness_impact"],
                        row["notes"],
                    )
                    for row in validations
                ],
            )
        con.execute("DELETE FROM literature.resolved_assets WHERE paper_id = ?", [paper_id])
        if assets:
            con.executemany(
                "INSERT INTO literature.resolved_assets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        f"resolved_{paper_id}_{index:03d}",
                        paper_id,
                        asset["asset_type"],
                        asset["source_url"],
                        asset["local_path"],
                        Path(asset["local_path"]).suffix.lstrip(".") if asset["local_path"] else "",
                        asset["resolution_method"],
                        asset["resolution_status"],
                        _checksum(Path(asset["local_path"])) if asset["local_path"] and Path(asset["local_path"]).exists() else "",
                        "manuscript" if asset["asset_type"] == "manuscript_pdf" else "si" if asset["asset_type"] == "supplementary" else "source_data",
                        "streamlit_row_level_oa_fetch",
                    )
                    for index, asset in enumerate(assets, start=1)
                ],
            )
        con.execute(
            """
            UPDATE literature.processing_queue
            SET asset_ready = ?, queue_status = ?, selected_for_ingestion = ?, notes = ?
            WHERE paper_id = ?
            """,
            [
                readiness in READY_STATUSES,
                "selected_for_ingestion" if readiness in READY_STATUSES else "pending",
                readiness in READY_STATUSES,
                "streamlit_row_level_oa_fetch",
                paper_id,
            ],
        )
        partition, priority, partition_notes = _queue_partition_for_row(
            {
                "title": candidate.title,
                "readiness_status": readiness,
                "final_score": final_score,
                "open_access_candidate": candidate.open_access_candidate,
                "asset_resolution_notes": "streamlit_row_level_oa_fetch",
            }
        )
        con.execute("DELETE FROM literature.queue_partition WHERE paper_id = ?", [paper_id])
        con.execute(
            "INSERT INTO literature.queue_partition VALUES (?, ?, ?, ?, ?, ?)",
            [paper_id, partition, readiness, candidate.open_access_candidate, priority, partition_notes],
        )
        con.execute(
            """
            UPDATE literature.candidate_papers
            SET already_in_local_corpus = ?
            WHERE paper_id = ?
            """,
            [bool(resolution_row["manuscript_local_path"] or _safe_json_list(resolution_row["supplementary_local_paths_json"]) or _safe_json_list(resolution_row["source_data_local_paths_json"])), paper_id],
        )
        if readiness == "blocked_manual_rescue":
            upload_target = (
                str((Path(resolution_row["manuscript_local_path"]).parent if resolution_row["manuscript_local_path"] else ROOT / "tmp" / paper_id / "manuscript"))
            )
            con.execute("DELETE FROM literature.blocked_assets WHERE paper_id = ?", [paper_id])
            con.execute(
                "INSERT INTO literature.blocked_assets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)",
                [
                    f"blocked_{paper_id}",
                    paper_id,
                    candidate.title,
                    candidate.doi,
                    candidate.journal,
                    resolution_row["canonical_article_url"],
                    "landing_page_only",
                    "Row-level OA fetch did not yield a usable local binary asset.",
                    manuscript_truth,
                    supp_truth,
                    source_truth,
                    True,
                    True,
                    True,
                    "Upload manuscript PDF or supplementary asset obtained externally.",
                    "manuscript_pdf",
                    upload_target,
                    False,
                    None,
                    "",
                    priority,
                    "streamlit_row_level_oa_fetch",
                ],
            )
        else:
            con.execute("DELETE FROM literature.blocked_assets WHERE paper_id = ?", [paper_id])
        con.commit()
        return {
            "paper_id": paper_id,
            "readiness_status": readiness,
            "queue_partition": partition,
            "downloaded_asset_count": sum(1 for asset in assets if asset["local_path"]),
            "validation_count": len(validations),
        }
    finally:
        con.close()


def load_oa_ready_df() -> pd.DataFrame:
    return _df(
        """
        WITH evidence_counts AS (
          SELECT source_id, COUNT(*) AS structured_evidence_rows
          FROM evidence.peak_assignment_evidence
          WHERE source_id LIKE 'src_%_manuscript'
          GROUP BY 1
        )
        SELECT
          q.paper_id,
          c.title,
          COALESCE(r.readiness_status, '') AS readiness_status,
          COALESCE(r.manuscript_asset_status, '') AS manuscript_asset_status,
          COALESCE(r.supplementary_asset_status, '') AS supplementary_asset_status,
          COALESCE(r.source_data_asset_status, '') AS source_data_asset_status,
          COALESCE(r.manuscript_local_path, '') AS manuscript_local_path,
          COALESCE(r.supplementary_local_paths_json, '[]') AS supplementary_local_paths_json,
          COALESCE(r.source_data_local_paths_json, '[]') AS source_data_local_paths_json,
          COALESCE(t.final_score, 0.0) AS final_score,
          COALESCE(y.yield_priority_class, '') AS yield_priority_class,
          COALESCE(y.rationale, '') AS yield_rationale,
          COALESCE(c.already_processed_in_evidence, FALSE) AS already_processed_in_evidence
        FROM literature.queue_partition q
        JOIN literature.candidate_papers c USING (paper_id)
        LEFT JOIN literature.paper_asset_resolution r USING (paper_id)
        LEFT JOIN literature.paper_triage t USING (paper_id)
        LEFT JOIN read_csv_auto('/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/gaira_oa_ready_controlled_ingest_v1/tables/oa_paper_yield_assessment.csv', ignore_errors=true) y
          USING (paper_id)
        WHERE q.queue_partition = 'oa_high_confidence'
          AND COALESCE(c.already_processed_in_evidence, FALSE) = FALSE
        ORDER BY t.final_score DESC, q.paper_id
        """
    )


def load_blocked_df() -> pd.DataFrame:
    return _df(
        """
        WITH asset_links AS (
          SELECT
            paper_id,
            MAX(COALESCE(remote_url, '')) AS canonical_or_asset_url,
            MAX(COALESCE(supplementary_files_json, '[]')) AS supplementary_files_json,
            MAX(COALESCE(source_data_links_json, '[]')) AS source_data_links_json
          FROM literature.paper_assets
          GROUP BY 1
        ),
        yield_guess AS (
          SELECT paper_id, MAX(yield_priority_class) AS yield_priority_class
          FROM (
            SELECT paper_id, yield_priority_class
            FROM read_csv_auto('/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/gaira_ready_paper_controlled_ingest_v1/tables/paper_yield_assessment.csv', ignore_errors=true)
            UNION ALL
            SELECT paper_id, yield_priority_class
            FROM read_csv_auto('/Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/gaira_oa_ready_controlled_ingest_v1/tables/oa_paper_yield_assessment.csv', ignore_errors=true)
          )
          GROUP BY 1
        )
        SELECT
          b.paper_id,
          b.title,
          b.doi,
          b.publisher,
          COALESCE(r.canonical_article_url, '') AS canonical_article_url,
          COALESCE(a.supplementary_files_json, '[]') AS supplementary_files_json,
          COALESCE(a.source_data_links_json, '[]') AS source_data_links_json,
          b.blocker_type,
          b.blocker_detail,
          b.suggested_manual_action,
          b.preferred_asset_type_to_rescue_first,
          b.local_upload_target_path,
          b.manual_upload_pending,
          b.resolved_manually,
          b.priority_score,
          COALESCE(r.readiness_status, '') AS readiness_status,
          COALESCE(y.yield_priority_class,
            CASE
              WHEN COALESCE(t.final_score, 0) >= 0.8 THEN 'high'
              WHEN COALESCE(t.final_score, 0) >= 0.65 THEN 'medium'
              ELSE 'low'
            END
          ) AS expected_yield
        FROM literature.blocked_assets b
        LEFT JOIN literature.paper_asset_resolution r USING (paper_id)
        LEFT JOIN asset_links a USING (paper_id)
        LEFT JOIN literature.paper_triage t USING (paper_id)
        LEFT JOIN yield_guess y USING (paper_id)
        WHERE COALESCE(r.readiness_status, 'not_ready') IN ('blocked_manual_rescue', 'not_ready')
           OR COALESCE(b.resolved_manually, FALSE) = FALSE
        ORDER BY b.priority_score DESC, b.paper_id
        """
    )


def load_asset_inventory_df() -> pd.DataFrame:
    return _df(
        """
        WITH truth_assets AS (
          SELECT
            paper_id,
            asset_type,
            local_path,
            validation_status AS asset_truth_status,
            source_url,
            '' AS duplicate_flag
          FROM literature.asset_truth_validation
        ),
        resolved_assets AS (
          SELECT
            paper_id,
            asset_type,
            local_path,
            resolution_status AS asset_truth_status,
            source_url,
            CASE WHEN resolution_status = 'duplicate_existing' THEN 'duplicate_existing' ELSE '' END AS duplicate_flag
          FROM literature.resolved_assets
          WHERE COALESCE(local_path, '') <> ''
        ),
        unioned AS (
          SELECT * FROM truth_assets
          UNION ALL
          SELECT * FROM resolved_assets
        )
        SELECT
          u.paper_id,
          c.title,
          u.asset_type,
          u.local_path,
          u.asset_truth_status,
          u.source_url,
          u.duplicate_flag,
          CASE
            WHEN EXISTS (
              SELECT 1
              FROM evidence.peak_assignment_evidence pae
              WHERE pae.source_id LIKE 'src_%'
                AND (
                  pae.manuscript_or_si = 'manuscript'
                  OR pae.manuscript_or_si = 'si'
                )
                AND pae.source_id IN (
                  SELECT source_id FROM registry.evidence_sources WHERE source_name = c.title
                )
            ) THEN TRUE
            ELSE FALSE
          END AS used_in_ingestion
        FROM unioned u
        LEFT JOIN literature.candidate_papers c USING (paper_id)
        ORDER BY c.title, u.asset_type, u.local_path
        """
    )


def load_evidence_impact_df() -> pd.DataFrame:
    return _df(
        """
        WITH source_map AS (
          SELECT
            es.source_id,
            es.source_name,
            MIN(es.source_kind) AS source_kind,
            MIN(cp.paper_id) AS paper_id
          FROM registry.evidence_sources es
          LEFT JOIN literature.candidate_papers cp
            ON cp.title = es.source_name
            OR es.source_id LIKE '%' || regexp_replace(cp.paper_id, '^paper_', '') || '%'
          WHERE es.source_id LIKE 'src_%_manuscript'
          GROUP BY es.source_id, es.source_name
        ),
        evidence_counts AS (
          SELECT
            pae.source_id,
            COUNT(*) AS structured_evidence_rows,
            SUM(CASE WHEN om.meaning_class = 'unresolved_signal' THEN 1 ELSE 0 END) AS unresolved_rows,
            SUM(CASE WHEN om.meaning_class = 'confounder_signal' THEN 1 ELSE 0 END) AS confounder_rows,
            COUNT(DISTINCT COALESCE(om.normalized_subfamily, '')) FILTER (WHERE COALESCE(om.normalized_subfamily, '') <> '') AS meanings_touched,
            string_agg(
              DISTINCT CASE
                WHEN pae.extraction_method IN ('table_assignment', 'table_text_assignment') THEN 'table'
                WHEN pae.extraction_method = 'caption_assignment' THEN 'caption'
                WHEN pae.extraction_method IN ('text_assignment', 'text_regex') THEN 'body'
                ELSE pae.extraction_method
              END,
              '|'
            ) AS extraction_modes
          FROM evidence.peak_assignment_evidence pae
          LEFT JOIN ontology.evidence_ontology_mappings om
            ON om.assignment_record_id = pae.assignment_record_id
          WHERE pae.source_id LIKE 'src_%_manuscript'
          GROUP BY 1
        ),
        strengthened_counts AS (
          SELECT m.source_id, COUNT(DISTINCT m.assignment_record_id) AS strengthened_support_rows
          FROM evidence.local_support_neighborhood_members m
          WHERE m.source_id LIKE 'src_%_manuscript'
          GROUP BY 1
        ),
        neighborhood_counts AS (
          SELECT source_id, COUNT(DISTINCT neighborhood_id) AS neighborhoods_affected
          FROM evidence.local_support_neighborhood_members
          WHERE source_id LIKE 'src_%_manuscript'
          GROUP BY 1
        ),
        motif_counts AS (
          SELECT m.source_id, COUNT(DISTINCT l.pattern_id) AS motifs_affected
          FROM evidence.local_support_neighborhood_members m
          JOIN evidence.neighborhood_motif_links l USING (neighborhood_id)
          WHERE m.source_id LIKE 'src_%_manuscript'
          GROUP BY 1
        ),
        condition_neighborhood_counts AS (
          SELECT source_id, COUNT(*) AS condition_neighborhood_links
          FROM evidence.condition_to_neighborhood_links
          GROUP BY 1
        ),
        condition_motif_counts AS (
          SELECT source_id, COUNT(*) AS condition_motif_links
          FROM evidence.condition_to_motif_links
          GROUP BY 1
        ),
        new_meaning_counts AS (
          SELECT
            pae.source_id,
            COUNT(DISTINCT om.normalized_subfamily) AS new_meanings_introduced
          FROM evidence.peak_assignment_evidence pae
          JOIN ontology.evidence_ontology_mappings om USING (assignment_record_id)
          WHERE pae.source_id LIKE 'src_%_manuscript'
            AND COALESCE(om.normalized_subfamily, '') <> ''
            AND NOT EXISTS (
              SELECT 1
              FROM evidence.peak_assignment_evidence other_pae
              JOIN ontology.evidence_ontology_mappings other_om USING (assignment_record_id)
              WHERE other_pae.source_id <> pae.source_id
                AND other_pae.source_id LIKE 'src_%_manuscript'
                AND other_om.normalized_subfamily = om.normalized_subfamily
            )
          GROUP BY 1
        )
        SELECT DISTINCT
          sm.paper_id,
          es.source_id,
          sm.source_name AS title,
          ec.structured_evidence_rows,
          COALESCE(sc.strengthened_support_rows, 0) AS strengthened_support_rows,
          COALESCE(ec.meanings_touched, 0) AS meanings_touched,
          COALESCE(nmc.new_meanings_introduced, 0) AS new_meanings_introduced,
          COALESCE(mc.motifs_affected, 0) AS motifs_affected,
          COALESCE(cnc.condition_neighborhood_links, 0) AS condition_neighborhood_links,
          COALESCE(cmc.condition_motif_links, 0) AS condition_motif_links,
          COALESCE(ec.unresolved_rows, 0) AS unresolved_rows,
          COALESCE(ec.confounder_rows, 0) AS confounder_rows,
          COALESCE(ec.extraction_modes, '') AS extraction_modes,
          sm.source_kind AS ingestion_run_id,
          '' AS last_ingestion_at,
          ROUND(
            (COALESCE(nmc.new_meanings_introduced, 0) * 3.0) +
            (COALESCE(ec.structured_evidence_rows, 0) * 1.0) +
            (COALESCE(mc.motifs_affected, 0) * 5.0) +
            ((COALESCE(cmc.condition_motif_links, 0) + COALESCE(cnc.condition_neighborhood_links, 0)) * 4.0),
            3
          ) AS impact_score
        FROM source_map sm
        JOIN registry.evidence_sources es USING (source_id)
        LEFT JOIN evidence_counts ec USING (source_id)
        LEFT JOIN strengthened_counts sc USING (source_id)
        LEFT JOIN motif_counts mc USING (source_id)
        LEFT JOIN condition_neighborhood_counts cnc USING (source_id)
        LEFT JOIN condition_motif_counts cmc USING (source_id)
        LEFT JOIN new_meaning_counts nmc USING (source_id)
        WHERE es.source_id LIKE 'src_%_manuscript'
          AND COALESCE(ec.structured_evidence_rows, 0) > 0
        ORDER BY ec.structured_evidence_rows DESC, es.source_id
        """
    )


def load_live_evidence_registry_df() -> pd.DataFrame:
    return load_evidence_impact_df()


def build_ingest_impact_summary(paper_id: str) -> dict[str, object]:
    registry = load_evidence_impact_df()
    matched = registry[
        registry["paper_id"].fillna("").eq(paper_id)
        | registry["source_id"].astype(str).str.contains(paper_id.replace("paper_", ""), regex=False)
    ]
    if matched.empty:
        return {
            "paper_id": paper_id,
            "source_id": "",
            "title": "",
            "structured_evidence_rows": 0,
            "strengthened_support_rows": 0,
            "meanings_touched": 0,
            "new_meanings_introduced": 0,
            "motifs_affected": 0,
            "condition_links_affected": 0,
            "unresolved_rows": 0,
            "confounder_rows": 0,
            "impact_score": 0.0,
            "rollback_supported": False,
            "rollback_note": "Undo ingest is not yet fully supported; current rollback readiness is source-level only.",
        }
    row = matched.iloc[0]
    return {
        "paper_id": paper_id,
        "source_id": row["source_id"],
        "title": row["title"],
        "structured_evidence_rows": int(row["structured_evidence_rows"]),
        "strengthened_support_rows": int(row["strengthened_support_rows"]),
        "meanings_touched": int(row["meanings_touched"]),
        "new_meanings_introduced": int(row["new_meanings_introduced"]),
        "motifs_affected": int(row["motifs_affected"]),
        "condition_links_affected": int(row["condition_neighborhood_links"]) + int(row["condition_motif_links"]),
        "unresolved_rows": int(row["unresolved_rows"]),
        "confounder_rows": int(row["confounder_rows"]),
        "impact_score": float(row["impact_score"]),
        "rollback_supported": False,
        "rollback_note": f"Rollback-ready metadata exists at source level (`{row['source_id']}`), but one-click undo is not yet implemented.",
    }


def load_paper_detail_df(source_or_paper_id: str) -> dict[str, pd.DataFrame]:
    if source_or_paper_id.startswith("src_"):
        source_ids = pd.DataFrame({"source_id": [source_or_paper_id]})
    else:
        source_ids = _df(
            """
            SELECT source_id, source_name
            FROM registry.evidence_sources
            WHERE source_id LIKE 'src_%_manuscript'
              AND (
                source_name = (SELECT title FROM literature.candidate_papers WHERE paper_id = ?)
                OR source_id LIKE ?
              )
            """,
            params=[source_or_paper_id, f"%{source_or_paper_id}%"],
        )
    source_id_list = source_ids["source_id"].tolist()
    if not source_id_list:
        return {"evidence": pd.DataFrame(), "neighborhoods": pd.DataFrame(), "motifs": pd.DataFrame(), "conditions": pd.DataFrame()}
    placeholders = ", ".join(["?"] * len(source_id_list))
    params = source_id_list
    evidence = _df(
        f"""
        SELECT pae.*, COALESCE(om.normalized_subfamily, '') AS normalized_subfamily,
               COALESCE(om.broader_family, '') AS broader_family,
               COALESCE(om.meaning_class, '') AS meaning_class,
               COALESCE(om.confounder_subclass, '') AS confounder_subclass,
               COALESCE(om.spectral_region, '') AS spectral_region
        FROM evidence.peak_assignment_evidence pae
        LEFT JOIN ontology.evidence_ontology_mappings om USING (assignment_record_id)
        WHERE pae.source_id IN ({placeholders})
        ORDER BY peak_center_cm
        """,
        params=params,
    )
    neighborhoods = _df(
        f"""
        SELECT DISTINCT m.neighborhood_id, n.canonical_peak_cm, n.dominant_normalized_subfamily,
               n.broader_family, n.meaning_class, n.local_ambiguity_score, n.motif_link_count
        FROM evidence.local_support_neighborhood_members m
        JOIN evidence.local_support_neighborhoods n USING (neighborhood_id)
        WHERE m.source_id IN ({placeholders})
        ORDER BY n.canonical_peak_cm
        """,
        params=params,
    )
    motifs = _df(
        f"""
        SELECT DISTINCT l.pattern_id, ap.pattern_label, ap.normalized_subfamily, ap.broader_family,
               ap.meaning_class, ap.confidence_score, ap.ambiguity_score
        FROM evidence.local_support_neighborhood_members m
        JOIN evidence.neighborhood_motif_links l USING (neighborhood_id)
        JOIN evidence.assignment_patterns ap ON ap.pattern_id = l.pattern_id
        WHERE m.source_id IN ({placeholders})
        ORDER BY ap.pattern_id
        """,
        params=params,
    )
    conditions = _df(
        f"""
        SELECT 'neighborhood' AS link_type, normalized_condition_label, condition_family,
               neighborhood_id AS target_id, support_strength, ambiguity_score
        FROM evidence.condition_to_neighborhood_links
        WHERE source_id IN ({placeholders})
        UNION ALL
        SELECT 'motif' AS link_type, normalized_condition_label, condition_family,
               pattern_id AS target_id, support_strength, ambiguity_score
        FROM evidence.condition_to_motif_links
        WHERE source_id IN ({placeholders})
        ORDER BY link_type, normalized_condition_label, target_id
        """,
        params=params + params,
    )
    return {"evidence": evidence, "neighborhoods": neighborhoods, "motifs": motifs, "conditions": conditions}


def _candidate_lookup_frames(connection: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registry = connection.sql(
        """
        SELECT c.paper_id, c.title, c.doi, c.title_key,
               c.already_in_local_corpus, c.already_processed_in_evidence,
               COALESCE(q.queue_status, '') AS queue_status,
               COALESCE(b.blocker_type, '') AS blocker_type
        FROM literature.candidate_papers c
        LEFT JOIN literature.processing_queue q USING (paper_id)
        LEFT JOIN literature.blocked_assets b USING (paper_id)
        """
    ).df()
    assets = connection.sql(
        """
        SELECT paper_id, manuscript_local_path, supplementary_local_paths_json, source_data_local_paths_json
        FROM literature.paper_asset_resolution
        """
    ).df()
    blocked = connection.sql("SELECT paper_id, blocker_type FROM literature.blocked_assets").df()
    queue = connection.sql("SELECT paper_id, queue_status, selected_for_ingestion FROM literature.processing_queue").df()
    return registry, assets, blocked, queue


def _status_for_candidate(candidate: CandidateRecord, registry_df: pd.DataFrame) -> tuple[str, str, str]:
    doi = candidate.doi or ""
    title_key = candidate.title_key
    exact = registry_df[(registry_df["doi"] == doi) & (registry_df["doi"] != "")]
    if not exact.empty:
        row = exact.iloc[0]
        if bool(row["already_processed_in_evidence"]):
            return "PROCESSED", row["paper_id"], "Already processed into structured evidence."
        if str(row["blocker_type"]):
            return "BLOCKED", row["paper_id"], f"Blocked: {row['blocker_type']}."
        if str(row["queue_status"]):
            return "QUEUED", row["paper_id"], f"Already in processing queue: {row['queue_status']}."
        if bool(row["already_in_local_corpus"]):
            return "LOCAL_ASSET_PRESENT", row["paper_id"], "Already present in the local corpus."
        return "IN_REGISTRY", row["paper_id"], "Already in candidate registry."
    title_match = None
    best_score = 0.0
    for _, row in registry_df.iterrows():
        score = SequenceMatcher(None, title_key, row["title_key"]).ratio()
        if score > best_score:
            best_score = score
            title_match = row
    if title_match is not None and best_score >= 0.93:
        label = "DUPLICATE_PROBABLE"
        detail = f"High title-similarity match to {title_match['paper_id']} (score {best_score:.2f})."
        return label, str(title_match["paper_id"]), detail
    return "NEW", "", "No existing DOI/title match found in registry."


def run_discovery_search(query: str) -> pd.DataFrame:
    records: list[dict] = []
    label = f"app_search:{query}"
    for fetcher in (_crossref_search, _europepmc_search, _pubmed_search):
        try:
            records.extend(fetcher(query, label))
        except Exception:
            continue
    candidates = _dedupe_records(records)
    local_map = _load_local_corpus_map()
    con = _connect(read_only=True)
    try:
        for candidate in candidates:
            _match_local_corpus(candidate, local_map)
            _enrich_candidate(candidate)
        _mark_existing_processing(con, candidates)
        registry_df, _, _, _ = _candidate_lookup_frames(con)
    finally:
        con.close()
    rows = []
    for candidate in candidates:
        triage = _triage(candidate)
        status_label, matched_paper_id, detail = _status_for_candidate(candidate, registry_df)
        payload = asdict(candidate)
        payload["sources"] = sorted(candidate.sources)
        payload["query_sources"] = sorted(candidate.query_sources)
        payload["disease_keywords"] = sorted(candidate.disease_keywords)
        payload["sample_keywords"] = sorted(candidate.sample_keywords)
        payload["spectral_keywords"] = sorted(candidate.spectral_keywords)
        recommended = {
            "NEW": "Add to registry",
            "IN_REGISTRY": "Inspect existing record",
            "LOCAL_ASSET_PRESENT": "Run/refresh truth validation",
            "PROCESSED": "Inspect evidence impact",
            "BLOCKED": "Use blocked-rescue workflow",
            "QUEUED": "Inspect queue state",
            "DUPLICATE_PROBABLE": "Review duplicate before adding",
        }[status_label]
        rows.append(
            {
                "paper_id": candidate.paper_id,
                "title": candidate.title,
                "doi": candidate.doi,
                "source": candidate.primary_source(),
                "year": candidate.year,
                "journal": candidate.journal,
                "open_access_candidate": candidate.open_access_candidate,
                "final_score": triage["final_score"],
                "duplicate_status": status_label,
                "matched_paper_id": matched_paper_id,
                "recommended_action": recommended,
                "detail": detail,
                "query_source": json.dumps(sorted(candidate.query_sources)),
                "already_in_local_corpus": candidate.already_in_local_corpus,
                "already_processed_in_evidence": candidate.already_processed_in_evidence,
                "_candidate_payload": json.dumps(payload),
            }
        )
    return pd.DataFrame(rows).sort_values(["final_score", "title"], ascending=[False, True]).reset_index(drop=True)


def add_search_candidates(candidate_payloads: list[str], queue_selected: bool) -> list[str]:
    if not candidate_payloads:
        return []
    con = _connect(read_only=False)
    added: list[str] = []
    try:
        for payload in candidate_payloads:
            raw = json.loads(payload)
            candidate = CandidateRecord(
                **{k: raw[k] for k in CandidateRecord.__dataclass_fields__.keys() if k in raw}
            )
            # dataclass fields serialized as lists; convert set fields back.
            candidate.sources = set(raw.get("sources", []))
            candidate.query_sources = set(raw.get("query_sources", []))
            candidate.disease_keywords = set(raw.get("disease_keywords", []))
            candidate.sample_keywords = set(raw.get("sample_keywords", []))
            candidate.spectral_keywords = set(raw.get("spectral_keywords", []))
            candidate.abstracts = list(raw.get("abstracts", []))
            candidate.authors = list(raw.get("authors", []))
            candidate.notes = list(raw.get("notes", []))
            from gaira.evidence_v1.literature_asset_truth_oa import _upsert_candidate

            paper_id, triage = _upsert_candidate(con, candidate)
            existing_assets = con.sql("SELECT COUNT(*) FROM literature.paper_assets WHERE paper_id = ?", params=[paper_id]).fetchone()[0]
            if not existing_assets:
                con.execute(
                    "INSERT INTO literature.paper_assets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        f"asset_{paper_id}_registry",
                        paper_id,
                        candidate.primary_source(),
                        "candidate_registry",
                        candidate.local_manuscript_path,
                        json.dumps(candidate.supplementary_files),
                        json.dumps(candidate.source_data_links),
                        candidate.figures_detected,
                        candidate.tables_detected,
                        candidate.remote_manuscript_urls[0] if candidate.remote_manuscript_urls else "",
                        candidate.local_manuscript_path,
                        Path(candidate.local_manuscript_path).suffix.lstrip(".") if candidate.local_manuscript_path else "",
                        "linked_existing" if candidate.local_manuscript_path else "not_attempted",
                        "Added from literature ops console search.",
                    ],
                )
            if queue_selected:
                con.execute(
                    """
                    UPDATE literature.processing_queue
                    SET queue_status = ?, selected_for_ingestion = ?, selection_reason = ?, notes = ?
                    WHERE paper_id = ?
                    """,
                    ["pending", False, "app_manual_queue", "Queued from literature ops console search.", paper_id],
                )
            added.append(paper_id)
        con.commit()
        return added
    finally:
        con.close()


def manual_upload_asset(paper_id: str, asset_type: str, uploaded_file_name: str, uploaded_bytes: bytes, notes: str = "") -> dict[str, str]:
    con = _connect(read_only=False)
    try:
        normalized_asset_type = "supplementary" if asset_type == "figures" else asset_type
        meta = con.sql(
            """
            SELECT c.title, COALESCE(c.doi, ''), COALESCE(c.open_access_candidate, FALSE), COALESCE(t.final_score, 0.0),
                   COALESCE(r.readiness_status, 'not_ready'), COALESCE(r.asset_resolution_notes, ''),
                   COALESCE(r.manuscript_local_path, ''), COALESCE(r.supplementary_local_paths_json, '[]'),
                   COALESCE(r.source_data_local_paths_json, '[]'), COALESCE(r.canonical_article_url, ''),
                   COALESCE(b.local_upload_target_path, '')
            FROM literature.candidate_papers c
            LEFT JOIN literature.paper_triage t USING (paper_id)
            LEFT JOIN literature.paper_asset_resolution r USING (paper_id)
            LEFT JOIN literature.blocked_assets b USING (paper_id)
            WHERE c.paper_id = ?
            """,
            params=[paper_id],
        ).fetchone()
        if not meta:
            raise ValueError(f"Unknown paper_id: {paper_id}")
        title, doi, oa_candidate, final_score, _, resolution_notes, manuscript_path, supp_json, src_json, canonical_url, upload_target = meta
        suffix = Path(uploaded_file_name).suffix or ".bin"
        if upload_target:
            target_path = Path(upload_target)
            if target_path.is_dir():
                target_path = target_path / uploaded_file_name
        else:
            base = Path(manuscript_path).parent.parent if manuscript_path else (ROOT / "tmp" / paper_id)
            subdir = {"manuscript_pdf": "manuscript", "supplementary": "supplementary", "source_data": "source_data"}[normalized_asset_type]
            target_path = base / subdir / uploaded_file_name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(uploaded_bytes)

        expected_type = "pdf" if normalized_asset_type == "manuscript_pdf" else target_path.suffix.lstrip(".") or "unknown"
        validation = _validate_local_asset(
            paper_id=paper_id,
            title=title,
            asset_type=normalized_asset_type,
            local_path=str(target_path),
            source_url=canonical_url,
            expected_file_type=expected_type,
            validation_index=999,
        )

        current_supp = json.loads(supp_json or "[]")
        current_src = json.loads(src_json or "[]")
        if normalized_asset_type == "manuscript_pdf":
            manuscript_path = str(target_path)
            manuscript_status = validation.validation_status
            supplementary_status = None
            source_status = None
        elif normalized_asset_type == "supplementary":
            if str(target_path) not in current_supp:
                current_supp.append(str(target_path))
            manuscript_status = None
            supplementary_status = validation.validation_status
            source_status = None
        else:
            if str(target_path) not in current_src:
                current_src.append(str(target_path))
            manuscript_status = None
            supplementary_status = None
            source_status = validation.validation_status

        current_row = {
            "title": title,
            "readiness_status": "",
            "final_score": float(final_score or 0.0),
            "open_access_candidate": bool(oa_candidate),
            "asset_resolution_notes": resolution_notes or "",
        }
        existing = con.sql(
            """
            SELECT manuscript_asset_status, supplementary_asset_status, source_data_asset_status
            FROM literature.paper_asset_resolution
            WHERE paper_id = ?
            """,
            params=[paper_id],
        ).fetchone()
        manuscript_asset_status = manuscript_status or (existing[0] if existing else "not_ready_unknown")
        supplementary_asset_status = supplementary_status or (existing[1] if existing else "not_ready_unknown")
        source_data_asset_status = source_status or (existing[2] if existing else "not_ready_unknown")
        readiness_candidates = [validation.readiness_impact, manuscript_asset_status, supplementary_asset_status, source_data_asset_status]
        if "source_data_ready" in readiness_candidates and "supplementary_ready" in readiness_candidates:
            readiness_status = "multi_asset_ready"
        elif "source_data_ready" in readiness_candidates:
            readiness_status = "source_data_ready"
        elif "supplementary_ready" in readiness_candidates and "manuscript_binary_ready" in readiness_candidates:
            readiness_status = "multi_asset_ready"
        elif "supplementary_ready" in readiness_candidates:
            readiness_status = "supplementary_ready"
        elif "manuscript_binary_ready" in readiness_candidates:
            readiness_status = "manuscript_binary_ready"
        elif "review_only_low_yield" in readiness_candidates:
            readiness_status = "review_only_low_yield"
        elif "platform_heavy_not_assignment_ready" in readiness_candidates:
            readiness_status = "platform_heavy_not_assignment_ready"
        elif "blocked_manual_rescue" in readiness_candidates:
            readiness_status = "blocked_manual_rescue"
        else:
            readiness_status = "not_ready"
        current_row["readiness_status"] = readiness_status
        queue_partition, priority_score, partition_notes = _queue_partition_for_row(current_row)

        exists_resolution = con.sql("SELECT COUNT(*) FROM literature.paper_asset_resolution WHERE paper_id = ?", params=[paper_id]).fetchone()[0]
        if exists_resolution:
            con.execute(
                """
                UPDATE literature.paper_asset_resolution
                SET manuscript_asset_status = ?,
                    supplementary_asset_status = ?,
                    source_data_asset_status = ?,
                    manuscript_local_path = ?,
                    supplementary_local_paths_json = ?,
                    source_data_local_paths_json = ?,
                    readiness_status = ?,
                    asset_resolution_notes = ?
                WHERE paper_id = ?
                """,
                [
                    manuscript_asset_status,
                    supplementary_asset_status,
                    source_data_asset_status,
                    manuscript_path,
                    json.dumps(sorted(set(current_supp))),
                    json.dumps(sorted(set(current_src))),
                    readiness_status,
                    "; ".join(filter(None, [resolution_notes, notes, "manual_upload_via_streamlit"])),
                    paper_id,
                ],
            )
        else:
            con.execute(
                "INSERT INTO literature.paper_asset_resolution VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    paper_id,
                    canonical_url,
                    bool(doi),
                    manuscript_asset_status,
                    supplementary_asset_status,
                    source_data_asset_status,
                    manuscript_path,
                    json.dumps(sorted(set(current_supp))),
                    json.dumps(sorted(set(current_src))),
                    readiness_status,
                    "; ".join(filter(None, [notes, "manual_upload_via_streamlit"])),
                ],
            )
        con.execute(
            """
            DELETE FROM literature.queue_partition WHERE paper_id = ?
            """,
            [paper_id],
        )
        con.execute(
            "INSERT INTO literature.queue_partition VALUES (?, ?, ?, ?, ?, ?)",
            [paper_id, queue_partition, readiness_status, bool(oa_candidate), priority_score, partition_notes],
        )
        con.execute(
            """
            UPDATE literature.processing_queue
            SET asset_ready = ?, notes = ?
            WHERE paper_id = ?
            """,
            [readiness_status in {"manuscript_binary_ready", "supplementary_ready", "source_data_ready", "multi_asset_ready"}, "; ".join(filter(None, [notes, "manual_upload_via_streamlit"])), paper_id],
        )
        if con.sql("SELECT COUNT(*) FROM literature.blocked_assets WHERE paper_id = ?", params=[paper_id]).fetchone()[0]:
            con.execute(
                """
                UPDATE literature.blocked_assets
                SET manuscript_status = CASE WHEN ? = 'manuscript_pdf' THEN ? ELSE manuscript_status END,
                    supplementary_status = CASE WHEN ? = 'supplementary' THEN ? ELSE supplementary_status END,
                    source_data_status = CASE WHEN ? = 'source_data' THEN ? ELSE source_data_status END,
                    manual_upload_pending = FALSE,
                    resolved_manually = TRUE,
                    resolved_manually_at = CURRENT_TIMESTAMP,
                    resolved_manually_notes = ?,
                    notes = ?
                WHERE paper_id = ?
                """,
                [
                    normalized_asset_type,
                    validation.validation_status,
                    normalized_asset_type,
                    validation.validation_status,
                    normalized_asset_type,
                    validation.validation_status,
                    notes,
                    "; ".join(filter(None, [notes, "manual_upload_via_streamlit"])),
                    paper_id,
                ],
            )
        con.commit()
        con.execute(
            """
            UPDATE literature.candidate_papers
            SET already_in_local_corpus = TRUE
            WHERE paper_id = ?
            """,
            [paper_id],
        )
        con.commit()
        next_action = {
            "manuscript_binary_ready": "Inspect OA / Queue and run controlled ingestion when appropriate.",
            "supplementary_ready": "Inspect OA / Queue and confirm whether the supplementary asset is assignment-grade before ingestion.",
            "source_data_ready": "Inspect asset inventory and decide whether the source-data asset is sufficient for controlled extraction.",
            "multi_asset_ready": "Paper is fully ready for controlled ingestion.",
            "review_only_low_yield": "Keep available for context, but do not prioritize ingestion.",
            "platform_heavy_not_assignment_ready": "Keep asset linked, but inspect manually before any ingestion attempt.",
            "blocked_manual_rescue": "Another manual rescue asset is still needed.",
        }.get(readiness_status, "Refresh the paper panel and review the updated truth/readiness state.")
        return {
            "paper_id": paper_id,
            "saved_to": str(target_path),
            "validation_status": validation.validation_status,
            "readiness_status": readiness_status,
            "queue_partition": queue_partition,
            "next_action": next_action,
        }
    finally:
        con.close()


def _source_id_candidates_for_paper(paper_id: str) -> list[str]:
    return [
        f"src_oa_ready_{paper_id}_manuscript",
        f"src_ready_{paper_id}_manuscript",
        paper_id,
    ]


def _post_ingestion_state_refresh(con: duckdb.DuckDBPyConnection, paper_ids: list[str]) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    for paper_id in paper_ids:
        source_candidates = _source_id_candidates_for_paper(paper_id)
        placeholders = ", ".join(["?"] * len(source_candidates))
        evidence_rows = con.sql(
            f"""
            SELECT COUNT(*)
            FROM evidence.peak_assignment_evidence
            WHERE source_id IN ({placeholders})
               OR assignment_record_id LIKE ?
            """,
            params=source_candidates + [f"%{paper_id}%"],
        ).fetchone()[0]
        if evidence_rows > 0:
            con.execute(
                """
                UPDATE literature.candidate_papers
                SET already_processed_in_evidence = TRUE
                WHERE paper_id = ?
                """,
                [paper_id],
            )
            con.execute(
                """
                UPDATE literature.processing_queue
                SET queue_status = 'processed',
                    selected_for_ingestion = FALSE,
                    ingestion_attempted = TRUE,
                    extraction_row_count = ?,
                    notes = 'Processed via literature ops console.'
                WHERE paper_id = ?
                """,
                [int(evidence_rows), paper_id],
            )
            outcomes[paper_id] = "ingested"
        else:
            con.execute(
                """
                UPDATE literature.processing_queue
                SET queue_status = 'skipped_low_value',
                    selected_for_ingestion = FALSE,
                    ingestion_attempted = TRUE,
                    extraction_row_count = 0,
                    notes = 'No assignment-grade evidence yielded during controlled ingest.'
                WHERE paper_id = ?
                """,
                [paper_id],
            )
            outcomes[paper_id] = "low_yield_context_only"
    con.commit()
    return outcomes


def trigger_oa_ingestion(selected_paper_ids: list[str]) -> dict[str, object]:
    cmd = [sys.executable, str(RUN_OA_INGEST_SCRIPT), *selected_paper_ids]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    outcomes = {}
    impact_summaries = {}
    if result.returncode == 0:
        con = _connect(read_only=False)
        try:
            outcomes = _post_ingestion_state_refresh(con, selected_paper_ids)
        finally:
            con.close()
        impact_summaries = {paper_id: build_ingest_impact_summary(paper_id) for paper_id in selected_paper_ids}
    return {
        "command": " ".join(cmd),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "outcomes": outcomes,
        "impact_summaries": impact_summaries,
    }

def trigger_ready_ingestion(selected_paper_ids: list[str] | None = None) -> dict[str, object]:
    cmd = [sys.executable, str(RUN_READY_INGEST_SCRIPT), *(selected_paper_ids or [])]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    outcomes = {}
    impact_summaries = {}
    if result.returncode == 0 and selected_paper_ids:
        con = _connect(read_only=False)
        try:
            outcomes = _post_ingestion_state_refresh(con, selected_paper_ids)
        finally:
            con.close()
        impact_summaries = {paper_id: build_ingest_impact_summary(paper_id) for paper_id in selected_paper_ids}
    return {
        "command": " ".join(cmd),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "outcomes": outcomes,
        "impact_summaries": impact_summaries,
    }


def load_activity_log_df() -> pd.DataFrame:
    con = _connect(read_only=True)
    try:
        blocked_attempts = con.sql(
            """
            SELECT last_attempted_at AS event_time, paper_id, 'asset_resolution_attempt' AS operation_type,
                   blocker_type AS detail, title
            FROM literature.blocked_assets
            WHERE last_attempted_at IS NOT NULL
            """
        ).df()
        manual_uploads = con.sql(
            """
            SELECT resolved_manually_at AS event_time, paper_id, 'manual_upload' AS operation_type,
                   COALESCE(resolved_manually_notes, '') AS detail, title
            FROM literature.blocked_assets
            WHERE resolved_manually_at IS NOT NULL
            """
        ).df()
        ingestion_runs = con.sql(
            """
            SELECT NULL::TIMESTAMP AS event_time, paper_id, 'queue_state' AS operation_type,
                   queue_status || ' / selected=' || CAST(selected_for_ingestion AS VARCHAR) || ' / attempted=' || CAST(ingestion_attempted AS VARCHAR) AS detail,
                   COALESCE(c.title, paper_id) AS title
            FROM literature.processing_queue q
            LEFT JOIN literature.candidate_papers c USING (paper_id)
            """
        ).df()
        combined = pd.concat([blocked_attempts, manual_uploads, ingestion_runs], ignore_index=True)
        if combined.empty:
            return combined
        return combined.sort_values(["event_time", "paper_id"], ascending=[False, True], na_position="last").reset_index(drop=True)
    finally:
        con.close()
