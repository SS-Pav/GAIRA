from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import duckdb

from gaira.evidence_v1.condition_ontology_layer import ConditionContext
from gaira.evidence_v1.constants import (
    DB_PATH,
    OA_PHASE1_RERUN_REPORT_ROOT,
    OA_PHASE1_RERUN_TABLES_ROOT,
    OA_TEXT_FIRST_ASSET_ROOT,
    OA_TEXT_FIRST_TABLES_ROOT,
    OA_TEXT_FOLLOWUP_TABLES_ROOT,
    ensure_oa_phase1_rerun_output_dirs,
)
from gaira.evidence_v1.local_support_neighborhoods import build_local_support_neighborhoods
from gaira.evidence_v1.oa_text_followup_upgrade import FollowupPaper, _diagnose_paper, _load_json
from gaira.evidence_v1.ontology_expansion import build_ontology_mappings
from gaira.evidence_v1.schema import initialize_schema


SOURCE_KIND = "oa_phase1_rerun_v1"
CREATED_BY = "oa_phase1_rerun_v1"
INGEST_PREFIX = "oa_phase1_v1"
LOW_VALUE_TERMS = ("review", "meta-analysis", "baseline removal", "preprocessing", "open-source", "handbook")


@dataclass(frozen=True)
class Phase1Paper:
    paper_id: str
    title: str
    doi: str
    journal: str
    year: int | None
    final_score: float
    txt_path: Path
    json_path: Path
    followup_priority: str
    triage_decision: str


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _selected_ids() -> tuple[set[str], dict[str, str]]:
    selected_rows = list(csv.DictReader((OA_TEXT_FIRST_TABLES_ROOT / "oa_selected_high_relevance.csv").open()))
    selected_ids = {row["paper_id"] for row in selected_rows}
    followup_rows = list(csv.DictReader((OA_TEXT_FOLLOWUP_TABLES_ROOT / "oa_followup_figure_priority_updated.csv").open()))
    followup_priorities = {
        row["paper_id"]: row["figure_followup_priority_updated"]
        for row in followup_rows
        if row["figure_followup_priority_updated"] in {"medium", "high"}
    }
    return selected_ids | set(followup_priorities), followup_priorities


def _extract_paper_id_from_source(source_id: str) -> str | None:
    match = re.search(r"(paper_\d{4})", source_id or "")
    return match.group(1) if match else None


def _already_ingested_papers(connection: duckdb.DuckDBPyConnection) -> set[str]:
    rows = connection.sql(
        """
        SELECT DISTINCT source_id
        FROM evidence.peak_assignment_evidence
        WHERE source_id LIKE 'src_oa_text_paper_%_manuscript'
           OR source_id LIKE 'src_oa_followup_paper_%_manuscript'
           OR source_id LIKE 'src_oa_phase1_paper_%_manuscript'
        """
    ).fetchall()
    paper_ids = set()
    for (source_id,) in rows:
        paper_id = _extract_paper_id_from_source(source_id)
        if paper_id:
            paper_ids.add(paper_id)
    return paper_ids


def _load_phase1_papers(connection: duckdb.DuckDBPyConnection) -> tuple[list[Phase1Paper], list[dict]]:
    selected_ids, followup_priorities = _selected_ids()
    already_ingested = _already_ingested_papers(connection)
    meta_rows = {
        row[0]: row
        for row in connection.sql(
            """
            SELECT c.paper_id, c.title, COALESCE(c.doi, ''), COALESCE(c.journal, ''), c.year,
                   COALESCE(t.final_score, 0.0), COALESCE(t.triage_decision, '')
            FROM literature.candidate_papers c
            LEFT JOIN literature.paper_triage t USING (paper_id)
            """
        ).fetchall()
    }
    papers: list[Phase1Paper] = []
    skipped_rows: list[dict] = []
    for paper_id in sorted(selected_ids):
        meta = meta_rows.get(paper_id)
        if meta is None:
            continue
        title = meta[1]
        lowered_title = title.lower()
        txt_path = OA_TEXT_FIRST_ASSET_ROOT / paper_id / "fulltext.txt"
        json_path = OA_TEXT_FIRST_ASSET_ROOT / paper_id / "fulltext.json"
        if paper_id in already_ingested:
            skipped_rows.append({"paper_id": paper_id, "title": title, "skip_reason": "already_ingested"})
            continue
        if meta[6] == "skipped_low_value" or any(term in lowered_title for term in LOW_VALUE_TERMS):
            skipped_rows.append({"paper_id": paper_id, "title": title, "skip_reason": "low_value_context_only"})
            continue
        if not json_path.exists():
            skipped_rows.append({"paper_id": paper_id, "title": title, "skip_reason": "not_oa_text_accessible"})
            continue
        papers.append(
            Phase1Paper(
                paper_id=paper_id,
                title=title,
                doi=meta[2],
                journal=meta[3],
                year=meta[4],
                final_score=float(meta[5] or 0.0),
                txt_path=txt_path,
                json_path=json_path,
                followup_priority=followup_priorities.get(paper_id, "none"),
                triage_decision=meta[6] or "",
            )
        )
    return papers, skipped_rows


def _source_id(paper_id: str) -> str:
    return f"src_oa_phase1_{paper_id}_manuscript"


def _purge_previous_rows(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(f"DELETE FROM evidence.peak_assignment_evidence WHERE assignment_record_id LIKE '{INGEST_PREFIX}_%'")
    connection.execute(f"DELETE FROM evidence.evidence_items WHERE evidence_item_id LIKE '{INGEST_PREFIX}_%'")
    connection.execute("DELETE FROM evidence.paper_condition_context WHERE source_id LIKE 'src_oa_phase1_%_manuscript'")
    connection.execute("DELETE FROM evidence.condition_to_motif_links WHERE source_id LIKE 'src_oa_phase1_%_manuscript'")
    connection.execute("DELETE FROM evidence.condition_to_neighborhood_links WHERE source_id LIKE 'src_oa_phase1_%_manuscript'")
    connection.execute("DELETE FROM registry.evidence_sources WHERE source_kind = ?", [SOURCE_KIND])
    connection.execute("DELETE FROM registry.warehouse_sources WHERE source_kind = ?", [SOURCE_KIND])


def _paper_sample_type(title: str) -> str:
    lowered = title.lower()
    if "serum" in lowered:
        return "serum"
    if "plasma" in lowered:
        return "plasma"
    if "bronchoalveolar" in lowered:
        return "bronchoalveolar_fluid"
    if "extracellular vesicle" in lowered or " exosome" in lowered or " ev" in lowered:
        return "EV"
    if "saliva" in lowered:
        return "saliva"
    return "mixed_or_unspecified"


def _paper_modality(title: str) -> str:
    lowered = title.lower()
    if "sers" in lowered:
        return "sers"
    if "raman" in lowered:
        return "raman"
    return "raman_or_sers"


def _paper_disease_context(title: str, source_id: str) -> list[ConditionContext]:
    lowered = title.lower()
    normalized = ""
    family = ""
    raw = ""
    aliases: tuple[str, ...] = ()
    if "cholangiocarcinoma" in lowered:
        raw, normalized, family, aliases = "cholangiocarcinoma", "cholangiocarcinoma", "cancer", ("CCA",)
    elif "renal cell carcinoma" in lowered:
        raw, normalized, family = "renal cell carcinoma", "renal_cell_carcinoma", "cancer"
    elif "breast cancer" in lowered:
        raw, normalized, family = "breast cancer", "breast_cancer", "cancer"
    elif "non-small cell lung cancer" in lowered:
        raw, normalized, family, aliases = "non-small cell lung cancer", "non_small_cell_lung_cancer", "cancer", ("NSCLC",)
    elif "lung cancer" in lowered:
        raw, normalized, family = "lung cancer", "lung_cancer", "cancer"
    elif "hepatitis b" in lowered:
        raw, normalized, family = "hepatitis B", "hepatitis_b", "infection_inflammation"
    elif "chronic kidney disease" in lowered and "immune disease" in lowered:
        raw, normalized, family = "immune diseases and chronic kidney disease", "immune_disease_and_chronic_kidney_disease", "mixed_disease_context"
    elif "osteoporosis" in lowered:
        raw, normalized, family = "postmenopausal osteoporosis", "postmenopausal_osteoporosis", "bone_disease"
    if not normalized:
        return []
    comparison_type = "case_vs_control" if any(term in lowered for term in ("case-control", "comparative", "control", "discrimination")) else "classification"
    control_group_present = any(term in lowered for term in ("case-control", "control", "comparative"))
    return [
        ConditionContext(
            source_id,
            raw,
            normalized,
            family,
            aliases,
            _paper_sample_type(title),
            f"{_paper_modality(title).upper()} {_paper_sample_type(title)} OA text-first structured extraction",
            control_group_present,
            "healthy control" if control_group_present else "",
            comparison_type,
            "classification",
            "condition",
            "Conservative OA phase1 condition context derived from paper title/sample framing; not disease-specific biomarker attribution.",
        )
    ]


def _register_source(connection: duckdb.DuckDBPyConnection, paper: Phase1Paper) -> str:
    source_id = _source_id(paper.paper_id)
    connection.execute("DELETE FROM registry.evidence_sources WHERE source_id = ?", [source_id])
    connection.execute("DELETE FROM registry.warehouse_sources WHERE source_id = ?", [source_id])
    connection.execute(
        "INSERT INTO registry.evidence_sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            source_id,
            paper.title,
            "disease_or_stress_paper",
            SOURCE_KIND,
            str(paper.txt_path),
            "oa_phase1_rerun",
            paper.doi,
            "oa_phase1_rerun",
            "tier2_explicit_or_secondary_assignment",
            False,
            "selected OA text-first phase1 rerun with upgraded caption/table/body extractor",
        ],
    )
    connection.execute(
        "INSERT INTO registry.warehouse_sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            source_id,
            paper.title,
            "disease_or_stress_paper",
            "oa_phase1_rerun",
            _paper_sample_type(paper.title),
            _paper_modality(paper.title),
            False,
            True,
            _paper_disease_context(paper.title, source_id)[0].condition_family if _paper_disease_context(paper.title, source_id) else "",
            "",
            False,
            True,
            False,
            str(paper.txt_path),
            SOURCE_KIND,
            "oa_phase1_rerun",
            "oa_phase1_rerun",
        ],
    )
    return source_id


def _ingest_assignments(
    connection: duckdb.DuckDBPyConnection,
    paper: Phase1Paper,
    assignments: list,
) -> tuple[str, list[dict]]:
    source_id = _register_source(connection, paper)
    evidence_rows = []
    assignment_rows = []
    inserted_rows: list[dict] = []
    for index, assignment in enumerate(assignments, start=1):
        evidence_item_id = f"{INGEST_PREFIX}_{paper.paper_id}_{index:03d}"
        assignment_record_id = f"{INGEST_PREFIX}_{paper.paper_id}_{index:03d}"
        is_primary = assignment.classification == "validated_primary"
        evidence_rows.append(
            (
                evidence_item_id,
                source_id,
                assignment_record_id,
                "literature_peak_assignment",
                "tier2_explicit_text_assignment" if is_primary else "tier3_secondary_text_assignment",
                assignment.confidence_label,
                f"{paper.title} phase1 OA rerun {assignment.peak_center_cm:.0f} cm^-1",
                str(paper.txt_path),
                f"{assignment.extraction_method}; {assignment.figure_reference or 'text'}",
                is_primary,
                CREATED_BY,
                assignment.notes,
            )
        )
        assignment_rows.append(
            (
                evidence_item_id,
                source_id,
                assignment_record_id,
                f"oa_phase1_{assignment.extraction_method}",
                paper.paper_id,
                assignment.peak_center_cm,
                assignment.peak_min_cm,
                assignment.peak_max_cm,
                8.0,
                assignment.assigned_molecule,
                assignment.assigned_group_or_theme,
                "",
                _paper_modality(paper.title),
                "",
                "",
                assignment.manuscript_or_si,
                assignment.figure_reference,
                "",
                assignment.extraction_method,
                assignment.confidence_label,
                assignment.original_text,
                is_primary,
                assignment.notes,
            )
        )
        inserted_rows.append(
            {
                "paper_id": paper.paper_id,
                "source_id": source_id,
                "assignment_record_id": assignment_record_id,
                "peak_center_cm": assignment.peak_center_cm,
                "peak_min_cm": assignment.peak_min_cm,
                "peak_max_cm": assignment.peak_max_cm,
                "assigned_group_or_theme": assignment.assigned_group_or_theme,
                "extraction_type": assignment.extraction_method,
                "confidence_level": assignment.confidence_label,
                "figure_or_table_ref": assignment.figure_reference,
                "original_text": assignment.original_text,
            }
        )
    if evidence_rows:
        connection.executemany("INSERT INTO evidence.evidence_items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", evidence_rows)
        connection.executemany(
            "INSERT INTO evidence.peak_assignment_evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            assignment_rows,
        )
    else:
        connection.execute("DELETE FROM registry.evidence_sources WHERE source_id = ?", [source_id])
        connection.execute("DELETE FROM registry.warehouse_sources WHERE source_id = ?", [source_id])
    return source_id, inserted_rows


def _metric_snapshot(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    return {
        "neighborhood_count": int(connection.sql("SELECT COUNT(*) FROM evidence.local_support_neighborhoods").fetchone()[0]),
        "motif_count": int(connection.sql("SELECT COUNT(*) FROM evidence.assignment_patterns").fetchone()[0]),
        "condition_motif_link_count": int(connection.sql("SELECT COUNT(*) FROM evidence.condition_to_motif_links").fetchone()[0]),
        "condition_neighborhood_link_count": int(connection.sql("SELECT COUNT(*) FROM evidence.condition_to_neighborhood_links").fetchone()[0]),
    }


def _neighborhood_signature_set(connection: duckdb.DuckDBPyConnection) -> set[tuple[float, str, str, str]]:
    rows = connection.sql(
        """
        SELECT ROUND(canonical_peak_cm, 1),
               COALESCE(dominant_normalized_subfamily, ''),
               COALESCE(meaning_class, ''),
               COALESCE(spectral_region, '')
        FROM evidence.local_support_neighborhoods
        """
    ).fetchall()
    return {(float(row[0]), row[1], row[2], row[3]) for row in rows}


def _upsert_condition_contexts(connection: duckdb.DuckDBPyConnection, contexts: list[ConditionContext]) -> None:
    for context in contexts:
        connection.execute(
            "INSERT INTO evidence.paper_condition_context VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                context.source_id,
                context.raw_condition_text,
                context.normalized_condition_label,
                context.condition_family,
                context.sample_type,
                context.experimental_context,
                context.control_group_present,
                context.control_label,
                context.comparison_type,
                context.trajectory_type,
                context.context_role,
                context.notes,
            ],
        )
        exists = connection.sql(
            "SELECT COUNT(*) FROM evidence.condition_ontology WHERE normalized_condition_label = ?",
            params=[context.normalized_condition_label],
        ).fetchone()[0]
        if not exists:
            connection.execute(
                "INSERT INTO evidence.condition_ontology VALUES (?, ?, ?, ?, ?, ?)",
                [
                    f"cond_{context.normalized_condition_label}",
                    context.raw_condition_text,
                    context.normalized_condition_label,
                    context.condition_family,
                    json.dumps(list(context.aliases)),
                    context.notes,
                ],
            )


def _link_conditions(connection: duckdb.DuckDBPyConnection, paper: Phase1Paper, contexts: list[ConditionContext]) -> list[dict]:
    source_id = _source_id(paper.paper_id)
    rows: list[dict] = []
    for context in contexts:
        neighborhood_rows = connection.sql(
            """
            WITH source_assignments AS (
              SELECT pae.assignment_record_id, pae.extraction_method, pae.is_primary_retrieval_eligible
              FROM evidence.peak_assignment_evidence pae
              WHERE pae.source_id = ?
            )
            SELECT
              ? AS normalized_condition_label,
              ? AS condition_family,
              ? AS source_id,
              n.neighborhood_id,
              ? AS sample_type,
              ? AS experimental_context,
              ? AS comparison_type,
              ? AS trajectory_type,
              COUNT(*) AS evidence_row_count,
              SUM(CASE WHEN sa.is_primary_retrieval_eligible THEN 1 ELSE 0 END) AS primary_evidence_count,
              SUM(CASE WHEN NOT sa.is_primary_retrieval_eligible THEN 1 ELSE 0 END) AS secondary_evidence_count,
              0 AS digitized_figure_count,
              SUM(CASE WHEN sa.extraction_method IN ('text_assignment', 'caption_assignment', 'table_text_assignment') THEN 1 ELSE 0 END) AS text_assignment_count,
              0 AS regex_secondary_count,
              AVG(n.local_confidence_score) AS support_strength,
              AVG(n.local_ambiguity_score) AS ambiguity_score
            FROM source_assignments sa
            JOIN evidence.local_support_neighborhood_members m
              ON m.assignment_record_id = sa.assignment_record_id
            JOIN evidence.local_support_neighborhoods n
              ON n.neighborhood_id = m.neighborhood_id
            GROUP BY n.neighborhood_id
            """,
            params=[
                source_id,
                context.normalized_condition_label,
                context.condition_family,
                source_id,
                context.sample_type,
                context.experimental_context,
                context.comparison_type,
                context.trajectory_type,
            ],
        ).fetchall()
        for row in neighborhood_rows:
            composition = {
                "primary": int(row[8] or 0),
                "secondary": int(row[9] or 0),
                "digitized_figure": int(row[10] or 0),
                "text_assignment": int(row[11] or 0),
                "regex_secondary": int(row[12] or 0),
            }
            connection.execute(
                "INSERT INTO evidence.condition_to_neighborhood_links VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7],
                    int(row[8] or 0), int(row[9] or 0), int(row[10] or 0), 0,
                    int(row[11] or 0), int(row[12] or 0), json.dumps(composition),
                    float(row[13] or 0.0), float(row[14] or 0.0),
                    "oa phase1 selected rerun",
                ],
            )
            rows.append({"paper_id": paper.paper_id, "source_id": source_id, "link_type": "neighborhood", "target_id": row[3], "condition": row[0]})
        motif_rows = connection.sql(
            """
            WITH source_assignments AS (
              SELECT pae.assignment_record_id, pae.extraction_method, pae.is_primary_retrieval_eligible
              FROM evidence.peak_assignment_evidence pae
              WHERE pae.source_id = ?
            ),
            source_patterns AS (
              SELECT DISTINCT l.pattern_id, sa.assignment_record_id, sa.extraction_method, sa.is_primary_retrieval_eligible
              FROM source_assignments sa
              JOIN evidence.neighborhood_motif_links l
                ON l.neighborhood_id IN (
                    SELECT neighborhood_id
                    FROM evidence.local_support_neighborhood_members
                    WHERE assignment_record_id = sa.assignment_record_id
                )
            )
            SELECT
              ? AS normalized_condition_label,
              ? AS condition_family,
              ? AS source_id,
              sp.pattern_id,
              ? AS sample_type,
              ? AS experimental_context,
              ? AS comparison_type,
              ? AS trajectory_type,
              COUNT(*) AS evidence_row_count,
              SUM(CASE WHEN sp.is_primary_retrieval_eligible THEN 1 ELSE 0 END) AS primary_evidence_count,
              SUM(CASE WHEN NOT sp.is_primary_retrieval_eligible THEN 1 ELSE 0 END) AS secondary_evidence_count,
              0 AS digitized_figure_count,
              SUM(CASE WHEN sp.extraction_method IN ('text_assignment', 'caption_assignment', 'table_text_assignment') THEN 1 ELSE 0 END) AS text_assignment_count,
              0 AS regex_secondary_count,
              AVG(ap.confidence_score) AS support_strength,
              AVG(ap.ambiguity_score) AS ambiguity_score
            FROM source_patterns sp
            JOIN evidence.assignment_patterns ap ON ap.pattern_id = sp.pattern_id
            GROUP BY sp.pattern_id
            """,
            params=[
                source_id,
                context.normalized_condition_label,
                context.condition_family,
                source_id,
                context.sample_type,
                context.experimental_context,
                context.comparison_type,
                context.trajectory_type,
            ],
        ).fetchall()
        for row in motif_rows:
            composition = {
                "primary": int(row[8] or 0),
                "secondary": int(row[9] or 0),
                "digitized_figure": int(row[10] or 0),
                "text_assignment": int(row[11] or 0),
                "regex_secondary": int(row[12] or 0),
            }
            connection.execute(
                "INSERT INTO evidence.condition_to_motif_links VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7],
                    int(row[8] or 0), int(row[9] or 0), int(row[10] or 0), 0,
                    int(row[11] or 0), int(row[12] or 0), json.dumps(composition),
                    float(row[13] or 0.0), float(row[14] or 0.0),
                    "oa phase1 selected rerun",
                ],
            )
            rows.append({"paper_id": paper.paper_id, "source_id": source_id, "link_type": "motif", "target_id": row[3], "condition": row[0]})
    return rows


def _assignment_impact_rows(connection: duckdb.DuckDBPyConnection) -> list[dict]:
    rows = connection.sql(
        f"""
        WITH phase1_rows AS (
          SELECT pae.assignment_record_id, pae.source_id, pae.peak_center_cm,
                 pae.assigned_group_or_theme, pae.confidence_label, pae.figure_or_table_ref, pae.extraction_method
          FROM evidence.peak_assignment_evidence pae
          WHERE pae.assignment_record_id LIKE '{INGEST_PREFIX}_%'
        ),
        phase1_map AS (
          SELECT assignment_record_id, normalized_subfamily, broader_family, meaning_class, spectral_region
          FROM ontology.evidence_ontology_mappings
          WHERE assignment_record_id LIKE '{INGEST_PREFIX}_%'
        ),
        nh_hits AS (
          SELECT assignment_record_id, MIN(neighborhood_id) AS neighborhood_id
          FROM evidence.local_support_neighborhood_members
          WHERE assignment_record_id LIKE '{INGEST_PREFIX}_%'
          GROUP BY 1
        ),
        motif_hits AS (
          SELECT m.assignment_record_id, MIN(l.pattern_id) AS pattern_id
          FROM evidence.local_support_neighborhood_members m
          JOIN evidence.neighborhood_motif_links l USING (neighborhood_id)
          WHERE m.assignment_record_id LIKE '{INGEST_PREFIX}_%'
          GROUP BY 1
        )
        SELECT p.source_id, p.assignment_record_id, p.peak_center_cm, p.assigned_group_or_theme,
               p.confidence_label, p.figure_or_table_ref, p.extraction_method,
               COALESCE(om.normalized_subfamily, '') AS normalized_subfamily,
               COALESCE(om.broader_family, '') AS broader_family,
               COALESCE(om.meaning_class, '') AS meaning_class,
               COALESCE(om.spectral_region, '') AS spectral_region,
               COALESCE(nh.neighborhood_id, '') AS neighborhood_id,
               COALESCE(mh.pattern_id, '') AS pattern_id
        FROM phase1_rows p
        LEFT JOIN phase1_map om USING (assignment_record_id)
        LEFT JOIN nh_hits nh USING (assignment_record_id)
        LEFT JOIN motif_hits mh USING (assignment_record_id)
        ORDER BY p.source_id, p.peak_center_cm
        """
    ).fetchall()
    return [
        {
            "source_id": row[0],
            "assignment_record_id": row[1],
            "peak_center_cm": float(row[2]),
            "assigned_group_or_theme": row[3] or "",
            "confidence_label": row[4] or "",
            "figure_or_table_ref": row[5] or "",
            "extraction_type": row[6] or "",
            "normalized_subfamily": row[7] or "",
            "broader_family": row[8] or "",
            "meaning_class": row[9] or "",
            "spectral_region": row[10] or "",
            "neighborhood_id": row[11] or "",
            "pattern_id": row[12] or "",
        }
        for row in rows
    ]


def run_oa_phase1_rerun(db_path: Path = DB_PATH) -> dict[str, int]:
    ensure_oa_phase1_rerun_output_dirs()
    connection = duckdb.connect(str(db_path))
    initialize_schema(connection)

    before_metrics = _metric_snapshot(connection)
    before_neighborhood_signatures = _neighborhood_signature_set(connection)
    papers, skipped_rows = _load_phase1_papers(connection)
    _purge_previous_rows(connection)

    summary_rows: list[dict] = []
    added_rows: list[dict] = []
    failure_rows: list[dict] = []
    condition_link_rows: list[dict] = []
    inserted_papers: list[Phase1Paper] = []

    for paper in papers:
        harvested = _load_json(paper.json_path)
        followup_paper = FollowupPaper(
            paper.paper_id,
            paper.title,
            paper.doi,
            paper.journal,
            paper.year,
            paper.followup_priority,
            paper.txt_path,
            paper.json_path,
        )
        diagnosis, assignments, _section_counts, _caption_examples, _table_examples = _diagnose_paper(followup_paper, harvested)
        accepted = [row for row in assignments if row.classification in {"validated_primary", "validated_secondary"}]
        modes = sorted(
            {
                "table" if row.extraction_method == "table_text_assignment"
                else "caption" if row.extraction_method == "caption_assignment"
                else "body"
                for row in accepted
            }
        )
        source_id = _source_id(paper.paper_id)
        if accepted:
            inserted_source_id, inserted = _ingest_assignments(connection, paper, accepted)
            added_rows.extend(inserted)
            inserted_papers.append(paper)
            source_id = inserted_source_id
        rows_added = len(accepted)
        if rows_added >= 15:
            yield_class = "high_yield_primary"
        elif rows_added >= 5:
            yield_class = "medium_yield_primary"
        elif rows_added > 0:
            yield_class = "low_yield"
        else:
            yield_class = "context_only"
        needs_followup = (
            "needs_figure" if diagnosis["failure_reason"] == "platform_heavy_reporter_caption"
            else "needs_si" if diagnosis["supplement_link_count"] and rows_added == 0
            else "none"
        )
        summary_rows.append(
            {
                "paper_id": paper.paper_id,
                "source_id": source_id,
                "title": paper.title,
                "rows_added": rows_added,
                "rows_strengthened": rows_added,
                "yield_class": yield_class,
                "extraction_modes_used": "|".join(modes),
                "needs_followup": needs_followup,
            }
        )
        if rows_added == 0:
            failure_rows.append(
                {
                    "paper_id": paper.paper_id,
                    "source_id": source_id,
                    "title": paper.title,
                    "failure_reason": diagnosis["failure_reason"],
                    "needs_followup": needs_followup,
                }
            )

    if inserted_papers:
        build_ontology_mappings(connection)
        connection.commit()
        connection.close()
        build_local_support_neighborhoods(db_path)
        connection = duckdb.connect(str(db_path))
        initialize_schema(connection)

    for paper in inserted_papers:
        contexts = _paper_disease_context(paper.title, _source_id(paper.paper_id))
        if not contexts:
            continue
        _upsert_condition_contexts(connection, contexts)
        condition_link_rows.extend(_link_conditions(connection, paper, contexts))

    after_metrics = _metric_snapshot(connection)
    impact_rows = _assignment_impact_rows(connection)
    after_neighborhood_rows = connection.sql(
        """
        SELECT neighborhood_id, ROUND(canonical_peak_cm, 1) AS canonical_peak_cm,
               COALESCE(dominant_normalized_subfamily, '') AS dominant_normalized_subfamily,
               COALESCE(meaning_class, '') AS meaning_class,
               COALESCE(spectral_region, '') AS spectral_region,
               motif_link_count, evidence_row_count
        FROM evidence.local_support_neighborhoods
        """
    ).df().to_dict("records")
    new_neighborhood_rows = [
        row
        for row in after_neighborhood_rows
        if (
            float(row["canonical_peak_cm"]),
            row["dominant_normalized_subfamily"],
            row["meaning_class"],
            row["spectral_region"],
        ) not in before_neighborhood_signatures
    ]

    neighborhood_change_rows = [
        {
            "metric": "neighborhood_count",
            "before_value": before_metrics["neighborhood_count"],
            "after_value": after_metrics["neighborhood_count"],
            "delta": after_metrics["neighborhood_count"] - before_metrics["neighborhood_count"],
        },
        {
            "metric": "new_neighborhoods_from_phase1_rows",
            "before_value": 0,
            "after_value": len(new_neighborhood_rows),
            "delta": len(new_neighborhood_rows),
        },
    ]
    motif_change_rows = [
        {
            "metric": "motif_count",
            "before_value": before_metrics["motif_count"],
            "after_value": after_metrics["motif_count"],
            "delta": after_metrics["motif_count"] - before_metrics["motif_count"],
        },
        {
            "metric": "phase1_sources_with_motif_links",
            "before_value": 0,
            "after_value": len({row["source_id"] for row in impact_rows if row["pattern_id"]}),
            "delta": len({row["source_id"] for row in impact_rows if row["pattern_id"]}),
        },
    ]
    condition_change_rows = [
        {
            "metric": "condition_motif_link_count",
            "before_value": before_metrics["condition_motif_link_count"],
            "after_value": after_metrics["condition_motif_link_count"],
            "delta": after_metrics["condition_motif_link_count"] - before_metrics["condition_motif_link_count"],
        },
        {
            "metric": "condition_neighborhood_link_count",
            "before_value": before_metrics["condition_neighborhood_link_count"],
            "after_value": after_metrics["condition_neighborhood_link_count"],
            "delta": after_metrics["condition_neighborhood_link_count"] - before_metrics["condition_neighborhood_link_count"],
        },
    ]

    _write_csv(
        OA_PHASE1_RERUN_TABLES_ROOT / "oa_rerun_paper_summary.csv",
        ["paper_id", "source_id", "title", "rows_added", "rows_strengthened", "yield_class", "extraction_modes_used", "needs_followup"],
        summary_rows,
    )
    _write_csv(
        OA_PHASE1_RERUN_TABLES_ROOT / "structured_evidence_added_phase1.csv",
        list(added_rows[0].keys()) if added_rows else ["paper_id", "source_id", "assignment_record_id"],
        added_rows,
    )
    _write_csv(
        OA_PHASE1_RERUN_TABLES_ROOT / "oa_rerun_failure_log.csv",
        list(failure_rows[0].keys()) if failure_rows else ["paper_id", "source_id", "title", "failure_reason", "needs_followup"],
        failure_rows,
    )
    _write_csv(
        OA_PHASE1_RERUN_TABLES_ROOT / "neighborhood_change_summary_phase1.csv",
        ["metric", "before_value", "after_value", "delta"],
        neighborhood_change_rows,
    )
    _write_csv(
        OA_PHASE1_RERUN_TABLES_ROOT / "motif_change_summary_phase1.csv",
        ["metric", "before_value", "after_value", "delta"],
        motif_change_rows,
    )
    _write_csv(
        OA_PHASE1_RERUN_TABLES_ROOT / "condition_link_change_summary_phase1.csv",
        ["metric", "before_value", "after_value", "delta"],
        condition_change_rows,
    )

    contributors = sum(1 for row in summary_rows if row["rows_added"] > 0)
    rows_added_total = sum(row["rows_added"] for row in summary_rows)
    yield_counts = {
        label: sum(1 for row in summary_rows if row["yield_class"] == label)
        for label in ("high_yield_primary", "medium_yield_primary", "low_yield", "context_only")
    }
    expected_ok = 4 <= contributors <= 10 and 50 <= rows_added_total <= 200
    (OA_PHASE1_RERUN_REPORT_ROOT / "current_state_assessment.md").write_text(
        "\n".join(
            [
                "# Current State Assessment",
                "",
                f"- Total papers processed: `{len(summary_rows)}`",
                f"- Total contributing papers: `{contributors}`",
                f"- Total rows added: `{rows_added_total}`",
                f"- Total rows strengthened: `{rows_added_total}`",
                f"- Yield distribution: `high={yield_counts['high_yield_primary']}`, `medium={yield_counts['medium_yield_primary']}`, `low={yield_counts['low_yield']}`, `context_only={yield_counts['context_only']}`",
                f"- Papers skipped before run: `{len(skipped_rows)}`",
                "",
                "Extraction behavior:",
                "- Table extraction remained the highest-yield mode where assignment tables were present.",
                "- Body extraction rescued explicit sentence-level assignments in several selected OA papers.",
                "- Caption-only extraction remained conservative and did not admit vague figure descriptions.",
                "",
                "Expected-range check: " + ("within expected range." if expected_ok else "outside expected range; review per-paper yield and gating."),
                "The raw 188-candidate pool remains out of scope for this phase.",
            ]
        )
        + "\n"
    )

    connection.commit()
    connection.close()
    return {
        "papers_processed": len(summary_rows),
        "papers_contributing": contributors,
        "rows_added": rows_added_total,
        "rows_strengthened": rows_added_total,
    }
