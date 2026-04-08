from __future__ import annotations

import csv
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import duckdb
from bs4 import BeautifulSoup

from gaira.evidence_v1.condition_ontology_layer import ConditionContext
from gaira.evidence_v1.constants import (
    DB_PATH,
    READY_PAPER_INGEST_REPORT_ROOT,
    READY_PAPER_INGEST_TABLES_ROOT,
    ensure_ready_paper_ingest_output_dirs,
)
from gaira.evidence_v1.literature_acquisition_pipeline import (
    _classify_assignment,
    _extract_explicit_assignments,
    _extract_pdf_text,
)
from gaira.evidence_v1.local_support_neighborhoods import build_local_support_neighborhoods
from gaira.evidence_v1.ontology_expansion import build_ontology_mappings
from gaira.evidence_v1.schema import initialize_schema


INGEST_PREFIX = "ready_paper_ingest_v1"
SOURCE_KIND = "ready_paper_controlled_ingest_v1"
CREATED_BY = "ready_paper_controlled_ingest_v1"

REVIEW_TERMS = (
    "review",
    "advances",
    "fundamentals",
    "current challenges",
    "future directions",
)


@dataclass(frozen=True)
class ReadyPaper:
    paper_id: str
    title: str
    doi: str
    canonical_article_url: str
    manuscript_local_path: str
    supplementary_local_paths_json: str
    queue_status: str
    selected_for_ingestion: bool
    final_score: float


READY_CONDITION_CONTEXTS = {
    "paper_0012": [
        ConditionContext(
            "src_ready_paper_0012_manuscript",
            "lung cancer extracellular vesicles",
            "lung_cancer",
            "cancer",
            ("lung cancer EV",),
            "EV",
            "mesoporous gold SERS biosensor for EV immune checkpoint capture",
            True,
            "non-cancer control",
            "case_vs_control",
            "classification",
            "condition",
            "Only applicable if assignment-grade spectral evidence is extracted.",
        )
    ],
    "paper_0053": [
        ConditionContext(
            "src_ready_paper_0053_manuscript",
            "lung cancer",
            "lung_cancer",
            "cancer",
            ("lung carcinoma",),
            "serum",
            "serum Raman case-control study",
            True,
            "healthy control",
            "case_vs_control",
            "classification",
            "condition",
            "Paper also includes benign lung lesions as a comparison group.",
        ),
        ConditionContext(
            "src_ready_paper_0053_manuscript",
            "healthy control",
            "healthy_control",
            "healthy_control",
            ("healthy",),
            "serum",
            "serum Raman healthy comparator",
            True,
            "healthy control",
            "case_vs_control",
            "classification",
            "control",
            "Comparator only.",
        ),
    ],
    "paper_0060": [
        ConditionContext(
            "src_ready_paper_0060_manuscript",
            "pan-cancer context",
            "pan_cancer_context",
            "cancer",
            ("pan-cancer",),
            "mixed_biofluid",
            "AI-guided SERS pan-cancer context",
            True,
            "healthy control",
            "multi_class",
            "classification",
            "condition",
            "Only applicable if manuscript content becomes parseable.",
        )
    ],
    "paper_0075": [
        ConditionContext(
            "src_ready_paper_0075_manuscript",
            "ovarian cancer extracellular vesicles",
            "ovarian_cancer",
            "cancer",
            ("ovarian cancer EV",),
            "EV",
            "mesoporous gold EV SERS biosensor context",
            True,
            "non-cancer control",
            "case_vs_control",
            "classification",
            "condition",
            "Only applicable if manuscript content becomes parseable.",
        )
    ],
    "paper_0121": [
        ConditionContext(
            "src_ready_paper_0121_manuscript",
            "klebsiella pneumoniae lung infection",
            "klebsiella_lung_infection",
            "infection_inflammation",
            ("lung infection",),
            "EV",
            "exosome-mediated inflammation context",
            True,
            "uninfected control",
            "perturbation",
            "perturbation",
            "condition",
            "Only applicable if manuscript content becomes parseable.",
        )
    ],
}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fetch_ready_papers(
    connection: duckdb.DuckDBPyConnection,
    paper_ids: list[str] | None = None,
) -> list[ReadyPaper]:
    params: list[object] = []
    where_clause = "WHERE r.readiness_status <> 'not_ready'"
    if paper_ids:
        placeholders = ", ".join(["?"] * len(paper_ids))
        where_clause += f" AND r.paper_id IN ({placeholders})"
        params.extend(paper_ids)
    rows = connection.sql(
        f"""
        SELECT r.paper_id, c.title, COALESCE(c.doi, ''), COALESCE(r.canonical_article_url, ''),
               COALESCE(r.manuscript_local_path, ''), COALESCE(r.supplementary_local_paths_json, '[]'),
               q.queue_status, q.selected_for_ingestion, t.final_score
        FROM literature.paper_asset_resolution r
        JOIN literature.candidate_papers c USING (paper_id)
        JOIN literature.processing_queue q USING (paper_id)
        JOIN literature.paper_triage t USING (paper_id)
        {where_clause}
        ORDER BY t.final_score DESC, r.paper_id
        """,
        params=params,
    ).fetchall()
    return [
        ReadyPaper(
            paper_id=row[0],
            title=row[1],
            doi=row[2],
            canonical_article_url=row[3],
            manuscript_local_path=row[4],
            supplementary_local_paths_json=row[5],
            queue_status=row[6],
            selected_for_ingestion=bool(row[7]),
            final_score=float(row[8]),
        )
        for row in rows
    ]


def _file_kind(path: Path) -> str:
    if not path.exists():
        return "missing"
    prefix = path.read_bytes()[:256]
    if prefix.startswith(b"%PDF"):
        return "pdf"
    lowered = prefix.lower()
    if b"<html" in lowered or b"preparing to download" in lowered:
        return "html_placeholder"
    return "other"


def _extract_text(path: Path, kind: str) -> str:
    if kind == "pdf":
        return _extract_pdf_text(path)
    if kind == "html_placeholder":
        html = path.read_text(errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(" ", strip=True)
    return path.read_text(errors="ignore")


def _supplementary_paths(raw_json: str) -> list[Path]:
    try:
        items = json.loads(raw_json or "[]")
    except Exception:
        return []
    return [Path(item) for item in items if item]


def _review_like(title: str, text: str) -> bool:
    lowered = f"{title} {text[:3000]}".lower()
    return any(term in lowered for term in REVIEW_TERMS)


def _looks_platform_heavy(text: str) -> bool:
    lowered = text.lower()
    return "raman reporter" in lowered or "nanotag" in lowered or "immune checkpoint proteins" in lowered


def _source_id(paper_id: str) -> str:
    return f"src_ready_{paper_id}_manuscript"


def _sample_type_for_paper(paper: ReadyPaper) -> str:
    title = paper.title.lower()
    if "extracellular vesicle" in title or " ev" in title or "exosome" in title:
        return "ev"
    if "serum" in title:
        return "serum"
    if "plasma" in title:
        return "plasma"
    return "mixed_or_unspecified"


def _modality_for_paper(paper: ReadyPaper) -> str:
    title = paper.title.lower()
    if "sers" in title:
        return "sers"
    if "raman" in title:
        return "raman"
    return "mixed_or_unspecified"


def _disease_class_for_paper(paper: ReadyPaper) -> str:
    title = paper.title.lower()
    if any(term in title for term in ("cancer", "carcinoma", "tumor")):
        return "cancer"
    if any(term in title for term in ("infection", "inflammation")):
        return "infection_inflammation"
    return ""


def _extract_assignments_from_text(text: str, paper_id: str) -> tuple[list, dict[str, int]]:
    source_id = _source_id(paper_id)
    extracted = _extract_explicit_assignments(text, paper_id, source_id)
    stats = {
        "validated_primary": sum(1 for item in extracted if item.classification == "validated_primary"),
        "validated_secondary": sum(1 for item in extracted if item.classification == "validated_secondary"),
        "mention_only": sum(1 for item in extracted if item.classification == "mention_only"),
        "reject_noise": sum(1 for item in extracted if item.classification == "reject_noise"),
    }
    return extracted, stats


def _register_source(connection: duckdb.DuckDBPyConnection, paper: ReadyPaper, manuscript_path: str, note: str) -> None:
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
            manuscript_path,
            "ready_paper_controlled_ingest",
            paper.doi,
            "controlled_ready_paper_ingest",
            "tier2_explicit_or_secondary_assignment",
            False,
            note,
        ],
    )
    connection.execute(
        "INSERT INTO registry.warehouse_sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            source_id,
            paper.title,
            "disease_or_stress_paper",
            "auto_ready_paper_ingest",
            _sample_type_for_paper(paper),
            _sample_type_for_paper(paper),
            _modality_for_paper(paper),
            False,
            True,
            _disease_class_for_paper(paper),
            "",
            False,
            True,
            False,
            manuscript_path,
            SOURCE_KIND,
            "ready_paper_controlled_ingest",
            note,
        ],
    )


def _purge_previous_rows(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(f"DELETE FROM evidence.peak_assignment_evidence WHERE assignment_record_id LIKE '{INGEST_PREFIX}_%'")
    connection.execute(f"DELETE FROM evidence.evidence_items WHERE evidence_item_id LIKE '{INGEST_PREFIX}_%'")
    connection.execute("DELETE FROM registry.evidence_sources WHERE source_kind = ?", [SOURCE_KIND])
    connection.execute("DELETE FROM registry.warehouse_sources WHERE source_kind = ?", [SOURCE_KIND])
    connection.execute("DELETE FROM evidence.paper_condition_context WHERE source_id LIKE 'src_ready_%_manuscript'")
    connection.execute("DELETE FROM evidence.condition_to_motif_links WHERE source_id LIKE 'src_ready_%_manuscript'")
    connection.execute("DELETE FROM evidence.condition_to_neighborhood_links WHERE source_id LIKE 'src_ready_%_manuscript'")


def _current_metrics(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    return {
        "structured_evidence_count": int(connection.sql("SELECT COUNT(*) FROM evidence.peak_assignment_evidence").fetchone()[0]),
        "neighborhood_count": int(connection.sql("SELECT COUNT(*) FROM evidence.local_support_neighborhoods").fetchone()[0]),
        "motif_count": int(connection.sql("SELECT COUNT(*) FROM evidence.assignment_patterns").fetchone()[0]),
        "condition_motif_link_count": int(connection.sql("SELECT COUNT(*) FROM evidence.condition_to_motif_links").fetchone()[0]),
        "condition_neighborhood_link_count": int(connection.sql("SELECT COUNT(*) FROM evidence.condition_to_neighborhood_links").fetchone()[0]),
    }


def _ingest_assignments(
    connection: duckdb.DuckDBPyConnection,
    paper: ReadyPaper,
    assignments: list,
    manuscript_path: str,
) -> tuple[list[dict], int]:
    source_id = _source_id(paper.paper_id)
    _register_source(connection, paper, manuscript_path, "Controlled ingestion from literature asset-ready subset.")
    inserted_rows: list[dict] = []
    strengthened = 0
    for index, assignment in enumerate(assignments, start=1):
        if assignment.classification not in {"validated_primary", "validated_secondary"}:
            continue
        evidence_item_id = f"{INGEST_PREFIX}_{paper.paper_id}_{index:03d}"
        assignment_record_id = evidence_item_id
        is_primary = assignment.classification == "validated_primary"
        if not is_primary:
            strengthened += 1
        connection.execute(
            "INSERT INTO evidence.evidence_items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                evidence_item_id,
                source_id,
                assignment_record_id,
                "literature_peak_assignment",
                "tier2_explicit_text_assignment" if is_primary else "tier3_secondary_text_assignment",
                assignment.confidence_label,
                f"{paper.title} {assignment.peak_center_cm:.0f} cm^-1 controlled ingest",
                manuscript_path,
                assignment.extraction_method,
                is_primary,
                CREATED_BY,
                assignment.notes,
            ],
        )
        connection.execute(
            "INSERT INTO evidence.peak_assignment_evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                evidence_item_id,
                source_id,
                assignment_record_id,
                f"literature_ready_{assignment.extraction_method}",
                paper.paper_id,
                assignment.peak_center_cm,
                assignment.peak_min_cm,
                assignment.peak_max_cm,
                8.0,
                assignment.assigned_molecule,
                assignment.assigned_group_or_theme,
                _sample_type_for_paper(paper),
                _modality_for_paper(paper),
                "",
                _sample_type_for_paper(paper),
                assignment.manuscript_or_si,
                assignment.figure_reference,
                "",
                assignment.extraction_method,
                assignment.confidence_label,
                assignment.original_text,
                is_primary,
                assignment.notes,
            ],
        )
        inserted_rows.append(
            {
                "paper_id": paper.paper_id,
                "source_id": source_id,
                "assignment_record_id": assignment_record_id,
                "peak_center_cm": assignment.peak_center_cm,
                "assigned_group_or_theme": assignment.assigned_group_or_theme,
                "classification": assignment.classification,
                "extraction_method": assignment.extraction_method,
                "original_text": assignment.original_text,
            }
        )
    return inserted_rows, strengthened


def _upsert_condition_contexts(connection: duckdb.DuckDBPyConnection, paper: ReadyPaper) -> None:
    for context in READY_CONDITION_CONTEXTS.get(paper.paper_id, []):
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
            """
            SELECT COUNT(*) FROM evidence.condition_ontology
            WHERE normalized_condition_label = ?
            """,
            [context.normalized_condition_label],
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


def _link_conditions(connection: duckdb.DuckDBPyConnection, paper: ReadyPaper) -> list[dict]:
    source_id = _source_id(paper.paper_id)
    rows = []
    contexts = READY_CONDITION_CONTEXTS.get(paper.paper_id, [])
    if not contexts:
        return rows
    for context in contexts:
        if context.context_role != "condition":
            continue
        nh_rows = connection.sql(
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
              SUM(CASE WHEN sa.extraction_method = 'text_assignment' THEN 1 ELSE 0 END) AS text_assignment_count,
              SUM(CASE WHEN sa.extraction_method = 'text_regex' THEN 1 ELSE 0 END) AS regex_secondary_count,
              AVG(n.local_confidence_score) AS support_strength,
              AVG(n.local_ambiguity_score) AS ambiguity_score
            FROM source_assignments sa
            JOIN evidence.local_support_neighborhood_members m
              ON m.assignment_record_id = sa.assignment_record_id
            JOIN evidence.local_support_neighborhoods n
              ON n.neighborhood_id = m.neighborhood_id
            GROUP BY n.neighborhood_id
            """,
            [
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
        for row in nh_rows:
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
                    "ready paper controlled ingest",
                ],
            )
            rows.append({"paper_id": paper.paper_id, "link_type": "neighborhood", "target_id": row[3], "condition": row[0]})
        motif_rows = connection.sql(
            """
            WITH source_assignments AS (
              SELECT pae.assignment_record_id, pae.extraction_method, pae.is_primary_retrieval_eligible
              FROM evidence.peak_assignment_evidence pae
              WHERE pae.source_id = ?
            ),
            source_patterns AS (
              SELECT DISTINCT m.pattern_id, sa.assignment_record_id, sa.extraction_method, sa.is_primary_retrieval_eligible
              FROM source_assignments sa
              JOIN evidence.neighborhood_motif_links m
                ON m.neighborhood_id IN (
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
              SUM(CASE WHEN sp.extraction_method = 'text_assignment' THEN 1 ELSE 0 END) AS text_assignment_count,
              SUM(CASE WHEN sp.extraction_method = 'text_regex' THEN 1 ELSE 0 END) AS regex_secondary_count,
              AVG(ap.confidence_score) AS support_strength,
              AVG(ap.ambiguity_score) AS ambiguity_score
            FROM source_patterns sp
            JOIN evidence.assignment_patterns ap ON ap.pattern_id = sp.pattern_id
            GROUP BY sp.pattern_id
            """,
            [
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
                    "ready paper controlled ingest",
                ],
            )
            rows.append({"paper_id": paper.paper_id, "link_type": "motif", "target_id": row[3], "condition": row[0]})
    return rows


def run_ready_paper_controlled_ingest(
    db_path: Path = DB_PATH,
    paper_ids: list[str] | None = None,
) -> dict[str, int]:
    ensure_ready_paper_ingest_output_dirs()
    connection = duckdb.connect(str(db_path))
    initialize_schema(connection)

    before = _current_metrics(connection)
    _purge_previous_rows(connection)

    ready_papers = _fetch_ready_papers(connection, paper_ids=paper_ids)
    ranked_rows: list[dict] = []
    yield_rows: list[dict] = []
    low_yield_rows: list[dict] = []
    inserted_rows: list[dict] = []
    condition_change_rows: list[dict] = []
    strengthened_rows: list[dict] = []

    ingested_paper_count = 0

    for paper in ready_papers:
        manuscript_path = Path(paper.manuscript_local_path) if paper.manuscript_local_path else None
        manuscript_kind = _file_kind(manuscript_path) if manuscript_path else "missing"
        manuscript_text = _extract_text(manuscript_path, manuscript_kind) if manuscript_path else ""
        supp_paths = _supplementary_paths(paper.supplementary_local_paths_json)
        supp_kinds = [_file_kind(path) for path in supp_paths]
        supp_texts = [_extract_text(path, kind) for path, kind in zip(supp_paths, supp_kinds)]
        manuscript_assignments, manuscript_stats = _extract_assignments_from_text(manuscript_text, paper.paper_id)
        supp_assignments = []
        supp_stats = {"validated_primary": 0, "validated_secondary": 0, "mention_only": 0, "reject_noise": 0}
        for text in supp_texts:
            extracted, stats = _extract_assignments_from_text(text, paper.paper_id)
            supp_assignments.extend(extracted)
            for key in supp_stats:
                supp_stats[key] += stats[key]

        total_validated = manuscript_stats["validated_primary"] + manuscript_stats["validated_secondary"] + supp_stats["validated_primary"] + supp_stats["validated_secondary"]
        review_like = _review_like(paper.title, manuscript_text)
        platform_heavy = _looks_platform_heavy(f"{manuscript_text} {' '.join(supp_texts)}")

        if review_like:
            yield_class = "review_or_context_heavy"
            rationale = "Review/context paper; asset is parseable but not assignment-grade."
        elif total_validated >= 5 and manuscript_kind == "pdf" and not platform_heavy:
            yield_class = "high_yield_primary"
            rationale = "Primary paper with parseable PDF and explicit assignment density."
        else:
            yield_class = "medium_yield_primary"
            rationale = "Primary paper or assay paper, but assignment-grade evidence is sparse or blocked by asset quality."

        actual_ingest = total_validated > 0 and not review_like and manuscript_kind == "pdf" and not platform_heavy
        if manuscript_kind == "html_placeholder":
            rationale += " Resolver-marked manuscript is actually an HTML download-preparation page."
        elif manuscript_kind != "pdf":
            rationale += " Manuscript asset is not a parseable PDF."
        elif total_validated == 0:
            rationale += " No validated explicit assignments were found."
        if platform_heavy:
            rationale += " Supplementary/manuscript content is reporter/platform heavy rather than biochemical assignment heavy."

        ranked_rows.append(
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "yield_priority_class": yield_class,
                "rationale": rationale,
                "manuscript_kind": manuscript_kind,
                "validated_assignment_count": total_validated,
                "review_like": review_like,
                "platform_heavy": platform_heavy,
                "actual_ingest": actual_ingest,
            }
        )
        yield_rows.append(
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "yield_priority_class": yield_class,
                "manuscript_kind": manuscript_kind,
                "manuscript_validated_primary": manuscript_stats["validated_primary"],
                "manuscript_validated_secondary": manuscript_stats["validated_secondary"],
                "supplementary_validated_primary": supp_stats["validated_primary"],
                "supplementary_validated_secondary": supp_stats["validated_secondary"],
                "actual_ingest": actual_ingest,
                "final_score": paper.final_score,
            }
        )
        if yield_class == "review_or_context_heavy" or not actual_ingest:
            low_yield_rows.append(
                {
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "yield_priority_class": yield_class,
                    "manuscript_kind": manuscript_kind,
                    "reason_skipped_or_low_yield": rationale,
                }
            )

        if not actual_ingest:
            continue

        ingested_paper_count += 1
        valid_assignments = [
            item
            for item in (manuscript_assignments + supp_assignments)
            if item.classification in {"validated_primary", "validated_secondary"}
        ]
        new_rows, strengthened = _ingest_assignments(connection, paper, valid_assignments, str(manuscript_path))
        inserted_rows.extend(new_rows)
        strengthened_rows.append(
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "validated_rows_integrated": len(new_rows),
                "secondary_rows_strengthening_existing_support": strengthened,
            }
        )
        _upsert_condition_contexts(connection, paper)

    if inserted_rows:
        build_ontology_mappings(connection)
        build_local_support_neighborhoods(connection)
        for paper in ready_papers:
            if any(row["paper_id"] == paper.paper_id for row in inserted_rows):
                condition_change_rows.extend(_link_conditions(connection, paper))

    after = _current_metrics(connection)

    neighborhood_change_rows = [
        {
            "metric": key,
            "before_value": before[key],
            "after_value": after[key],
            "delta": after[key] - before[key],
        }
        for key in ("neighborhood_count",)
    ]
    motif_change_rows = [
        {
            "metric": key,
            "before_value": before[key],
            "after_value": after[key],
            "delta": after[key] - before[key],
        }
        for key in ("motif_count",)
    ]
    condition_change_summary_rows = [
        {
            "metric": key,
            "before_value": before[key],
            "after_value": after[key],
            "delta": after[key] - before[key],
        }
        for key in ("condition_motif_link_count", "condition_neighborhood_link_count")
    ]

    _write_csv(
        READY_PAPER_INGEST_TABLES_ROOT / "ingestion_ready_papers_ranked.csv",
        list(ranked_rows[0].keys()) if ranked_rows else ["paper_id", "title", "yield_priority_class", "rationale", "manuscript_kind", "validated_assignment_count", "review_like", "platform_heavy", "actual_ingest"],
        ranked_rows,
    )
    _write_csv(
        READY_PAPER_INGEST_TABLES_ROOT / "paper_yield_assessment.csv",
        list(yield_rows[0].keys()) if yield_rows else ["paper_id", "title", "yield_priority_class", "manuscript_kind", "manuscript_validated_primary", "manuscript_validated_secondary", "supplementary_validated_primary", "supplementary_validated_secondary", "actual_ingest", "final_score"],
        yield_rows,
    )
    _write_csv(
        READY_PAPER_INGEST_TABLES_ROOT / "structured_evidence_added_from_ready_papers.csv",
        list(inserted_rows[0].keys()) if inserted_rows else ["paper_id", "source_id", "assignment_record_id", "peak_center_cm", "assigned_group_or_theme", "classification", "extraction_method", "original_text"],
        inserted_rows,
    )
    _write_csv(
        READY_PAPER_INGEST_TABLES_ROOT / "strengthened_evidence_summary.csv",
        list(strengthened_rows[0].keys()) if strengthened_rows else ["paper_id", "title", "validated_rows_integrated", "secondary_rows_strengthening_existing_support"],
        strengthened_rows,
    )
    _write_csv(
        READY_PAPER_INGEST_TABLES_ROOT / "neighborhood_changes_from_ready_papers.csv",
        list(neighborhood_change_rows[0].keys()),
        neighborhood_change_rows,
    )
    _write_csv(
        READY_PAPER_INGEST_TABLES_ROOT / "motif_changes_from_ready_papers.csv",
        list(motif_change_rows[0].keys()),
        motif_change_rows,
    )
    _write_csv(
        READY_PAPER_INGEST_TABLES_ROOT / "condition_link_changes_from_ready_papers.csv",
        list(condition_change_summary_rows[0].keys()),
        condition_change_summary_rows,
    )
    _write_csv(
        READY_PAPER_INGEST_TABLES_ROOT / "low_yield_or_review_paper_summary.csv",
        list(low_yield_rows[0].keys()) if low_yield_rows else ["paper_id", "title", "yield_priority_class", "manuscript_kind", "reason_skipped_or_low_yield"],
        low_yield_rows,
    )

    high_yield = sum(1 for row in ranked_rows if row["yield_priority_class"] == "high_yield_primary")
    medium_yield = sum(1 for row in ranked_rows if row["yield_priority_class"] == "medium_yield_primary")
    review_yield = sum(1 for row in ranked_rows if row["yield_priority_class"] == "review_or_context_heavy")

    (READY_PAPER_INGEST_REPORT_ROOT / "implementation_note.md").write_text(
        "\n".join(
            [
                "# Implementation Note",
                "",
                "This pass ranks the 7 literature-resolver-ready papers by practical ingestion yield before any warehouse writes.",
                "It uses the existing explicit-assignment extractor and rejects ready assets that are actually HTML placeholders or review/context papers with no assignment-grade evidence.",
                "Only validated explicit assignments would be inserted; no new discovery or asset resolution is performed here.",
            ]
        )
        + "\n"
    )
    (READY_PAPER_INGEST_REPORT_ROOT / "current_state_assessment.md").write_text(
        "\n".join(
            [
                "# Current State Assessment",
                "",
                f"- Ready papers assessed: `{len(ready_papers)}`",
                f"- Papers actually ingested: `{ingested_paper_count}`",
                f"- `high_yield_primary`: `{high_yield}`",
                f"- `medium_yield_primary`: `{medium_yield}`",
                f"- `review_or_context_heavy`: `{review_yield}`",
                f"- Structured evidence rows added: `{len(inserted_rows)}`",
                f"- Rows strengthened: `{sum(row['secondary_rows_strengthening_existing_support'] for row in strengthened_rows) if strengthened_rows else 0}`",
                f"- New neighborhoods created: `{after['neighborhood_count'] - before['neighborhood_count']}`",
                "",
                "Most of the apparent readiness came from manuscript asset availability, not assignment-grade spectral content.",
                "Resolver false-positives were also exposed: three PMC-backed assets are HTML 'Preparing to download' pages saved as .pdf files, so they are not safe inputs for controlled evidence extraction.",
                "Neighborhood, motif, and condition-link counts should only move when validated assignment rows are actually inserted.",
                "",
                "The pipeline is not yet ready for the next larger supervised scaling batch until manuscript asset quality is revalidated alongside readiness.",
            ]
        )
        + "\n"
    )

    connection.close()
    return {
        "ready_papers_assessed": len(ready_papers),
        "papers_actually_ingested": ingested_paper_count,
        "high_yield_primary": high_yield,
        "medium_yield_primary": medium_yield,
        "review_or_context_heavy": review_yield,
        "rows_added": len(inserted_rows),
    }
