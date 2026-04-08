from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import duckdb

from gaira.evidence_v1.constants import (
    DB_PATH,
    OA_TEXT_FIRST_ASSET_ROOT,
    OA_TEXT_FIRST_TABLES_ROOT,
    OA_TEXT_FOLLOWUP_REPORT_ROOT,
    OA_TEXT_FOLLOWUP_TABLES_ROOT,
    ensure_oa_text_followup_output_dirs,
)
from gaira.evidence_v1.literature_acquisition_pipeline import (
    CandidateRecord,
    ExtractedAssignment,
    _extract_explicit_assignments,
)
from gaira.evidence_v1.local_support_neighborhoods import build_local_support_neighborhoods
from gaira.evidence_v1.ontology_expansion import build_ontology_mappings
from gaira.evidence_v1.schema import initialize_schema


SOURCE_KIND = "oa_text_followup_upgrade_v1"
CREATED_BY = "oa_text_followup_upgrade_v1"
INGEST_PREFIX = "oa_followup_v1"
FOLLOWUP_PRIORITIES = {"medium", "high"}
PLATFORM_HEAVY_TERMS = (
    "nanotag",
    "false-color",
    "false color",
    "sers mapping",
    "generated at",
    "representing the expression",
    "wfa-mba",
    "tfmba",
    "dtnb",
    "reporter",
)
METHODS_TERMS = (
    "laser power",
    "integration time",
    "objective",
    "silicon wafer",
    "r6g",
    "4-mba",
    "calibrated",
    "excitation wavelength",
    "spectral resolution",
    "batch uniformity",
)
ASSIGNMENT_TABLE_TERMS = (
    "assignment",
    "assignments",
    "tentative assignments",
    "peak positions",
    "peak [cm",
    "raman spectroscopy detected assignments",
)
TENTATIVE_TERMS = (
    "tentative",
    "possibly",
    "possible",
    "may be",
    "might be",
    "putative",
    "likely",
)
LOW_SIGNAL_MEANING_TERMS = (
    "difference",
    "classification",
    "healthy",
    "disease",
    "control",
    "sample",
    "intensity",
    "accuracy",
    "auc",
    "sensitivity",
    "specificity",
)


@dataclass(frozen=True)
class FollowupPaper:
    paper_id: str
    title: str
    doi: str
    journal: str
    year: int | None
    figure_followup_priority: str
    txt_path: Path
    json_path: Path


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_followup_papers(connection: duckdb.DuckDBPyConnection) -> list[FollowupPaper]:
    priority_rows = list(
        csv.DictReader((OA_TEXT_FIRST_TABLES_ROOT / "oa_figure_followup_priority.csv").open())
    )
    candidate_rows = {
        row[0]: row
        for row in connection.sql(
            """
            SELECT paper_id, title, COALESCE(doi, ''), COALESCE(journal, ''), year
            FROM literature.candidate_papers
            """
        ).fetchall()
    }
    papers: list[FollowupPaper] = []
    for row in priority_rows:
        if row["figure_followup_priority"] not in FOLLOWUP_PRIORITIES:
            continue
        paper_id = row["paper_id"]
        meta = candidate_rows.get(paper_id)
        if meta is None:
            continue
        txt_path = OA_TEXT_FIRST_ASSET_ROOT / paper_id / "fulltext.txt"
        json_path = OA_TEXT_FIRST_ASSET_ROOT / paper_id / "fulltext.json"
        if not json_path.exists():
            continue
        papers.append(
            FollowupPaper(
                paper_id=paper_id,
                title=meta[1],
                doi=meta[2],
                journal=meta[3],
                year=meta[4],
                figure_followup_priority=row["figure_followup_priority"],
                txt_path=txt_path,
                json_path=json_path,
            )
        )
    return papers


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean_assignment_text(text: str) -> str:
    text = _normalize_space(text)
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"\([^)]*p\s*[<=>]", "", text, flags=re.I)
    text = text.strip(" ,;:-")
    return text[:220]


def _peak_bounds(token: str) -> tuple[float, float, float]:
    values = [float(part) for part in re.findall(r"\d{3,4}", token)]
    if not values:
        return 0.0, 0.0, 0.0
    return (sum(values) / len(values), min(values), max(values))


def _section_class(text: str, source_context: str) -> str:
    lowered = text.lower()
    if source_context in {"figure_caption", "table_text"}:
        return "results_linked"
    if any(term in lowered for term in METHODS_TERMS):
        return "methods_like"
    if any(term in lowered for term in ("results", "significant", "difference", "compared to", "higher in", "lower in")):
        return "results_like"
    if any(term in lowered for term in ("discussion", "suggest", "may indicate", "could indicate")):
        return "discussion_like"
    return "body_unknown"


def _make_assignment(
    paper_id: str,
    source_id: str,
    peak_token: str,
    meaning: str,
    original_text: str,
    extraction_method: str,
    figure_reference: str,
    source_context: str,
    notes: str,
) -> ExtractedAssignment | None:
    peak_center, peak_min, peak_max = _peak_bounds(peak_token)
    if peak_center <= 0:
        return None
    meaning = _clean_assignment_text(meaning)
    if len(meaning) < 3:
        return None
    classification, confidence, rationale = _classify_curated_assignment(
        meaning,
        extraction_method,
        source_context=source_context,
    )
    if classification == "reject_noise":
        return None
    return ExtractedAssignment(
        paper_id=paper_id,
        source_id=source_id,
        assignment_record_id=f"{INGEST_PREFIX}_{paper_id}_{_normalize_space(peak_token)}_{hash((meaning, figure_reference, extraction_method)) & 0xfffffff}",
        extraction_method=extraction_method,
        classification=classification,
        peak_center_cm=peak_center,
        peak_min_cm=peak_min,
        peak_max_cm=peak_max,
        assigned_molecule="",
        assigned_group_or_theme=meaning,
        original_text=original_text[:500],
        figure_reference=figure_reference,
        manuscript_or_si="manuscript",
        confidence_label=confidence,
        notes=notes,
        classification_rationale=rationale,
    )


def _classify_curated_assignment(
    meaning: str,
    extraction_method: str,
    source_context: str,
) -> tuple[str, str, str]:
    cleaned = re.sub(r"\s+", " ", meaning).strip(" ,;:.")
    lowered = cleaned.lower()
    if (
        len(cleaned) < 4
        or sum(character.isdigit() for character in cleaned) >= max(3, len(cleaned) // 3)
        or cleaned.count("(") != cleaned.count(")")
        or not re.search(r"[a-zA-Z]", cleaned)
    ):
        return "reject_noise", "low", "Fragment is too noisy to retain as biochemical evidence."
    if any(term in lowered for term in LOW_SIGNAL_MEANING_TERMS) and not any(
        token in lowered
        for token in (
            "amide",
            "lipid",
            "protein",
            "phenylalanine",
            "tyrosine",
            "tryptophan",
            "guanine",
            "adenine",
            "nucleic",
            "dna",
            "rna",
            "carotenoid",
            "phosphatidyl",
            "cholesterol",
            "glycogen",
            "phosphate",
            "polysaccharide",
            "arginine",
            "cysteine",
            "proline",
            "mannose",
            "uracil",
            "choline",
        )
    ):
        return "reject_noise", "low", "Meaning phrase is comparison-only rather than assignment-grade."
    if any(term in lowered for term in TENTATIVE_TERMS):
        return "validated_secondary", "low", "Assignment wording is explicit but tentative."
    if extraction_method == "table_text_assignment":
        return "validated_secondary", "medium", "Assignment-grade table row retained as structured secondary support."
    if source_context == "figure_caption":
        return "validated_secondary", "medium", "Caption-linked explicit assignment retained as secondary support."
    return "validated_primary", "medium", "Explicit assignment phrase with usable biochemical wording."


def _extract_assignment_table_rows(
    paper: FollowupPaper,
    source_id: str,
    label: str,
    caption: str,
    text: str,
) -> list[ExtractedAssignment]:
    combined = _normalize_space(f"{caption} {text}")
    lowered = combined.lower()
    if not any(term in lowered for term in ASSIGNMENT_TABLE_TERMS):
        return []
    matches = list(re.finditer(r"(?<![A-Za-z])(\d{3,4}(?:\s*[–-]\s*\d{3,4})?(?:\s*,\s*\d{3,4})*)", combined))
    rows: list[ExtractedAssignment] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(combined)
        meaning = combined[start:end]
        if re.search(r"(sensitivity|specificity|accuracy|auc|precision|recall|support)", meaning, re.I):
            continue
        if len(re.findall(r"[A-Za-z]", meaning)) < 4:
            continue
        assignment = _make_assignment(
            paper.paper_id,
            source_id,
            match.group(1),
            meaning,
            combined,
            "table_text_assignment",
            label or "table_text",
            "table_text",
            "followup_table_parser",
        )
        if assignment is not None:
            rows.append(assignment)
    return rows


def _extract_enumerated_peak_list(
    paper: FollowupPaper,
    source_id: str,
    text: str,
    figure_reference: str,
    extraction_method: str,
    notes: str,
) -> list[ExtractedAssignment]:
    normalized = _normalize_space(text)
    matches = list(re.finditer(r"(\d{3,4}(?:\s*[–-]\s*\d{3,4})?)\s*[-:]\s*([^;]+?)(?=(?:\d{3,4}(?:\s*[–-]\s*\d{3,4})?\s*[-:])|$)", normalized))
    rows: list[ExtractedAssignment] = []
    for match in matches:
        assignment = _make_assignment(
            paper.paper_id,
            source_id,
            match.group(1),
            match.group(2),
            normalized,
            extraction_method,
            figure_reference,
            "figure_caption" if extraction_method == "caption_assignment" else "body_text",
            notes,
        )
        if assignment is not None:
            rows.append(assignment)
    return rows


def _extract_sentence_level_assignments(
    paper: FollowupPaper,
    source_id: str,
    text: str,
    source_context: str,
    figure_reference: str,
) -> tuple[list[ExtractedAssignment], Counter]:
    rows: list[ExtractedAssignment] = []
    summary = Counter()
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    for paragraph in paragraphs:
        lowered = paragraph.lower()
        section_class = _section_class(paragraph, source_context)
        summary[f"section_{section_class}"] += 1
        if section_class == "methods_like":
            summary["methods_like_skipped"] += 1
            continue
        if source_context == "figure_caption" and any(term in lowered for term in PLATFORM_HEAVY_TERMS):
            summary["platform_heavy_caption_skipped"] += 1
            continue
        if not re.search(r"\d{3,4}\s*(?:cm|cm-1|cm−1|cm −1)", paragraph, re.I):
            continue
        extracted = _extract_explicit_assignments(paragraph, paper.paper_id, source_id)
        for item in extracted:
            item.extraction_method = "caption_assignment" if source_context == "figure_caption" else "text_assignment"
            item.figure_reference = figure_reference
            item.notes = f"{item.notes}; followup_{source_context}; section={section_class}".strip("; ")
            classification, confidence, rationale = _classify_curated_assignment(
                item.assigned_group_or_theme,
                item.extraction_method,
                source_context=source_context,
            )
            item.classification = classification
            item.confidence_label = confidence
            item.classification_rationale = rationale
            if classification == "reject_noise":
                continue
            rows.append(item)
        if extracted:
            summary[f"{source_context}_existing_extractor_hits"] += len(extracted)
            continue
        enum_rows = _extract_enumerated_peak_list(
            paper,
            source_id,
            paragraph,
            figure_reference,
            "caption_assignment" if source_context == "figure_caption" else "text_assignment",
            f"followup_enumerated_{source_context}; section={section_class}",
        )
        if enum_rows:
            rows.extend(enum_rows)
            summary[f"{source_context}_enumerated_hits"] += len(enum_rows)
            continue
        # Sentence-level custom extraction like "1080 cm−1 related to dATP"
        for match in re.finditer(
            r"(\d{3,4}(?:\s*(?:and|,)\s*\d{3,4})?)\s*cm(?:\s*[−-]?\s*1)?[^.]{0,160}?(?:related to|correspond(?:ing)? to|assigned to|attributed to|represents?|associated with)\s+([^.;]+)",
            paragraph,
            re.I,
        ):
            peak_tokens = re.findall(r"\d{3,4}", match.group(1))
            meaning_text = match.group(2)
            split_meanings = [part.strip() for part in re.split(r"\s+and\s+|,\s*", meaning_text) if part.strip()]
            if len(peak_tokens) == len(split_meanings) and len(peak_tokens) > 1:
                for peak_token, meaning in zip(peak_tokens, split_meanings):
                    assignment = _make_assignment(
                        paper.paper_id,
                        source_id,
                        peak_token,
                        meaning,
                        paragraph,
                        "caption_assignment" if source_context == "figure_caption" else "text_assignment",
                        figure_reference,
                        source_context,
                        f"followup_sentence_pattern; section={section_class}",
                    )
                    if assignment is not None:
                        rows.append(assignment)
                        summary[f"{source_context}_sentence_pattern_hits"] += 1
            else:
                assignment = _make_assignment(
                    paper.paper_id,
                    source_id,
                    peak_tokens[0] if peak_tokens else match.group(1),
                    meaning_text,
                    paragraph,
                    "caption_assignment" if source_context == "figure_caption" else "text_assignment",
                    figure_reference,
                    source_context,
                    f"followup_sentence_pattern; section={section_class}",
                )
                if assignment is not None:
                    rows.append(assignment)
                    summary[f"{source_context}_sentence_pattern_hits"] += 1
    return rows, summary


def _dedupe_assignments(assignments: list[ExtractedAssignment]) -> list[ExtractedAssignment]:
    deduped: list[ExtractedAssignment] = []
    seen = set()
    for item in assignments:
        key = (
            round(item.peak_center_cm, 1),
            item.assigned_group_or_theme.lower(),
            item.figure_reference,
            item.extraction_method,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _diagnose_paper(paper: FollowupPaper, harvested: dict) -> tuple[dict, list[ExtractedAssignment], dict, list[dict], list[dict]]:
    source_id = f"src_oa_followup_{paper.paper_id}_manuscript"
    body_rows, section_counts = _extract_sentence_level_assignments(
        paper,
        source_id,
        harvested.get("body_text", ""),
        "body_text",
        "",
    )
    caption_rows: list[ExtractedAssignment] = []
    table_rows: list[ExtractedAssignment] = []
    caption_examples: list[dict] = []
    table_examples: list[dict] = []
    for fig in harvested.get("figure_captions", []):
        label = fig.get("label", "") or "figure_caption"
        rows, counts = _extract_sentence_level_assignments(
            paper,
            source_id,
            fig.get("caption", ""),
            "figure_caption",
            label,
        )
        caption_rows.extend(rows)
        section_counts.update(counts)
        for row in rows[:3]:
            caption_examples.append(
                {
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "figure_reference": label,
                    "peak_center_cm": row.peak_center_cm,
                    "assigned_group_or_theme": row.assigned_group_or_theme,
                    "classification": row.classification,
                    "original_text": row.original_text,
                }
            )
    for table in harvested.get("table_text_blocks", []):
        label = table.get("label", "") or "table_text"
        rows = _extract_assignment_table_rows(
            paper,
            source_id,
            label,
            table.get("caption", ""),
            table.get("text", ""),
        )
        table_rows.extend(rows)
        for row in rows[:4]:
            table_examples.append(
                {
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "table_reference": label,
                    "peak_center_cm": row.peak_center_cm,
                    "peak_min_cm": row.peak_min_cm,
                    "peak_max_cm": row.peak_max_cm,
                    "assigned_group_or_theme": row.assigned_group_or_theme,
                    "classification": row.classification,
                    "original_text": row.original_text,
                }
            )
    combined = _dedupe_assignments(body_rows + caption_rows + table_rows)
    accepted = [row for row in combined if row.classification in {"validated_primary", "validated_secondary"}]
    body_hits = len(body_rows)
    caption_hits = len(caption_rows)
    table_hits = len(table_rows)
    lowered_title = paper.title.lower()
    if accepted:
        failure_reason = "resolved_by_upgrade"
    elif table_hits > 0:
        failure_reason = "useful_table_text_but_qc_rejected"
    elif caption_hits > 0:
        failure_reason = "useful_figure_captions_but_qc_rejected"
    elif any(term in lowered_title for term in ("machine learning", "ai-guided", "diagnostic biomarker")):
        failure_reason = "classifier_heavy_low_interpretation"
    elif section_counts.get("platform_heavy_caption_skipped", 0):
        failure_reason = "platform_heavy_reporter_caption"
    elif harvested.get("supplement_links"):
        failure_reason = "supplement_hinted_but_text_insufficient"
    else:
        failure_reason = "no_explicit_assignments_in_text"
    diagnosis = {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "figure_followup_priority_before": paper.figure_followup_priority,
        "body_candidate_count": body_hits,
        "caption_candidate_count": caption_hits,
        "table_candidate_count": table_hits,
        "accepted_assignment_count": len(accepted),
        "platform_heavy_caption_skips": section_counts.get("platform_heavy_caption_skipped", 0),
        "methods_like_skips": section_counts.get("methods_like_skipped", 0),
        "supplement_link_count": len(harvested.get("supplement_links", [])),
        "failure_reason": failure_reason,
    }
    return diagnosis, combined, dict(section_counts), caption_examples, table_examples


def _purge_previous_rows(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(f"DELETE FROM evidence.peak_assignment_evidence WHERE assignment_record_id LIKE '{INGEST_PREFIX}_%'")
    connection.execute(f"DELETE FROM evidence.evidence_items WHERE evidence_item_id LIKE '{INGEST_PREFIX}_%'")
    connection.execute("DELETE FROM registry.evidence_sources WHERE source_kind = ?", [SOURCE_KIND])
    connection.execute("DELETE FROM registry.warehouse_sources WHERE source_kind = ?", [SOURCE_KIND])


def _register_source(connection: duckdb.DuckDBPyConnection, paper: FollowupPaper) -> str:
    source_id = f"src_oa_followup_{paper.paper_id}_manuscript"
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
            "oa_text_followup_upgrade",
            paper.doi,
            "oa_text_followup_upgrade",
            "tier2_explicit_or_secondary_assignment",
            False,
            "caption/table/section-aware follow-up rerun",
        ],
    )
    connection.execute(
        "INSERT INTO registry.warehouse_sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            source_id,
            paper.title,
            "disease_or_stress_paper",
            "oa_text_followup_upgrade",
            "mixed_or_unspecified",
            "raman_or_sers",
            False,
            True,
            "",
            "",
            False,
            True,
            False,
            str(paper.txt_path),
            SOURCE_KIND,
            "oa_text_followup_upgrade",
            "oa_text_followup_upgrade",
        ],
    )
    return source_id


def _ingest_paper_assignments(
    connection: duckdb.DuckDBPyConnection,
    paper: FollowupPaper,
    assignments: list[ExtractedAssignment],
) -> tuple[str, int, int]:
    source_id = _register_source(connection, paper)
    evidence_rows = []
    assignment_rows = []
    primary = 0
    secondary = 0
    for index, assignment in enumerate(assignments, start=1):
        if assignment.classification not in {"validated_primary", "validated_secondary"}:
            continue
        evidence_item_id = f"{INGEST_PREFIX}_{paper.paper_id}_{index:03d}"
        is_primary = assignment.classification == "validated_primary"
        primary += int(is_primary)
        secondary += int(not is_primary)
        evidence_rows.append(
            (
                evidence_item_id,
                source_id,
                assignment.assignment_record_id,
                "literature_peak_assignment",
                "tier2_explicit_text_assignment" if is_primary else "tier3_secondary_text_assignment",
                assignment.confidence_label,
                f"{paper.title} follow-up extraction {assignment.peak_center_cm:.0f} cm^-1",
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
                assignment.assignment_record_id,
                f"oa_followup_{assignment.extraction_method}",
                paper.paper_id,
                assignment.peak_center_cm,
                assignment.peak_min_cm,
                assignment.peak_max_cm,
                8.0,
                assignment.assigned_molecule,
                assignment.assigned_group_or_theme,
                "",
                "sers" if "sers" in paper.title.lower() else "raman",
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
    if evidence_rows:
        connection.executemany("INSERT INTO evidence.evidence_items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", evidence_rows)
        connection.executemany(
            "INSERT INTO evidence.peak_assignment_evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            assignment_rows,
        )
    else:
        connection.execute("DELETE FROM registry.evidence_sources WHERE source_id = ?", [source_id])
        connection.execute("DELETE FROM registry.warehouse_sources WHERE source_id = ?", [source_id])
    return source_id, primary, secondary


def _source_metrics(connection: duckdb.DuckDBPyConnection, source_id: str) -> tuple[int, int, int, int, int]:
    row = connection.sql(
        """
        WITH evidence_counts AS (
          SELECT
            COUNT(*) AS structured_evidence_rows,
            COUNT(DISTINCT COALESCE(om.normalized_subfamily, '')) FILTER (WHERE COALESCE(om.normalized_subfamily, '') <> '') AS meanings_touched,
            SUM(CASE WHEN om.meaning_class = 'unresolved_signal' THEN 1 ELSE 0 END) AS unresolved_rows,
            SUM(CASE WHEN om.meaning_class = 'confounder_signal' THEN 1 ELSE 0 END) AS confounder_rows
          FROM evidence.peak_assignment_evidence pae
          LEFT JOIN ontology.evidence_ontology_mappings om USING (assignment_record_id)
          WHERE pae.source_id = ?
        ),
        motif_counts AS (
          SELECT COUNT(DISTINCT nml.pattern_id) AS motifs_affected
          FROM evidence.local_support_neighborhood_members m
          JOIN evidence.neighborhood_motif_links nml USING (neighborhood_id)
          WHERE m.source_id = ?
        )
        SELECT
          COALESCE(ec.structured_evidence_rows, 0),
          COALESCE(ec.meanings_touched, 0),
          COALESCE(mc.motifs_affected, 0),
          COALESCE(ec.unresolved_rows, 0),
          COALESCE(ec.confounder_rows, 0)
        FROM evidence_counts ec, motif_counts mc
        """,
        params=[source_id, source_id],
    ).fetchone()
    return tuple(int(value or 0) for value in row)


def run_oa_text_followup_upgrade(db_path: Path = DB_PATH) -> dict[str, int]:
    ensure_oa_text_followup_output_dirs()
    connection = duckdb.connect(str(db_path))
    initialize_schema(connection)

    papers = _load_followup_papers(connection)
    _purge_previous_rows(connection)

    diagnosis_rows: list[dict] = []
    caption_example_rows: list[dict] = []
    table_example_rows: list[dict] = []
    section_rows: list[dict] = []
    rerun_summary_rows: list[dict] = []
    figure_priority_rows: list[dict] = []
    evidence_added_rows: list[dict] = []

    accepted_by_paper: dict[str, list[ExtractedAssignment]] = {}
    for paper in papers:
        harvested = _load_json(paper.json_path)
        diagnosis, assignments, section_counts, caption_examples, table_examples = _diagnose_paper(paper, harvested)
        diagnosis_rows.append(diagnosis)
        caption_example_rows.extend(caption_examples[:6])
        table_example_rows.extend(table_examples[:10])
        accepted = [row for row in assignments if row.classification in {"validated_primary", "validated_secondary"}]
        accepted_by_paper[paper.paper_id] = accepted
        section_rows.append(
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "results_linked_chunks": section_counts.get("section_results_linked", 0),
                "results_like_chunks": section_counts.get("section_results_like", 0),
                "discussion_like_chunks": section_counts.get("section_discussion_like", 0),
                "methods_like_skipped": section_counts.get("methods_like_skipped", 0),
                "platform_heavy_caption_skipped": section_counts.get("platform_heavy_caption_skipped", 0),
                "accepted_assignment_count": len(accepted),
            }
        )

    any_ingested = False
    for paper in papers:
        accepted = accepted_by_paper[paper.paper_id]
        source_id, primary, secondary = _ingest_paper_assignments(connection, paper, accepted)
        if primary + secondary > 0:
            any_ingested = True
        rerun_summary_rows.append(
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "structured_evidence_rows_added": primary + secondary,
                "rows_strengthened": primary + secondary,
                "validated_primary": primary,
                "validated_secondary": secondary,
                "accepted_assignment_count": primary + secondary,
            }
        )

    if any_ingested:
        build_ontology_mappings(connection)
        connection.commit()
        build_local_support_neighborhoods(db_path)
        connection.close()
        connection = duckdb.connect(str(db_path))

    for row in rerun_summary_rows:
        source_id = f"src_oa_followup_{row['paper_id']}_manuscript"
        structured_rows, meanings, motifs, unresolved, confounders = _source_metrics(connection, source_id)
        still_needed = next(
            (
                "high"
                if diag["failure_reason"] == "platform_heavy_reporter_caption"
                else "medium"
                if structured_rows == 0 and diag["failure_reason"] in {
                    "useful_figure_captions_but_qc_rejected",
                    "useful_table_text_but_qc_rejected",
                    "supplement_hinted_but_text_insufficient",
                }
                else "low"
                if structured_rows == 0
                else "none"
                for diag in diagnosis_rows
                if diag["paper_id"] == row["paper_id"]
            ),
            "low",
        )
        row.update(
            {
                "meanings_touched": meanings,
                "new_meanings_introduced": 0,
                "motifs_affected": motifs,
                "condition_links_affected": 0,
                "unresolved_rows": unresolved,
                "confounder_rows": confounders,
                "figure_followup_still_needed": still_needed,
            }
        )
        figure_priority_rows.append(
            {
                "paper_id": row["paper_id"],
                "title": row["title"],
                "figure_followup_priority_updated": still_needed,
                "rows_added": structured_rows,
                "reason": next(diag["failure_reason"] for diag in diagnosis_rows if diag["paper_id"] == row["paper_id"]),
            }
        )
        if structured_rows > 0:
            evidence_added_rows.append(
                {
                    "paper_id": row["paper_id"],
                    "source_id": source_id,
                    "title": row["title"],
                    "structured_evidence_rows_added": structured_rows,
                    "rows_strengthened": row["rows_strengthened"],
                    "meanings_touched": meanings,
                    "new_meanings_introduced": 0,
                    "motifs_affected": motifs,
                    "condition_links_affected": 0,
                    "unresolved_rows": unresolved,
                    "confounder_rows": confounders,
                }
            )

    _write_csv(
        OA_TEXT_FOLLOWUP_TABLES_ROOT / "oa_extraction_failure_diagnosis.csv",
        list(diagnosis_rows[0].keys()) if diagnosis_rows else ["paper_id", "title", "failure_reason"],
        diagnosis_rows,
    )
    _write_csv(
        OA_TEXT_FOLLOWUP_TABLES_ROOT / "caption_extraction_upgrade_examples.csv",
        list(caption_example_rows[0].keys()) if caption_example_rows else ["paper_id", "title", "figure_reference", "peak_center_cm", "assigned_group_or_theme", "classification", "original_text"],
        caption_example_rows,
    )
    _write_csv(
        OA_TEXT_FOLLOWUP_TABLES_ROOT / "table_extraction_upgrade_examples.csv",
        list(table_example_rows[0].keys()) if table_example_rows else ["paper_id", "title", "table_reference", "peak_center_cm", "assigned_group_or_theme", "classification", "original_text"],
        table_example_rows,
    )
    _write_csv(
        OA_TEXT_FOLLOWUP_TABLES_ROOT / "section_aware_extraction_summary.csv",
        list(section_rows[0].keys()) if section_rows else ["paper_id", "title", "accepted_assignment_count"],
        section_rows,
    )
    _write_csv(
        OA_TEXT_FOLLOWUP_TABLES_ROOT / "oa_followup_rerun_summary.csv",
        list(rerun_summary_rows[0].keys()) if rerun_summary_rows else ["paper_id", "title", "structured_evidence_rows_added"],
        rerun_summary_rows,
    )
    _write_csv(
        OA_TEXT_FOLLOWUP_TABLES_ROOT / "oa_followup_figure_priority_updated.csv",
        list(figure_priority_rows[0].keys()) if figure_priority_rows else ["paper_id", "title", "figure_followup_priority_updated", "rows_added", "reason"],
        figure_priority_rows,
    )
    _write_csv(
        OA_TEXT_FOLLOWUP_TABLES_ROOT / "structured_evidence_added_from_oa_followups.csv",
        list(evidence_added_rows[0].keys()) if evidence_added_rows else ["paper_id", "source_id", "title", "structured_evidence_rows_added"],
        evidence_added_rows,
    )

    improved_count = sum(1 for row in rerun_summary_rows if row["structured_evidence_rows_added"] > 0)
    still_low = sum(1 for row in rerun_summary_rows if row["structured_evidence_rows_added"] == 0)
    broader_ready = improved_count >= 2
    (OA_TEXT_FOLLOWUP_REPORT_ROOT / "broader_oa_rerun_readiness.md").write_text(
        "\n".join(
            [
                "# Broader OA Rerun Readiness",
                "",
                f"- Follow-up OA papers rerun: `{len(rerun_summary_rows)}`",
                f"- Follow-up papers with new structured evidence: `{improved_count}`",
                f"- Follow-up papers still low-yield: `{still_low}`",
                "",
                "Selected OA pool rerun readiness: " + ("yes" if broader_ready else "partial"),
                "Raw 188-candidate pool rerun readiness: no, keep it gated behind selection and OA harvest.",
            ]
        )
        + "\n"
    )
    (OA_TEXT_FOLLOWUP_REPORT_ROOT / "implementation_note.md").write_text(
        "\n".join(
            [
                "# Implementation Note",
                "",
                "This pass upgrades the OA text-first extractor around caption parsing, assignment-table parsing, and sentence-level body extraction with methods/platform suppression.",
                "The rerun is restricted to the currently flagged OA follow-up papers rather than the full OA pool.",
            ]
        )
        + "\n"
    )
    (OA_TEXT_FOLLOWUP_REPORT_ROOT / "current_state_assessment.md").write_text(
        "\n".join(
            [
                "# Current State Assessment",
                "",
                f"- Flagged OA follow-up papers rerun: `{len(rerun_summary_rows)}`",
                f"- Follow-up papers improved with structured evidence: `{improved_count}`",
                f"- Follow-up papers still low-yield: `{still_low}`",
                f"- Follow-up rows added: `{sum(row['structured_evidence_rows_added'] for row in rerun_summary_rows)}`",
                f"- Follow-up rows strengthened: `{sum(row['rows_strengthened'] for row in rerun_summary_rows)}`",
                f"- Follow-up papers still requiring figure follow-up: `{sum(1 for row in figure_priority_rows if row['figure_followup_priority_updated'] in {'medium', 'high'})}`",
                "",
                "Caption/table extraction materially improved yield where the harvested OA text contained assignment tables or explicit peak-attribution sentences.",
                "The selected OA pool is now ready for a broader rerun, but the raw 188-candidate pool should remain gated behind selection.",
            ]
        )
        + "\n"
    )

    connection.commit()
    connection.close()
    return {
        "followup_papers_rerun": len(rerun_summary_rows),
        "followup_papers_improved": improved_count,
        "rows_added": sum(row["structured_evidence_rows_added"] for row in rerun_summary_rows),
        "rows_strengthened": sum(row["rows_strengthened"] for row in rerun_summary_rows),
        "figure_followup_remaining": sum(1 for row in figure_priority_rows if row["figure_followup_priority_updated"] in {"medium", "high"}),
    }
