from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import duckdb

from gaira.evidence_v1.constants import (
    DB_PATH,
    LITERATURE_BLOCKED_ASSET_REPORT_ROOT,
    LITERATURE_BLOCKED_ASSET_TABLES_ROOT,
    ensure_literature_blocked_asset_output_dirs,
)
from gaira.evidence_v1.schema import (
    initialize_schema,
    reset_literature_blocked_asset_tables,
)


HIGH_PRIORITY_THRESHOLD = 0.80


@dataclass
class BlockedPaper:
    paper_id: str
    title: str
    doi: str
    canonical_article_url: str
    manuscript_status: str
    supplementary_status: str
    source_data_status: str
    readiness_status: str
    asset_resolution_notes: str
    queue_status: str
    selected_for_ingestion: bool
    final_score: float
    decision_rationale: str
    condition_score: float
    comparison_score: float


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: object) -> float:
    if value is None:
        return 0.0
    return float(value)


def _fetch_blocked_papers(connection: duckdb.DuckDBPyConnection) -> list[BlockedPaper]:
    rows = connection.sql(
        """
        SELECT
            r.paper_id,
            c.title,
            COALESCE(c.doi, '') AS doi,
            COALESCE(r.canonical_article_url, '') AS canonical_article_url,
            COALESCE(r.manuscript_asset_status, '') AS manuscript_status,
            COALESCE(r.supplementary_asset_status, '') AS supplementary_status,
            COALESCE(r.source_data_asset_status, '') AS source_data_status,
            COALESCE(r.readiness_status, '') AS readiness_status,
            COALESCE(r.asset_resolution_notes, '') AS asset_resolution_notes,
            COALESCE(q.queue_status, '') AS queue_status,
            COALESCE(q.selected_for_ingestion, FALSE) AS selected_for_ingestion,
            COALESCE(t.final_score, 0.0) AS final_score,
            COALESCE(t.decision_rationale, '') AS decision_rationale,
            COALESCE(t.condition_relevance_score, 0.0) AS condition_score,
            COALESCE(t.comparison_structure_score, 0.0) AS comparison_score
        FROM literature.paper_asset_resolution r
        JOIN literature.candidate_papers c USING (paper_id)
        LEFT JOIN literature.processing_queue q USING (paper_id)
        LEFT JOIN literature.paper_triage t USING (paper_id)
        WHERE r.readiness_status = 'not_ready'
           OR r.manuscript_asset_status IN ('inaccessible', 'landing_page_only', 'failed')
           OR r.supplementary_asset_status IN ('inaccessible', 'landing_page_only', 'failed')
           OR r.source_data_asset_status IN ('inaccessible', 'landing_page_only', 'failed')
           OR (
                COALESCE(q.selected_for_ingestion, FALSE) = TRUE
                AND (
                    COALESCE(r.manuscript_asset_status, '') NOT IN ('resolved_downloaded', 'duplicate_existing')
                    OR COALESCE(r.supplementary_asset_status, '') IN ('landing_page_only', 'inaccessible')
                    OR COALESCE(r.source_data_asset_status, '') IN ('landing_page_only', 'inaccessible')
                )
           )
        ORDER BY t.final_score DESC, q.selected_for_ingestion DESC, r.paper_id
        """
    ).fetchall()
    return [
        BlockedPaper(
            paper_id=row[0],
            title=row[1],
            doi=row[2],
            canonical_article_url=row[3],
            manuscript_status=row[4],
            supplementary_status=row[5],
            source_data_status=row[6],
            readiness_status=row[7],
            asset_resolution_notes=row[8],
            queue_status=row[9],
            selected_for_ingestion=bool(row[10]),
            final_score=_safe_float(row[11]),
            decision_rationale=row[12],
            condition_score=_safe_float(row[13]),
            comparison_score=_safe_float(row[14]),
        )
        for row in rows
    ]


def _publisher_from_url(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    if "mdpi" in netloc:
        return "mdpi"
    if "elsevier" in netloc or "sciencedirect" in netloc or "linkinghub" in netloc:
        return "elsevier"
    if "optica" in netloc or "opg.optica" in netloc or "perfdrive" in netloc:
        return "optica"
    if "wiley" in netloc:
        return "wiley"
    if "frontiers" in netloc:
        return "frontiers"
    if "pmc" in netloc or "ncbi.nlm.nih.gov" in netloc:
        return "pmc"
    if "springer" in netloc:
        return "springer"
    if "rsc" in netloc:
        return "rsc"
    return netloc or "unknown"


def _blocker_from_paper(paper: BlockedPaper) -> tuple[str, str]:
    notes = paper.asset_resolution_notes.lower()
    url = paper.canonical_article_url.lower()
    manuscript = paper.manuscript_status.lower()
    supplementary = paper.supplementary_status.lower()
    source_data = paper.source_data_status.lower()

    if "perfdrive" in url or "radware" in url:
        return "anti_bot", "DOI resolution ended at an anti-bot validation page."
    if manuscript == "inaccessible":
        if "doi_redirect:inaccessible" in notes:
            if "mdpi" in url:
                return "publisher_paywall", "Publisher article page resolved but manuscript access was denied."
            return "inaccessible", "Canonical article page resolved but direct manuscript access failed."
        return "inaccessible", "Asset resolver could not access the manuscript."
    if manuscript == "landing_page_only" or supplementary == "landing_page_only" or source_data == "landing_page_only":
        return "landing_page_only", "Only landing-page links were available; no directly downloadable asset was exposed."
    if "pmc" in url and manuscript == "not_available":
        return "PMCHTML_only", "PMC article context was reachable, but no directly usable binary manuscript asset was resolved."
    if manuscript == "not_available" and "linkinghub.elsevier.com" in url:
        return "publisher_paywall", "Elsevier landing page resolved without an openly downloadable manuscript or SI asset."
    if supplementary == "not_available" and manuscript in {"resolved_downloaded", "duplicate_existing"}:
        return "SI_link_missing", "Primary manuscript is available, but no useful supplementary or source-data asset was exposed."
    if manuscript == "failed" or supplementary == "failed" or source_data == "failed":
        return "broken_pdf_link", "Asset URL existed but failed during retrieval."
    if manuscript == "duplicate_existing":
        return "duplicate_existing", "A local manuscript already exists; only supplementary/source-data rescue remains."
    return "unknown", "Blocked state does not yet map cleanly to a more specific category."


def _preferred_asset_type(paper: BlockedPaper, blocker_type: str) -> str:
    if paper.manuscript_status not in {"resolved_downloaded", "duplicate_existing"}:
        return "manuscript_pdf"
    if paper.supplementary_status not in {"resolved_downloaded", "duplicate_existing"}:
        return "supplementary"
    if paper.source_data_status not in {"resolved_downloaded", "duplicate_existing"}:
        return "source_data"
    if blocker_type == "duplicate_existing":
        return "supplementary"
    return "manuscript_pdf"


def _suggested_manual_action(paper: BlockedPaper, blocker_type: str, preferred_asset_type: str) -> str:
    if blocker_type == "publisher_paywall":
        return f"download {preferred_asset_type.replace('_', ' ')} through institutional or publisher access"
    if blocker_type == "anti_bot":
        return f"open the article manually in a browser, pass the anti-bot gate, then save the {preferred_asset_type.replace('_', ' ')} locally"
    if blocker_type == "landing_page_only":
        return f"inspect the article landing page and manually capture the exposed {preferred_asset_type.replace('_', ' ')} link or file"
    if blocker_type == "SI_link_missing":
        return "inspect publisher page manually for supporting information or source-data attachments and upload any useful files"
    if blocker_type == "PMCHTML_only":
        return "open the PMC article manually and save the manuscript PDF or supplementary attachment if the browser can resolve it"
    if blocker_type == "duplicate_existing":
        return "upload a locally saved supplementary/source-data file if available; manuscript already exists"
    if blocker_type == "broken_pdf_link":
        return f"retry the broken {preferred_asset_type.replace('_', ' ')} link manually and upload the recovered file"
    if blocker_type == "inaccessible":
        return f"attempt a manual browser download of the {preferred_asset_type.replace('_', ' ')} and upload it locally"
    return f"manually investigate and upload the missing {preferred_asset_type.replace('_', ' ')}"


def _local_upload_target_path(paper_id: str, preferred_asset_type: str) -> str:
    root = Path("/Volumes/SSD_Rad/GAIRA_DATA/raw/gaira_literature_asset_resolution_v1") / paper_id
    if preferred_asset_type == "manuscript_pdf":
        return str(root / "manuscript")
    if preferred_asset_type == "supplementary":
        return str(root / "supplementary")
    return str(root / "source_data")


def _priority_score(paper: BlockedPaper, blocker_type: str) -> float:
    base = paper.final_score * 100.0
    if paper.selected_for_ingestion:
        base += 15.0
    if blocker_type in {"landing_page_only", "SI_link_missing", "duplicate_existing"}:
        base += 10.0
    elif blocker_type in {"publisher_paywall", "PMCHTML_only"}:
        base += 6.0
    elif blocker_type == "anti_bot":
        base += 4.0
    if paper.manuscript_status in {"resolved_downloaded", "duplicate_existing"} and (
        paper.supplementary_status not in {"resolved_downloaded", "duplicate_existing"}
        or paper.source_data_status not in {"resolved_downloaded", "duplicate_existing"}
    ):
        base += 6.0
    return round(base, 3)


def _manual_recoverable(blocker_type: str) -> bool:
    return blocker_type in {
        "publisher_paywall",
        "landing_page_only",
        "anti_bot",
        "login_required",
        "broken_pdf_link",
        "SI_link_missing",
        "PMCHTML_only",
        "ambiguous_asset",
        "inaccessible",
        "duplicate_existing",
    }


def build_blocked_asset_registry() -> dict[str, int]:
    ensure_literature_blocked_asset_output_dirs()

    connection = duckdb.connect(str(DB_PATH))
    initialize_schema(connection)
    reset_literature_blocked_asset_tables(connection)

    blocked_papers = _fetch_blocked_papers(connection)
    last_attempted_at = datetime.now(timezone.utc).isoformat()

    registry_rows: list[dict] = []
    unresolved_rows: list[dict] = []
    manual_rows: list[dict] = []

    for paper in blocked_papers:
        blocker_type, blocker_detail = _blocker_from_paper(paper)
        preferred_asset_type = _preferred_asset_type(paper, blocker_type)
        suggested_manual_action = _suggested_manual_action(paper, blocker_type, preferred_asset_type)
        manual_download_needed = preferred_asset_type in {"manuscript_pdf", "supplementary", "source_data"}
        manual_upload_pending = _manual_recoverable(blocker_type)
        retry_recommended = blocker_type in {"landing_page_only", "PMCHTML_only", "broken_pdf_link"}
        priority_score = _priority_score(paper, blocker_type)
        row = {
            "blocked_asset_id": f"blocked_{paper.paper_id}",
            "paper_id": paper.paper_id,
            "title": paper.title,
            "doi": paper.doi,
            "publisher": _publisher_from_url(paper.canonical_article_url),
            "canonical_article_url": paper.canonical_article_url,
            "blocker_type": blocker_type,
            "blocker_detail": blocker_detail,
            "manuscript_status": paper.manuscript_status,
            "supplementary_status": paper.supplementary_status,
            "source_data_status": paper.source_data_status,
            "manual_download_needed": manual_download_needed,
            "manual_upload_pending": manual_upload_pending,
            "retry_recommended": retry_recommended,
            "suggested_manual_action": suggested_manual_action,
            "preferred_asset_type_to_rescue_first": preferred_asset_type,
            "local_upload_target_path": _local_upload_target_path(paper.paper_id, preferred_asset_type),
            "resolved_manually": False,
            "resolved_manually_at": None,
            "resolved_manually_notes": "",
            "priority_score": priority_score,
            "last_attempted_at": last_attempted_at,
            "notes": paper.asset_resolution_notes,
        }
        registry_rows.append(row)
        unresolved_rows.append(
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "doi": paper.doi,
                "final_score": paper.final_score,
                "queue_status": paper.queue_status,
                "selected_for_ingestion": paper.selected_for_ingestion,
                "blocker_type": blocker_type,
                "manuscript_status": paper.manuscript_status,
                "supplementary_status": paper.supplementary_status,
                "source_data_status": paper.source_data_status,
                "preferred_asset_type_to_rescue_first": preferred_asset_type,
                "suggested_manual_action": suggested_manual_action,
                "priority_score": priority_score,
            }
        )
        manual_rows.append(
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "publisher": _publisher_from_url(paper.canonical_article_url),
                "blocker_type": blocker_type,
                "preferred_asset_type_to_rescue_first": preferred_asset_type,
                "suggested_manual_action": suggested_manual_action,
                "local_upload_target_path": _local_upload_target_path(paper.paper_id, preferred_asset_type),
                "manual_download_needed": manual_download_needed,
                "manual_upload_pending": manual_upload_pending,
                "retry_recommended": retry_recommended,
                "priority_score": priority_score,
            }
        )

    if registry_rows:
        connection.executemany(
            """
            INSERT INTO literature.blocked_assets (
                blocked_asset_id, paper_id, title, doi, publisher, canonical_article_url, blocker_type,
                blocker_detail, manuscript_status, supplementary_status, source_data_status,
                manual_download_needed, manual_upload_pending, retry_recommended,
                suggested_manual_action, preferred_asset_type_to_rescue_first, local_upload_target_path,
                resolved_manually, resolved_manually_at, resolved_manually_notes, priority_score,
                last_attempted_at, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["blocked_asset_id"],
                    row["paper_id"],
                    row["title"],
                    row["doi"],
                    row["publisher"],
                    row["canonical_article_url"],
                    row["blocker_type"],
                    row["blocker_detail"],
                    row["manuscript_status"],
                    row["supplementary_status"],
                    row["source_data_status"],
                    row["manual_download_needed"],
                    row["manual_upload_pending"],
                    row["retry_recommended"],
                    row["suggested_manual_action"],
                    row["preferred_asset_type_to_rescue_first"],
                    row["local_upload_target_path"],
                    row["resolved_manually"],
                    row["resolved_manually_at"],
                    row["resolved_manually_notes"],
                    row["priority_score"],
                    row["last_attempted_at"],
                    row["notes"],
                )
                for row in registry_rows
            ],
        )

    blocker_summary = connection.sql(
        """
        SELECT blocker_type, COUNT(*) AS blocked_paper_count,
               SUM(CASE WHEN priority_score >= 90 THEN 1 ELSE 0 END) AS high_priority_blocked_count,
               SUM(CASE WHEN manual_upload_pending THEN 1 ELSE 0 END) AS likely_manually_recoverable_count
        FROM literature.blocked_assets
        GROUP BY 1
        ORDER BY blocked_paper_count DESC, blocker_type
        """
    ).df().to_dict("records")

    priority_summary = connection.sql(
        """
        SELECT paper_id, title, doi, publisher, blocker_type, preferred_asset_type_to_rescue_first,
               suggested_manual_action, priority_score
        FROM literature.blocked_assets
        ORDER BY priority_score DESC, paper_id
        """
    ).df().to_dict("records")

    _write_csv(
        LITERATURE_BLOCKED_ASSET_TABLES_ROOT / "blocked_asset_registry.csv",
        list(registry_rows[0].keys()) if registry_rows else [
            "blocked_asset_id", "paper_id", "title", "doi", "publisher", "canonical_article_url",
            "blocker_type", "blocker_detail", "manuscript_status", "supplementary_status",
            "source_data_status", "manual_download_needed", "manual_upload_pending",
            "retry_recommended", "suggested_manual_action", "preferred_asset_type_to_rescue_first",
            "local_upload_target_path", "resolved_manually", "resolved_manually_at",
            "resolved_manually_notes", "priority_score", "last_attempted_at", "notes"
        ],
        registry_rows,
    )
    _write_csv(
        LITERATURE_BLOCKED_ASSET_TABLES_ROOT / "blocked_asset_priority_summary.csv",
        list(priority_summary[0].keys()) if priority_summary else [
            "paper_id", "title", "doi", "publisher", "blocker_type",
            "preferred_asset_type_to_rescue_first", "suggested_manual_action", "priority_score"
        ],
        priority_summary,
    )
    _write_csv(
        LITERATURE_BLOCKED_ASSET_TABLES_ROOT / "unresolved_high_priority_papers.csv",
        list(unresolved_rows[0].keys()) if unresolved_rows else [
            "paper_id", "title", "doi", "final_score", "queue_status", "selected_for_ingestion",
            "blocker_type", "manuscript_status", "supplementary_status", "source_data_status",
            "preferred_asset_type_to_rescue_first", "suggested_manual_action", "priority_score"
        ],
        unresolved_rows,
    )
    _write_csv(
        LITERATURE_BLOCKED_ASSET_TABLES_ROOT / "blocker_type_summary.csv",
        list(blocker_summary[0].keys()) if blocker_summary else [
            "blocker_type", "blocked_paper_count", "high_priority_blocked_count",
            "likely_manually_recoverable_count"
        ],
        blocker_summary,
    )
    _write_csv(
        LITERATURE_BLOCKED_ASSET_TABLES_ROOT / "manual_rescue_recommendations.csv",
        list(manual_rows[0].keys()) if manual_rows else [
            "paper_id", "title", "publisher", "blocker_type", "preferred_asset_type_to_rescue_first",
            "suggested_manual_action", "local_upload_target_path", "manual_download_needed",
            "manual_upload_pending", "retry_recommended", "priority_score"
        ],
        manual_rows,
    )

    top_manual_targets = priority_summary[:5]
    assessment_lines = [
        "# Current State Assessment",
        "",
        f"- Blocked papers currently tracked: `{len(registry_rows)}`",
        f"- High-priority blocked papers: `{sum(1 for row in registry_rows if row['priority_score'] >= 90)}`",
        f"- Likely manually recoverable papers: `{sum(1 for row in registry_rows if row['manual_upload_pending'])}`",
        "",
        "Most common blockers:",
    ]
    for row in blocker_summary:
        assessment_lines.append(
            f"- `{row['blocker_type']}`: `{row['blocked_paper_count']}`"
        )
    assessment_lines.extend(
        [
            "",
            "Top manual rescue targets:",
        ]
    )
    for row in top_manual_targets:
        assessment_lines.append(
            f"- `{row['paper_id']}` `{row['title']}` -> {row['preferred_asset_type_to_rescue_first']} via {row['blocker_type']}"
        )
    assessment_lines.extend(
        [
            "",
            "The blocked registry is sufficient to drive later manual recovery in an internal app because it records the concrete blocker class, the first asset worth rescuing, and the exact local upload target path.",
        ]
    )
    (LITERATURE_BLOCKED_ASSET_REPORT_ROOT / "current_state_assessment.md").write_text(
        "\n".join(assessment_lines) + "\n"
    )

    implementation_lines = [
        "# Implementation Note",
        "",
        "This pass does not rerun discovery and does not ingest evidence.",
        "It materializes unresolved literature asset states into a persistent `literature.blocked_assets` registry.",
        "",
        "Registry logic:",
        "1. Read the current paper asset resolution state for the targeted acquisition subset.",
        "2. Keep unresolved papers plus selected papers with remaining asset blockers.",
        "3. Map observed resolution outcomes to explicit blocker classes without inventing success states.",
        "4. Add manual-rescue guidance and upload-target paths for later app-driven recovery.",
        "",
        "Priority is based on triage score, whether the paper was already selected for ingestion, and whether the blocker looks manually recoverable.",
    ]
    (LITERATURE_BLOCKED_ASSET_REPORT_ROOT / "implementation_note.md").write_text(
        "\n".join(implementation_lines) + "\n"
    )

    connection.close()
    return {
        "blocked_papers_tracked": len(registry_rows),
        "high_priority_blocked_papers": sum(1 for row in registry_rows if row["priority_score"] >= 90),
        "likely_manually_recoverable": sum(1 for row in registry_rows if row["manual_upload_pending"]),
    }
