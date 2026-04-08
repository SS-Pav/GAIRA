from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from gaira.evidence_v1.constants import (
    PATTERN_REFINEMENT_QA_ROOT,
    PATTERN_REFINEMENT_REPORT_ROOT,
    PATTERN_REFINEMENT_TABLES_ROOT,
    ensure_pattern_refinement_output_dirs,
)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


def write_example_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def generate_pattern_granularity_qa(
    db_path: Path,
    before_counts: dict[str, float],
    after_counts: dict[str, float],
    before_examples: dict[str, dict],
    after_examples: dict[str, dict],
) -> dict[str, int]:
    ensure_pattern_refinement_output_dirs()
    with duckdb.connect(str(db_path), read_only=True) as connection:
        pattern_count_before_after = pd.DataFrame(
            [
                {"metric": "pattern_count", "before": before_counts["pattern_count"], "after": after_counts["pattern_count"]},
                {"metric": "avg_pattern_size", "before": before_counts["avg_pattern_size"], "after": after_counts["avg_pattern_size"]},
                {"metric": "avg_core_size", "before": before_counts["avg_core_size"], "after": after_counts["avg_core_size"]},
                {"metric": "same_family_multi_pattern_cases", "before": before_counts["same_family_multi_pattern_cases"], "after": after_counts["same_family_multi_pattern_cases"]},
            ]
        )
        pattern_count_before_after.to_csv(PATTERN_REFINEMENT_TABLES_ROOT / "pattern_count_before_after.csv", index=False)

        patterns_per_family = connection.sql(
            """
            SELECT normalized_family, COUNT(*) AS pattern_count
            FROM evidence.assignment_patterns
            GROUP BY normalized_family
            ORDER BY normalized_family
            """
        ).df()
        patterns_per_family.to_csv(PATTERN_REFINEMENT_TABLES_ROOT / "patterns_per_family.csv", index=False)

        size_distribution = connection.sql(
            """
            SELECT total_member_count, COUNT(*) AS pattern_count
            FROM evidence.assignment_patterns
            GROUP BY total_member_count
            ORDER BY total_member_count
            """
        ).df()
        size_distribution.to_csv(PATTERN_REFINEMENT_TABLES_ROOT / "pattern_size_distribution.csv", index=False)

        core_distribution = connection.sql(
            """
            SELECT core_member_count, COUNT(*) AS pattern_count
            FROM evidence.assignment_patterns
            GROUP BY core_member_count
            ORDER BY core_member_count
            """
        ).df()
        core_distribution.to_csv(PATTERN_REFINEMENT_TABLES_ROOT / "core_member_distribution.csv", index=False)

        coherence_summary = connection.sql(
            """
            SELECT
                pattern_id,
                pattern_label,
                normalized_family,
                total_member_count,
                core_member_count,
                ROUND(coherence_score, 6) AS coherence_score,
                ROUND(support_strength_score, 6) AS support_strength_score,
                ROUND(confidence_score, 6) AS confidence_score,
                ROUND(ambiguity_score, 6) AS ambiguity_score
            FROM evidence.assignment_patterns
            ORDER BY coherence_score DESC, confidence_score DESC, pattern_id
            """
        ).df()
        coherence_summary.to_csv(PATTERN_REFINEMENT_TABLES_ROOT / "pattern_coherence_summary.csv", index=False)

        separability_summary = connection.sql(
            """
            SELECT
                pattern_id,
                pattern_label,
                normalized_family,
                ROUND(separability_score, 6) AS separability_score,
                ROUND(coherence_score, 6) AS coherence_score,
                ROUND(ambiguity_score, 6) AS ambiguity_score
            FROM evidence.assignment_patterns
            ORDER BY normalized_family, separability_score DESC, pattern_id
            """
        ).df()
        separability_summary.to_csv(PATTERN_REFINEMENT_TABLES_ROOT / "pattern_separability_summary.csv", index=False)

        refined_examples = connection.sql(
            """
            WITH roles AS (
                SELECT
                    ap.pattern_id,
                    ap.pattern_label,
                    ap.normalized_family,
                    ap.coherence_score,
                    ap.separability_score,
                    ap.support_strength_score,
                    apm.member_role,
                    apm.canonical_peak_cm
                FROM evidence.assignment_patterns ap
                JOIN evidence.assignment_pattern_members apm
                  ON apm.pattern_id = ap.pattern_id
            )
            SELECT
                pattern_id,
                pattern_label,
                normalized_family,
                ROUND(coherence_score, 6) AS coherence_score,
                ROUND(separability_score, 6) AS separability_score,
                ROUND(support_strength_score, 6) AS support_strength_score,
                STRING_AGG(CASE WHEN member_role = 'core' THEN CAST(ROUND(canonical_peak_cm, 1) AS VARCHAR) END, ', ' ORDER BY canonical_peak_cm) AS core_members_cm,
                STRING_AGG(CASE WHEN member_role = 'supporting' THEN CAST(ROUND(canonical_peak_cm, 1) AS VARCHAR) END, ', ' ORDER BY canonical_peak_cm) AS supporting_members_cm
            FROM roles
            GROUP BY 1, 2, 3, 4, 5, 6
            ORDER BY coherence_score DESC, separability_score DESC, pattern_id
            LIMIT 20
            """
        ).df()
        refined_examples.to_csv(PATTERN_REFINEMENT_TABLES_ROOT / "pattern_examples_refined.csv", index=False)

        same_family_multi_pattern_examples = connection.sql(
            """
            WITH multi AS (
                SELECT normalized_family
                FROM evidence.assignment_patterns
                GROUP BY normalized_family
                HAVING COUNT(*) > 1
            )
            SELECT
                ap.pattern_id,
                ap.pattern_label,
                ap.normalized_family,
                ap.total_member_count,
                ap.core_member_count,
                ROUND(ap.coherence_score, 6) AS coherence_score,
                ROUND(ap.separability_score, 6) AS separability_score,
                ROUND(ap.support_strength_score, 6) AS support_strength_score
            FROM evidence.assignment_patterns ap
            JOIN multi
              ON multi.normalized_family = ap.normalized_family
            ORDER BY ap.normalized_family, ap.pattern_id
            """
        ).df()
        same_family_multi_pattern_examples.to_csv(PATTERN_REFINEMENT_TABLES_ROOT / "same_family_multi_pattern_examples.csv", index=False)

        total_patterns = int(connection.sql("SELECT COUNT(*) FROM evidence.assignment_patterns").fetchone()[0])
        avg_pattern_size = float(connection.sql("SELECT COALESCE(AVG(total_member_count), 0) FROM evidence.assignment_patterns").fetchone()[0])
        avg_core_size = float(connection.sql("SELECT COALESCE(AVG(core_member_count), 0) FROM evidence.assignment_patterns").fetchone()[0])
        same_family_cases = int(connection.sql("SELECT COUNT(*) FROM (SELECT normalized_family FROM evidence.assignment_patterns GROUP BY normalized_family HAVING COUNT(*) > 1)").fetchone()[0])

    retrieval_lines = ["# Retrieval Before/After Pattern Granularity", ""]
    for name in before_examples:
        before_payload = before_examples[name]
        after_payload = after_examples[name]
        before_titles = [item.get("pattern_label", item.get("title")) for item in before_payload.get("pattern_results", [])[:3]]
        after_titles = [item.get("pattern_label", item.get("title")) for item in after_payload.get("pattern_results", [])[:5]]
        retrieval_lines.extend(
            [
                f"## {name}",
                "",
                f"- Before top patterns: {before_titles}",
                f"- After top patterns: {after_titles}",
                "",
            ]
        )
    _write_text(PATTERN_REFINEMENT_QA_ROOT / "retrieval_before_after_patterns.md", "\n".join(retrieval_lines))

    implementation_lines = [
        "# Pattern Granularity Refinement Note",
        "",
        "- Replaced family-wide connected components with smaller seed-centered co-occurrence motifs.",
        "- Patterns now require at least 3 core peaks and are capped at 12 total members.",
        "- Added `coherence_score`, `separability_score`, and `support_strength_score` to the pattern layer.",
        "- Retrieval now scores patterns using completeness, coherence, support strength, separability, and ambiguity.",
        "",
    ]
    _write_text(PATTERN_REFINEMENT_REPORT_ROOT / "implementation_note.md", "\n".join(implementation_lines))

    discriminative = after_counts["same_family_multi_pattern_cases"] > before_counts["same_family_multi_pattern_cases"]
    assessment_lines = [
        "# Current State Assessment",
        "",
        f"- Patterns before: {before_counts['pattern_count']}",
        f"- Patterns after: {after_counts['pattern_count']}",
        f"- Average pattern size before: {before_counts['avg_pattern_size']:.3f}",
        f"- Average pattern size after: {after_counts['avg_pattern_size']:.3f}",
        f"- Average core size after: {after_counts['avg_core_size']:.3f}",
        f"- Same-family multi-pattern cases after: {after_counts['same_family_multi_pattern_cases']}",
        "",
        f"- More discriminative than before: {'yes' if discriminative else 'limited'}",
        "- Spectroscopic meaning improved when the same family now exposes multiple smaller motifs instead of one family-spanning bundle.",
        "- Remaining weakness: separability is still bounded by the quality of the underlying cluster layer and the reference-heavy support mix.",
        "",
    ]
    _write_text(PATTERN_REFINEMENT_REPORT_ROOT / "current_state_assessment.md", "\n".join(assessment_lines))

    return {
        "pattern_count": total_patterns,
        "same_family_multi_pattern_cases": same_family_cases,
        "avg_pattern_size": round(avg_pattern_size, 6),
        "avg_core_size": round(avg_core_size, 6),
    }
