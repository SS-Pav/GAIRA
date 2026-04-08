from __future__ import annotations

import csv
import json
from pathlib import Path

import duckdb

from gaira.evidence_v1.constants import (
    CLEANUP_QA_ROOT,
    DB_PATH,
    PATTERN_QA_ROOT,
    SOURCE_AUDIT_REPORT_ROOT,
    SOURCE_AUDIT_TABLES_ROOT,
    ensure_source_audit_output_dirs,
)
from gaira.evidence_v1.retrieval import PeakListRetrievalEngine


REFERENCE_DOMINANT_THRESHOLD = 0.60
REFERENCE_HEAVY_THRESHOLD = 0.75
MIXED_EVIDENCE_THRESHOLD = 0.20
NON_REFERENCE_MEANINGFUL_THRESHOLD = 0.18


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _markdown_table(columns: list[str], rows: list[list[object]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join("" if value is None else str(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _fraction(count: float, total: float) -> float:
    return 0.0 if total <= 0 else count / total


def _dominance_flags(curated: int, explicit: int, reference: int, mention: int) -> dict[str, bool | float]:
    total = curated + explicit + reference + mention
    curated_fraction = _fraction(curated, total)
    explicit_fraction = _fraction(explicit, total)
    reference_fraction = _fraction(reference, total)
    mention_fraction = _fraction(mention, total)
    sorted_fractions = sorted(
        [
            ("curated", curated_fraction),
            ("explicit", explicit_fraction),
            ("reference", reference_fraction),
            ("mention", mention_fraction),
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    top_label, top_fraction = sorted_fractions[0]
    second_fraction = sorted_fractions[1][1]
    is_reference_only = reference > 0 and curated == 0 and explicit == 0 and mention == 0
    is_reference_dominant = (
        reference_fraction >= REFERENCE_DOMINANT_THRESHOLD and reference_fraction > second_fraction
    )
    is_curated_dominant = top_label == "curated" and curated_fraction >= REFERENCE_DOMINANT_THRESHOLD
    is_explicit_dominant = top_label == "explicit" and explicit_fraction >= REFERENCE_DOMINANT_THRESHOLD
    nonzero_major = sum(1 for value in (curated_fraction, explicit_fraction, reference_fraction, mention_fraction) if value >= MIXED_EVIDENCE_THRESHOLD)
    is_mixed_evidence = not is_reference_only and not is_reference_dominant and nonzero_major >= 2
    is_reference_heavy_but_not_only = reference_fraction >= REFERENCE_HEAVY_THRESHOLD and not is_reference_only
    return {
        "curated_fraction": curated_fraction,
        "explicit_fraction": explicit_fraction,
        "reference_fraction": reference_fraction,
        "aligned_mention_fraction": mention_fraction,
        "is_reference_dominant": is_reference_dominant,
        "is_curated_dominant": is_curated_dominant,
        "is_explicit_dominant": is_explicit_dominant,
        "is_mixed_evidence": is_mixed_evidence,
        "is_reference_only": is_reference_only,
        "is_reference_heavy_but_not_only": is_reference_heavy_but_not_only,
    }


def _non_reference_proxy(curated: int, explicit: int, mention: int, nonref_source_diversity: int) -> float:
    proxy = (
        min(0.55, 0.18 * curated)
        + min(0.30, 0.10 * explicit)
        + min(0.04, 0.01 * mention)
        + min(0.11, 0.03 * nonref_source_diversity)
    )
    return min(1.0, proxy)


def main() -> None:
    ensure_source_audit_output_dirs()
    cluster_rows_out: list[dict] = []
    pattern_rows_out: list[dict] = []

    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        cluster_base = con.sql(
            """
            SELECT
                pmc.cluster_id,
                pmc.normalized_family,
                pmc.normalized_meaning_label,
                pmc.canonical_peak_cm,
                pmc.confidence_score,
                pmc.ambiguity_score,
                pms.support_kind,
                pms.source_id,
                pms.raw_label
            FROM evidence.peak_meaning_clusters pmc
            JOIN evidence.peak_meaning_support pms
              ON pmc.cluster_id = pms.cluster_id
            ORDER BY pmc.cluster_id
            """
        ).fetchall()
        cluster_map: dict[str, dict] = {}
        for cluster_id, family, label, peak, confidence, ambiguity, support_kind, source_id, raw_label in cluster_base:
            row = cluster_map.setdefault(
                cluster_id,
                {
                    "cluster_id": cluster_id,
                    "normalized_family": family,
                    "normalized_meaning_label": label,
                    "canonical_peak_cm": float(peak),
                    "original_confidence": float(confidence),
                    "original_ambiguity": float(ambiguity),
                    "curated_support_count": 0,
                    "explicit_source_backed_support_count": 0,
                    "reference_support_count": 0,
                    "aligned_mention_support_count": 0,
                    "curated_sources": set(),
                    "explicit_sources": set(),
                    "reference_entities": set(),
                    "all_source_ids": set(),
                },
            )
            if source_id:
                row["all_source_ids"].add(source_id)
            if support_kind == "curated_assignment":
                row["curated_support_count"] += 1
                if source_id:
                    row["curated_sources"].add(source_id)
            elif support_kind == "source_backed_assignment":
                row["explicit_source_backed_support_count"] += 1
                if source_id:
                    row["explicit_sources"].add(source_id)
            elif support_kind == "reference_peak":
                row["reference_support_count"] += 1
                if raw_label:
                    row["reference_entities"].add(raw_label.strip())
            elif support_kind == "aligned_mention":
                row["aligned_mention_support_count"] += 1
        for row in cluster_map.values():
            flags = _dominance_flags(
                row["curated_support_count"],
                row["explicit_source_backed_support_count"],
                row["reference_support_count"],
                row["aligned_mention_support_count"],
            )
            nonref_source_diversity = len(row["curated_sources"] | row["explicit_sources"])
            row.update(flags)
            row["unique_curated_sources"] = len(row["curated_sources"])
            row["unique_source_backed_papers"] = len(row["explicit_sources"])
            row["unique_reference_entities"] = len(row["reference_entities"])
            row["total_unique_source_ids"] = len(row["all_source_ids"])
            row["non_reference_support_score_proxy"] = _non_reference_proxy(
                row["curated_support_count"],
                row["explicit_source_backed_support_count"],
                row["aligned_mention_support_count"],
                nonref_source_diversity,
            )
            row["reference_contribution_share"] = row["reference_fraction"]
            row["meaningful_without_reference"] = row["non_reference_support_score_proxy"] >= NON_REFERENCE_MEANINGFUL_THRESHOLD
            cluster_rows_out.append(row)

        pattern_base = con.sql(
            """
            SELECT
                ap.pattern_id,
                ap.normalized_family,
                ap.pattern_label,
                ap.confidence_score,
                ap.ambiguity_score,
                apm.cluster_id
            FROM evidence.assignment_patterns ap
            JOIN evidence.assignment_pattern_members apm
              ON ap.pattern_id = apm.pattern_id
            ORDER BY ap.pattern_id, apm.cluster_id
            """
        ).fetchall()
        pattern_map: dict[str, dict] = {}
        for pattern_id, family, label, confidence, ambiguity, cluster_id in pattern_base:
            pattern = pattern_map.setdefault(
                pattern_id,
                {
                    "pattern_id": pattern_id,
                    "normalized_family": family,
                    "pattern_label": label,
                    "original_confidence": float(confidence),
                    "original_ambiguity": float(ambiguity),
                    "cluster_ids": [],
                    "curated_support_count": 0,
                    "explicit_source_backed_support_count": 0,
                    "reference_support_count": 0,
                    "aligned_mention_support_count": 0,
                    "curated_sources": set(),
                    "explicit_sources": set(),
                    "reference_entities": set(),
                    "all_source_ids": set(),
                },
            )
            pattern["cluster_ids"].append(cluster_id)
        for pattern in pattern_map.values():
            for cluster_id in pattern["cluster_ids"]:
                cluster = cluster_map[cluster_id]
                pattern["curated_support_count"] += cluster["curated_support_count"]
                pattern["explicit_source_backed_support_count"] += cluster["explicit_source_backed_support_count"]
                pattern["reference_support_count"] += cluster["reference_support_count"]
                pattern["aligned_mention_support_count"] += cluster["aligned_mention_support_count"]
                pattern["curated_sources"].update(cluster["curated_sources"])
                pattern["explicit_sources"].update(cluster["explicit_sources"])
                pattern["reference_entities"].update(cluster["reference_entities"])
                pattern["all_source_ids"].update(cluster["all_source_ids"])
            flags = _dominance_flags(
                pattern["curated_support_count"],
                pattern["explicit_source_backed_support_count"],
                pattern["reference_support_count"],
                pattern["aligned_mention_support_count"],
            )
            nonref_source_diversity = len(pattern["curated_sources"] | pattern["explicit_sources"])
            pattern.update(flags)
            pattern["unique_curated_sources"] = len(pattern["curated_sources"])
            pattern["unique_source_backed_papers"] = len(pattern["explicit_sources"])
            pattern["unique_reference_entities"] = len(pattern["reference_entities"])
            pattern["total_unique_source_ids"] = len(pattern["all_source_ids"])
            pattern["non_reference_support_score_proxy"] = _non_reference_proxy(
                pattern["curated_support_count"],
                pattern["explicit_source_backed_support_count"],
                pattern["aligned_mention_support_count"],
                nonref_source_diversity,
            )
            pattern["reference_contribution_share"] = pattern["reference_fraction"]
            pattern["meaningful_without_reference"] = pattern["non_reference_support_score_proxy"] >= NON_REFERENCE_MEANINGFUL_THRESHOLD
            pattern_rows_out.append(pattern)

    cluster_csv = SOURCE_AUDIT_TABLES_ROOT / "cluster_source_composition.csv"
    pattern_csv = SOURCE_AUDIT_TABLES_ROOT / "pattern_source_composition.csv"
    family_csv = SOURCE_AUDIT_TABLES_ROOT / "family_source_composition_summary.csv"
    dominance_csv = SOURCE_AUDIT_TABLES_ROOT / "reference_dominance_summary.csv"
    nonref_csv = SOURCE_AUDIT_TABLES_ROOT / "non_reference_support_audit.csv"

    cluster_fields = [
        "cluster_id",
        "normalized_family",
        "normalized_meaning_label",
        "canonical_peak_cm",
        "curated_support_count",
        "explicit_source_backed_support_count",
        "reference_support_count",
        "aligned_mention_support_count",
        "curated_fraction",
        "explicit_fraction",
        "reference_fraction",
        "aligned_mention_fraction",
        "unique_curated_sources",
        "unique_source_backed_papers",
        "unique_reference_entities",
        "total_unique_source_ids",
        "is_reference_dominant",
        "is_curated_dominant",
        "is_explicit_dominant",
        "is_mixed_evidence",
        "is_reference_only",
        "is_reference_heavy_but_not_only",
        "original_confidence",
        "original_ambiguity",
        "non_reference_support_score_proxy",
        "reference_contribution_share",
        "meaningful_without_reference",
    ]
    with cluster_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cluster_fields)
        writer.writeheader()
        for row in sorted(cluster_rows_out, key=lambda item: (item["normalized_family"], item["canonical_peak_cm"], item["cluster_id"])):
            writer.writerow({field: row[field] for field in cluster_fields})

    pattern_fields = [
        "pattern_id",
        "normalized_family",
        "pattern_label",
        "curated_support_count",
        "explicit_source_backed_support_count",
        "reference_support_count",
        "aligned_mention_support_count",
        "curated_fraction",
        "explicit_fraction",
        "reference_fraction",
        "aligned_mention_fraction",
        "unique_curated_sources",
        "unique_source_backed_papers",
        "unique_reference_entities",
        "total_unique_source_ids",
        "is_reference_dominant",
        "is_curated_dominant",
        "is_explicit_dominant",
        "is_mixed_evidence",
        "is_reference_only",
        "is_reference_heavy_but_not_only",
        "original_confidence",
        "original_ambiguity",
        "non_reference_support_score_proxy",
        "reference_contribution_share",
        "meaningful_without_reference",
    ]
    with pattern_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pattern_fields)
        writer.writeheader()
        for row in sorted(pattern_rows_out, key=lambda item: (item["normalized_family"], item["pattern_id"])):
            writer.writerow({field: row[field] for field in pattern_fields})

    family_summary_rows = []
    families = sorted({row["normalized_family"] for row in cluster_rows_out} | {row["normalized_family"] for row in pattern_rows_out})
    for family in families:
        family_clusters = [row for row in cluster_rows_out if row["normalized_family"] == family]
        family_patterns = [row for row in pattern_rows_out if row["normalized_family"] == family]
        avg = lambda rows, key: (sum(row[key] for row in rows) / len(rows)) if rows else 0.0
        family_summary_rows.append(
            {
                "normalized_family": family,
                "total_clusters": len(family_clusters),
                "total_patterns": len(family_patterns),
                "avg_curated_fraction": avg(family_clusters, "curated_fraction"),
                "avg_explicit_fraction": avg(family_clusters, "explicit_fraction"),
                "avg_reference_fraction": avg(family_clusters, "reference_fraction"),
                "avg_aligned_mention_fraction": avg(family_clusters, "aligned_mention_fraction"),
                "cluster_reference_dominant_fraction": avg(family_clusters, "is_reference_dominant"),
                "cluster_mixed_evidence_fraction": avg(family_clusters, "is_mixed_evidence"),
                "cluster_meaningful_non_reference_fraction": avg(family_clusters, "meaningful_without_reference"),
                "pattern_reference_dominant_fraction": avg(family_patterns, "is_reference_dominant"),
                "pattern_mixed_evidence_fraction": avg(family_patterns, "is_mixed_evidence"),
                "pattern_meaningful_non_reference_fraction": avg(family_patterns, "meaningful_without_reference"),
            }
        )
    with family_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(family_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(family_summary_rows)

    dominance_rows = [
        {
            "object_type": "cluster",
            "total_objects": len(cluster_rows_out),
            "reference_dominant_count": sum(1 for row in cluster_rows_out if row["is_reference_dominant"]),
            "reference_only_count": sum(1 for row in cluster_rows_out if row["is_reference_only"]),
            "reference_heavy_but_not_only_count": sum(1 for row in cluster_rows_out if row["is_reference_heavy_but_not_only"]),
            "mixed_evidence_count": sum(1 for row in cluster_rows_out if row["is_mixed_evidence"]),
            "meaningful_without_reference_count": sum(1 for row in cluster_rows_out if row["meaningful_without_reference"]),
        },
        {
            "object_type": "pattern",
            "total_objects": len(pattern_rows_out),
            "reference_dominant_count": sum(1 for row in pattern_rows_out if row["is_reference_dominant"]),
            "reference_only_count": sum(1 for row in pattern_rows_out if row["is_reference_only"]),
            "reference_heavy_but_not_only_count": sum(1 for row in pattern_rows_out if row["is_reference_heavy_but_not_only"]),
            "mixed_evidence_count": sum(1 for row in pattern_rows_out if row["is_mixed_evidence"]),
            "meaningful_without_reference_count": sum(1 for row in pattern_rows_out if row["meaningful_without_reference"]),
        },
    ]
    with dominance_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dominance_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dominance_rows)

    nonref_rows = []
    for row in cluster_rows_out:
        nonref_rows.append(
            {
                "object_type": "cluster",
                "object_id": row["cluster_id"],
                "normalized_family": row["normalized_family"],
                "label": row["normalized_meaning_label"],
                "original_confidence": row["original_confidence"],
                "non_reference_only_confidence_proxy": row["non_reference_support_score_proxy"],
                "reference_contribution_share": row["reference_contribution_share"],
                "meaningful_without_reference": row["meaningful_without_reference"],
            }
        )
    for row in pattern_rows_out:
        nonref_rows.append(
            {
                "object_type": "pattern",
                "object_id": row["pattern_id"],
                "normalized_family": row["normalized_family"],
                "label": row["pattern_label"],
                "original_confidence": row["original_confidence"],
                "non_reference_only_confidence_proxy": row["non_reference_support_score_proxy"],
                "reference_contribution_share": row["reference_contribution_share"],
                "meaningful_without_reference": row["meaningful_without_reference"],
            }
        )
    with nonref_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(nonref_rows[0].keys()))
        writer.writeheader()
        writer.writerows(nonref_rows)

    engine = PeakListRetrievalEngine(str(DB_PATH))
    retrieval_specs = [
        ("serum", [725.0, 1004.0, 1450.0, 1660.0], "serum", "sers"),
        ("ev", [785.0, 1095.0, 1452.0, 1658.0], "ev", "sers"),
        ("pathogen", [669.0, 772.0, 1063.0, 1447.0], "pathogen", "raman"),
    ]
    pattern_lookup = {row["pattern_id"]: row for row in pattern_rows_out}
    cluster_lookup = {row["cluster_id"]: row for row in cluster_rows_out}
    retrieval_lines = [
        "# Retrieval Source Dependency Audit",
        "",
        "## Thresholds",
        "",
        f"- `reference_dominant`: reference fraction >= {REFERENCE_DOMINANT_THRESHOLD:.2f} and largest support share",
        f"- `reference_heavy_but_not_only`: reference fraction >= {REFERENCE_HEAVY_THRESHOLD:.2f} with some non-reference support",
        f"- `mixed_evidence`: at least two support types each >= {MIXED_EVIDENCE_THRESHOLD:.2f} and no dominant type",
        f"- `meaningful_without_reference`: non-reference support proxy >= {NON_REFERENCE_MEANINGFUL_THRESHOLD:.2f}",
        "",
    ]
    for name, peaks, domain, modality in retrieval_specs:
        payload = engine.search(peaks, domain_hint=domain, modality_hint=modality, tolerance_cm=10.0, top_k=5)
        retrieval_lines.extend([f"## {name}", ""])
        retrieval_lines.append("### Top Patterns")
        retrieval_lines.append("")
        pattern_rows_md = []
        for item in payload.get("pattern_results", [])[:5]:
            audit = pattern_lookup[item["pattern_id"]]
            pattern_rows_md.append(
                [
                    item["pattern_label"],
                    round(item["score"], 6),
                    audit["reference_fraction"],
                    audit["is_reference_dominant"],
                    audit["is_mixed_evidence"],
                    audit["meaningful_without_reference"],
                ]
            )
        retrieval_lines.append(
            _markdown_table(
                ["pattern_label", "score", "reference_fraction", "reference_dominant", "mixed_evidence", "meaningful_without_reference"],
                pattern_rows_md,
            )
        )
        retrieval_lines.extend(["", "### Top Clusters", ""])
        cluster_rows_md = []
        for item in payload.get("support_bundle_results", [])[:5]:
            audit = cluster_lookup[item["cluster_id"]]
            cluster_rows_md.append(
                [
                    item["title"],
                    round(item["score"], 6),
                    audit["reference_fraction"],
                    audit["is_reference_dominant"],
                    audit["is_mixed_evidence"],
                    audit["meaningful_without_reference"],
                ]
            )
        retrieval_lines.append(
            _markdown_table(
                ["cluster_title", "score", "reference_fraction", "reference_dominant", "mixed_evidence", "meaningful_without_reference"],
                cluster_rows_md,
            )
        )
        retrieval_lines.append("")
    _write_text(SOURCE_AUDIT_REPORT_ROOT / "retrieval_source_dependency_audit.md", "\n".join(retrieval_lines))

    ref_dom_clusters = [row for row in cluster_rows_out if row["is_reference_dominant"]]
    ref_dom_patterns = [row for row in pattern_rows_out if row["is_reference_dominant"]]
    mixed_patterns = sorted(
        [row for row in pattern_rows_out if row["is_mixed_evidence"]],
        key=lambda item: (-item["non_reference_support_score_proxy"], item["pattern_id"]),
    )[:5]
    reference_patterns = sorted(
        pattern_rows_out,
        key=lambda item: (-item["reference_fraction"], item["non_reference_support_score_proxy"], item["pattern_id"]),
    )[:5]
    family_rows_md = [
        [
            row["normalized_family"],
            row["total_clusters"],
            row["total_patterns"],
            round(row["avg_curated_fraction"], 3),
            round(row["avg_explicit_fraction"], 3),
            round(row["avg_reference_fraction"], 3),
            round(row["cluster_reference_dominant_fraction"], 3),
            round(row["pattern_reference_dominant_fraction"], 3),
            round(row["pattern_meaningful_non_reference_fraction"], 3),
        ]
        for row in family_summary_rows
    ]
    summary_lines = [
        "# Source Composition Audit Summary",
        "",
        f"- Reference-dominant clusters: {len(ref_dom_clusters)} / {len(cluster_rows_out)}",
        f"- Reference-dominant patterns: {len(ref_dom_patterns)} / {len(pattern_rows_out)}",
        "",
        "## Family Breakdown",
        "",
        _markdown_table(
            [
                "family",
                "clusters",
                "patterns",
                "avg_curated_fraction",
                "avg_explicit_fraction",
                "avg_reference_fraction",
                "cluster_reference_dom_fraction",
                "pattern_reference_dom_fraction",
                "pattern_meaningful_nonref_fraction",
            ],
            family_rows_md,
        ),
        "",
        "## Stronger Mixed-Support Patterns",
        "",
        _markdown_table(
            [
                "pattern_label",
                "family",
                "curated_fraction",
                "explicit_fraction",
                "reference_fraction",
                "nonref_proxy",
            ],
            [
                [
                    row["pattern_label"],
                    row["normalized_family"],
                    round(row["curated_fraction"], 3),
                    round(row["explicit_fraction"], 3),
                    round(row["reference_fraction"], 3),
                    round(row["non_reference_support_score_proxy"], 3),
                ]
                for row in mixed_patterns
            ],
        ),
        "",
        "## RamanBioLib-Driven Patterns",
        "",
        _markdown_table(
            [
                "pattern_label",
                "family",
                "reference_fraction",
                "curated_fraction",
                "explicit_fraction",
                "nonref_proxy",
            ],
            [
                [
                    row["pattern_label"],
                    row["normalized_family"],
                    round(row["reference_fraction"], 3),
                    round(row["curated_fraction"], 3),
                    round(row["explicit_fraction"], 3),
                    round(row["non_reference_support_score_proxy"], 3),
                ]
                for row in reference_patterns
            ],
        ),
        "",
        "## Notes",
        "",
        "- `non_reference_support_score_proxy` is an audit-only diagnostic: curated and explicit supports count more than aligned mentions; RamanBioLib is excluded.",
        "- Current production confidence was not modified in this pass.",
    ]
    _write_text(SOURCE_AUDIT_REPORT_ROOT / "implementation_note.md", "\n".join(summary_lines))

    print(
        json.dumps(
            {
                "cluster_count": len(cluster_rows_out),
                "pattern_count": len(pattern_rows_out),
                "reference_dominant_clusters": len(ref_dom_clusters),
                "reference_dominant_patterns": len(ref_dom_patterns),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
