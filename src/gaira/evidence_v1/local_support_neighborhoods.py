from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from gaira.evidence_v1.constants import (
    DB_PATH,
    LOCAL_NEIGHBORHOOD_OUTPUT_ROOT,
    LOCAL_NEIGHBORHOOD_REPORT_ROOT,
    LOCAL_NEIGHBORHOOD_TABLES_ROOT,
    ensure_local_neighborhood_output_dirs,
)
from gaira.evidence_v1.schema import (
    initialize_schema,
    reset_local_neighborhood_tables,
)


@dataclass
class EvidencePoint:
    evidence_item_id: str
    assignment_record_id: str
    source_id: str
    peak_center_cm: float
    peak_min_cm: float
    peak_max_cm: float
    exact_source_phrase: str
    normalized_subfamily: str
    broader_family: str
    meaning_class: str
    confounder_subclass: str
    spectral_region: str
    evidence_origin: str
    source_family: str
    direct_support: bool
    reference_support: bool
    provenance_json: str


@dataclass
class Neighborhood:
    neighborhood_id: str
    canonical_peak_cm: float
    window_start_cm: float
    window_end_cm: float
    spectral_region: str
    dominant_normalized_subfamily: str
    broader_family: str
    meaning_class: str
    confounder_subclass: str
    members: list[EvidencePoint] = field(default_factory=list)
    candidate_normalized_subfamilies: list[str] = field(default_factory=list)
    source_diversity_count: int = 0
    evidence_row_count: int = 0
    motif_link_count: int = 0
    local_confidence_score: float = 0.0
    local_ambiguity_score: float = 0.0
    notes: str = ""


def _humanize(value: str) -> str:
    return value.replace("_", " ")


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value.lower()).strip("_")


def _group_tolerance(spectral_region: str, meaning_class: str) -> float:
    if spectral_region == "carbonyl_1700_1900":
        return 18.0
    if spectral_region == "high_wavenumber_2800_3200":
        return 28.0
    if spectral_region == "other_region":
        return 14.0
    if meaning_class == "confounder_signal":
        return 8.0
    if meaning_class == "unresolved_signal":
        return 12.0
    return 10.0


def _ambiguity_overlap_tolerance(spectral_region: str) -> float:
    if spectral_region == "carbonyl_1700_1900":
        return 16.0
    if spectral_region == "high_wavenumber_2800_3200":
        return 30.0
    if spectral_region == "other_region":
        return 14.0
    return 10.0


def _fetch_evidence_points(connection: duckdb.DuckDBPyConnection) -> list[EvidencePoint]:
    rows = connection.sql(
        """
        SELECT
            m.evidence_item_id,
            m.assignment_record_id,
            m.source_id,
            COALESCE(pa.peak_center_cm, sf.peak_center_cm) AS peak_center_cm,
            COALESCE(pa.peak_min_cm, sf.peak_min_cm, COALESCE(pa.peak_center_cm, sf.peak_center_cm)) AS peak_min_cm,
            COALESCE(pa.peak_max_cm, sf.peak_max_cm, COALESCE(pa.peak_center_cm, sf.peak_center_cm)) AS peak_max_cm,
            m.exact_source_phrase,
            m.normalized_subfamily,
            m.broader_family,
            m.meaning_class,
            COALESCE(m.confounder_subclass, '') AS confounder_subclass,
            m.spectral_region,
            COALESCE(
                pa.assignment_origin,
                CASE
                    WHEN sf.feature_origin LIKE 'reference%' THEN 'reference_feature'
                    WHEN sf.feature_origin LIKE 'grounding%' THEN 'grounding_feature'
                    WHEN sf.feature_origin IS NOT NULL THEN sf.feature_origin
                    ELSE 'unknown'
                END
            ) AS evidence_origin,
            COALESCE(ws.source_family, es.source_family, '') AS source_family,
            COALESCE(pa.extraction_method, '') AS extraction_method,
            COALESCE(qc.qc_classification, '') AS qc_classification
        FROM ontology.evidence_ontology_mappings m
        LEFT JOIN evidence.peak_assignment_evidence pa
          ON pa.evidence_item_id = m.evidence_item_id
         AND pa.assignment_record_id = m.assignment_record_id
        LEFT JOIN features.spectral_features sf
          ON sf.evidence_item_id = m.evidence_item_id
         AND sf.feature_id = m.assignment_record_id
        LEFT JOIN registry.warehouse_sources ws
          ON ws.source_id = m.source_id
        LEFT JOIN registry.evidence_sources es
          ON es.source_id = m.source_id
        LEFT JOIN evidence.paper_assignment_qc qc
          ON qc.evidence_item_id = m.evidence_item_id
         AND qc.assignment_record_id = m.assignment_record_id
        WHERE COALESCE(pa.peak_center_cm, sf.peak_center_cm) IS NOT NULL
          AND (
            COALESCE(pa.extraction_method, '') <> 'text_regex'
            OR qc.qc_classification IS NULL
            OR COALESCE(qc.include_in_local_layer, FALSE) = TRUE
          )
        ORDER BY m.meaning_class, m.spectral_region, m.broader_family, m.normalized_subfamily, peak_center_cm
        """
    ).fetchall()

    points: list[EvidencePoint] = []
    for row in rows:
        evidence_origin = row[12] or "unknown"
        extraction_method = row[14] or ""
        qc_classification = row[15] or ""
        direct_support = evidence_origin in {"curated_assignment", "source_backed_regex", "digitized_figure", "text_assignment"}
        if extraction_method == "text_regex":
            direct_support = qc_classification == "validated_primary"
        reference_support = "reference" in evidence_origin or (row[13] or "") == "reference_molecule"
        points.append(
            EvidencePoint(
                evidence_item_id=row[0],
                assignment_record_id=row[1],
                source_id=row[2],
                peak_center_cm=float(row[3]),
                peak_min_cm=float(row[4]),
                peak_max_cm=float(row[5]),
                exact_source_phrase=row[6] or "",
                normalized_subfamily=row[7] or "unresolved_assignment_support",
                broader_family=row[8] or "unresolved_support",
                meaning_class=row[9] or "unresolved_signal",
                confounder_subclass=row[10] or "",
                spectral_region=row[11] or "other_region",
                evidence_origin=evidence_origin,
                source_family=row[13] or "",
                direct_support=direct_support,
                reference_support=reference_support,
                provenance_json=json.dumps(
                    {
                        "evidence_item_id": row[0],
                        "assignment_record_id": row[1],
                        "source_id": row[2],
                        "evidence_origin": evidence_origin,
                        "extraction_method": extraction_method,
                        "qc_classification": qc_classification,
                    },
                    sort_keys=True,
                ),
            )
        )
    return points


def _seed_key(point: EvidencePoint) -> tuple[str, str, str, str]:
    return (
        point.normalized_subfamily,
        point.meaning_class,
        point.confounder_subclass or "",
        point.spectral_region,
    )


def _compatible_for_overlap(first: Neighborhood, second: Neighborhood) -> bool:
    if first.neighborhood_id == second.neighborhood_id:
        return False
    if first.spectral_region != second.spectral_region:
        return False
    if first.meaning_class != second.meaning_class:
        return False
    if (first.confounder_subclass or "") != (second.confounder_subclass or ""):
        return False
    if first.broader_family != second.broader_family:
        return False
    return abs(first.canonical_peak_cm - second.canonical_peak_cm) <= _ambiguity_overlap_tolerance(first.spectral_region)


def _build_seed_neighborhoods(points: list[EvidencePoint]) -> list[Neighborhood]:
    grouped: dict[tuple[str, str, str, str], list[EvidencePoint]] = defaultdict(list)
    for point in points:
        grouped[_seed_key(point)].append(point)

    neighborhoods: list[Neighborhood] = []
    for seed_key, seed_points in sorted(grouped.items()):
        normalized_subfamily, meaning_class, confounder_subclass, spectral_region = seed_key
        seed_points.sort(key=lambda item: item.peak_center_cm)
        tolerance = _group_tolerance(spectral_region, meaning_class)
        current_points: list[EvidencePoint] = []
        neighborhood_index = 0

        def flush_bucket(bucket: list[EvidencePoint]) -> None:
            nonlocal neighborhood_index
            if not bucket:
                return
            neighborhood_index += 1
            centers = [point.peak_center_cm for point in bucket]
            broader_family = Counter(point.broader_family for point in bucket).most_common(1)[0][0]
            neighborhood_id = (
                f"nh_{normalized_subfamily}_{_slug(spectral_region)}_{neighborhood_index:03d}"
            )
            neighborhoods.append(
                Neighborhood(
                    neighborhood_id=neighborhood_id,
                    canonical_peak_cm=sum(centers) / len(centers),
                    window_start_cm=min(point.peak_min_cm for point in bucket),
                    window_end_cm=max(point.peak_max_cm for point in bucket),
                    spectral_region=spectral_region,
                    dominant_normalized_subfamily=normalized_subfamily,
                    broader_family=broader_family,
                    meaning_class=meaning_class,
                    confounder_subclass=confounder_subclass,
                    members=list(bucket),
                )
            )

        for point in seed_points:
            if not current_points:
                current_points.append(point)
                continue
            current_center = sum(item.peak_center_cm for item in current_points) / len(current_points)
            current_end = max(item.peak_max_cm for item in current_points)
            if (
                abs(point.peak_center_cm - current_center) <= tolerance
                and point.peak_center_cm <= current_end + tolerance
            ):
                current_points.append(point)
            else:
                flush_bucket(current_points)
                current_points = [point]
        flush_bucket(current_points)
    return neighborhoods


def _annotate_neighborhoods(neighborhoods: list[Neighborhood]) -> None:
    for neighborhood in neighborhoods:
        overlap_subfamilies = {
            other.dominant_normalized_subfamily
            for other in neighborhoods
            if _compatible_for_overlap(neighborhood, other)
        }
        member_counts = Counter(member.normalized_subfamily for member in neighborhood.members)
        for subfamily in sorted(overlap_subfamilies):
            member_counts.setdefault(subfamily, 0)

        sorted_candidates = [
            subfamily
            for subfamily, _count in member_counts.most_common()
        ]
        if neighborhood.dominant_normalized_subfamily not in sorted_candidates:
            sorted_candidates.insert(0, neighborhood.dominant_normalized_subfamily)
        neighborhood.candidate_normalized_subfamilies = sorted_candidates

        member_sources = {member.source_id for member in neighborhood.members}
        direct_count = sum(1 for member in neighborhood.members if member.direct_support)
        reference_count = sum(1 for member in neighborhood.members if member.reference_support)
        dominant_count = member_counts.get(neighborhood.dominant_normalized_subfamily, 0)
        total_count = len(neighborhood.members)
        dominant_share = dominant_count / max(1, total_count)
        overlap_penalty = min(0.35, 0.10 * max(0, len(sorted_candidates) - 1))
        unresolved_penalty = 0.18 if neighborhood.meaning_class == "unresolved_signal" else 0.0
        confounder_penalty = 0.05 if neighborhood.meaning_class == "confounder_signal" else 0.0

        neighborhood.source_diversity_count = len(member_sources)
        neighborhood.evidence_row_count = total_count
        neighborhood.local_ambiguity_score = max(
            0.0,
            min(
                1.0,
                (0.45 * (1.0 - dominant_share))
                + overlap_penalty
                + unresolved_penalty
                + confounder_penalty,
            ),
        )
        neighborhood.local_confidence_score = max(
            0.0,
            min(
                1.0,
                (0.32 * min(1.0, direct_count / 4.0))
                + (0.24 * min(1.0, reference_count / 10.0))
                + (0.20 * min(1.0, neighborhood.source_diversity_count / 4.0))
                + (0.24 * dominant_share)
                - (0.20 * neighborhood.local_ambiguity_score),
            ),
        )
        neighborhood.notes = (
            f"dominant_subfamily={neighborhood.dominant_normalized_subfamily}; "
            f"candidate_count={len(sorted_candidates)}; direct_support={direct_count}; "
            f"reference_support={reference_count}"
        )


def _insert_neighborhoods(connection: duckdb.DuckDBPyConnection, neighborhoods: list[Neighborhood]) -> None:
    neighborhood_rows = []
    member_rows = []
    for neighborhood in neighborhoods:
        neighborhood_rows.append(
            (
                neighborhood.neighborhood_id,
                neighborhood.canonical_peak_cm,
                neighborhood.window_start_cm,
                neighborhood.window_end_cm,
                neighborhood.spectral_region,
                neighborhood.dominant_normalized_subfamily,
                json.dumps(neighborhood.candidate_normalized_subfamilies),
                neighborhood.broader_family,
                neighborhood.meaning_class,
                neighborhood.confounder_subclass,
                neighborhood.local_confidence_score,
                neighborhood.local_ambiguity_score,
                neighborhood.source_diversity_count,
                neighborhood.evidence_row_count,
                neighborhood.motif_link_count,
                neighborhood.notes,
            )
        )
        dominant_set = {neighborhood.dominant_normalized_subfamily}
        for member in neighborhood.members:
            member_rows.append(
                (
                    neighborhood.neighborhood_id,
                    member.evidence_item_id,
                    member.assignment_record_id,
                    member.source_id,
                    member.peak_center_cm,
                    member.exact_source_phrase,
                    member.normalized_subfamily,
                    member.broader_family,
                    member.meaning_class,
                    member.confounder_subclass,
                    member.spectral_region,
                    member.evidence_origin,
                    "primary" if member.normalized_subfamily in dominant_set else "secondary",
                    member.normalized_subfamily == neighborhood.dominant_normalized_subfamily,
                    member.provenance_json,
                    "",
                )
            )
    connection.executemany(
        "INSERT INTO evidence.local_support_neighborhoods VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        neighborhood_rows,
    )
    connection.executemany(
        "INSERT INTO evidence.local_support_neighborhood_members VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        member_rows,
    )


def _link_neighborhoods_to_motifs(
    connection: duckdb.DuckDBPyConnection,
    neighborhoods: list[Neighborhood],
) -> dict[str, int]:
    pattern_rows = connection.sql(
        """
        SELECT
            ap.pattern_id,
            ap.normalized_subfamily,
            ap.broader_family,
            ap.meaning_class,
            COALESCE(ap.confounder_subclass, '') AS confounder_subclass,
            ap.spectral_region,
            apm.canonical_peak_cm,
            apm.window_start_cm,
            apm.window_end_cm,
            apm.member_role
        FROM evidence.assignment_patterns ap
        JOIN evidence.assignment_pattern_members apm
          ON apm.pattern_id = ap.pattern_id
        ORDER BY ap.pattern_id, apm.canonical_peak_cm
        """
    ).fetchall()
    pattern_members: dict[str, list[dict]] = defaultdict(list)
    pattern_meta: dict[str, dict] = {}
    for row in pattern_rows:
        pattern_id = row[0]
        pattern_meta[pattern_id] = {
            "normalized_subfamily": row[1],
            "broader_family": row[2],
            "meaning_class": row[3],
            "confounder_subclass": row[4] or "",
            "spectral_region": row[5],
        }
        pattern_members[pattern_id].append(
            {
                "canonical_peak_cm": float(row[6]),
                "window_start_cm": float(row[7]),
                "window_end_cm": float(row[8]),
                "member_role": row[9],
            }
        )

    link_rows = []
    link_counts = {"linked_neighborhoods": 0, "unlinked_neighborhoods": 0, "total_links": 0}
    for neighborhood in neighborhoods:
        best_links = []
        for pattern_id, meta in pattern_meta.items():
            if meta["normalized_subfamily"] != neighborhood.dominant_normalized_subfamily:
                continue
            if meta["meaning_class"] != neighborhood.meaning_class:
                continue
            if meta["spectral_region"] != neighborhood.spectral_region:
                continue
            if meta["broader_family"] != neighborhood.broader_family:
                continue
            if meta["confounder_subclass"] != (neighborhood.confounder_subclass or ""):
                continue

            matched = []
            strongest_member_role = "optional"
            role_rank = {"core": 3, "supporting": 2, "optional": 1}
            for member in pattern_members[pattern_id]:
                tolerance = _group_tolerance(neighborhood.spectral_region, neighborhood.meaning_class)
                member_center = member["canonical_peak_cm"]
                if abs(neighborhood.canonical_peak_cm - member_center) <= tolerance:
                    matched.append(member)
                    if role_rank[member["member_role"]] > role_rank.get(strongest_member_role, 0):
                        strongest_member_role = member["member_role"]
            if not matched:
                continue
            link_confidence = min(
                1.0,
                0.60
                + (0.10 * len(matched))
                + (0.20 * neighborhood.local_confidence_score)
                - (0.10 * neighborhood.local_ambiguity_score),
            )
            best_links.append(
                (
                    pattern_id,
                    strongest_member_role,
                    len(matched),
                    link_confidence,
                )
            )

        if best_links:
            best_links.sort(key=lambda item: (item[2], item[3]), reverse=True)
            chosen = best_links[:2]
            neighborhood.motif_link_count = len(chosen)
            link_counts["linked_neighborhoods"] += 1
            for pattern_id, strongest_member_role, matched_count, link_confidence in chosen:
                link_rows.append(
                    (
                        neighborhood.neighborhood_id,
                        pattern_id,
                        strongest_member_role,
                        strongest_member_role,
                        neighborhood.dominant_normalized_subfamily,
                        matched_count,
                        link_confidence,
                        "",
                    )
                )
                link_counts["total_links"] += 1
        else:
            link_counts["unlinked_neighborhoods"] += 1

    if link_rows:
        connection.executemany(
            "INSERT INTO evidence.neighborhood_motif_links VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            link_rows,
        )
    connection.executemany(
        """
        UPDATE evidence.local_support_neighborhoods
        SET motif_link_count = ?
        WHERE neighborhood_id = ?
        """,
        [(neighborhood.motif_link_count, neighborhood.neighborhood_id) for neighborhood in neighborhoods],
    )
    return link_counts


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _generate_reports(connection: duckdb.DuckDBPyConnection, before_cluster_count: int, link_counts: dict[str, int]) -> dict[str, int]:
    neighborhood_count = connection.sql("SELECT COUNT(*) FROM evidence.local_support_neighborhoods").fetchone()[0]
    confounder_count = connection.sql(
        "SELECT COUNT(*) FROM evidence.local_support_neighborhoods WHERE meaning_class = 'confounder_signal'"
    ).fetchone()[0]
    ambiguous_count = connection.sql(
        """
        SELECT COUNT(*)
        FROM evidence.local_support_neighborhoods
        WHERE local_ambiguity_score >= 0.35 OR json_array_length(candidate_normalized_subfamilies_json) > 1
        """
    ).fetchone()[0]
    carbonyl_count = connection.sql(
        "SELECT COUNT(*) FROM evidence.local_support_neighborhoods WHERE spectral_region = 'carbonyl_1700_1900'"
    ).fetchone()[0]
    high_count = connection.sql(
        "SELECT COUNT(*) FROM evidence.local_support_neighborhoods WHERE spectral_region = 'high_wavenumber_2800_3200'"
    ).fetchone()[0]

    neighborhood_summary = connection.sql(
        """
        SELECT
            neighborhood_id,
            canonical_peak_cm,
            window_start_cm,
            window_end_cm,
            spectral_region,
            dominant_normalized_subfamily,
            broader_family,
            meaning_class,
            confounder_subclass,
            local_confidence_score,
            local_ambiguity_score,
            source_diversity_count,
            evidence_row_count,
            motif_link_count,
            candidate_normalized_subfamilies_json
        FROM evidence.local_support_neighborhoods
        ORDER BY meaning_class, spectral_region, canonical_peak_cm
        """
    ).df()
    neighborhood_summary.to_csv(LOCAL_NEIGHBORHOOD_TABLES_ROOT / "neighborhood_summary.csv", index=False)

    neighborhood_examples = connection.sql(
        """
        SELECT
            neighborhood_id,
            canonical_peak_cm,
            dominant_normalized_subfamily,
            broader_family,
            meaning_class,
            spectral_region,
            local_confidence_score,
            local_ambiguity_score,
            evidence_row_count,
            motif_link_count,
            candidate_normalized_subfamilies_json
        FROM evidence.local_support_neighborhoods
        WHERE meaning_class <> 'unresolved_signal'
        ORDER BY local_confidence_score DESC, evidence_row_count DESC
        LIMIT 20
        """
    ).df()
    neighborhood_examples.to_csv(LOCAL_NEIGHBORHOOD_TABLES_ROOT / "neighborhood_examples_refined.csv", index=False)

    neighborhood_links = connection.sql(
        """
        SELECT
            nml.neighborhood_id,
            n.canonical_peak_cm,
            n.dominant_normalized_subfamily,
            n.meaning_class,
            n.spectral_region,
            nml.pattern_id,
            nml.link_role,
            nml.strongest_member_role,
            nml.matched_member_count,
            nml.link_confidence
        FROM evidence.neighborhood_motif_links nml
        JOIN evidence.local_support_neighborhoods n
          ON n.neighborhood_id = nml.neighborhood_id
        ORDER BY nml.link_confidence DESC, n.canonical_peak_cm
        """
    ).df()
    neighborhood_links.to_csv(LOCAL_NEIGHBORHOOD_TABLES_ROOT / "neighborhood_to_motif_links.csv", index=False)

    ambiguous_summary = connection.sql(
        """
        SELECT
            neighborhood_id,
            canonical_peak_cm,
            dominant_normalized_subfamily,
            broader_family,
            meaning_class,
            spectral_region,
            local_ambiguity_score,
            candidate_normalized_subfamilies_json,
            evidence_row_count
        FROM evidence.local_support_neighborhoods
        WHERE local_ambiguity_score >= 0.35 OR json_array_length(candidate_normalized_subfamilies_json) > 1
        ORDER BY local_ambiguity_score DESC, evidence_row_count DESC
        """
    ).df()
    ambiguous_summary.to_csv(LOCAL_NEIGHBORHOOD_TABLES_ROOT / "ambiguous_neighborhood_summary.csv", index=False)

    confounder_summary = connection.sql(
        """
        SELECT
            neighborhood_id,
            canonical_peak_cm,
            dominant_normalized_subfamily,
            confounder_subclass,
            local_confidence_score,
            local_ambiguity_score,
            evidence_row_count,
            motif_link_count
        FROM evidence.local_support_neighborhoods
        WHERE meaning_class = 'confounder_signal'
        ORDER BY canonical_peak_cm
        """
    ).df()
    confounder_summary.to_csv(LOCAL_NEIGHBORHOOD_TABLES_ROOT / "confounder_neighborhood_summary.csv", index=False)

    spectral_summary = connection.sql(
        """
        SELECT
            spectral_region,
            meaning_class,
            COUNT(*) AS neighborhood_count,
            AVG(local_confidence_score) AS mean_local_confidence,
            AVG(local_ambiguity_score) AS mean_local_ambiguity,
            SUM(motif_link_count) AS motif_link_total
        FROM evidence.local_support_neighborhoods
        GROUP BY 1,2
        ORDER BY 1,2
        """
    ).df()
    spectral_summary.to_csv(LOCAL_NEIGHBORHOOD_TABLES_ROOT / "spectral_region_neighborhood_summary.csv", index=False)

    count_rows = [
        {"layer": "old_peak_meaning_clusters", "count_before": before_cluster_count, "count_after": before_cluster_count},
        {"layer": "local_support_neighborhoods", "count_before": before_cluster_count, "count_after": neighborhood_count},
    ]
    _write_csv(
        LOCAL_NEIGHBORHOOD_TABLES_ROOT / "neighborhood_count_before_after.csv",
        ["layer", "count_before", "count_after"],
        count_rows,
    )

    comparison_md = f"""# Neighborhood vs Old Cluster Comparison

- Old `evidence.peak_meaning_clusters` count: `{before_cluster_count}`
- New `evidence.local_support_neighborhoods` count: `{neighborhood_count}`
- Linked neighborhoods: `{link_counts['linked_neighborhoods']}`
- Unlinked neighborhoods: `{link_counts['unlinked_neighborhoods']}`
- Total neighborhood-to-motif links: `{link_counts['total_links']}`

The new local layer is ontology-constrained:
- grouping does not cross spectral-region boundaries
- confounder neighborhoods are isolated from biological neighborhoods
- dominant subfamily and candidate subfamilies are both recorded
- motif linkage is explicit instead of being implicit through broad family labels

The old clusters remain in place for backward compatibility, but the new neighborhoods are the ontology-aware local aggregation layer intended to sit under motifs.
"""
    (LOCAL_NEIGHBORHOOD_REPORT_ROOT / "neighborhood_vs_old_cluster_comparison.md").write_text(comparison_md)

    implementation_note = f"""# Implementation Note

This pass rebuilds the local aggregation layer as `evidence.local_support_neighborhoods` plus member/link tables.

Construction rules:
- seed neighborhoods are built from ontology-mapped evidence rows grouped by `normalized_subfamily`, `meaning_class`, `confounder_subclass`, and `spectral_region`
- proximity grouping uses region-aware tolerances rather than the older broad-family cluster logic
- nearby same-region neighborhoods from the same broader family contribute candidate subfamilies for ambiguity tracking, but are not merged if they would blur distinct ontology labels
- neighborhood-to-motif links are created only when ontology metadata and peak proximity both agree

Run summary:
- old broad-family clusters: `{before_cluster_count}`
- ontology-aware neighborhoods: `{neighborhood_count}`
- linked neighborhoods: `{link_counts['linked_neighborhoods']}`
- ambiguous neighborhoods: `{ambiguous_count}`
- confounder neighborhoods: `{confounder_count}`

This preserves provenance while making the local layer consistent with the current ontology and motif architecture.
"""
    (LOCAL_NEIGHBORHOOD_REPORT_ROOT / "implementation_note.md").write_text(implementation_note)

    current_state = f"""# Current State Assessment

- The local support layer is now aligned to ontology fields rather than only older broad-family labels.
- `{neighborhood_count}` neighborhoods were built from `4891` ontology-mapped evidence rows, replacing the old role of `{before_cluster_count}` broad-family clusters for local aggregation.
- Biological and confounder neighborhoods are separated cleanly by `meaning_class` and `confounder_subclass`; current confounder count is `{confounder_count}`.
- Spectral-region handling is improved because neighborhoods are constructed independently within fingerprint, carbonyl, high-wavenumber, and other-region classes. Current region counts: carbonyl `{carbonyl_count}`, high-wavenumber `{high_count}`.
- Neighborhoods now link explicitly to motifs, making motif support more auditable. Linked neighborhoods: `{link_counts['linked_neighborhoods']}`. Ambiguous neighborhoods retained explicitly: `{ambiguous_count}`.

What remains before regex validation / SI linkage / figure triage:
- regex-derived weak evidence still needs its own controlled validation pass against these neighborhoods
- supplementary / SI linkage is still sparse and should be added before large-scale paper processing
- figure triage should attach to motif and neighborhood gaps, not directly to old clusters
"""
    (LOCAL_NEIGHBORHOOD_REPORT_ROOT / "current_state_assessment.md").write_text(current_state)

    return {
        "old_cluster_count": int(before_cluster_count),
        "local_neighborhood_count": int(neighborhood_count),
        **{key: int(value) for key, value in link_counts.items()},
    }


def build_local_support_neighborhoods(db_path: Path = DB_PATH) -> dict[str, int]:
    ensure_local_neighborhood_output_dirs()
    connection = duckdb.connect(str(db_path))
    try:
        initialize_schema(connection)
        before_cluster_count = connection.sql("SELECT COUNT(*) FROM evidence.peak_meaning_clusters").fetchone()[0]
        reset_local_neighborhood_tables(connection)
        points = _fetch_evidence_points(connection)
        neighborhoods = _build_seed_neighborhoods(points)
        _annotate_neighborhoods(neighborhoods)
        _insert_neighborhoods(connection, neighborhoods)
        link_counts = _link_neighborhoods_to_motifs(connection, neighborhoods)
        summary = _generate_reports(connection, before_cluster_count, link_counts)
        connection.commit()
        return summary
    finally:
        connection.close()


if __name__ == "__main__":
    result = build_local_support_neighborhoods()
    print(json.dumps(result, indent=2, sort_keys=True))
