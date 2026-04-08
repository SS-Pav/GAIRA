from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

import duckdb


STRONG_EDGE_THRESHOLD = 0.38
MEDIUM_EDGE_THRESHOLD = 0.24
MAX_CORE_SIZE = 6
MAX_TOTAL_SIZE = 12


@dataclass
class ClusterContext:
    cluster_id: str
    normalized_subfamily: str
    broader_family: str
    meaning_class: str
    confounder_subclass: str
    spectral_region: str
    canonical_peak_cm: float
    window_start_cm: float
    window_end_cm: float
    confidence_score: float
    ambiguity_score: float
    source_diversity_count: int
    curated_assignment_count: int
    explicit_assignment_count: int
    reference_support_count: int
    linked_evidence_ids: set[str]
    linked_source_ids: set[str]
    contexts: set[str]

    @property
    def direct_support_count(self) -> int:
        return self.curated_assignment_count + self.explicit_assignment_count


def _humanize(value: str) -> str:
    return value.replace("_", " ")


def _group_key(cluster: ClusterContext) -> str:
    return "|".join(
        [
            cluster.normalized_subfamily,
            cluster.broader_family,
            cluster.meaning_class,
            cluster.confounder_subclass or "none",
            cluster.spectral_region,
        ]
    )


def _fetch_cluster_contexts(connection: duckdb.DuckDBPyConnection) -> dict[str, dict[str, ClusterContext]]:
    cluster_rows = connection.sql(
        """
        SELECT
            c.cluster_id,
            COALESCE(o.normalized_subfamily, c.normalized_family) AS normalized_subfamily,
            COALESCE(o.broader_family, c.normalized_family) AS broader_family,
            COALESCE(
                o.meaning_class,
                CASE
                    WHEN c.normalized_family = 'unresolved_assignment_support' THEN 'unresolved_signal'
                    ELSE 'biological_signal'
                END
            ) AS meaning_class,
            COALESCE(o.confounder_subclass, '') AS confounder_subclass,
            COALESCE(o.spectral_region, 'fingerprint_400_1800') AS spectral_region,
            c.canonical_peak_cm,
            c.window_start_cm,
            c.window_end_cm,
            c.confidence_score,
            c.ambiguity_score,
            c.source_diversity_count,
            c.curated_assignment_count,
            c.explicit_assignment_count,
            c.reference_support_count,
            c.linked_evidence_ids_json,
            c.linked_source_ids_json
        FROM evidence.peak_meaning_clusters c
        LEFT JOIN ontology.cluster_ontology_mappings o
          ON o.cluster_id = c.cluster_id
        WHERE COALESCE(o.normalized_subfamily, c.normalized_family) <> 'unresolved_assignment_support'
        ORDER BY normalized_subfamily, c.canonical_peak_cm, c.cluster_id
        """
    ).fetchall()

    grouped: dict[str, dict[str, ClusterContext]] = defaultdict(dict)
    for row in cluster_rows:
        cluster = ClusterContext(
            cluster_id=row[0],
            normalized_subfamily=row[1],
            broader_family=row[2],
            meaning_class=row[3],
            confounder_subclass=row[4],
            spectral_region=row[5],
            canonical_peak_cm=float(row[6]),
            window_start_cm=float(row[7]),
            window_end_cm=float(row[8]),
            confidence_score=float(row[9]),
            ambiguity_score=float(row[10]),
            source_diversity_count=int(row[11]),
            curated_assignment_count=int(row[12]),
            explicit_assignment_count=int(row[13]),
            reference_support_count=int(row[14]),
            linked_evidence_ids=set(json.loads(row[15])),
            linked_source_ids=set(json.loads(row[16])),
            contexts=set(),
        )
        grouped[_group_key(cluster)][cluster.cluster_id] = cluster

    support_rows = connection.sql(
        """
        SELECT
            pms.cluster_id,
            pms.support_kind,
            pms.raw_label,
            pms.evidence_item_id,
            pms.source_id,
            COALESCE(pa.study_family, '') AS study_family
        FROM evidence.peak_meaning_support pms
        LEFT JOIN evidence.peak_assignment_evidence pa
          ON pa.evidence_item_id = pms.evidence_item_id
        WHERE pms.is_primary_support = TRUE
        """
    ).fetchall()
    cluster_lookup = {
        cluster_id: cluster
        for grouped_clusters in grouped.values()
        for cluster_id, cluster in grouped_clusters.items()
    }
    for cluster_id, support_kind, raw_label, evidence_item_id, source_id, study_family in support_rows:
        cluster = cluster_lookup.get(cluster_id)
        if cluster is None:
            continue
        if support_kind == "reference_peak":
            context_id = f"reference:{(raw_label or source_id).strip().lower()}"
        else:
            context_id = f"context:{(study_family or source_id or evidence_item_id).strip().lower()}"
        if context_id:
            cluster.contexts.add(context_id)
    return grouped


def _pair_metrics(cluster_a: ClusterContext, cluster_b: ClusterContext) -> tuple[float, int]:
    context_overlap = len(cluster_a.contexts & cluster_b.contexts)
    evidence_overlap = len(cluster_a.linked_evidence_ids & cluster_b.linked_evidence_ids)
    source_overlap = len(cluster_a.linked_source_ids & cluster_b.linked_source_ids)
    if context_overlap == 0 and evidence_overlap == 0 and source_overlap == 0:
        return 0.0, 0

    containment = context_overlap / max(1, min(len(cluster_a.contexts), len(cluster_b.contexts)))
    jaccard = context_overlap / max(1, len(cluster_a.contexts | cluster_b.contexts))
    source_ratio = source_overlap / max(1, min(len(cluster_a.linked_source_ids), len(cluster_b.linked_source_ids)))
    evidence_ratio = evidence_overlap / max(1, min(len(cluster_a.linked_evidence_ids), len(cluster_b.linked_evidence_ids)))
    closeness = max(0.0, 1.0 - abs(cluster_a.canonical_peak_cm - cluster_b.canonical_peak_cm) / 70.0)
    direct_bonus = 0.08 if cluster_a.direct_support_count > 0 and cluster_b.direct_support_count > 0 else 0.0
    weight = min(
        1.0,
        (0.34 * containment)
        + (0.18 * jaccard)
        + (0.18 * source_ratio)
        + (0.12 * evidence_ratio)
        + (0.10 * closeness)
        + (0.08 * min(cluster_a.confidence_score, cluster_b.confidence_score))
        + direct_bonus,
    )
    overlap_count = context_overlap + evidence_overlap + source_overlap
    return weight, overlap_count


def _build_pair_cache(grouped_clusters: dict[str, ClusterContext]) -> dict[tuple[str, str], tuple[float, int]]:
    cluster_ids = sorted(grouped_clusters)
    pair_cache: dict[tuple[str, str], tuple[float, int]] = {}
    for index, cluster_id in enumerate(cluster_ids):
        for other_id in cluster_ids[index + 1 :]:
            pair_cache[(cluster_id, other_id)] = _pair_metrics(grouped_clusters[cluster_id], grouped_clusters[other_id])
    return pair_cache


def _pair_weight(pair_cache: dict[tuple[str, str], tuple[float, int]], first: str, second: str) -> float:
    return pair_cache.get(tuple(sorted((first, second))), (0.0, 0))[0]


def _mean_pair_weight(node_ids: list[str], pair_cache: dict[tuple[str, str], tuple[float, int]]) -> float:
    weights = []
    for index, cluster_id in enumerate(node_ids):
        for other_id in node_ids[index + 1 :]:
            weights.append(_pair_weight(pair_cache, cluster_id, other_id))
    return sum(weights) / max(1, len(weights))


def _weighted_degree(cluster_id: str, candidate_ids: list[str], pair_cache: dict[tuple[str, str], tuple[float, int]]) -> float:
    return sum(_pair_weight(pair_cache, cluster_id, other_id) for other_id in candidate_ids if other_id != cluster_id)


def _thresholds(cluster: ClusterContext) -> tuple[int, float]:
    if cluster.meaning_class == "confounder_signal":
        return 1, 0.15
    if cluster.spectral_region == "carbonyl_1700_1900":
        return 1, 0.12
    if cluster.spectral_region == "high_wavenumber_2800_3200":
        return 1, 0.12
    return 2, 0.22


def _select_core_members(
    seed_id: str,
    grouped_clusters: dict[str, ClusterContext],
    pair_cache: dict[tuple[str, str], tuple[float, int]],
    used_ids: set[str],
) -> list[str]:
    seed_cluster = grouped_clusters[seed_id]
    min_core, coherence_threshold = _thresholds(seed_cluster)
    if min_core == 1 and len(grouped_clusters) == 1:
        return [seed_id]

    candidate_ids = [seed_id]
    scored_neighbors = []
    for other_id, cluster in grouped_clusters.items():
        if other_id == seed_id or other_id in used_ids:
            continue
        weight = _pair_weight(pair_cache, seed_id, other_id)
        if weight >= MEDIUM_EDGE_THRESHOLD or min_core == 1:
            scored_neighbors.append((weight, cluster.direct_support_count, cluster.confidence_score, -cluster.ambiguity_score, other_id))
    scored_neighbors.sort(reverse=True)

    for weight, _direct, _confidence, _neg_ambiguity, other_id in scored_neighbors:
        proposed = candidate_ids + [other_id]
        if len(proposed) > MAX_CORE_SIZE:
            break
        mean_weight = _mean_pair_weight(proposed, pair_cache)
        if weight >= STRONG_EDGE_THRESHOLD or mean_weight >= coherence_threshold:
            candidate_ids = proposed

    if len(candidate_ids) < min_core:
        return []
    if len(candidate_ids) > 1 and _mean_pair_weight(candidate_ids, pair_cache) < coherence_threshold:
        return []
    return sorted(candidate_ids, key=lambda item: grouped_clusters[item].canonical_peak_cm)


def _select_support_members(
    core_ids: list[str],
    grouped_clusters: dict[str, ClusterContext],
    pair_cache: dict[tuple[str, str], tuple[float, int]],
    used_ids: set[str],
) -> tuple[list[str], list[str]]:
    if not core_ids:
        return [], []
    first_cluster = grouped_clusters[core_ids[0]]
    supporting = []
    optional = []
    scored = []
    for cluster_id, cluster in grouped_clusters.items():
        if cluster_id in used_ids or cluster_id in core_ids:
            continue
        weights = [_pair_weight(pair_cache, cluster_id, core_id) for core_id in core_ids]
        max_weight = max(weights) if weights else 0.0
        mean_weight = sum(weights) / max(1, len(weights))
        if max_weight < 0.12:
            continue
        scored.append((mean_weight, max_weight, cluster.direct_support_count, cluster.confidence_score, -cluster.ambiguity_score, cluster_id))
    scored.sort(reverse=True)

    for mean_weight, max_weight, _direct, _conf, _neg_amb, cluster_id in scored:
        total_if_added = len(core_ids) + len(supporting) + len(optional) + 1
        if total_if_added > MAX_TOTAL_SIZE:
            break
        if first_cluster.meaning_class == "confounder_signal":
            if max_weight >= 0.12:
                optional.append(cluster_id)
            continue
        if mean_weight >= 0.20 or max_weight >= STRONG_EDGE_THRESHOLD:
            supporting.append(cluster_id)
        elif mean_weight >= 0.14:
            optional.append(cluster_id)

    return (
        sorted(supporting, key=lambda item: grouped_clusters[item].canonical_peak_cm),
        sorted(optional, key=lambda item: grouped_clusters[item].canonical_peak_cm),
    )


def _pattern_support_strength(member_clusters: list[ClusterContext]) -> float:
    avg_conf = sum(cluster.confidence_score for cluster in member_clusters) / max(1, len(member_clusters))
    direct_density = sum(cluster.direct_support_count for cluster in member_clusters) / max(1, 2 * len(member_clusters))
    source_div = len(set().union(*(cluster.linked_source_ids for cluster in member_clusters)))
    evidence_div = len(set().union(*(cluster.linked_evidence_ids for cluster in member_clusters)))
    return min(
        1.0,
        (0.45 * avg_conf)
        + (0.25 * min(1.0, direct_density))
        + (0.15 * min(1.0, source_div / 6.0))
        + (0.15 * min(1.0, evidence_div / 18.0)),
    )


def _coherence_score(core_ids: list[str], support_ids: list[str], pair_cache: dict[tuple[str, str], tuple[float, int]]) -> float:
    core_weight = _mean_pair_weight(core_ids, pair_cache)
    support_weights = []
    for support_id in support_ids:
        for core_id in core_ids:
            support_weights.append(_pair_weight(pair_cache, support_id, core_id))
    support_weight = sum(support_weights) / max(1, len(support_weights))
    if not support_ids:
        support_weight = core_weight
    return min(1.0, (0.72 * core_weight) + (0.28 * support_weight))


def _overlap_ratio(first: set[str], second: set[str]) -> float:
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def _pattern_label(cluster: ClusterContext, index: int) -> str:
    label = _humanize(cluster.normalized_subfamily)
    if cluster.spectral_region == "carbonyl_1700_1900":
        label = f"{label} carbonyl-region"
    elif cluster.spectral_region == "high_wavenumber_2800_3200":
        label = f"{label} high-wavenumber"
    elif cluster.spectral_region == "other_region":
        label = f"{label} other-region"
    if cluster.meaning_class == "confounder_signal":
        label = f"{label} confounder"
    return f"{label} motif {index}"


def build_assignment_patterns(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    clusters_by_group = _fetch_cluster_contexts(connection)
    pattern_rows = []
    member_rows = []
    total_patterns = 0
    patterns_per_group: dict[str, int] = {}

    for group_key, grouped_clusters in clusters_by_group.items():
        exemplar = next(iter(grouped_clusters.values()))
        pair_cache = _build_pair_cache(grouped_clusters)
        candidate_ids = sorted(grouped_clusters)
        weighted_degrees = {
            cluster_id: _weighted_degree(cluster_id, candidate_ids, pair_cache)
            for cluster_id in candidate_ids
        }
        seed_order = sorted(
            candidate_ids,
            key=lambda cluster_id: (
                grouped_clusters[cluster_id].direct_support_count,
                weighted_degrees[cluster_id],
                grouped_clusters[cluster_id].confidence_score,
                len(grouped_clusters[cluster_id].contexts),
                cluster_id,
            ),
            reverse=True,
        )

        used_ids: set[str] = set()
        built_patterns: list[dict] = []
        min_core, coherence_threshold = _thresholds(exemplar)
        for seed_id in seed_order:
            if seed_id in used_ids:
                continue
            seed_cluster = grouped_clusters[seed_id]
            if weighted_degrees[seed_id] < max(0.12, coherence_threshold) and seed_cluster.direct_support_count == 0 and seed_cluster.confidence_score < 0.10:
                continue
            core_ids = _select_core_members(seed_id, grouped_clusters, pair_cache, used_ids)
            if len(core_ids) < min_core:
                continue
            supporting_ids, optional_ids = _select_support_members(core_ids, grouped_clusters, pair_cache, used_ids)
            member_ids = core_ids + supporting_ids + optional_ids
            member_clusters = [grouped_clusters[cluster_id] for cluster_id in member_ids]
            coherence = _coherence_score(core_ids, supporting_ids + optional_ids, pair_cache)
            if len(core_ids) > 1 and coherence < coherence_threshold:
                continue

            support_strength = _pattern_support_strength(member_clusters)
            ambiguity = min(
                1.0,
                (0.60 * (sum(cluster.ambiguity_score for cluster in member_clusters) / len(member_clusters)))
                + (0.20 * max(0.0, coherence_threshold - coherence))
                + (0.20 * (1.0 if exemplar.meaning_class == "unresolved_signal" else 0.0)),
            )
            confidence = min(
                1.0,
                (0.40 * support_strength)
                + (0.30 * coherence)
                + (0.20 * min(1.0, len(core_ids) / max(2.0, float(min_core + 1))))
                + (0.10 * min(1.0, len(set().union(*(cluster.linked_source_ids for cluster in member_clusters))) / 6.0)),
            )
            built_patterns.append(
                {
                    "core_ids": core_ids,
                    "supporting_ids": supporting_ids,
                    "optional_ids": optional_ids,
                    "member_ids": member_ids,
                    "coherence": coherence,
                    "support_strength": support_strength,
                    "ambiguity": ambiguity,
                    "confidence": confidence,
                }
            )
            used_ids.update(member_ids)

        if not built_patterns:
            # Fallback: keep one provisional motif for small but interpretable groups.
            ordered = sorted(grouped_clusters.values(), key=lambda item: (-item.confidence_score, -item.direct_support_count, item.canonical_peak_cm))
            if ordered:
                fallback_ids = [ordered[0].cluster_id]
                if len(ordered) > 1 and (exemplar.meaning_class == "confounder_signal" or exemplar.spectral_region != "fingerprint_400_1800"):
                    fallback_ids.append(ordered[1].cluster_id)
                member_clusters = [grouped_clusters[cluster_id] for cluster_id in fallback_ids]
                built_patterns.append(
                    {
                        "core_ids": fallback_ids[:1] if exemplar.meaning_class == "confounder_signal" else fallback_ids,
                        "supporting_ids": fallback_ids[1:] if exemplar.meaning_class == "confounder_signal" else [],
                        "optional_ids": [],
                        "member_ids": fallback_ids,
                        "coherence": _coherence_score(fallback_ids[:1] if len(fallback_ids) == 1 else fallback_ids, [], pair_cache) if len(fallback_ids) > 1 else 0.5,
                        "support_strength": _pattern_support_strength(member_clusters),
                        "ambiguity": sum(cluster.ambiguity_score for cluster in member_clusters) / len(member_clusters),
                        "confidence": sum(cluster.confidence_score for cluster in member_clusters) / len(member_clusters),
                    }
                )

        member_sets = [set(pattern["member_ids"]) for pattern in built_patterns]
        separability_scores = []
        for index, member_set in enumerate(member_sets):
            others = [other_set for other_index, other_set in enumerate(member_sets) if other_index != index]
            if not others:
                separability_scores.append(1.0)
                continue
            max_overlap = max(_overlap_ratio(member_set, other_set) for other_set in others)
            separability_scores.append(max(0.0, 1.0 - max_overlap))

        patterns_per_group[group_key] = len(built_patterns)
        for pattern_index, (pattern_data, separability) in enumerate(
            sorted(
                zip(built_patterns, separability_scores),
                key=lambda item: (-item[0]["support_strength"], -item[0]["coherence"], -len(item[0]["core_ids"])),
            ),
            start=1,
        ):
            core_ids = pattern_data["core_ids"]
            supporting_ids = pattern_data["supporting_ids"]
            optional_ids = pattern_data["optional_ids"]
            member_ids = pattern_data["member_ids"]
            member_clusters = [grouped_clusters[cluster_id] for cluster_id in member_ids]
            pattern_sources = set().union(*(cluster.linked_source_ids for cluster in member_clusters))
            pattern_evidence = set().union(*(cluster.linked_evidence_ids for cluster in member_clusters))

            pattern_id = f"pattern_{exemplar.normalized_subfamily}_{pattern_index:02d}"
            pattern_rows.append(
                (
                    pattern_id,
                    exemplar.normalized_subfamily,
                    exemplar.broader_family,
                    exemplar.broader_family,
                    exemplar.meaning_class,
                    exemplar.confounder_subclass,
                    exemplar.spectral_region,
                    _pattern_label(exemplar, pattern_index),
                    "ontology_subfamily_motif",
                    True,
                    pattern_data["coherence"] < max(0.20, coherence_threshold),
                    len(pattern_sources) < 2 or len(core_ids) < min_core,
                    len(core_ids),
                    len(member_ids),
                    len(pattern_sources),
                    len(pattern_evidence),
                    pattern_data["ambiguity"],
                    pattern_data["confidence"],
                    pattern_data["coherence"],
                    separability,
                    pattern_data["support_strength"],
                    (
                        f"Ontology-aligned motif for {exemplar.normalized_subfamily}; "
                        f"broader_family={exemplar.broader_family}, meaning_class={exemplar.meaning_class}, "
                        f"spectral_region={exemplar.spectral_region}, core={len(core_ids)}, total={len(member_ids)}."
                    ),
                )
            )
            total_patterns += 1

            def _member_row(cluster_id: str, member_role: str, member_weight: float, ambiguity_contribution: float) -> tuple:
                cluster = grouped_clusters[cluster_id]
                return (
                    pattern_id,
                    cluster.cluster_id,
                    cluster.normalized_subfamily,
                    cluster.broader_family,
                    cluster.meaning_class,
                    cluster.confounder_subclass,
                    cluster.spectral_region,
                    cluster.canonical_peak_cm,
                    cluster.window_start_cm,
                    cluster.window_end_cm,
                    member_role,
                    member_weight,
                    cluster.direct_support_count + cluster.reference_support_count,
                    cluster.curated_assignment_count,
                    cluster.explicit_assignment_count,
                    cluster.reference_support_count,
                    ambiguity_contribution,
                )

            for cluster_id in core_ids:
                member_rows.append(_member_row(cluster_id, "core", 1.0, grouped_clusters[cluster_id].ambiguity_score))
            for cluster_id in supporting_ids:
                core_weights = [_pair_weight(pair_cache, cluster_id, core_id) for core_id in core_ids]
                ambiguity_contribution = min(
                    1.0,
                    0.5 * grouped_clusters[cluster_id].ambiguity_score + 0.5 * (1.0 - (sum(core_weights) / max(1, len(core_weights)))),
                )
                member_rows.append(_member_row(cluster_id, "supporting", 0.62, ambiguity_contribution))
            for cluster_id in optional_ids:
                core_weights = [_pair_weight(pair_cache, cluster_id, core_id) for core_id in core_ids]
                ambiguity_contribution = min(
                    1.0,
                    0.55 * grouped_clusters[cluster_id].ambiguity_score + 0.45 * (1.0 - (sum(core_weights) / max(1, len(core_weights)))),
                )
                member_rows.append(_member_row(cluster_id, "optional", 0.35, ambiguity_contribution))

    represented_keys = {
        (
            row[1],  # normalized_subfamily
            row[4],  # meaning_class
            row[5],  # confounder_subclass
            row[6],  # spectral_region
        )
        for row in pattern_rows
    }
    supplemental_rows = connection.sql(
        """
        SELECT
            e.normalized_subfamily,
            e.broader_family,
            e.meaning_class,
            e.confounder_subclass,
            e.spectral_region,
            AVG(COALESCE(p.peak_center_cm, 0.0)) AS canonical_peak_cm,
            MIN(COALESCE(p.peak_center_cm, 0.0)) AS min_peak_cm,
            MAX(COALESCE(p.peak_center_cm, 0.0)) AS max_peak_cm,
            COUNT(*) AS evidence_count,
            COUNT(DISTINCT e.source_id) AS source_diversity,
            STRING_AGG(DISTINCT e.evidence_item_id, ', ' ORDER BY e.evidence_item_id) AS evidence_items_csv
        FROM ontology.evidence_ontology_mappings e
        LEFT JOIN evidence.peak_assignment_evidence p
          ON p.evidence_item_id = e.evidence_item_id
        WHERE e.normalized_subfamily <> 'unresolved_assignment_support'
        GROUP BY 1,2,3,4,5
        ORDER BY e.normalized_subfamily
        """
    ).fetchall()
    for (
        normalized_subfamily,
        broader_family,
        meaning_class,
        confounder_subclass,
        spectral_region,
        canonical_peak_cm,
        min_peak_cm,
        max_peak_cm,
        evidence_count,
        source_diversity,
        evidence_items_csv,
    ) in supplemental_rows:
        key = (normalized_subfamily, meaning_class, confounder_subclass or "", spectral_region)
        if key in represented_keys:
            continue
        if meaning_class != "confounder_signal" and spectral_region == "fingerprint_400_1800":
            continue
        pattern_id = f"pattern_{normalized_subfamily}_supplemental_01"
        pattern_rows.append(
            (
                pattern_id,
                normalized_subfamily,
                broader_family,
                broader_family,
                meaning_class,
                confounder_subclass,
                spectral_region,
                f"{_humanize(normalized_subfamily)} motif supplemental",
                "ontology_evidence_singleton",
                False,
                True,
                True,
                1,
                1,
                int(source_diversity),
                int(evidence_count),
                0.55 if meaning_class == "unresolved_signal" else 0.20,
                0.38 if meaning_class == "confounder_signal" else 0.32,
                0.30,
                1.0,
                0.22,
                "Supplemental ontology motif created from evidence-level mappings not represented in cluster-derived motifs.",
            )
        )
        synthetic_cluster_id = f"synthetic_{normalized_subfamily}_supplemental"
        member_rows.append(
            (
                pattern_id,
                synthetic_cluster_id,
                normalized_subfamily,
                broader_family,
                meaning_class,
                confounder_subclass,
                spectral_region,
                float(canonical_peak_cm or 0.0),
                float(min_peak_cm or canonical_peak_cm or 0.0),
                float(max_peak_cm or canonical_peak_cm or 0.0),
                "core",
                1.0,
                int(evidence_count),
                0,
                0,
                0,
                0.55 if meaning_class == "unresolved_signal" else 0.20,
            )
        )
        total_patterns += 1

    connection.executemany(
        "INSERT INTO evidence.assignment_patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        pattern_rows,
    )
    connection.executemany(
        "INSERT INTO evidence.assignment_pattern_members VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        member_rows,
    )

    return {
        "assignment_patterns_loaded": total_patterns,
        "assignment_pattern_members_loaded": len(member_rows),
        **{f"patterns_in_{group_key}": count for group_key, count in sorted(patterns_per_group.items())},
    }
