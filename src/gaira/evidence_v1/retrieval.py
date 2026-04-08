from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, UTC
from hashlib import sha1

import duckdb


@dataclass
class PeakMeaningResult:
    document_id: str
    cluster_id: str
    title: str
    normalized_meaning_label: str
    normalized_family: str
    score: float
    confidence_score: float
    ambiguity_score: float
    mixed_family_flag: bool
    overlapping_family_count: int
    source_diversity_count: int
    curated_assignment_count: int
    explicit_assignment_count: int
    reference_support_count: int
    aligned_mention_support_count: int
    score_components: dict
    matched_query_peaks: list[dict]
    support_summary: dict
    provenance: dict
    applicable_context_node_ids: list[str]
    applicable_context_edge_ids: list[str]


@dataclass
class AssignmentPatternResult:
    pattern_id: str
    pattern_label: str
    normalized_family: str
    score: float
    pattern_completeness: float
    pattern_confidence: float
    pattern_ambiguity: float
    pattern_coherence: float
    pattern_separability: float
    pattern_support_strength: float
    source_diversity: int
    evidence_count: int
    matched_clusters: list[dict]
    matched_core_members: list[dict]
    matched_supporting_members: list[dict]
    matched_optional_members: list[dict]
    cluster_level_details: list[dict]
    context_modifiers: list[dict]


class PeakListRetrievalEngine:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _table_has_rows(self, table_name: str) -> bool:
        with duckdb.connect(self.db_path, read_only=True) as connection:
            return bool(connection.sql(f"SELECT COUNT(*) > 0 FROM {table_name}").fetchone()[0])

    def _domain_factor(self, query_domain: str | None, doc_domain: str | None) -> float:
        if not query_domain:
            return 1.0
        doc_domain = (doc_domain or "").lower()
        query_domain = query_domain.lower()
        if doc_domain == query_domain:
            return 1.08
        if query_domain == "plasma" and doc_domain == "serum":
            return 0.98
        if doc_domain in {"generic", "reference", ""}:
            return 0.93
        return 0.78

    def _modality_factor(self, query_modality: str | None, doc_modality: str | None) -> float:
        if not query_modality:
            return 1.0
        doc_modality = (doc_modality or "").lower()
        query_modality = query_modality.lower()
        if doc_modality == query_modality:
            return 1.05
        if doc_modality in {"", "unknown"}:
            return 1.0
        return 0.82

    def _fetch_peak_meaning_documents(self) -> list[dict]:
        with duckdb.connect(self.db_path, read_only=True) as connection:
            rows = connection.sql(
                """
                SELECT
                    document_id,
                    cluster_id,
                    normalized_meaning_label,
                    normalized_family,
                    title,
                    summary_text,
                    canonical_peak_cm,
                    window_start_cm,
                    window_end_cm,
                    domain_hint,
                    modality_hint,
                    direct_retrieval_eligible,
                    confidence_score,
                    ambiguity_score,
                    mixed_family_flag,
                    overlapping_family_count,
                    source_diversity_count,
                    curated_assignment_count,
                    explicit_assignment_count,
                    reference_support_count,
                    aligned_mention_support_count,
                    score_components_json,
                    applicable_context_node_ids_json,
                    applicable_context_edge_ids_json,
                    support_summary_json,
                    provenance_json
                FROM retrieval.peak_meaning_documents
                WHERE direct_retrieval_eligible = TRUE
                ORDER BY canonical_peak_cm, cluster_id
                """
            ).fetchall()
        return [
            {
                "document_id": row[0],
                "cluster_id": row[1],
                "normalized_meaning_label": row[2],
                "normalized_family": row[3],
                "title": row[4],
                "summary_text": row[5],
                "canonical_peak_cm": float(row[6]),
                "window_start_cm": float(row[7]),
                "window_end_cm": float(row[8]),
                "domain_hint": row[9],
                "modality_hint": row[10],
                "confidence_score": float(row[12]),
                "ambiguity_score": float(row[13]),
                "mixed_family_flag": bool(row[14]),
                "overlapping_family_count": int(row[15]),
                "source_diversity_count": int(row[16]),
                "curated_assignment_count": int(row[17]),
                "explicit_assignment_count": int(row[18]),
                "reference_support_count": int(row[19]),
                "aligned_mention_support_count": int(row[20]),
                "score_components": json.loads(row[21]),
                "applicable_context_node_ids": json.loads(row[22]),
                "applicable_context_edge_ids": json.loads(row[23]),
                "support_summary": json.loads(row[24]),
                "provenance": json.loads(row[25]),
            }
            for row in rows
        ]

    def _fetch_context_details(self, node_ids: list[str], edge_ids: list[str]) -> list[dict]:
        if not node_ids and not edge_ids:
            return []
        with duckdb.connect(self.db_path, read_only=True) as connection:
            node_lookup = {}
            if node_ids:
                node_df = connection.sql(
                    f"""
                    SELECT node_id, node_type, node_label, notes
                    FROM context.context_nodes
                    WHERE node_id IN ({", ".join(["?"] * len(node_ids))})
                    """,
                    params=node_ids,
                ).df()
                node_lookup = {
                    row["node_id"]: {
                        "node_type": row["node_type"],
                        "node_label": row["node_label"],
                        "notes": row["notes"],
                    }
                    for _, row in node_df.iterrows()
                }
            if not edge_ids:
                return []
            edge_df = connection.sql(
                f"""
                SELECT edge_id, source_node_id, target_node_id, edge_type, weight, evidence_basis, notes
                FROM context.context_edges
                WHERE edge_id IN ({", ".join(["?"] * len(edge_ids))})
                ORDER BY edge_id
                """,
                params=edge_ids,
            ).df()
        context_items = []
        for _, row in edge_df.iterrows():
            context_items.append(
                {
                    "edge_id": row["edge_id"],
                    "edge_type": row["edge_type"],
                    "weight": float(row["weight"]),
                    "evidence_basis": row["evidence_basis"],
                    "notes": row["notes"],
                    "source_node": {"node_id": row["source_node_id"], **node_lookup.get(row["source_node_id"], {})},
                    "target_node": {"node_id": row["target_node_id"], **node_lookup.get(row["target_node_id"], {})},
                }
            )
        return context_items

    def _fetch_assignment_patterns(self) -> list[dict]:
        with duckdb.connect(self.db_path, read_only=True) as connection:
            column_df = connection.sql("DESCRIBE evidence.assignment_patterns").df()
            pattern_columns = set(column_df["column_name"].astype(str).tolist())
            has_refinement_metrics = {"coherence_score", "separability_score", "support_strength_score"}.issubset(pattern_columns)
            extra_select = (
                "ap.coherence_score, ap.separability_score, ap.support_strength_score,"
                if has_refinement_metrics
                else "NULL AS coherence_score, NULL AS separability_score, NULL AS support_strength_score,"
            )
            rows = connection.sql(
                f"""
                SELECT
                    ap.pattern_id,
                    ap.normalized_subfamily,
                    ap.normalized_family,
                    ap.broader_family,
                    ap.meaning_class,
                    ap.confounder_subclass,
                    ap.spectral_region,
                    ap.pattern_label,
                    ap.pattern_type,
                    ap.canonical_multi_peak,
                    ap.loose_constellation,
                    ap.provisional,
                    ap.core_member_count,
                    ap.total_member_count,
                    ap.source_diversity,
                    ap.evidence_count,
                    ap.ambiguity_score,
                    ap.confidence_score,
                    {extra_select}
                    ap.notes,
                    apm.cluster_id,
                    apm.normalized_subfamily,
                    apm.broader_family,
                    apm.meaning_class,
                    apm.confounder_subclass,
                    apm.spectral_region,
                    apm.canonical_peak_cm,
                    apm.window_start_cm,
                    apm.window_end_cm,
                    apm.member_role,
                    apm.member_weight,
                    apm.evidence_support_count,
                    apm.curated_support_count,
                    apm.explicit_support_count,
                    apm.reference_support_count,
                    apm.ambiguity_contribution
                FROM evidence.assignment_patterns ap
                JOIN evidence.assignment_pattern_members apm
                  ON ap.pattern_id = apm.pattern_id
                ORDER BY ap.pattern_id, apm.canonical_peak_cm, apm.cluster_id
                """
            ).fetchall()
        grouped: dict[str, dict] = {}
        for row in rows:
            pattern = grouped.setdefault(
                row[0],
                {
                    "pattern_id": row[0],
                    "normalized_subfamily": row[1],
                    "normalized_family": row[2],
                    "broader_family": row[3],
                    "meaning_class": row[4],
                    "confounder_subclass": row[5],
                    "spectral_region": row[6],
                    "pattern_label": row[7],
                    "pattern_type": row[8],
                    "canonical_multi_peak": bool(row[9]),
                    "loose_constellation": bool(row[10]),
                    "provisional": bool(row[11]),
                    "core_member_count": int(row[12]),
                    "total_member_count": int(row[13]),
                    "source_diversity": int(row[14]),
                    "evidence_count": int(row[15]),
                    "ambiguity_score": float(row[16]),
                    "confidence_score": float(row[17]),
                    "coherence_score": float(row[18] or 0.0),
                    "separability_score": float(row[19] or 0.0),
                    "support_strength_score": float(row[20] or 0.0),
                    "notes": row[21],
                    "members": [],
                },
            )
            pattern["members"].append(
                {
                    "cluster_id": row[22],
                    "normalized_subfamily": row[23],
                    "broader_family": row[24],
                    "meaning_class": row[25],
                    "confounder_subclass": row[26],
                    "spectral_region": row[27],
                    "canonical_peak_cm": float(row[28]),
                    "window_start_cm": float(row[29]),
                    "window_end_cm": float(row[30]),
                    "member_role": row[31],
                    "member_weight": float(row[32]),
                    "evidence_support_count": int(row[33]),
                    "curated_support_count": int(row[34]),
                    "explicit_support_count": int(row[35]),
                    "reference_support_count": int(row[36]),
                    "ambiguity_contribution": float(row[37]),
                }
            )
        return list(grouped.values())

    def _score_bundle(self, document: dict, query_peaks: list[float], tolerance_cm: float) -> PeakMeaningResult | None:
        matched_query_peaks = []
        effective_window_start = document["window_start_cm"] - tolerance_cm
        effective_window_end = document["window_end_cm"] + tolerance_cm
        for query_peak in query_peaks:
            if effective_window_start <= query_peak <= effective_window_end:
                distance = abs(query_peak - document["canonical_peak_cm"])
                closeness = max(0.0, 1.0 - (distance / max(tolerance_cm, 1.0)))
                matched_query_peaks.append(
                    {
                        "query_peak_cm": query_peak,
                        "canonical_peak_cm": document["canonical_peak_cm"],
                        "window_start_cm": document["window_start_cm"],
                        "window_end_cm": document["window_end_cm"],
                        "distance_cm": round(distance, 6),
                        "closeness": round(closeness, 6),
                    }
                )
        if not matched_query_peaks:
            return None

        overlap_count = len(matched_query_peaks)
        overlap_score = sum(item["closeness"] for item in matched_query_peaks)
        support_counts = document["score_components"].get("support_counts", {})
        reference_grounding_count = int(support_counts.get("reference_grounding_count", 0))
        serum_grounding_count = int(support_counts.get("serum_grounding_count", 0))
        trust_score = (
            0.60 * document["confidence_score"]
            + 0.12 * min(1.0, document["curated_assignment_count"] / 2.0)
            + 0.10 * min(1.0, document["explicit_assignment_count"] / 3.0)
            + 0.06 * min(1.0, reference_grounding_count / 3.0)
            + 0.06 * min(1.0, serum_grounding_count / 3.0)
            + 0.04 * min(1.0, document["source_diversity_count"] / 4.0)
            + 0.02 * min(1.0, document["reference_support_count"] / 12.0)
            + 0.01 * min(1.0, document["aligned_mention_support_count"] / 3.0)
        )
        ambiguity_multiplier = max(0.30, 1.0 - 0.50 * document["ambiguity_score"])
        direct_support_count = (
            document["curated_assignment_count"]
            + document["explicit_assignment_count"]
            + reference_grounding_count
            + serum_grounding_count
        )
        if direct_support_count == 0:
            evidence_basis_multiplier = 0.62
        elif document["curated_assignment_count"] > 0:
            evidence_basis_multiplier = 1.08
        else:
            evidence_basis_multiplier = 0.96
        score = ((1.8 * overlap_count) + overlap_score + (2.2 * trust_score)) * ambiguity_multiplier * evidence_basis_multiplier

        return PeakMeaningResult(
            document_id=document["document_id"],
            cluster_id=document["cluster_id"],
            title=document["title"],
            normalized_meaning_label=document["normalized_meaning_label"],
            normalized_family=document["normalized_family"],
            score=score,
            confidence_score=document["confidence_score"],
            ambiguity_score=document["ambiguity_score"],
            mixed_family_flag=document["mixed_family_flag"],
            overlapping_family_count=document["overlapping_family_count"],
            source_diversity_count=document["source_diversity_count"],
            curated_assignment_count=document["curated_assignment_count"],
            explicit_assignment_count=document["explicit_assignment_count"],
            reference_support_count=document["reference_support_count"],
            aligned_mention_support_count=document["aligned_mention_support_count"],
            score_components=document["score_components"],
            matched_query_peaks=matched_query_peaks,
            support_summary=document["support_summary"],
            provenance=document["provenance"],
            applicable_context_node_ids=document["applicable_context_node_ids"],
            applicable_context_edge_ids=document["applicable_context_edge_ids"],
        )

    def _score_pattern(
        self,
        pattern: dict,
        cluster_documents: dict[str, dict],
        query_peaks: list[float],
        domain_hint: str | None,
        modality_hint: str | None,
        tolerance_cm: float,
    ) -> AssignmentPatternResult | None:
        matched_members = []
        matched_cluster_details = []
        context_node_ids: set[str] = set()
        context_edge_ids: set[str] = set()
        core_total = sum(1 for member in pattern["members"] if member["member_role"] == "core")
        supporting_total = sum(1 for member in pattern["members"] if member["member_role"] == "supporting")
        optional_total = sum(1 for member in pattern["members"] if member["member_role"] in {"optional", "ambiguous"})

        for member in pattern["members"]:
            effective_window_start = member["window_start_cm"] - tolerance_cm
            effective_window_end = member["window_end_cm"] + tolerance_cm
            matched_peaks = []
            for query_peak in query_peaks:
                if effective_window_start <= query_peak <= effective_window_end:
                    distance = abs(query_peak - member["canonical_peak_cm"])
                    closeness = max(0.0, 1.0 - (distance / max(tolerance_cm, 1.0)))
                    matched_peaks.append(
                        {
                            "query_peak_cm": query_peak,
                            "canonical_peak_cm": member["canonical_peak_cm"],
                            "distance_cm": round(distance, 6),
                            "closeness": round(closeness, 6),
                        }
                    )
            if not matched_peaks:
                continue
            cluster_doc = cluster_documents.get(member["cluster_id"])
            if cluster_doc:
                context_node_ids.update(cluster_doc["applicable_context_node_ids"])
                context_edge_ids.update(cluster_doc["applicable_context_edge_ids"])
            matched_members.append(
                {
                    **member,
                    "matched_query_peaks": matched_peaks,
                }
            )
            if cluster_doc:
                matched_cluster_details.append(
                    {
                        "cluster_id": cluster_doc["cluster_id"],
                        "title": cluster_doc["title"],
                        "normalized_meaning_label": cluster_doc["normalized_meaning_label"],
                        "confidence_score": cluster_doc["confidence_score"],
                        "ambiguity_score": cluster_doc["ambiguity_score"],
                        "curated_assignment_count": cluster_doc["curated_assignment_count"],
                        "explicit_assignment_count": cluster_doc["explicit_assignment_count"],
                        "reference_support_count": cluster_doc["reference_support_count"],
                    }
                )
        if not matched_members:
            return None

        matched_core = [member for member in matched_members if member["member_role"] == "core"]
        matched_supporting = [member for member in matched_members if member["member_role"] == "supporting"]
        matched_optional = [member for member in matched_members if member["member_role"] in {"optional", "ambiguous"}]
        completeness = (
            (0.7 * (len(matched_core) / max(1, core_total)))
            + (0.2 * (len(matched_supporting) / max(1, supporting_total or 1)))
            + (0.1 * (len(matched_optional) / max(1, optional_total or 1)))
        )
        matched_cluster_confidence = sum(
            cluster_documents[member["cluster_id"]]["confidence_score"]
            for member in matched_members
            if member["cluster_id"] in cluster_documents
        ) / max(1, len(matched_members))
        matched_cluster_ambiguity = sum(
            cluster_documents[member["cluster_id"]]["ambiguity_score"]
            for member in matched_members
            if member["cluster_id"] in cluster_documents
        ) / max(1, len(matched_members))
        boost = 1.0 + (0.18 * max(0, len(matched_core) - 1)) + (0.06 * len(matched_supporting))
        domain_factor = self._domain_factor(domain_hint, None)
        modality_factor = self._modality_factor(modality_hint, None)
        pattern_confidence = min(
            1.0,
            (0.28 * pattern["confidence_score"])
            + (0.24 * pattern["support_strength_score"])
            + (0.20 * pattern["coherence_score"])
            + (0.18 * matched_cluster_confidence)
            + (0.10 * completeness),
        )
        pattern_ambiguity = min(
            1.0,
            (0.45 * pattern["ambiguity_score"])
            + (0.30 * matched_cluster_ambiguity)
            + (0.15 * max(0.0, 0.45 - pattern["separability_score"]))
            + (0.10 * max((member["member_role"] == "ambiguous") for member in matched_members)),
        )
        score = (
            (1.55 * completeness)
            + (1.00 * pattern_confidence)
            + (0.45 * pattern["coherence_score"])
            + (0.30 * pattern["separability_score"])
            + (0.25 * len(matched_members) / max(1, pattern["total_member_count"]))
        ) * max(0.30, 1.0 - 0.45 * pattern_ambiguity) * boost * domain_factor * modality_factor

        context_modifiers = self._fetch_context_details(sorted(context_node_ids), sorted(context_edge_ids))
        return AssignmentPatternResult(
            pattern_id=pattern["pattern_id"],
            pattern_label=pattern["pattern_label"],
            normalized_family=pattern["normalized_family"],
            score=score,
            pattern_completeness=completeness,
            pattern_confidence=pattern_confidence,
            pattern_ambiguity=pattern_ambiguity,
            pattern_coherence=pattern["coherence_score"],
            pattern_separability=pattern["separability_score"],
            pattern_support_strength=pattern["support_strength_score"],
            source_diversity=pattern["source_diversity"],
            evidence_count=pattern["evidence_count"],
            matched_clusters=matched_members,
            matched_core_members=matched_core,
            matched_supporting_members=matched_supporting,
            matched_optional_members=matched_optional,
            cluster_level_details=matched_cluster_details,
            context_modifiers=context_modifiers,
        )

    def _search_peak_meaning_bundles(
        self,
        query_peaks: list[float],
        domain_hint: str | None,
        modality_hint: str | None,
        tolerance_cm: float,
        top_k: int,
    ) -> dict:
        documents = self._fetch_peak_meaning_documents()
        results: list[PeakMeaningResult] = []
        for document in documents:
            scored = self._score_bundle(document, query_peaks, tolerance_cm)
            if scored is None:
                continue
            scored.score *= self._domain_factor(domain_hint, document["domain_hint"])
            scored.score *= self._modality_factor(modality_hint, document["modality_hint"])
            results.append(scored)
        results.sort(key=lambda item: (-item.score, item.cluster_id))

        support_bundle_results = []
        for item in results[:top_k]:
            context_modifiers = self._fetch_context_details(item.applicable_context_node_ids, item.applicable_context_edge_ids)
            support_bundle_results.append(
                {
                    "document_id": item.document_id,
                    "cluster_id": item.cluster_id,
                    "title": item.title,
                    "normalized_meaning_label": item.normalized_meaning_label,
                    "normalized_family": item.normalized_family,
                    "score": round(item.score, 6),
                    "confidence_score": round(item.confidence_score, 6),
                    "ambiguity_score": round(item.ambiguity_score, 6),
                    "mixed_family_flag": item.mixed_family_flag,
                    "overlapping_family_count": item.overlapping_family_count,
                    "source_diversity_count": item.source_diversity_count,
                    "curated_assignment_count": item.curated_assignment_count,
                    "explicit_assignment_count": item.explicit_assignment_count,
                    "reference_support_count": item.reference_support_count,
                    "aligned_mention_support_count": item.aligned_mention_support_count,
                    "score_components": item.score_components,
                    "matched_query_peaks": item.matched_query_peaks,
                    "evidence_family_support_summary": item.support_summary,
                    "applicable_context_modifiers": context_modifiers,
                    "provenance": item.provenance,
                }
            )
        context_graph_results = [
            modifier
            for bundle in support_bundle_results
            for modifier in bundle["applicable_context_modifiers"]
        ]
        pattern_results = []
        if self._table_has_rows("evidence.assignment_patterns"):
            cluster_documents = {document["cluster_id"]: document for document in documents}
            patterns = self._fetch_assignment_patterns()
            scored_patterns = []
            for pattern in patterns:
                scored = self._score_pattern(pattern, cluster_documents, query_peaks, domain_hint, modality_hint, tolerance_cm)
                if scored is not None:
                    scored_patterns.append(scored)
            scored_patterns.sort(key=lambda item: (-item.score, item.pattern_id))
            for item in scored_patterns[:top_k]:
                pattern_results.append(
                    {
                        "pattern_id": item.pattern_id,
                        "pattern_label": item.pattern_label,
                        "normalized_family": item.normalized_family,
                        "score": round(item.score, 6),
                        "pattern_completeness": round(item.pattern_completeness, 6),
                        "pattern_confidence": round(item.pattern_confidence, 6),
                        "pattern_ambiguity": round(item.pattern_ambiguity, 6),
                        "pattern_coherence": round(item.pattern_coherence, 6),
                        "pattern_separability": round(item.pattern_separability, 6),
                        "pattern_support_strength": round(item.pattern_support_strength, 6),
                        "source_diversity": item.source_diversity,
                        "evidence_count": item.evidence_count,
                        "matched_clusters": item.matched_clusters,
                        "matched_core_members": item.matched_core_members,
                        "matched_supporting_members": item.matched_supporting_members,
                        "matched_optional_members": item.matched_optional_members,
                        "cluster_level_details": item.cluster_level_details,
                        "context_modifiers": item.context_modifiers,
                    }
                )
        return {
            "query_peaks": query_peaks,
            "domain_hint": domain_hint,
            "modality_hint": modality_hint,
            "tolerance_cm": tolerance_cm,
            "pattern_results": pattern_results,
            "support_bundle_results": support_bundle_results,
            "context_graph_results": context_graph_results,
            "direct_results": pattern_results or support_bundle_results,
            "context_results": context_graph_results,
        }

    def _legacy_search(
        self,
        query_peaks: list[float],
        domain_hint: str | None,
        modality_hint: str | None,
        tolerance_cm: float,
        top_k: int,
    ) -> dict:
        with duckdb.connect(self.db_path, read_only=True) as connection:
            rows = connection.sql(
                """
                SELECT document_id, title, summary_text, domain_hint, modality_hint
                FROM retrieval.retrieval_documents
                WHERE direct_retrieval_eligible = TRUE
                LIMIT 0
                """
            ).fetchall()
        return {
            "query_peaks": query_peaks,
            "domain_hint": domain_hint,
            "modality_hint": modality_hint,
            "tolerance_cm": tolerance_cm,
            "support_bundle_results": [],
            "pattern_results": [],
            "context_graph_results": [],
            "direct_results": [],
            "context_results": [],
            "notes": "Legacy retrieval documents exist but Phase 1 refinement peak-meaning bundles were not built.",
        }

    def search(
        self,
        query_peaks: list[float],
        domain_hint: str | None = None,
        modality_hint: str | None = None,
        tolerance_cm: float = 10.0,
        top_k: int = 8,
    ) -> dict:
        if self._table_has_rows("retrieval.peak_meaning_documents"):
            return self._search_peak_meaning_bundles(query_peaks, domain_hint, modality_hint, tolerance_cm, top_k)
        return self._legacy_search(query_peaks, domain_hint, modality_hint, tolerance_cm, top_k)

    def persist_run(self, result: dict, top_k: int) -> str:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        suffix = sha1(
            json.dumps(
                {
                    "query_peaks": result["query_peaks"],
                    "domain_hint": result.get("domain_hint"),
                    "modality_hint": result.get("modality_hint"),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:8]
        run_id = f"retrieval_run_{stamp}_{suffix}"
        direct_payload = result.get("support_bundle_results", result.get("direct_results", []))
        if result.get("pattern_results"):
            direct_payload = result["pattern_results"]
        context_payload = result.get("context_graph_results", result.get("context_results", []))
        with duckdb.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO retrieval.retrieval_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    run_id,
                    json.dumps(result["query_peaks"]),
                    result.get("domain_hint"),
                    result.get("modality_hint"),
                    result.get("tolerance_cm"),
                    top_k,
                    len(direct_payload),
                    json.dumps(direct_payload, sort_keys=True),
                    json.dumps(context_payload, sort_keys=True),
                    datetime.now(UTC),
                    "peak_meaning_retrieval_phase1_refinement",
                ],
            )
        return run_id
