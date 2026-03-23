from __future__ import annotations

import pandas as pd

from gaira.query_routing import classify_knowledge_family, classify_support_family, routing_weight


def _weight_tier1(domain: str, row: pd.Series) -> tuple[float, str]:
    dataset_id = str(row.get("source_dataset_id", ""))
    base_score = float(row.get("score", 0.0))

    if domain == "serum":
        if dataset_id == "serum_ag_colloids_grounding":
            if base_score >= 0.9:
                return 1.30, "serum query boost for study-matched Ag-colloid grounding"
            return 1.20, "serum query boost for study-matched Ag-colloid grounding"
        if dataset_id == "ramanbiolib":
            return 1.00, "neutral broad molecular grounding"
        return 0.98, "slight neutral serum fallback"

    if domain == "ev":
        if dataset_id == "ramanbiolib":
            return 1.00, "neutral broad shared grounding for EV"
        if dataset_id == "serum_ag_colloids_grounding":
            if base_score >= 0.95:
                return 0.95, "retain very strong cross-domain serum grounding visibility"
            if base_score >= 0.80:
                return 0.85, "mild penalty for strong but serum-specific grounding"
            return 0.72, "default penalty for serum-specific grounding under EV queries"
        return 0.95, "slight penalty for non-EV study-specific grounding"

    return 1.0, "no domain-specific reranking"


def _routing_family_for_row(tier: str, row: pd.Series) -> str:
    row_dict = row.to_dict()
    if tier == "tier1":
        return classify_support_family(row_dict)
    if str(row.get("result_type", "")) == "semantic_region_support":
        return classify_knowledge_family(row_dict)
    if tier == "tier2":
        if str(row.get("evidence_tier", "")).startswith("tier2_knowledge"):
            return classify_knowledge_family(row_dict)
        return classify_support_family(row_dict)
    return "shared_generic"


def _ev_target_match_weight(
    row: pd.Series,
    query_source_dataset_id: str | None,
) -> tuple[float, str] | None:
    if not query_source_dataset_id:
        return None

    target_dataset_id = str(row.get("target_dataset_id", "")).strip()
    if not target_dataset_id:
        return None

    target_ids = {part.strip() for part in target_dataset_id.split(",") if part.strip()}
    if query_source_dataset_id in target_ids:
        return 1.22, "EV same-dataset support bonus"

    if any(
        token in target_ids
        for token in ["small2023_ev", "shine_ev_sers", "diabetes_plasma_ev_sers"]
    ):
        return 0.95, "retain cross-EV support visibility with slight same-dataset preference"

    return None


def _weight_tier2(
    domain: str,
    row: pd.Series,
    query_source_dataset_id: str | None = None,
) -> tuple[float, str]:
    dataset_id = str(row.get("source_dataset_id", ""))
    result_type = str(row.get("result_type", ""))
    base_score = float(row.get("score", 0.0))

    if domain == "serum":
        if dataset_id == "serum_ag_colloids_literature_grounding":
            if result_type == "digitized_support_spectrum":
                return 0.90, "serum support-only digitized spectrum kept below primary evidence"
            return 1.10, "serum query boost for serum literature support"
        return 1.00, "neutral support weight"

    if domain == "ev":
        ev_target_weight = _ev_target_match_weight(
            row=row,
            query_source_dataset_id=query_source_dataset_id,
        )
        if ev_target_weight is not None and dataset_id != "serum_ag_colloids_literature_grounding":
            return ev_target_weight
        if dataset_id == "serum_ag_colloids_literature_grounding":
            if result_type == "digitized_support_spectrum":
                if base_score >= 0.75:
                    return 0.55, "retain unusually strong digitized serum support but keep penalized for EV"
                return 0.35, "strong penalty for serum digitized support under EV queries"
            if base_score >= 0.75:
                return 0.70, "retain unusually strong serum literature support under EV queries"
            return 0.45, "default penalty for serum literature support under EV queries"
        return 1.00, "neutral support weight"

    return 1.0, "no domain-specific reranking"


def rerank_grounding_hits(
    df: pd.DataFrame,
    domain: str,
    tier: str,
    query_source_dataset_id: str | None = None,
    query_routing_family: str | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    reranked_rows = []
    for row in df.to_dict(orient="records"):
        row_series = pd.Series(row)
        if tier == "tier1":
            weight, reason = _weight_tier1(domain=domain, row=row_series)
        elif tier == "tier2":
            weight, reason = _weight_tier2(
                domain=domain,
                row=row_series,
                query_source_dataset_id=query_source_dataset_id,
            )
        else:
            raise ValueError(f"Unsupported tier '{tier}'.")

        support_family = _routing_family_for_row(tier=tier, row=row_series)
        routing_relevance_weight = routing_weight(
            query_routing_family,
            support_family,
            channel="knowledge" if str(row_series.get("evidence_tier", "")).startswith("tier2_knowledge") else "support",
        )
        base_score = float(row_series["score"])
        reranked_score = base_score * weight * routing_relevance_weight
        reranked_rows.append(
            {
                **row,
                "base_score": base_score,
                "domain_relevance_weight": weight,
                "routing_relevance_weight": routing_relevance_weight,
                "support_family": support_family,
                "reranked_score": reranked_score,
                "rerank_reason": (
                    f"{reason}; routing_family={query_routing_family or 'legacy'}; "
                    f"candidate_family={support_family}; routing_weight={routing_relevance_weight:.2f}"
                ),
            }
        )

    reranked_df = pd.DataFrame(reranked_rows)
    return reranked_df.sort_values(
        ["reranked_score", "base_score", "source_dataset_id", "source_label"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
