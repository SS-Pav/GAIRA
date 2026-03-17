from __future__ import annotations

import pandas as pd


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


def _weight_tier2(domain: str, row: pd.Series) -> tuple[float, str]:
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


def rerank_grounding_hits(df: pd.DataFrame, domain: str, tier: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    reranked_rows = []
    for row in df.to_dict(orient="records"):
        row_series = pd.Series(row)
        if tier == "tier1":
            weight, reason = _weight_tier1(domain=domain, row=row_series)
        elif tier == "tier2":
            weight, reason = _weight_tier2(domain=domain, row=row_series)
        else:
            raise ValueError(f"Unsupported tier '{tier}'.")

        base_score = float(row_series["score"])
        reranked_score = base_score * weight
        reranked_rows.append(
            {
                **row,
                "base_score": base_score,
                "domain_relevance_weight": weight,
                "reranked_score": reranked_score,
                "rerank_reason": reason,
            }
        )

    reranked_df = pd.DataFrame(reranked_rows)
    return reranked_df.sort_values(
        ["reranked_score", "base_score", "source_dataset_id", "source_label"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
