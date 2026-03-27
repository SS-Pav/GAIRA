from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from gaira.config import get_database_path
from gaira.grounding_search import SpectrumQuery
from gaira.inference import (
    GAIRAInferenceEngine,
    InferenceRequest,
    load_ev_class_mean_query,
    load_grounding_class_mean_query,
    load_serum_class_mean_query,
)


@lru_cache(maxsize=1)
def get_engine() -> GAIRAInferenceEngine:
    return GAIRAInferenceEngine(get_database_path(), theme_layer_version="v3")


def build_database_request(
    domain: str,
    dataset_id: str,
    class_label: str,
    family_label: str,
    processing_version: str | None = None,
) -> InferenceRequest:
    db_path = get_database_path()
    if domain == "grounding":
        return load_grounding_class_mean_query(db_path, dataset_id, class_label, experiment_family=family_label, processing_version=processing_version)
    if domain == "ev":
        return load_ev_class_mean_query(db_path, dataset_id, class_label, family_label, processing_version=processing_version)
    return load_serum_class_mean_query(db_path, dataset_id, class_label, family_label, processing_version=processing_version)


def build_uploaded_request(
    *,
    domain: str,
    query_id: str,
    query_label: str,
    family_label: str,
    source_dataset_id: str,
    x_values: list[float],
    y_values: list[float],
    sample_type: str,
    modality: str,
    substrate_context: str | None,
    use_case_domain: str | None,
) -> InferenceRequest:
    query = SpectrumQuery(
        query_id=query_id,
        query_label=query_label,
        query_family=family_label,
        source_dataset_id=source_dataset_id,
        x=x_values,
        y=y_values,
        notes="Uploaded processed spectrum",
    )
    return InferenceRequest(
        domain=domain,
        query_id=query_id,
        query_label=query_label,
        query_family=family_label,
        source_dataset_id=source_dataset_id,
        spectrum_query=query,
        sample_type=sample_type,
        modality=modality,
        substrate_context=substrate_context,
        use_case_domain=use_case_domain,
    )


def run_general_inference(request: InferenceRequest) -> dict[str, Any]:
    engine = get_engine()
    return engine.run_inference(replace(request, disable_query_routing=True, forced_query_family=None))


def run_query_aware_inference(
    request: InferenceRequest,
    *,
    sample_type: str | None,
    modality: str | None,
    substrate_context: str | None,
    use_case_domain: str | None,
    forced_query_family: str | None = None,
) -> dict[str, Any]:
    engine = get_engine()
    routed_request = replace(
        request,
        disable_query_routing=False,
        sample_type=sample_type or request.sample_type,
        modality=modality or request.modality,
        substrate_context=substrate_context or request.substrate_context,
        use_case_domain=use_case_domain or request.use_case_domain,
        forced_query_family=forced_query_family,
    )
    return engine.run_inference(routed_request)


def concise_result_summary(result: dict[str, Any]) -> str:
    dominant = ", ".join(result.get("dominant_themes", [])[:3]) or "No dominant themes"
    cautions = ", ".join(result.get("biochemical_global_caveats", [])[:3]) or "No global caveats"
    family = result.get("query_routing_family") or "legacy"
    return f"Routing family: {family}. Dominant themes: {dominant}. Main cautions: {cautions}."


def _humanize_label(value: str) -> str:
    return str(value).replace("_associated", "").replace("_caution", "").replace("_", " ").strip().title()


def synthesize_result_summary(result: dict[str, Any]) -> str:
    positive_df = format_theme_table(result, "positive")
    caution_df = format_theme_table(result, "caution")
    top_theme = _humanize_label(str(positive_df.iloc[0]["theme_name"])) if not positive_df.empty else "non-specific signal"
    confidence = float(positive_df.iloc[0]["confidence"]) if not positive_df.empty else 0.0
    top_cautions = [_humanize_label(str(row["theme_name"])) for _, row in caution_df.head(2).iterrows()] if not caution_df.empty else []
    family = result.get("query_routing_family") or "legacy"
    support_hits = int(result.get("family_matched_support_hits", 0))
    if top_cautions:
        caution_text = " and ".join(top_cautions[:2]).lower()
        return (
            f"GAIRAM v1 reads this spectrum as primarily {top_theme.lower()} with "
            f"{'moderate' if confidence >= 0.15 else 'limited'} support. "
            f"Routing family is {family}, with {support_hits} family-matched support hits. "
            f"Interpretation remains constrained by {caution_text}."
        )
    return (
        f"GAIRAM v1 reads this spectrum as primarily {top_theme.lower()} with "
        f"{'moderate' if confidence >= 0.15 else 'limited'} support. "
        f"Routing family is {family}."
    )


def format_evidence_hits(result: dict[str, Any], key: str, label_key: str) -> list[dict[str, str]]:
    rows = []
    for item in result.get(key, [])[:5]:
        rows.append(
            {
                "label": str(item.get(label_key) or item.get("source_label") or item.get("document_id") or ""),
                "source": str(item.get("source_dataset_id") or item.get("document_id") or ""),
                "family": str(item.get("support_family") or item.get("source_family") or item.get("result_type") or ""),
                "detail": str(item.get("notes") or item.get("matched_tokens") or ""),
            }
        )
    return rows


def format_theme_table(result: dict[str, Any], category: str) -> pd.DataFrame:
    rows = [
        {
            "theme_name": row["theme_name"],
            "display_name": _humanize_label(str(row["theme_name"])),
            "score": float(row["score"]),
            "confidence": float(row["confidence"]),
            "specificity_index": float(row.get("specificity_index", 0.0)),
        }
        for row in result.get("biochemical_theme_outputs", [])
        if row.get("category") == category
    ]
    return (
        pd.DataFrame(rows).sort_values("score", ascending=False)
        if rows
        else pd.DataFrame(columns=["theme_name", "display_name", "score", "confidence", "specificity_index"])
    )


def compare_results(general_result: dict[str, Any], routed_result: dict[str, Any]) -> dict[str, Any]:
    general_themes = {row["theme_name"]: float(row["score"]) for row in general_result.get("biochemical_theme_outputs", []) if row.get("category") == "positive"}
    routed_themes = {row["theme_name"]: float(row["score"]) for row in routed_result.get("biochemical_theme_outputs", []) if row.get("category") == "positive"}
    changed_themes = []
    for theme_name in sorted(set(general_themes) | set(routed_themes)):
        delta = routed_themes.get(theme_name, 0.0) - general_themes.get(theme_name, 0.0)
        if abs(delta) >= 0.01:
            changed_themes.append({"theme_name": theme_name, "delta": round(delta, 4)})

    general_support = [str(row.get("source_label", "")) for row in general_result.get("tier2_support_hits", [])[:5]]
    routed_support = [str(row.get("source_label", "")) for row in routed_result.get("tier2_support_hits", [])[:5]]
    added_support = [label for label in routed_support if label and label not in general_support]
    removed_support = [label for label in general_support if label and label not in routed_support]
    general_cautions = {row["theme_name"]: float(row["score"]) for row in general_result.get("biochemical_theme_outputs", []) if row.get("category") == "caution"}
    routed_cautions = {row["theme_name"]: float(row["score"]) for row in routed_result.get("biochemical_theme_outputs", []) if row.get("category") == "caution"}
    changed_cautions = []
    for theme_name in sorted(set(general_cautions) | set(routed_cautions)):
        delta = routed_cautions.get(theme_name, 0.0) - general_cautions.get(theme_name, 0.0)
        if abs(delta) >= 0.01:
            changed_cautions.append({"theme_name": _humanize_label(theme_name), "delta": round(delta, 4)})

    return {
        "changed_themes": changed_themes[:8],
        "changed_cautions": changed_cautions[:8],
        "added_support": added_support[:5],
        "removed_support": removed_support[:5],
        "routing_family": routed_result.get("query_routing_family") or "legacy",
        "family_matched_support_hits": int(routed_result.get("family_matched_support_hits", 0)),
        "family_matched_context_hits": int(routed_result.get("family_matched_context_hits", 0)),
    }
