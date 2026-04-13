"""
Confidence composition for GAIRA responses.

Produces a structured, transparent confidence summary based on the
evidence mix — not a calibrated probability. Describes *how well
grounded* the answer is, not *how correct* it is.
"""
from __future__ import annotations

from typing import Sequence

from gaira.retrieval.text_query_retriever import RetrievedItem


# Canonical bucket mapping: raw source_tier → counting bucket
_CANONICAL_BUCKET = {
    # Current names
    "grounding_component": "grounded",
    "evidence_rules":      "grounded",
    "context_source":      "context",
    "benchmark_summary":   "benchmark",
    "analysis_summary":    "benchmark",
    "meta_summary":        "meta",
    # Legacy names (kept for safety)
    "grounded_evidence":   "grounded",
    "domain_context":      "context",
    "spectral_query":      "benchmark",
}


def _count_buckets(retrieved_items: Sequence[RetrievedItem]) -> dict[str, int]:
    """Count items by canonical bucket."""
    buckets: dict[str, int] = {"grounded": 0, "context": 0, "benchmark": 0, "meta": 0}
    for item in retrieved_items:
        raw = item.source_tier or "unknown"
        bucket = _CANONICAL_BUCKET.get(raw, "meta")
        buckets[bucket] = buckets.get(bucket, 0) + 1
    return buckets


def compose_confidence(
    retrieved_items: Sequence[RetrievedItem],
    packet: dict,
    section_links: dict[str, list[dict]] | None = None,
) -> dict:
    """Compose a confidence summary for the current answer."""
    total = len(retrieved_items)
    buckets = _count_buckets(retrieved_items)

    grounded = buckets["grounded"]
    context = buckets["context"]
    benchmark = buckets["benchmark"]
    meta = buckets["meta"]

    unique_sources = set(item.source for item in retrieved_items)
    source_diversity = len(unique_sources)

    # Section support sparsity
    sparse_sections = []
    well_supported_sections = []
    if section_links:
        for field, supports in section_links.items():
            if not supports:
                sparse_sections.append(field)
            elif len(supports) >= 2 and supports[0]["support_score"] > 0.15:
                well_supported_sections.append(field)

    # Contradiction signal
    texts_combined = " ".join(item.text.lower() for item in retrieved_items)
    conflict_terms = [
        ("enriched", "depleted"), ("increased", "decreased"),
        ("opposite", "consistent"), ("variable", "consistent"),
    ]
    contradiction_signals = [
        f"{pos}/{neg}" for pos, neg in conflict_terms
        if pos in texts_combined and neg in texts_combined
    ]
    has_contradiction = len(contradiction_signals) > 0

    # Overall label
    if total == 0:
        label = "no evidence"
        explanation = "No evidence was retrieved for this query."
    elif grounded / total >= 0.5 and source_diversity >= 3:
        label = "strongly grounded"
        explanation = (
            f"Majority of evidence ({grounded}/{total}) is grounded, "
            f"drawn from {source_diversity} distinct sources."
        )
    elif (grounded + context) / total >= 0.5:
        label = "well grounded"
        explanation = (
            f"Evidence is a mix of grounded ({grounded}) and contextual ({context}) items. "
            f"{source_diversity} distinct sources contribute."
        )
    elif benchmark / total >= 0.5:
        label = "benchmark-supported"
        explanation = (
            f"Answer relies primarily on benchmark/analysis summaries ({benchmark}/{total}). "
            "Interpretation is descriptive rather than directly evidence-grounded."
        )
    elif grounded >= 1:
        label = "partially grounded"
        explanation = (
            f"Some grounded evidence ({grounded}/{total}), "
            f"but majority is contextual or summary-derived."
        )
    else:
        label = "weakly grounded"
        explanation = (
            "No grounded evidence items retrieved. "
            "Answer is based on context and summaries only."
        )

    if has_contradiction:
        explanation += (
            f" Evidence contains potentially conflicting signals "
            f"({', '.join(contradiction_signals[:3])}). "
            "The answer should acknowledge this explicitly."
        )

    if sparse_sections:
        section_names = [s.replace("_", " ") for s in sparse_sections]
        explanation += f" Sections with weak evidence support: {', '.join(section_names)}."

    return {
        "label": label,
        "explanation": explanation,
        "grounded_count": grounded,
        "context_count": context,
        "benchmark_count": benchmark,
        "meta_count": meta,
        "total": total,
        "source_diversity": source_diversity,
        "has_contradiction": has_contradiction,
        "contradiction_signals": contradiction_signals,
        "sparse_sections": sparse_sections,
        "well_supported_sections": well_supported_sections,
    }
