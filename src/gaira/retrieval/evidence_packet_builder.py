"""
Evidence packet builder for GAIRA text queries.

Converts retrieved items into the packet format expected by
prompt_builder.build_prompt(). Phase 4: tier-aware ordering —
grounded evidence first, meta-summaries last.
"""
from __future__ import annotations

from typing import Sequence

from gaira.retrieval.text_query_retriever import RetrievedItem


# Preferred ordering: evidence items are sorted by tier priority,
# then by retrieval score within each tier.
_TIER_SORT_ORDER = {
    "grounding_component": 0,
    "evidence_rules": 1,
    "context_source": 2,
    "benchmark_summary": 3,
    "analysis_summary": 4,
    "meta_summary": 5,
    # Legacy aliases
    "grounded_evidence": 0,
    "domain_context": 2,
    "spectral_query": 4,
}

STANDARD_CAVEATS = [
    "Raman/SERS peak assignments are many-to-many: a single spectral region "
    "may correspond to multiple molecular origins.",
    "Literature support varies in specificity — some components have strong "
    "multi-source grounding while others rely on fewer studies.",
    "Substrate and sample preparation differences between studies may alter "
    "the expression of certain biochemical themes.",
]

DEFAULT_DOMAIN_CONTEXT = (
    "Evidence is drawn from a curated GAIRA registry of Raman/SERS literature, "
    "with strongest coverage for serum SERS and liver-related conditions "
    "(HCC, CCA, NAFLD). "
    "Evidence is structured around 8 BSV biochemical components: "
    "membrane_lipid, protein_backbone, aromatic_amino_acid, "
    "purine_nucleotide, pyrimidine_nucleotide, glycan_carbohydrate, "
    "redox_metabolite, nucleic_acid_backbone."
)


def build_packet(
    query: str,
    retrieved_items: Sequence[RetrievedItem],
    domain_context: str | None = None,
    extra_caveats: Sequence[str] | None = None,
) -> dict:
    """Build prompt-ready evidence packet from retrieved items.

    Items are reordered so grounded evidence appears first in the prompt,
    followed by domain context and benchmark data. Meta-summaries go last.
    Deduplicates near-identical text.

    Returns:
        dict with keys: evidence, provenance, caveats, domain_context, tier_summary
    """
    # Sort by tier priority, then by score within tier
    sorted_items = sorted(
        retrieved_items,
        key=lambda it: (_TIER_SORT_ORDER.get(it.source_tier, 9), -it.retrieval_score),
    )

    # Deduplicate
    evidence = []
    seen: set[str] = set()
    for item in sorted_items:
        key = item.text[:100].lower().strip()
        if key in seen:
            continue
        seen.add(key)
        evidence.append({
            "title": item.title,
            "text": item.text,
            "source": item.source,
        })

    # Provenance (unique sources, ordered by first appearance)
    sources_seen: set[str] = set()
    provenance = []
    for item in sorted_items:
        if item.source not in sources_seen:
            sources_seen.add(item.source)
            provenance.append(item.source)

    # Caveats
    caveats = list(STANDARD_CAVEATS)
    if extra_caveats:
        caveats.extend(extra_caveats)

    # Tier summary for transparency
    tier_counts: dict[str, int] = {}
    for item in sorted_items:
        tier_counts[item.source_tier] = tier_counts.get(item.source_tier, 0) + 1

    return {
        "evidence": evidence,
        "provenance": provenance,
        "caveats": caveats,
        "domain_context": domain_context or DEFAULT_DOMAIN_CONTEXT,
        "tier_summary": tier_counts,
    }
