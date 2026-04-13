"""
GAIRA LFM v1 — Full Corpus Retrieval Quality Audit

Tests retrieval quality after expanding to the full GAIRA structured evidence corpus.
Verifies grounding components surface, context sources appear, meta docs are suppressed,
and results are biologically relevant.

Run:
    PYTHONPATH=src python scripts/test_gaira_full_corpus_retrieval.py
"""
from __future__ import annotations

from gaira.retrieval.text_query_retriever import TextQueryRetriever
from gaira.retrieval.source_registry import TIER_DISPLAY_NAMES


QUERIES = [
    "What biochemical changes are associated with HCC in serum SERS spectra?",
    "How do CCA and HCC differ biochemically in serum Raman/SERS evidence?",
    "What evidence supports purine nucleotide changes in liver disease?",
    "What glycan-related themes appear in Raman/SERS evidence?",
    "What aromatic amino acid signals are associated with cancer serum SERS?",
    "What nucleic-acid-related evidence appears in EV Raman/SERS?",
    "Compare healthy vs liver disease biochemical composition in GAIRA evidence.",
    "What serum-specific caveats matter when interpreting liver SERS evidence?",
]


def main():
    print("=" * 85)
    print("GAIRA Full Corpus Retrieval Quality Audit")
    print("=" * 85)

    retriever = TextQueryRetriever()
    n_sections = retriever.load_sources()
    n_docs = len(retriever.documents)
    print(f"\nCorpus: {n_docs} documents, {n_sections} sections\n")

    # Coverage by tier
    tier_docs: dict[str, int] = {}
    tier_secs: dict[str, int] = {}
    for s in retriever.source_summary():
        t = s["tier"]
        tier_docs[t] = tier_docs.get(t, 0) + 1
        tier_secs[t] = tier_secs.get(t, 0) + s["n_sections"]

    print("Source coverage by tier:")
    for tier in sorted(tier_docs.keys()):
        label = TIER_DISPLAY_NAMES.get(tier, tier)
        print(f"  {label:24s}  {tier_docs[tier]:2d} docs  {tier_secs[tier]:3d} sections")
    print(f"  {'TOTAL':24s}  {n_docs:2d} docs  {n_sections:3d} sections")

    # Run queries
    results = []
    for qi, query in enumerate(QUERIES, 1):
        print(f"\n{'─' * 85}")
        print(f"Q{qi}: {query}")
        print("─" * 85)

        items = retriever.retrieve(query, top_k=10)

        if not items:
            print("  (no results)")
            results.append({"tiers": {}, "n": 0, "diversity": 0})
            continue

        tiers: dict[str, int] = {}
        for i, item in enumerate(items, 1):
            t = item.source_tier
            tiers[t] = tiers.get(t, 0) + 1
            tier_label = TIER_DISPLAY_NAMES.get(t, t)[:12]
            dname = item.source_display_name or "?"
            section = (item.title or "")[:50]
            print(f"  {i:2d}. [{tier_label:12s}] {item.retrieval_score:5.1f}  "
                  f"{dname:24s}  {section}")

        # Quality flags
        has_grounding = any(t in ("grounding_component", "evidence_rules") for t in tiers)
        has_context = "context_source" in tiers
        has_benchmark = "benchmark_summary" in tiers
        meta_count = tiers.get("meta_summary", 0)
        diversity = len(set(item.source for item in items))

        flags = []
        if has_grounding:
            flags.append("GROUNDING")
        if has_context:
            flags.append("CONTEXT")
        if has_benchmark:
            flags.append("BENCHMARK")
        if meta_count > len(items) // 2:
            flags.append("META_DOMINATED")
        flags.append(f"div={diversity}")

        tier_str = " ".join(f"{TIER_DISPLAY_NAMES.get(k,'?')[:8]}={v}" for k, v in sorted(tiers.items()))
        print(f"\n  Tiers: {tier_str}")
        print(f"  Flags: {' | '.join(flags)}")

        results.append({"tiers": tiers, "n": len(items), "diversity": diversity,
                         "has_grounding": has_grounding, "has_context": has_context})

    # Overall
    print(f"\n{'=' * 85}")
    print("AUDIT SUMMARY")
    print("=" * 85)

    n_q = len(QUERIES)
    n_grounding = sum(1 for r in results if r.get("has_grounding"))
    n_context = sum(1 for r in results if r.get("has_context"))
    n_meta_dom = sum(1 for r in results if r.get("tiers", {}).get("meta_summary", 0) > r["n"] // 2)
    avg_div = sum(r["diversity"] for r in results) / n_q if n_q else 0

    print(f"  Queries:               {n_q}")
    print(f"  With grounding:        {n_grounding}/{n_q}")
    print(f"  With context:          {n_context}/{n_q}")
    print(f"  Meta-dominated:        {n_meta_dom}/{n_q}")
    print(f"  Avg source diversity:  {avg_div:.1f}")

    if n_meta_dom == 0 and n_grounding >= n_q * 0.7:
        print(f"\n  VERDICT: Retrieval quality is good.")
    elif n_meta_dom == 0:
        print(f"\n  VERDICT: Retrieval quality is acceptable but grounding coverage is limited.")
    else:
        print(f"\n  VERDICT: Meta docs dominate {n_meta_dom} queries — needs tuning.")

    print("=" * 85)


if __name__ == "__main__":
    main()
