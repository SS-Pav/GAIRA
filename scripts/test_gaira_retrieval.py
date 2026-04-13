"""
GAIRA LFM v1 — Phase 4 Retrieval Test

Tests the tiered retriever and evidence packet builder.
Shows source tier, score, and diversity per query.

Run:
    PYTHONPATH=src python scripts/test_gaira_retrieval.py
"""
from __future__ import annotations

from gaira.retrieval.text_query_retriever import TextQueryRetriever
from gaira.retrieval.evidence_packet_builder import build_packet


QUERIES = [
    "What biochemical changes are associated with HCC in serum SERS?",
    "How do Au and AgNP substrates differ for Raman spectroscopy?",
    "What are the BSV components and what do they measure?",
    "Which spectral regions distinguish CCA from HCC?",
    "What role does the nucleic acid backbone axis play in disease classification?",
]


def main():
    print("=" * 75)
    print("GAIRA Retrieval Test — Phase 4 (evidence-tiered)")
    print("=" * 75)

    retriever = TextQueryRetriever()
    n_sections = retriever.load_sources()
    print(f"\nLoaded {len(retriever.documents)} documents, {n_sections} sections\n")

    print("Sources by tier:")
    for s in retriever.source_summary():
        print(f"  [{s['tier']:20s}] {s['n_sections']:3d} sec  {s['path']}")

    for query in QUERIES:
        print(f"\n{'=' * 75}")
        print(f"QUERY: {query}")
        print("=" * 75)

        items = retriever.retrieve(query, top_k=8)

        if not items:
            print("  No results found.")
            continue

        for i, item in enumerate(items, 1):
            print(f"\n  [{i}] score={item.retrieval_score:6.1f}  "
                  f"tier={item.source_tier:20s}  source={item.source}")
            if item.title:
                print(f"      title: {item.title[:70]}")
            print(f"      text:  {item.text[:120]}...")

        packet = build_packet(query, items)
        tiers = packet["tier_summary"]
        tier_str = ", ".join(f"{k}={v}" for k, v in sorted(tiers.items()))
        print(f"\n  Packet: {len(packet['evidence'])} evidence, "
              f"{len(packet['provenance'])} provenance  |  tiers: {tier_str}")

    print(f"\n{'=' * 75}")
    print("Retrieval test complete.")
    print("=" * 75)


if __name__ == "__main__":
    main()
