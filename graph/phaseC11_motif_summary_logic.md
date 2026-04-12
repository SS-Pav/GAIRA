# Phase C1.1 — Motif Summary Logic

## What Is Shown
For each query result, the top 8-12 motifs by member count are displayed with:
- Motif ID
- Subfamily (e.g., "tryptophan", "lipid", "collagen")
- Family (e.g., "protein_support", "lipid_support")
- Member count (evidence rows in this motif for this query)
- Condition enrichment assessment

## Condition Enrichment Labels

| Label | Criteria | Meaning |
|---|---|---|
| condition-enriched | 3-19 members | Concentrated, potentially condition-specific |
| broadly-shared | 20+ members | Ubiquitous across many conditions |
| sparse | < 3 members | Weak signal |

## Interpretation Guidance
- **condition-enriched** motifs are the most informative for disease-specific interpretation
- **broadly-shared** motifs represent background biology (e.g., protein backbone is always present)
- **sparse** motifs may be real but need more evidence before relying on them
