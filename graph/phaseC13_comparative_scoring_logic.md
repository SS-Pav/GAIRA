# Phase C1.3 — Comparative Scoring Logic

## Single-Condition Mode
Same as C1.2: support + specificity blend.

## Comparative Mode (pairwise / one-vs-rest)
For each theme, compute:

| Metric | Formula |
|---|---|
| support_in_query | direct * 3.0 + inferred * 1.5 + motifs * 2.0 + sources |
| support_in_comparator | comparator_direct * 3.0 |
| enrichment_ratio | (query_direct + 1) / (comparator_direct + 1) |
| contrast_score | support_in_query - support_in_comparator |

### Interpretation Labels

| Label | Condition |
|---|---|
| enriched | enrichment_ratio >= 2.0 |
| depleted | enrichment_ratio <= 0.5 |
| shared | 0.8 <= enrichment_ratio <= 1.25 |
| associated | everything else |

### Score Modifiers
- Enriched themes: final_score * 1.3 (boosted)
- Depleted themes: final_score * 0.5 (downweighted)

### Motif Interpretation

| Label | Condition |
|---|---|
| enriched | query >= 3 AND comparator == 0, OR query > 2x comparator |
| comparator-associated | comparator >= 3 AND query == 0, OR comparator > 2x query |
| shared | neither condition dominates |
| query-associated | single-condition mode (no comparison) |

## Design Principle
"Enriched" means graph-enriched relative to comparator evidence — NOT clinically validated.
