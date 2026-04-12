# Phase C1.5 — Normalization Logic

## Raw Enrichment (preserved from C1.4)
```
enrichment_ratio = (query_direct + 1) / (comparator_direct + 1)
```
+1 smoothing to avoid division by zero.

## Normalized Enrichment (NEW in C1.5)
```
query_direct_norm = query_direct / total_query_evidence_rows
comp_direct_norm = comparator_direct / total_comparator_evidence_rows
norm_enrichment_ratio = query_direct_norm / max(comp_direct_norm, 0.0001)
```

### Why Normalize
Raw counts can be misleading when query and comparator have different total evidence sizes. A theme with 20 query rows out of 100 total (20%) is more enriched than 20 out of 500 (4%), even though raw counts are identical.

### Denominator Choice
- Query denominator: `total_query_evidence_rows` (all evidence linked to the query condition)
- Comparator denominator: `total_comparator_evidence_rows` (all evidence linked to comparator)

### When Normalized Enrichment Is Used
The interpretation logic in C1.5 uses BOTH raw and normalized:
- `enriched` requires: `norm_enrichment >= 2.0 AND raw_enrichment >= 1.5`
- `depleted` requires: `norm_enrichment <= 0.5 AND raw_enrichment <= 0.67`
- This dual threshold prevents false enrichment from size imbalance

### Display
Both raw and normalized enrichment appear in the UI theme table as "Raw" and "Norm" columns.
