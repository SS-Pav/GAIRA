# Phase C1.5 — Comparator Warning Rules

## Feature-Level Coverage Flags

| Flag | Condition | Effect |
|---|---|---|
| `comparator_absent` | Theme/motif has 0 comparator direct support | NOT labeled "enriched"; confidence forced LOW |
| `insufficient_comparator` | Theme has < 2 comparator direct support | NOT labeled "enriched"; confidence capped MEDIUM |
| `low_query_support` | Theme has < 3 query direct support | NOT labeled "enriched" (insufficient basis) |
| (empty) | Adequate coverage on both sides | Normal enrichment logic applies |

## Global Comparator Adequacy

| Adequacy | Condition |
|---|---|
| adequate | comparator >= 5 rows AND >= 2 sources |
| sparse | comparator 2-4 rows OR < 2 sources |
| very_sparse | comparator < 2 rows |

## Caveats Generated

| Trigger | Caveat Text |
|---|---|
| `very_sparse` adequacy | "Comparator evidence is very sparse (N rows). Comparative results are unreliable." |
| `sparse` adequacy | "Comparator evidence is sparse (N rows). Enrichment ratios may be unstable." |
| themes with `comparator_absent` | "Comparator has no evidence for: X, Y. These cannot be interpreted as enriched." |
| themes with `insufficient_comparator` | "Some themes have insufficient comparator support for reliable enrichment ratios." |
| evidence_balance < 0.3 | "Evidence balance is low (0.XX). Query and comparator pools differ substantially." |

## Evidence Balance Metric
```
balance = 2 * min(query_rows, comparator_rows) / (query_rows + comparator_rows)
```
- 1.0 = perfectly balanced
- 0.0 = completely one-sided
- < 0.3 triggers warning
