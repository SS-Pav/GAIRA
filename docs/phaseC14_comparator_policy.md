# Phase C1.4 — Comparator Resolution Policy

## How comparators are resolved

### Pairwise: `Compare X vs Y`
- Both X and Y are resolved to their canonical condition names
- Each condition's evidence is retrieved independently via the same Cypher template
- Evidence pools are compared theme-by-theme and motif-by-motif

### One-vs-rest: `What is enriched in X vs rest?`
- X is the query condition
- Comparator = all conditions except X (background pool)
- Background is retrieved via a modified Cypher that excludes the query condition

### Sample type matching
- Sample type is NOT currently constrained at query time
- Instead, the dominant sample type is **inferred** from retrieved evidence and **reported** in the comparator summary
- If mixed matrices contributed substantially, a caveat is generated
- Future: explicit sample-type filtering via query syntax (e.g., "Compare HCC serum vs healthy serum")

### Comparator adequacy assessment

| Adequacy | Condition |
|---|---|
| adequate | comparator evidence >= 5 rows AND >= 2 sources |
| sparse | comparator evidence 2-4 rows OR < 2 sources |
| very_sparse | comparator evidence < 2 rows |

When comparator is sparse:
- Enrichment ratios are flagged as potentially unstable
- Confidence is capped at medium (no high confidence with sparse comparator)
- A caveat explains the limitation
