# Phase C1.4 — Coverage-Aware Confidence Rules

## Thresholds

| Parameter | Value | Effect |
|---|---|---|
| MIN_COMPARATOR_EVIDENCE | 5 | Below: comparator flagged sparse |
| MIN_COMPARATOR_SOURCES | 2 | Below: comparator lacks diversity |
| MIN_THEME_SUPPORT_FOR_ENRICHMENT | 3 | Theme needs >= 3 direct to claim enrichment |

## Coverage Flags on Themes

| Flag | Condition | Meaning |
|---|---|---|
| (empty) | Normal coverage | Enrichment ratio is reliable |
| low_comparator | Comparator sparse + enrichment >= 2x | Enrichment may reflect denominator collapse |
| low_query_support | Query direct < 3 | Insufficient query evidence to claim enrichment |

## Confidence Adjustments

- If coverage_flag = "low_comparator" and confidence was "high" -> downgrade to "medium"
- If comparator_adequacy = "sparse" or "very_sparse" -> enriched themes get smaller boost (1.1x instead of 1.3x)
- If comparator_adequacy = "very_sparse" -> add explicit caveat about unstable ratios

## Design Principle
The system should never present a high enrichment ratio (e.g., "10x enriched") without making the absolute numbers visible. A user should always be able to see whether "10x" means "30 vs 3" or "2 vs 0.2".
