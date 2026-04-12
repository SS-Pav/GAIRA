# Phase C1.1 — Caveat Rules

## When Caveats Are Generated

| Condition | Caveat Text |
|---|---|
| Evidence rows < 5 | "Low evidence support — interpret with caution." |
| Sources < 2 | "Evidence from fewer than 2 independent sources — single-source bias possible." |
| 2+ of top-3 themes are broad | "Top themes (X, Y) are broad biochemical classes... may reflect shared background biology." |
| Top-3 FGs include generic node | "Dominant functional group(s) (X) are structurally generic. Multiple biochemical origins possible." |
| Zero inferred support on chemistry query | "No inferred chemistry→biology mappings contributed." |
| 3+ of top-5 motifs are broadly-shared | "Most supporting motifs are broadly shared across many conditions." |
| Top-3 themes have specificity < 0.3 | "Top-ranked themes have low condition specificity." |

## Design Principles
- Caveats are additive — multiple can fire for the same query
- They are deterministic and reproducible
- They warn about overinterpretation, not about data quality
- They do NOT suppress results — they annotate them
