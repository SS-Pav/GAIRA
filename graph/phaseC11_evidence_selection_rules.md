# Phase C1.1 — Evidence Row Selection Rules

## Goal
Surface the most interpretable, highest-quality evidence rows in the demo — not just the first N.

## Quality Scoring Per Row

| Signal | Weight | Rationale |
|---|---|---|
| Meaning length (per 40 chars) | +1.0, cap 2.0 | Longer meanings are more informative |
| chemistry_plus_biomolecule level | +3.0 | Strongest assignment type |
| theme_only level | +2.0 | Good thematic content |
| chemistry_only level | +1.5 | Chemistry-only still useful |
| context_only level | +0.5 | Weak assignment |
| Fragment (< 10 chars or ends "...") | -1.0 | Truncated/noisy |
| Classifier contamination | -2.0 | Methods text leaked into meaning |

## Source Diversity
After scoring, selection prefers diverse sources:
- Max 2 rows from same source
- Only applies when candidate pool > max display count

## Display Count
Default: 8 rows per query result.
