# Phase C1.1 — Scoring Logic

## Score Decomposition

Every theme receives three scores:

### 1. Support Score
Measures raw evidence depth.

| Signal | Weight |
|---|---|
| Direct evidence edge | +3.0 per evidence row |
| Inferred mapping edge | +1.5 per mapping |
| Motif linkage | +2.0 per linked motif |
| Source diversity | +1.0 per source (capped at 10) |

### 2. Specificity Score
Measures how concentrated the evidence is for this theme relative to others.

```
evidence_share = theme_evidence_rows / total_evidence_rows
theme_concentration = 1 / total_themes
specificity_raw = evidence_share / theme_concentration
```

Broad theme penalty: themes in {protein, lipid, nucleic acid, carbohydrate, amino acid, membrane} are multiplied by 0.4 to reflect their ubiquitous nature.

Capped at 5.0.

### 3. Final Score
Blends support and specificity using geometric mean:

```
final = sqrt(support * max(specificity, 0.1))
```

This ensures that high support alone does not dominate — a theme also needs reasonable specificity to rank highly.

## Generic Hub Downweighting
Nodes in {stretch, stretching, vibration, breathing, deformation, ring} have their final score multiplied by 0.5.

## Confidence Levels

| Level | Criteria |
|---|---|
| high | direct >= 10, sources >= 3, specificity >= 0.5 |
| medium | direct >= 3 OR sources >= 2 |
| low | everything else |
