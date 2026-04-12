# Phase C1.3 — Motif Contribution Logic

## Single-Condition Mode
Each motif shows:
- subfamily, family
- member count (evidence rows in this motif for this query)
- enrichment: broadly-shared / condition-enriched / sparse
- interpretation: query-associated

## Comparative Mode
Each motif additionally shows:
- comparator_members (evidence in comparator condition)
- interpretation:
  - **enriched**: significantly more evidence in query than comparator
  - **comparator-associated**: more in comparator
  - **shared**: roughly equal
  - **query-associated**: single mode (no comparator)

## Why Motifs Matter for Interpretation
A theme like "lipid" may rank high globally because lipid SERS signal is ubiquitous. But the *specific motifs* within the lipid theme may differ between conditions. The motif table helps explain:
- Which lipid subfamilies are present (phospholipid vs cholesterol vs fatty acid)
- Whether those subfamilies are condition-enriched or background
- Whether a nucleic acid motif that looks visually prominent in the graph is actually condition-specific
