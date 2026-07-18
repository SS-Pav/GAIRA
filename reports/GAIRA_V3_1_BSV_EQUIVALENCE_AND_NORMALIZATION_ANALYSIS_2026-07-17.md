# GAIRA V3.1 — BSV Equivalence & Normalization Analysis

**Date:** 2026-07-17 · Experiment: `tools/build_diabetes_equivalence.py` → `data/generated/diabetes_equivalence/`.

## The core question, answered
> Did the historical "better-looking" diabetes radar come from **axis-wise normalization**, a **different upstream BSV engine**, or **both**?

**Answer: primarily normalization; the engine difference is real but confined to a single axis (G10 redox).**

## Three-path experiment (same 63 EV spectra)
- **Path B** — V3 plain `build_report`.
- **Path A** — `build_report_diabetes` (historical engine: tightened G10 window + co-band-gated thiol boost).
- **Historical** — saved 1322 (=Path A engine) and 1304 (=plain engine) tables.

### Engine effect (Path A vs Path B, identical spectra)
| Axes | Pearson | max abs diff |
| --- | --- | --- |
| G01–G09, G11 (10 axes) | **1.0000** | **0.0000** |
| G10 redox | 0.9999 | 0.0273 |

The diabetes engine changes **only** the redox axis. Historical 1304 vs 1322: **G10 only** (max 0.060).

### Input-spectra effect (V3 autoresearch export vs historical .mat mean spectra)
Historical-vs-V3 residuals (aligned) are small and spread (per-axis max abs 0.01–0.09); this is the different mean-spectrum source, not an engine change.

## Normalization variants (redox dominance rank; 1 = most dominant on OWD profile)
| Variant | Redox rank | Note |
| --- | --- | --- |
| Raw BSV | **1** | redox dominates by magnitude — the sparse/redox-heavy radar |
| Historical cohort z (exact, from 1322) | 3 | sterol & metabolite outrank redox |
| Robust cohort z (median/MAD) | 5 | strongest de-dominance |
| V3 frozen global | 2 | already de-dominated by the frozen calibration |
| Cohort-standardized global | 1–3 | downstream visualization only |

## Effect sizes (exact historical 1322, OWD vs NWD)
Sterol/neutral-lipid **d=+2.45** (top), Metabolic-small-molecule −1.45, Sulfur/thiol/redox +1.44, Pyrimidine −1.41, Aromatic +1.32. The **strongest cohort difference is sterol, not redox** — the raw magnitude radar hid this under redox dominance; the z-score reveals the multiaxis structure.

## Reproduction fidelity
Historical `diabetes_zscore_2group.csv` reproduced from the saved 1322 BSV with the documented formula: **max abs diff = 1.6e-15** (exact).

## Conclusion
The historical radar looked better because **axis-wise cohort z-score normalization** re-expresses each axis in comparable units and exposes multiaxis group structure (sterol/metabolite/redox/pyrimidine/aromatic). The upstream engine difference is **redox-only** (tightened thione window + co-band gating), a secondary correction — it does not create the multiaxis balance. Normalization is the driver; the engine change is a small G10 refinement.
