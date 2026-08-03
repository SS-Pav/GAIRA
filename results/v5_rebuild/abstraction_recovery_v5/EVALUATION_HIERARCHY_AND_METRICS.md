# V5 Evaluation Hierarchy & Metrics

*How V5 measures whether broader chemistry survives when exact Ag-SERS identity is lost. Additive
to V4; every matched value reproduces V4/V3. Frozen atlas `09ed804a…` unchanged. Computed by
`code/abstraction_analysis.py`. Subclass is an **evaluation overlay only** — never a new GAIRA axis.*

The ladder (each level asks a different question; identity is the strictest):

```
LEVEL 0  exact analyte identity      "the correct molecule vs all others"         (inherited, V4)
LEVEL 1  NMF component evidence       "same emergent basis components"
LEVEL 2  MSS motif                    "same spectroscopic chemical motif"
LEVEL 3  molecular subclass           "same chemical subclass (overlay)"
LEVEL 4  broad biochemical theme      "same GAIRA interpretive theme"
LEVEL 5  perturbation                 "correct functional response"               (inherited)
LEVEL 6  matrix                       "visible in serum"                          (inherited, secondary)
```

## Level 0 — exact analyte identity (inherited, unchanged)
V4 rank-1 + jackknife-stable: **latent 7/51, MSS 3/51, theme 4/51**. Reproduced here bit-for-bit
(same frozen `z/m/b`). Not recomputed with new thresholds; cited as the strict baseline.

## Level 1 — NMF component evidence
For analyte *i*: `comp_top3_overlap = |top3(z_R) ∩ top3(z_S)|/3`; `comp_mass_retained` = Ag-SERS
mass on the Raman-top-3 components; mismatched null over other analytes' SERS.
**Recovered ⇔ overlap ≥ 2/3 AND > null95.** A component id is an emergent basis pattern, **not** a
chemical subclass. Result: **2/51**.

## Level 2 — MSS motif recovery *(the primary chemistry test)*
Expected motif(s) assigned from **molecular chemistry + Raman activation** (overlay), NEVER because
a motif is high in Ag-SERS. Per analyte: motif rank in Raman & Ag-SERS; top-1/3/5 inclusion;
enrichment over the serum-blank background; enrichment over out-of-family analytes' score for that
motif (null95). Graded, distinguishing:
- **present** (expected motif in Ag-SERS top-3): **19/48**;
- **enriched** (> out-group null95): 6/48;
- **specific / recovered** (top-3 AND > null95 AND > background): **2/48**;
- **assignment unavailable**: phosphate, creatinine, urea (3).
90/95/99 null sensitivity reported. **Presence is common; specific recovery is rare.**

## Level 3 — molecular subclass (LOAO classification)
A versioned chemical overlay (`analyte_classification_overlay.csv`, provenance in
`ANALYTE_CLASSIFICATION_PROVENANCE.md`); multi-label where justified; `mixed`/`unassigned` allowed;
**singleton subclasses flagged exploratory** and excluded from the primary accuracy denominator
(their class centroid vanishes under leave-one-analyte-out). 10 subclasses have ≥2 members.

Two evaluations, both **leave-one-analyte-out at the analyte level** (no replicate leakage — uses
per-analyte means):
1. **Nearest-neighbour same-class retrieval:** each Ag-SERS's nearest *other* Raman analyte — does
   it share the subclass/family/theme? Chance = Σ n_c(n_c−1)/N(N−1).
2. **Nearest-centroid classification** in latent / MSS / theme space (true-class centroid excludes
   the held-out analyte): accuracy, **balanced accuracy, macro-F1**, confusion, **permutation null
   (label-shuffle)**, bootstrap CI.

**Control — Raman→Raman (same modality):** classify a held-out Raman analyte by Raman centroids.
This proves the classes are separable and isolates the modality gap.

## Level 4 — broad biochemical theme
Expected theme(s) from the curated map (purine→nucleic_purine, sulfur→sulfur_antioxidant, …).
Per analyte: expected-theme rank, top-1/2/3 inclusion, enrichment over **family-mismatched** null95,
and whether it exceeds the serum-blank background (common-mode correction). Graded:
- **present** (expected theme in top-3): **25/51**;
- **enriched**: 5/51;
- **specific / recovered**: **1/51**.
**Raw theme cosine is never used alone as recovery evidence.**

## Levels 5–6 — perturbation & matrix (inherited, unchanged)
Perturbation only for adenine, ergothioneine, urate (functional confirmation). Matrix (serum) kept
separate and secondary; never used to define pure-analyte subclass/motif recovery.

## Statistical design
Analyte-level LOAO throughout (per-analyte means → replicates never split across folds).
Permutation nulls (label-shuffle for classification; label-shuffle at fixed NN for retrieval).
Bootstrap CIs over analytes. Balanced accuracy + macro-F1 for class imbalance. Chance-adjusted
top-k. Simple interpretable models only (nearest-centroid, nearest-neighbour) — no deep learning,
no model shopping; the goal is to characterise resolution, not maximise classification.

## The result in one line
Within Raman, the taxonomy is separable and **abstraction helps** (subclass 0.23 → family 0.35 →
theme 0.42 balanced accuracy). Across the modality gap to Ag-SERS it **collapses to near chance**
(0.09–0.13; NN retrieval at/below chance). The expected motif/theme is often **present** (top-3,
40–50%) but that presence is the shared Ag background, not class-discriminative recovery
(**specific ≤ 2/48**). **Presence ≠ recovery.** Only functional perturbation (3 analytes) provides
class-specific recovery beyond exact identity.
