# GAIRA — Pure Ag-SERS Abstraction Recovery (V5)

### From exact molecular identity to recoverable biochemical abstraction

*Additive analysis on the frozen atlas `09ed804a40836f4a05a91ba10900cded`. Reproduces V4 exact-
identity counts and V3/V4 matched values; adds only abstraction-level evaluation. Subclass is an
evaluation overlay, never a new GAIRA axis. Tables `tables/`, figures `figures/`, cards `analytes/`.
Method: `EVALUATION_HIERARCHY_AND_METRICS.md`; overlay provenance:
`ANALYTE_CLASSIFICATION_PROVENANCE.md`; audit: `CURRENT_STATE_AUDIT.md`.*

Tags: **[obs]** observation · **[metric]** · **[interp]** · **[infer]** · **[spec]** · **[lim]**.

## 1 · Executive summary
When exact analyte identity is lost after Ag-SERS, does the correct **broader** chemistry survive?
The honest answer: **the expected motif/theme is often PRESENT (top-3: MSS 40%, theme 49%), but
that presence is not analyte-SPECIFIC** — specific, null-and-background-adjusted recovery is rare at
every level (component 2/51, MSS 2/48, theme 1/51), and cross-modal subclass/family classification
is **at chance**. A same-modality **Raman→Raman control** proves the taxonomy is separable and that
abstraction *does* help within Raman (subclass 0.23 → family 0.35 → theme 0.42 balanced accuracy) —
but the **Ag-SERS modality gap collapses it to ~chance**. **Presence ≠ recovery.** Only functional
perturbation (3 analytes) provides class-specific recovery beyond exact identity. **[interp]**

## 2 · Scientific question
"When exact identity is lost, does GAIRA still recover the correct broader chemical information —
component, motif, subclass, or theme — and at what level of abstraction does meaningful information
survive?" **[interp]**

## 3 · Why exact recovery is not the only useful endpoint
A serum SERS assay rarely needs the exact molecule; a correct *class* (purine vs lipid vs protein)
is often actionable. So we measure graceful degradation up the hierarchy, not just identity. **[interp]**

## 4 · Dataset & provenance
51 matched Raman↔Ag-SERS analytes; 265 SERS spectra / 5 replicates. Per-analyte means used for
classification → **no replicate leakage**. Serum blank (`load_serum_baseline`) for background
correction. **[obs]**

## 5 · Frozen atlas confirmation
`z/m/b` computed by identical frozen calls to V4 ⇒ exact-identity reproduces V4 (latent 7, MSS 3,
theme 4). Fingerprint verified `09ed804a…`. No frozen asset modified. **[metric]**

## 6 · Existing analyte-identity results (Level 0, inherited)
Latent 7/51, MSS 3/51, theme 4/51 (rank-1 + jackknife-stable, V4). Unchanged. **[obs]**

## 7 · Evaluation hierarchy
Exact identity → NMF component → MSS motif → molecular subclass → broad theme → perturbation →
matrix. Each a distinct question; see `EVALUATION_HIERARCHY_AND_METRICS.md`. **[metric]**

## 8 · NMF component recovery (Level 1)
Raman↔Ag-SERS top-3 component overlap, mass retention, mismatched null. **Recovered: 2/51.** Emergent
components rarely survive the surface reshaping; a component id is not a chemical subclass. **[metric]**

## 9 · MSS motif recovery (Level 2)
Expected motif from chemistry + Raman activation (not Ag-SERS height). Graded: **present (top-3)
19/48**; enriched (>out-group null95) 6/48; **specific (top-3 & >null & >background) 2/48**;
unassigned 3 (phosphate, creatinine, urea). Presence is common; specificity is rare. **[metric]**

## 10 · Molecular subclass overlay
Versioned chemical overlay (`analyte_classification_overlay.csv`); 10 subclasses ≥2 members, 15
exploratory singletons; multi-label where justified; phosphate/creatinine/urea MSS-unassigned. NOT a
GAIRA axis. **[metric]**

## 11 · Subclass classification results (LOAO)
Nearest-centroid, analyte-level LOAO, latent/MSS/theme spaces: **balanced accuracy ~0.03–0.18, all
permutation p non-significant** (0.22–0.81). Nearest-neighbour same-class retrieval is **at/below
chance** (subclass 0.02 vs 0.05; family 0.08 vs 0.15). **Raman→Raman control: 0.23 / 0.35 / 0.42** —
the classes ARE separable within Raman and abstraction helps; the Ag-SERS gap collapses it. **[metric]**

## 12 · Broad-family recovery
Cross-modal family classification ~chance (best space balanced acc 0.13–0.18, ns); NN same-family
0.08 vs chance 0.15. Family structure does not transfer across the modality gap. **[metric]**

## 13 · Biochemical-theme recovery (Level 4)
Expected theme: **present (top-3) 25/51**, enriched 5/51, **specific 1/51**. Raw theme presence is
dominated by the shared Ag background; only 1 analyte's expected theme is enriched above
family-mismatched null AND above the blank. **[metric]** Raw cosine never used alone. **[interp]**

## 14 · Purine-attractor correction
The serum blank is purine-dominant (share 0.27) before any analyte (V4). Non-purines are pulled
toward purine (Δpurine>0 for 36/51). A handful of non-purines still show their **expected** motif in
top-3 despite the pull — genuine motif presence separable from attraction — but this remains
*presence*, not specific recovery. **[interp]**

## 15 · Recovery counts by abstraction level
exact 7/51 · component 2/51 · **MSS present 19/48 / specific 2/48** · subclass (LOAO ~chance) ·
**theme present 25/51 / specific 1/51** · perturbation 3/51 · matrix 9/51. **Highest defensible level
per analyte:** 7 exact · 1 component/motif-specific · 2 perturbation-only · **22 broad-presence-only
(non-specific)** · 19 none. **[metric]**

## 16 · Family-specific findings
Purines/oxopurines and the CoA purine-cofactors dominate what little specific recovery exists
(their purine theme is legitimate chemistry). Amino acids, sugars, lipids, pyrimidines reach only
non-specific presence or none. See `family_abstraction_breakdown.csv`. **[obs]**

## 17 · Representative analytes
adenine, ergothioneine, urate, xanthine, hypoxanthine, creatinine, glucose, tyrosine, uracil, oleate,
n-acetylglucosamine, albumin — spectra → themes → evidence in Figure 10. **[obs]**

## 18 · Adenine perturbation
Static exact identity weak (latent 0.36, not recovered); expected purine motif present; **dose→purine
ρ=0.996 (functional recovery).** Ag-SERS reshapes the fingerprint, but purine-level dose behaviour is
recoverable. No molecular ID claimed from the static spectrum. **[obs/interp]**

## 19 · Ergothioneine perturbation
Static identity weak; **dose→sulfur ρ=0.927.** Functional sulfur chemistry recoverable. **[obs]**

## 20 · Urate/uricase perturbation
**Motif-specific depletion** (oxopurine_carbonyl Δ=−0.060) stronger and more specific than the broad
purine-theme change — directional, at the motif layer. **[obs]**

## 21 · Relationship to serum recovery
Pure abstraction recovery does **not** monotonically predict serum-strong tier (Figure 12); matrix is
a separate property. Kept strictly separate. **[metric]**

## 22 · What GAIRA can / cannot claim for pure Ag-SERS
**Can:** report that a broad biochemical theme/motif is *present* in the top-3 for ~40–50% of analytes
(broad interpretation); specifically recover exact identity for a strong-chemisorber minority (7/51);
functionally validate 3 analytes. **Cannot:** claim molecular identification from motif/theme presence;
claim class-discriminative subclass/family recovery from pure Ag-SERS (at chance); infer perturbation
or matrix for untested analytes. **[interp]**

## 23 · Recommended architecture
Treat Ag-SERS as an **observation channel that preserves broad presence but not analyte identity**;
build a learned Raman→SERS observation model to close the modality gap (the Raman→Raman control shows
the information exists in principle); prioritise **dynamic perturbation (DART)** as the route to
class-specific recovery. **[spec]**

## 24 · Limitations
Raman-trained atlas; surface-selection effects; purine attractor; low exact identity; subclass
imbalance (15 exploratory singletons); 3 perturbation cases; matrix competition; no Au-SERS
observation model; confidence ≠ identifiability; cross-modal centroid classification depends on the
global-shift confound (reported alongside the Raman control). **[lim]**

## 25 · Conclusions
Broad biochemical **presence** frequently survives Ag-SERS (top-3 motif/theme for ~40–50%), but it is
**not analyte-specific** and **not class-discriminative** — abstraction raises apparent presence via a
shared attractor, while the modality gap (proven by the Raman→Raman control) collapses genuine
class recovery to chance. Meaningful, *specific* recovery beyond a strong-chemisorber minority comes
only from **functional perturbation**. **[interp]**

## 26 · Methods & formulas
`EVALUATION_HIERARCHY_AND_METRICS.md`. Nearest-centroid / nearest-neighbour, analyte-level LOAO,
permutation nulls, bootstrap CIs, balanced accuracy + macro-F1, chance-adjusted top-k. Deterministic
(fixed seeds); reruns identical.

## 27 · Reproduction
```bash
python results/v5_rebuild/abstraction_recovery_v5/code/build_overlay.py
python results/v5_rebuild/abstraction_recovery_v5/code/abstraction_analysis.py
python results/v5_rebuild/abstraction_recovery_v5/code/make_figures_v5.py
python results/v5_rebuild/abstraction_recovery_v5/code/make_cards_v5.py
python results/v5_rebuild/abstraction_recovery_v5/code/make_report_v5_pdf.py
```
Interactive: `streamlit run gaira_foundation_explorer_v5/app.py`. Frozen atlas unchanged.
