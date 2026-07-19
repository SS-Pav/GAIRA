# V5 Phase 1 (V5.1) — Preprocessing & Comparability

**Question.** Can Raman and Ag-SERS reference spectra share one preprocessing pipeline and be jointly analyzed without erasing chemistry?

**Methods.** Canonical loader → 202 Raman (RamanBioLib) + 69 Ag-SERS (metabolite-63 63 @633 nm + adenine 6 @785 nm). Six deterministic candidate pipelines (baseline ∈ {none,asls,poly3} × smooth ∈ {none,savgol} × norm ∈ {l2,area,snv,robust}) applied onto a common 520–1750 cm⁻¹, 2 cm⁻¹ grid. Metrics: window coverage; matched-analyte cross-modality cosine vs a within-corpus null; modality leakage (5-fold CV accuracy predicting modality from 10 PCs); band preservation. 7 analytes matched across modalities (adenine, arginine, asparagine, cytochrome c, glutathione, histidine, tryptophan). Figures in `figures/`, metrics in `tables/`.

**Results.**
| Pipeline | cov Raman | cov SERS | matched x-mod cosine | null cosine | modality leakage CV acc |
| --- | --- | --- | --- | --- | --- |
| P0 raw+L2 | 1.00 | 0.997 | **0.527** | 0.410 | 0.823 |
| P1 asls+L2 | 1.00 | 0.997 | 0.483 | 0.332 | 0.841 |
| P2 asls+savgol+L2 | 1.00 | 0.997 | 0.484 | 0.333 | 0.841 |
| P3 asls+savgol+SNV | 1.00 | 0.997 | 0.253 | 0.022 | 0.856 |
| P4 poly+savgol+L2 | 1.00 | 0.997 | 0.248 | 0.037 | 0.860 |
| P5 asls+savgol+area | 1.00 | 0.997 | 0.484 | 0.333 | **0.745** |

- Spectra **share a defensible common window** (≥99.7% coverage of 520–1750 in both modalities).
- **Same analyte, different modality → low similarity** (cosine 0.25–0.53). SERS surface-selection rules reshape the spectrum; the Raman and Ag-SERS forms of the same molecule are only weakly alike (see `matched_analyte_crossmodality.png`).
- Matched **> null** in every pipeline (shared chemical information exists) but the **gap is modest** and absolute similarity is low.
- **Modality leaks into unsupervised structure**: CV accuracy predicting modality from 10 PCs is 0.82–0.86, above the 0.745 majority baseline (P5/area sits at baseline). Modality is a detectable structural separator (`pca_modality_leakage.png`).
- **Preprocessing is consequential**: SNV (P3) drives matched cosine to 0.25 while L2/area keep ~0.48–0.53 — no single pipeline is unambiguously best.

**Interpretation (as a spectroscopist/chemometrician).** This is expected SERS physics: enhancement is analyte-, orientation-, and substrate-dependent, so a molecule's Ag-SERS fingerprint is not a scaled Raman spectrum. The data confirm (a) a shared preprocessing *window* is defensible, (b) some cross-modality chemical signal exists (matched > null), but (c) it is too weak and, critically, **only 7 analytes are measured in both modalities** — far too few to *estimate* a cross-mode transformation. Forcing Raman and Ag-SERS into one representation now would let modality dominate emergent structure (H6 leakage).

**Limitations.** Raman spectra are digitized/normalized literature across **9 excitation wavelengths** (an additional nuisance axis); metabolite-63 is averaged/bg-subtracted; the Gobbato pure-metabolite corpus (53 analytes measured as BOTH pure Raman powder AND pure Ag-SERS — the ideal matched set) is not yet loaded; amino-acid xlsx not yet parsed. Leakage accuracy is against an imbalanced (0.745) baseline.

**Decision.** **Do NOT build a shared Raman/Ag-SERS observation space yet.** Adopt **modality-stratified analysis** as the working default. Retain a consistent preprocessing *window + pipeline family* (asls baseline + conservative savgol + L2, applied per modality), but keep normalization choice explicit and modality-specific. This is a legitimate scientific endpoint — no cross-mode correction is invented.

**Next action (before Phase 2 observation-layer fitting).** Load the **Gobbato pure-metabolite corpus** (265 Ag-SERS + 153 Raman powders) to expand matched Raman↔Ag-SERS analytes from 7 toward ~50; only then is a Phase-2 observation model (alignment/reliability weighting) estimable. Until then, Phase 2 decision = **D (insufficient overlap → acquire/parse more matched data)**.

## Answers to the 9 gate questions
1. **Can Raman and Ag-SERS share one preprocessing pipeline?** They can share a **window and pipeline family**, but normalization should be modality-specific; a single identical pipeline is not clearly best.
2. **Are spectra comparable after preprocessing?** Only weakly across modalities (matched cosine 0.25–0.53); comparable within a modality.
3. **Which preprocessing choices retained?** Common window 520–1750/2 cm⁻¹; ASLS baseline; conservative Savitzky–Golay; L2 (per-modality) as the default, SNV as an inspected alternative.
4. **Which rejected?** A single cross-modality normalization as "the" pipeline; treating Raman and Ag-SERS intensities as directly comparable.
5. **What analytes are matched across modalities?** 7: adenine, arginine, asparagine, cytochrome c, glutathione, histidine, tryptophan.
6. **What metadata gaps remain?** Excitation heterogeneity within Raman (9 domains); Gobbato + amino-acid spectra not yet loaded; ORC-Ag excitation/concentration unknown.
7. **Can we proceed to the observation-layer phase?** **Not yet.**
8. **If yes, why?** N/A.
9. **If not, what must be fixed first?** Expand matched Raman↔Ag-SERS analytes (load Gobbato pure metabolites) and resolve excitation stratification; only then estimate a cross-mode observation model. If matched overlap stays small, keep modality-stratified analysis and treat cross-mode transfer as a data-acquisition gap.
