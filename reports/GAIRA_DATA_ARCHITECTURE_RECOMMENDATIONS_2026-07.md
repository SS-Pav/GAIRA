# GAIRA Data Architecture Recommendations

**Date:** 2026-07 · Assessment only (no implementation).

## 1. One canonical accounting system (highest priority)
Adopt the five-role taxonomy and NEVER conflate its counters:
`unique_datasets · unique_analytes · unique_patients · unique_biological_samples · independent_measured_spectra · technical_spectra · augmented_spectra · processed_duplicates`.
The registries in `data_audit/` are the seed. Every future claim ("202 analytes", ">180k spectra") must cite the role + counter. Concretely: 202 = **141 unique Raman reference compounds**; >180k = **~760 independent human samples**.

## 2. Mode-aware calibration (do NOT keep one universal scale)
Current global coordinates are fit on **275 biological Ag-colloid SERS spectra** — a single substrate. Evidence that harmonization is unsafe:
- The European adenine set shows G01 varies materially across cAg/cAu/sAg/sAu (CV ~0.13) and 532 vs 785 (0.105 vs 0.181); the demo is blind to all of it.
- RamanBioLib grounding is **spontaneous Raman**; the biological corpus is **Ag-SERS** — different observation physics.
Recommended architecture:
```
Shared biochemical ontology (axes, meaning)         <- unchanged, modality-independent
   |
   +-- Raman reference coordinates          (RamanBioLib, powder/CaF2)
   +-- Ag-colloid SERS reference coordinates (serum/EV/adenine/ergothioneine)  <- current V3 scale
   +-- Au-surface SERS reference coordinates (European sAu/cAu; ovarian AuNP)
   +-- DART-Met porous-Au coordinates        (LAB_DATA/Cracked_Au)
   |
   +-- modality-specific observation models  (per substrate x excitation)
   +-- common higher-level biochemical coordinates (after per-mode calibration)
   +-- explicit nuisance metadata            (substrate, laser, instrument, lab)
```
Keep ONE ontology; stratify the numerical calibration by substrate/modality; harmonize only after per-mode calibration, with nuisance metadata carried explicitly.

## 3. Activate or retire the dormant production physics engine
`src/gaira/substrate` (42 source-backed, bounded, conflict-aware effects) and `src/gaira/atlas` are imported by nothing. Either wire them into inference (replacing the 5 demo heuristics) or explicitly deprecate them — do not leave the most rigorous layer dormant while shipping unvalidated heuristics.

## 4. Substrate/modality validation program (data GAPS)
To validate substrate/physics layers, acquire/curate:
- **Paired Raman↔SERS** of the same reference analytes (only ovarian is paired today, and IDs don't cross-map).
- **Au vs Ag** and **colloid vs planar** references (European set has these but no matched analyte panel across all).
- **Excitation-matched** standards (532 & 785).
- **Per-analyte replicate** reference spectra (most references are 1 spectrum/analyte → no reproducibility estimate).
- **Isotope/enzyme** perturbations beyond uric acid (only ¹⁵N-UA + uricase exist).

## 5. Provenance fixes
- Resolve `SER-CCA-58` duplicate and `SER-LM-11` id mismatch upstream.
- Decide `hcc_serum` (registered but skipped) and `LAB_DATA/Cracked_Au` (wired nowhere).
- Rename `serum_ag_colloids_grounding` (empty raw folder; data in DB) to remove the missing-data illusion and the grounding/calibration role conflation.
- Flag small2023 augmented rows as non-independent in every downstream count.
- Fix pilot5 covid "spectrum-as-patient" (309 ≠ ~103 patients).

## 6. What is missing to build a truly globally calibrated GAIRA
1. A **modality/substrate-stratified reference library** with replicate spectra.
2. **Cross-modality anchors** (same analyte, Raman + Ag-SERS + Au-SERS).
3. **Excitation and instrument metadata** on every spectrum (present in European adenine and stroke; absent elsewhere).
4. **Independent-sample-level** biological cohorts with clinical metadata (not spectrum-level inflation).
5. A **validated substrate observation model** (the dormant production engine is the closest existing asset).
Until these exist, GAIRA's global coordinates should be described as an **Ag-colloid-SERS biological reference prototype**, not a universal scale.
