# GAIRA — Current State & Architecture (V4 canonical)

**Date:** 2026-07-18 · Branch `gaira-v4-grounding-architecture-and-substrate-validation`. Supersedes earlier statements that implied axes emerged from UMAP, that >180k spectra are molecular grounding, that serum spike-ins calibrated the axes, that the physics atlas modifies inference, or that substrate rules are validated corrections. Historical docs are retained and marked superseded here, not deleted.

## 1. Molecular grounding corpus
Pure/reference analyte spectral evidence only. ~**719 full-spectrum measured references** + peak-level evidence: RamanBioLib (202 rows/141 compounds, Raman), amino-acid (20, Raman), adenine (~16, Ag-SERS), metabolite-63 (63, Ag-SERS 633 nm), Gobbato pure metabolites (265 Ag-SERS + 153 Raman powders), Ag-flake (24 metabolites, **peak tables only**).

## 2. Every direct grounding source
See `data_audit/v4_direct_grounding_sources.csv`. Six sources; RamanBioLib is the backbone.

## 3. Raman vs Ag-SERS vs Au-SERS vs Ag-flake evidence
**Raman ≈ 375** (RamanBioLib 202 + AA 20 + Gobbato powders 153). **Ag-SERS ≈ 344** (adenine 16 + metabolite-63 63 + Gobbato SERS 265). **Au-SERS = 0** (none exist). **ORC-Ag (Ag-flake) = 24 analytes as peak tables** (no spectra). Cross-modality numeric BSVs are NOT directly comparable.

## 4. Why 202 rows = 141 unique RamanBioLib chemicals
The table is digitized RamanBioLib (DOI 10.1002/jrs.1734, "140 components"); 202 = compound × substrate × laser rows; 61 rows are the same compound re-measured (33 on a second substrate glass-Raman + metal-ring, 6 same-substrate). Unique compounds = 141.

## 5. Serum Ag-colloid corpus
Gobbato/Bonifacio (Trieste) "Adsorption of Serum Components on Ag Colloids", 907 SERS spectra (785 nm): 265 pure-metabolite Ag-SERS, 153 pure-metabolite Raman powders, 270 serum spike-ins, 20 uricase design, 73 isotopic, 81 donor serum, + fit refs.

## 6. Serum-Ag: grounding vs perturbation evaluation
**Grounding:** 265 pure Ag-SERS + 153 pure Raman metabolite standards. **Perturbation evaluation (NOT grounding):** uricase (20), serum spikes (270), isotope (73), serum baselines. **Biological challenge:** 81 donor serum. **Candidate-only:** literature assignments. (Registry: `data_audit/v4_serum_ag_analyte_condition_registry.csv`.)

## 7. Metabolite-63
**Ag citrate colloid (Lee–Meisel), 633 nm, 63 pure-analyte averaged spectra** (716 pts, ~500–2000 cm⁻¹). NOT Au (the task premise was wrong; repo already labels it Ag). Direct Ag-SERS grounding.

## 8. Ag-flake metabolite-24
**24 metabolites (not 23), ORC-roughened Ag (not flakes), peak tables only** (454 peaks; qualitative vs/s/w/vw + `*` exclusive flag; excitation/conc not in SI). Role: **peak-level SERS grounding** for MSS band construction / collision — never full-spectrum/BSV.

## 9. Are the 11 axes emergent or curated?
**Curated.** Hard-coded constants (`base2/schema.py:18` BIOLOGY_AXES_V11 "from axis design doc §2"; demo `config.py:73` BSV_AXES). 6 of 11 are inherited splits of an older curated 8-axis list (`PROJECTION_V11_TO_V8`). **Not emergent from UMAP.**

## 10. Exact role of UMAP
**Visualization/clustering of already-computed embeddings/class-means only.** It never defines axes or BSV (the cluster script states "NOT a scoring/BSV/calibration phase"; ev_latent_map: "themes painted onto clusters afterward").

## 11. Motifs / BSV / MSS / retrieval / interpretation
- **BSV = deterministic band/motif scoring** (band window-max → noisy-OR over curated axis mappings). No learned encoder.
- **MSS**: demo runs before BSV as a small additive push (weight 0.25, motif-dominant); production base2 BSV has no MSS term (base3 MSS is a separate deterministic discriminant layer).
- **Retrieval (grounding_search)** = supporting evidence only; **never sets BSV** (production `inference.py` emits no `bsv`).
- **Learned CNN encoder** exists (`embedding/model.py`) but is **not used at inference** — non-load-bearing.
- Interpretation = domain context (EV/serum caveats + ranking weights), never alters coordinates.

## 12. Why controlled perturbation data must not fit axis weights
Fitting axes/centers/scales on serum spike-ins/uricase/dose data would make the "evaluation" circular. These are **held-out tests** that a model grounded independently responds correctly. The V3 frozen calibration is fit on the biological range, NOT on perturbations. (Registry: `data_audit/v4_controlled_perturbation_evaluation_registry.csv`.)

## 13. How biological datasets are used
13 datasets, ~180k spectra but **~760 independent human samples** (>99% technical/augmented). Used for range/projection (serum-liver 212, EV 63, SHINE reduced) and nuisance diagnostics — NOT as molecular grounding.

## 14. Demo–production divergence
Two deterministic engines. Production `base2` (50 motifs, 39 mappings, dual-status regime, 11↔8 projection) is richer than the demo (11 motifs). Top-axis agrees 5/6 on reference/serum/synthetic; diverges on EV mixtures. Demo touches no DuckDB; production is DuckDB-driven (185,686 biosample + 468 grounding + 202 reference). Recommended canonical engine: **production base2/base3**, with the demo as presentation layer. (Report: reconciliation.)

## 15. Substrate-layer maturity
Demo = 5 heuristic multipliers/caveats (unvalidated; **no cross-substrate utility** — European adenine test). Production = 42 source-backed bounded conflict-aware effects but **dormant** (imported by nothing). No validated correction exists; adopt metadata-only stratification + activate the production engine.

## 16. Physics-atlas maturity
8 curated literature prose regions; **UI captions only, 0 numeric BSV effect**. Testable upgrades (collision/ambiguity, OOD warning) belong in **confidence/caveats**, never BSV.

## 17. What V4 tested
Architecture code trace (axes curated, UMAP viz-only, BSV deterministic); demo↔production reconciliation (adapter, comparison packet); substrate cross-substrate/rule-ablation (European adenine); atlas numeric-effect (none); metabolite-63 substrate (Ag not Au); Ag-flake extraction (24, peak-only); perturbation-evaluation reframing.

## 18. What remains unvalidated
Substrate/modality corrections (no paired multi-substrate references); Au-SERS grounding (none exists); cross-modality BSV comparability; the production engine's real-data behavior at scale; MSS-as-BSV-contributor in the demo (should be support-only); the 53 serum spike-ins and European set as formal evaluations (unwired).

## 19. Corrected future architecture
```
              SHARED disease-independent BIOCHEMICAL ONTOLOGY (curated 11 axes)
                                   |
   Raman obs. model  |  Ag-SERS obs. model  |  Au-SERS obs. model  |  ORC-Ag/other-SERS
                                   |
              substrate/mode-specific biochemical EVIDENCE extraction
                                   |
                     biochemical theme representation (BSV)
                                   |
              analyte-level MSS SUPPORT (secondary; never defines an axis)
                                   |
                     domain-aware interpretation (caveats/ranking)
```
One ontology; per-modality observation models; MSS as secondary support; controlled perturbation sets strictly held-out.

## 20. Concrete V5 plan
1. **Shared axis-constant module** imported by both demo and production (retire duplicate constants).
2. **Adopt substrate/excitation metadata** on every grounding + biological spectrum; stratify (Raman/Ag-colloid/Au-colloid/ORC-Ag/unknown).
3. **Activate the production 42-effect substrate engine** as the observation model (bounded, source-backed); retire demo heuristics.
4. **Wire metabolite-63 (Ag 633) + Gobbato pure metabolites (265) + Ag-flake peaks (24)** as multi-substrate grounding with modality tags.
5. **Acquire Au-SERS pure-analyte references** (the one true grounding gap) + paired cross-substrate panels.
6. **Formalize Grounded Perturbation Tests** (adenine/ergothioneine/hypoxanthine/uricase/European) as a held-out evaluation suite; keep uricase inconsistency visible.
7. **Build an inference-safe collision/OOD confidence layer** from Ag-flake exclusive flags + shared-band map (never touches BSV).
8. Defer any learned encoder until the grounded scaffold + evaluation suite are stable.

## Canonical conclusion
- **Truly grounded:** ~719 pure-analyte Raman + Ag-SERS reference spectra (+ Ag-flake peak evidence).
- **Only curated:** the 11 axes, motif band windows, MSS→axis map.
- **Directly measured:** RamanBioLib, amino-acid, adenine, metabolite-63, Gobbato pure metabolites.
- **Evaluated independently (held-out):** adenine/ergothioneine dose, hypoxanthine/uricase/isotope, 53 serum spikes, European inter-instrument adenine.
- **Biological challenge:** the 13 mixture datasets (~760 independent human samples).
- **Substrate-specific:** all SERS grounding + the (dormant) production substrate engine.
- **Currently learned:** only the non-load-bearing CNN encoder + UMAP visualizations.
- **Not yet learned:** everything defining axes/BSV/MSS (deterministic by design).
- **Architecture GAIRA should commit to:** a shared disease-independent biochemical ontology, direct molecular grounding across Raman and multiple SERS substrate domains, substrate/modality-specific observation models converting spectra to biochemical evidence, MSS as secondary analyte support, domain context for interpretation, and controlled serum/enzyme/isotope/dose datasets kept strictly as held-out perturbation evaluations that never define or fit the axes.
