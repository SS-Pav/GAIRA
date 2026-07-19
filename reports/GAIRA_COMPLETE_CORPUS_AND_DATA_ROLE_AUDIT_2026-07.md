# GAIRA Complete Corpus & Data-Role Audit

**Date:** 2026-07 · Read-only forensic audit. Branch `gaira-data-and-physics-audit-2026-07`. Canonical tables in `data_audit/`; regeneration scripts in `audits/corpus_audit/`; 13/13 reconciliation checks pass (`data_audit/reconciliation.json`).

## Executive table (one page)
| Role | Datasets | Unique analytes | Independent measured spectra | Technical/augmented spectra | Patients / samples | Raman | SERS | Currently used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Molecular grounding** | 6 | ~141 (RamanBioLib) + 63 (met-63) + 20 (AA), ~200 distinct | 202 reference + **468 grounding (DB)** / 160 class-summary | 0 | 0 (pure compounds) | 202 ref + 20 AA | 448 grounding (adenine 16, met 64, serum-Ag 368) | **YES** (202 table, warehouse map, adenine live) |
| **Calibration** | 7 | ~80 small molecules + 3 drugs | ~4,046 (**~3,726 SERS + 320 Raman**) | 7,032 raw European (vs 3,516 ILS) | serum pools/donors | 320 | ~3,726 | **partial** (adenine, ergothioneine, serum SAEL) |
| **Biological mixtures** | 13 | 0 (mixtures, not references) | **~760 independent human** | **~180,000** (technical + augmented) | ~760 human + cell/microbial | ~4k (covid/ovarian/ucla/sv) | ~176k | serum-liver 212 + EV 63 + SHINE (reduced) |
| **Supporting literature** | 4 | 0 | 0 | 0 | 0 | 0 | 0 | warehouse 30 rows/15 papers; knowledge (288 assignments) |
| **Substrate/physics** | demo 5 rules + prod 42-effect engine + atlas | 0 | 0 | 0 | 0 | — | — | 5 heuristic rules + prose atlas (wired); prod engine DORMANT |

## The single most important conclusion
GAIRA has a **small, heterogeneous, mostly-Raman molecular grounding layer** (~141 unique digitized RamanBioLib compounds + ~150 measured reference spectra), a **modest set of controlled calibration datasets** (7; only 3 GAIRA axes — purine G01, purine-metabolite G02, redox G10 — have supportive/partial calibration), and a **much larger biological mixture corpus** (~180,000 spectra) that is **not molecular grounding**. Of those ~180k, **>99% are technical map scans, single-vesicle points, experimenter-averaged replicates, or augmented synthetic mixtures**; independent human biological samples number only **~760**. Dataset count, spectrum count, sample count, patient count, and analyte count are all different and must never be merged.

## Runtime split (critical)
- **V3.1 demo** reads small bundled CSV/parquet extracts only (202 table, warehouse CSV, adenine raw, serum/EV mean spectra, frozen calibration JSON). It **never reads the production DuckDB**.
- **Production `src/gaira`** is entirely DuckDB-driven (`interim/gaira.duckdb`: 185,686 biosample spectra, 468 grounding, 202 reference, knowledge/peak-assignment tables) via `inference.py`, `grounding_search.py`, EV/serum context, SAEL, expected-BSV. **The demo and production share almost no runtime data path.**
- **Dormant assets:** the production `src/gaira/substrate` (42 source-backed effects, bounded [0.40,1.15], conflict-aware) and `src/gaira/atlas` engines are **imported by nothing** — the most rigorous physics layer in the repo is unused.

## Plain-language answers (25)
1. **Grounding corpus?** Digitized RamanBioLib (Raman) + a handful of measured reference sources (adenine, amino-acid, metabolite-63, serum-Ag).
2. **Unique analytes?** ~141 in the 202-row RamanBioLib table (+63 metabolite-63, +20 amino-acid; ~200 distinct with overlap).
3. **Analytes with accessible spectra?** All 202 RamanBioLib rows have raw spectra (parquet, 0 precomputed-only); the measured grounding sources add 468 grounding spectra (DB).
4. **Spectra per analyte?** RamanBioLib: 1 per (compound×substrate×laser), 1–4 per compound; reference panels: 1 per analyte; adenine 12–17.
5. **Raman vs SERS grounding?** RamanBioLib = spontaneous **Raman**; measured grounding = mostly **SERS** (adenine 16, metabolite-63 64, serum-Ag 368) + 20 Raman amino-acids.
6. **202-analyte table?** 202 digitized RamanBioLib rows = **141 unique compounds** (61 duplicate/multi-substrate rows); 6 of 11 GAIRA axes are inherited splits.
7. **43-source warehouse?** 43 registry rows but **28 unique source_ids**; 30 "disease/stress paper" rows = 15 papers duplicated (context, not spectra); only 4 sources carry measured-spectra counts.
8. **Serum Ag-colloid dataset?** The Gobbato/Bonifacio (Trieste) "Adsorption of Serum Components on Ag Colloids" study — 907 SERS spectra (785 nm), mechanistic serum-metabolite work.
9. **Which analytes/interventions?** Controlled: hypoxanthine spike, uric-acid uricase depletion, ¹⁵N-UA isotope, 53-metabolite serum spikes, ergothioneine spike (cspp). Candidate-only: literature assignments.
10. **How used?** Grounding (serum_ag_colloids_grounding = 368 in DB) + calibration (the SAEL uricase/hypoxanthine/ergothioneine contrasts, incl. the honestly-preserved inconsistent uricase result).
11. **Calibration datasets?** 7: adenine (×2), ergothioneine, serum-Ag uricase, cspp, metabolite-63, amino-acid-20 (+OTC drugs).
12. **Calibration spectra/analytes?** ~4,046 measured (~3,726 SERS + 320 Raman); ~80 small molecules + 3 drugs.
13. **Which axes calibrated?** Only **G01, G02, G10** (supportive/partial); G03/G04/G08/G09 not tested; G05/G06/G07/G11 insufficient.
14. **Biological datasets?** 13 (diabetes-EV, SHINE, small2023, cca/hcc serum, covid, ovarian, saliva, coeliac, stroke, mycoplasma, single-vesicle, Cracked_Au).
15. **Per-dataset counts?** See biological audit + `biological_dataset_registry.csv`.
16. **>180,000 figure?** small2023 105,140 (augmented) + diabetes 31,834 (technical) + SHINE 23,646 (technical) = ~89%.
17. **Independent vs technical/augmented?** ~760 independent human samples vs ~180,000 total (~200× inflation).
18. **Substrate awareness?** 5 hard-coded heuristic multipliers/caveats (demo) + a dormant 42-effect production engine.
19. **Raman vs SERS awareness?** A binary `substrate` string only; no Au/planar/excitation model.
20. **Physics atlas?** 8 curated literature prose regions (config + app.py).
21. **Atlas affects numbers?** **No** — UI caveats only.
22. **Substrate/physics validatable?** Not with current single-substrate data; the European multi-substrate set is the vehicle but needs a model the demo lacks.
23. **Demonstrated-value layers?** None validated; thiol boost + diabetes co-band gate are suggestive.
24. **Heuristic-only layers?** All demo substrate rules + atlas + collision.
25. **Missing data for a true global system?** Paired Raman↔SERS references, Au/planar substrate references, excitation-matched standards, per-analyte replicate reference spectra, and metadata-complete biological cohorts (see architecture report).
