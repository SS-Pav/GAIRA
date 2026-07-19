# GAIRA V4 — Serum Ag-Colloid Deep Audit

**Date:** 2026-07-18 · Source: Gobbato/Bonifacio (Trieste) "Adsorption of Serum Components on Ag Colloids" (`raw/serum_ag_colloids/` 907 spectra, B&WTek 785 nm) + `cspp_serum/`. Registry: `data_audit/v4_serum_ag_analyte_condition_registry.csv`.

## Contents (907 measured SERS spectra + Raman powders)
| Component | n | Type |
| --- | --- | --- |
| SERS metabolites (53 pure) | 265 | **pure metabolite SERS standards** (adenine, hypoxanthine, uric acid, xanthine, guanine, ergothioneine, glucose, urea, AAs…) |
| Raman metabolites (51 powders) | 153 | **pure metabolite Raman** |
| dataset uricase | 20 | serum ± hypoxanthine spike ± uricase (4 conditions ×5) |
| isotopic | 73 | UA vs ¹⁵N-UA ± HSA ± ultrafiltration |
| SERS metabolites for fitting | 30 | hypoxanthine / UA-free / UA-bound fit refs |
| SERS serum Merck | 15 | commercial serum baseline |
| SERS spiked serum Merck | 270 | 53 metabolites spiked into serum ×5 |
| donors serum SERS | 81 | 81 healthy-donor serum |
| digitized literature | 0 (3 csv) | literature, not measured |

## Role separation (the critical V4 distinction)
- **DIRECT GROUNDING** (defensible analyte spectral evidence): the **265 pure-metabolite Ag-SERS** + **153 pure-metabolite Raman** spectra. These are controlled pure-analyte references → legitimate grounding (with Ag-SERS/Raman modality tags).
- **PERTURBATION EVALUATION** (test, not grounding): the uricase design (20), hypoxanthine/ergothioneine serum spikes, 53 serum spike-ins (270), isotope (73), fit refs (30), serum baselines (15). These are **serum/background comparisons that test whether GAIRA detects the expected direction** — they must NOT define or fit axes.
- **BIOLOGICAL CHALLENGE**: 81 donor serum spectra (unknown-mixture cohort).
- **CANDIDATE ASSIGNMENT ONLY**: the literature band table + digitized spectra.

## Explicit answers
- **Legitimate direct grounding?** The 265 pure Ag-SERS + 153 pure Raman metabolite spectra.
- **Only challenge/evaluation?** The uricase, spike-in, isotope, and serum-baseline conditions (~460 spectra).
- **Literature-derived?** The `literature/` assignment table + 3 digitized CSVs.
- **Pure metabolite spectra present?** **Yes** — 265 SERS + 153 Raman pure standards (this corrects the earlier framing that treated serum-Ag as only a serum study).
- **Are the 368 DuckDB grounding spectra = the 907 raw?** The 368 (`serum_ag_colloids_grounding` in `grounding_metadata`) are a **processed subset/summary** of the 907 raw (the warehouse peak-support summary counts 64 class-summaries). Not identical; a derived grounding extract.
- **Analytes among the 368?** The pure-metabolite panel (53 metabolites) is the grounding-relevant subset.
- **Are the 53 spike-ins used for grounding, evaluation, or neither?** The **pure** 53 are grounding-eligible; the **serum spike-in** versions are perturbation evaluation; currently the demo uses only 3 SAEL contrasts (uricase/hypoxanthine) — the 53-panel is **not wired** for either.
- **Uricase depletion:** honestly **inconsistent** (6/11 axes wrong direction, n=5/5); preserved, not laundered.
