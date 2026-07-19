# GAIRA Serum Ag-Colloid Dataset Audit

**Date:** 2026-07 · Read-only (archives extracted only to audit temp). Source: `/Volumes/SSD_Rad/GAIRA_DATA/raw/serum_ag_colloids/` + `cspp_serum/`.

## What it actually is
**Gobbato, Fornasaro, Sergo & Bonifacio (Univ. Trieste): "Adsorption of Serum Components on Ag Colloids — On the Biochemical Interpretation of SERS of Human Serum."** A mechanistic SERS-of-serum study arguing that the dominant serum SERS bands (638, 725 cm⁻¹) arise from **hypoxanthine and uric acid** adsorbing on the Ag colloid — not a generic "serum fingerprint."
- Instrument: **B&WTek i-Raman Plus BWS465-785S, 785 nm**; each `.txt` = a single raw acquisition (2048-pixel export), Raman-shift axis −310…+3270 cm⁻¹.
- **907 measured `.txt` spectra** (from `dataset_spectral_data.zip`) + digitized-literature CSVs.

## Composition (direct `find | wc -l`)
| Subfolder | n spectra | Content |
| --- | --- | --- |
| dataset uricase | 20 | **core controlled design**: serum ± hypoxanthine spike ± uricase (4 conditions × 5 rep) |
| isotopic | 73 | UA vs ¹⁵N-UA ± HSA ± ultrafiltration (binding mechanism) |
| SERS metabolites for fitting | 30 | hypoxanthine / UA-free / UA-bound fit references |
| SERS metabolites | 265 | **53 pure metabolites × 5 rep** |
| Raman metabolites | 153 | 51 metabolite powders, normal Raman |
| SERS serum Merck | 15 | commercial serum baseline (3×5) |
| SERS spiked serum Merck | 270 | **53 metabolites spiked into serum ×5** + control |
| donors serum SERS | 81 | 81 healthy-donor serum (1 each) |
| digitized literature | 0 (3 csv) | De Gelder 2007 / Kim 1987 / Stewart 1999 — NOT measured here |

**Empty raw sibling folders** `serum_ag_colloids_grounding/` and `serum_ag_colloids_literature_grounding/` (0 files) correspond to data **processed upstream**: the DB `grounding_metadata` carries `serum_ag_colloids_grounding = 368` grounding spectra (and the warehouse peak-support summary counts 64 class-summaries). So the empty folders are not missing data — the grounding was built into `interim/gaira.duckdb`.

## Uricase controlled design (the calibration core)
| condition_id | intervention | serum | enzyme | control | n_spectra | n_samples | n_rep |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Serumspiked_Prot1 | hypoxanthine spike | Sigma | none | spiked test | 5 | 1 | 5 |
| Serumspiked+Enzyme_Prot1 | hypoxanthine spike + uricase | Sigma | uricase | enzyme-treated | 5 | 1 | 5 |
| SerumSigma_Prot1 | neat serum | Sigma | none | control | 5 | 1 | 5 |
| SerumSigma+Enzyme_Prot1 | neat serum + uricase | Sigma | uricase | enzyme control | 5 | 1 | 5 |

cspp_serum companion (`Figure-7`, 150 spectra): background (Bkg, 50) / **ergothioneine 25 µM** (50) / **hypoxanthine 50 µM** (50).

## PLAIN ANSWER
- **Analytes the dataset actually contains (measured):** 53 pure metabolites (incl. adenine, hypoxanthine, uric acid, xanthine, guanine, **ergothioneine**, glucose, urea, albumin, amino acids, nucleotides) as pure SERS standards AND spiked into serum; plus isotopic uric-acid.
- **Experimentally CONTROLLED interventions:** hypoxanthine spike, uric-acid **uricase depletion**, ¹⁵N-UA isotope, 53-metabolite serum spikes, ergothioneine serum spike (cspp).
- **Candidate assignments only (NOT controlled):** the `literature/` band-assignment table and `digitized literature spectra/` — literature interpretations, not spiked measurements.
- **Role in GAIRA:** serves as **grounding** (serum_ag_colloids_grounding = 368/64 in DB/warehouse) AND **calibration** (the uricase/hypoxanthine/ergothioneine contrasts → the demo's Mode-2 "uric acid" SAEL contrasts, incl. the honestly-preserved **inconsistent** uricase-depletion result). It is a serum-matrix Ag-colloid SERS dataset (single serum pool ×5 rep for the uricase design → low independent-sample count).
- **Name confusion risk:** "serum_ag_colloids_grounding" (empty raw folder) can read as missing; it is actually the Gobbato dataset processed into the DB. The name conflates grounding + calibration + mechanistic roles.
