# GAIRA Calibration Dataset Audit

**Date:** 2026-07 · Read-only. Full table: `data_audit/calibration_dataset_registry.csv`, `data_audit/axis_calibration_coverage.csv`.

## Calibration datasets (7)
| Dataset | Analyte / behaviour | Modality | Substrate | Excitation | n spectra | Levels/conditions | Used V3.1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adenine_sers_control | adenine dose | SERS | bAgNPs | 785 | 17 CSV | 6–7 conc + stability | **yes (live)** |
| european_multi_instrument_adenine | adenine reproducibility | SERS | **sAg/sAu/cAg/cAu** | **532;785** | 3516 (ILS) / 7032 raw | 43 conc, 15 labs, blank | ablation only |
| ergothioneine_serum | ergothioneine dose (redox) | SERS | cAg | 785 | 55 | 11 conc × 5 | **yes (live)** |
| serum_ag_colloids (Gobbato uricase) | hypoxanthine spike + UA uricase depletion | SERS | Ag colloid | 785 | 20 (uricase design) | 4 conditions × 5 | yes (cached SAEL) |
| cspp_serum | ergothioneine + hypoxanthine serum spikes | SERS | Ag colloid | 785 | 150 (Fig-7) | Bkg/Erg25/Hyp50 | yes (cached SAEL) |
| sers_metabolite_63 (NIHMS1547448) | 63-metabolite reference panel | SERS | Ag | — | 63 | 1/analyte | grounding (64) |
| amino_acid_raman_grounding | 20-analyte AA/metabolite panel | **Raman** | powder/CaF₂ | — | 20 | 1/analyte | grounding (20) |
| (otc_drugs) | 3 OTC drugs | Raman | pure | — | 300 | 50/drug-form | NO |
| (metabolite_sers63_support) | **NOT spectra** — 64 .fit + 63 .peaks Gaussian fit products | — | — | — | 0 | — | — |

## Totals (measured spectra only; fit products excluded)
- **Calibration datasets: 7** (adenine×2, ergothioneine, serum-uricase, cspp, metabolite-63, amino-acid-20; +OTC as a non-biochemical extra).
- **SERS calibration spectra ≈ 3,726** (17 adenine + 3,516 European-ILS + 55 ERG + 75 serum-protocol + 63 metabolite-63) — or **≈ 7,242** counting the 7,032 raw European `.txt` instead of the 3,516-row ILS matrix.
- **Raman calibration spectra = 320** (20 amino-acid + 300 OTC drugs).
- **Serum-matrix calibration:** serum_protocol_comparison (75), serum_ag_colloids uricase (20), cspp (150). **Buffer/pure standards:** adenine, metabolite-63, amino-acid-20, OTC.
- Cross-instrument: only European adenine (15 labs). Cross-substrate: only European adenine (4 substrates). Depletion/enzyme: only serum_ag_colloids uricase. Isotope: only serum_ag_colloids (¹⁵N-UA).

## Axis calibration coverage (verdicts)
| Axis | Calibration support | Verdict |
| --- | --- | --- |
| G01 Purine-nuc | adenine (×2), serum spike | **supportive** (dose Spearman 0.83) |
| G02 Purine-met | uricase/hypoxanthine/cspp | **partially_supportive** (hypoxanthine agree; **uricase depletion inconsistent**, preserved) |
| G10 Redox | ergothioneine, cspp | **supportive** (dose Spearman 0.94) |
| G05 Glycan, G06 Protein, G07 Aromatic, G11 Metabolite | reference panels only (no dose/perturbation) | **insufficient** |
| G03 Pyrimidine, G04 Nuc-phosphate, G08 Lipid, G09 Sterol | none | **not_tested** |

**Only 3 of 11 axes have supportive/partially-supportive calibration** (G01, G02, G10). The uricase-depletion contrast is honestly recorded as **inconsistent** and NOT converted into support. The rich European 4-substrate × 2-laser adenine set is **not exploited** because the demo substrate layer is blind to Au/planar/excitation.
