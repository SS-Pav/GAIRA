# GAIRA V4 — Direct Molecular Grounding Corpus

**Date:** 2026-07-18 · Registries: `data_audit/v4_direct_grounding_sources.csv`, `v4_grounding_analyte_registry.csv`, `v4_metabolite63_analytes.csv`, `v4_ag_flakes_metabolite24_peak_registry.csv`.

## Direct grounding sources (measured pure-analyte evidence only)
| Source | Publication | Modality/substrate | Excitation | Unique analytes | Measured spectra | Raw |
| --- | --- | --- | --- | --- | --- | --- |
| RamanBioLib | DOI 10.1002/jrs.1734 | **Raman** (CaF₂/glass/metal-ring) | 785/1064/532/488 | **141** | 202 rows | parquet 272,902 pts |
| amino_acid_raman_grounding | curated AA panel | **Raman** powder | — | 20 | 20 | xlsx |
| adenine_sers_control | bAgNP | **Ag-SERS** | 785 | 1 | 12–17 | CSV |
| sers_metabolite_63 | PMC6989628 | **Ag citrate colloid** (Lee–Meisel) | **633** | **63** | 63 | xlsx (avg) |
| serum_ag_colloids (Gobbato pure) | Trieste | **Ag-SERS** (265) + **Raman** powder (153) | 785 | 53 (+51 Raman) | 418 | 907 txt |
| ag_flakes_metabolites_24 | S1386142523012726 | **ORC-roughened Ag** | not in SI | **24** | 454 peaks (peak-level only) | DOCX |

## Totals — kept SEPARATE (do not collapse)
| Counter | Value |
| --- | --- |
| Unique chemicals (approx, de-duplicated across sources) | ~240–280 (RamanBioLib 141 + metabolite-63 63 + Gobbato 53 + AA 20 + Ag-flake 24, with overlaps) |
| compound × substrate × excitation observations | 202 (RamanBioLib) + others |
| Independent measured spectra (full-spectrum) | RamanBioLib 202 + AA 20 + adenine ~16 + metabolite-63 63 + Gobbato pure 418 ≈ **719** |
| Peak-level-only evidence (NOT full spectra) | Ag-flake 454 peaks / 24 analytes |
| **Raman spectra** | 202 (RamanBioLib) + 20 (AA) + 153 (Gobbato powders) = **375** |
| **Ag-SERS spectra** (colloid) | adenine 16 + metabolite-63 63 + Gobbato SERS 265 = **344** |
| **Au-SERS spectra** | **0** (metabolite-63 is Ag, not Au; the corpus has NO Au-SERS pure-analyte grounding) |
| **Ag-flake / ORC-Ag SERS** | 24 analytes as peak tables (no spectra) |
| **other/unknown SERS** | — |

## Key corrections
1. **There is NO Au-SERS molecular grounding.** metabolite-63 (the presumed Au dataset) is **Ag citrate colloid, 633 nm**. The multi-substrate ambition (Raman / Ag-SERS / Au-SERS) currently has **Raman + Ag-SERS only**; Au-SERS grounding must be acquired.
2. The grounding is **modality-heterogeneous** (Raman 785/1064/532/488 + Ag-SERS 785/633) and must carry substrate+excitation metadata; numeric BSVs across modalities are NOT directly comparable.
3. Ag-flake adds **lipid/sphingolipid peak evidence** (a thin class) but only at peak level.
