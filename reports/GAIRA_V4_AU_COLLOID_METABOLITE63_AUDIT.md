# GAIRA V4 — "Au-colloid" Metabolite-63 Audit  →  CORRECTION: it is Ag-colloid

**Date:** 2026-07-18 · Source: `raw/sers_metabolite_63/` (PMC6989628 / NIHMS1547448, Sherman et al., *Talanta*). Registry: `data_audit/v4_metabolite63_analytes.csv`.

## The task's premise is wrong — the substrate is SILVER, not gold
Direct evidence from the paper PDF:
- Abstract: *"untargeted approach utilizing **citrate-capped silver nanoparticles**."*
- Methods: several substrates were prescreened (incl. Au colloids) but **Ag citrate colloids (Lee–Meisel)** were **selected** for the database (90.6 mg AgNO₃ / 500 mL, 1% trisodium citrate, boil 20 min; aggregated with NaBr).
- Conclusion: *"the widely used **Lee and Meisel silver colloids** were used."*
Gold was only a **rejected prescreen candidate**. **metabolite-63 is Ag-SERS.** The repo already labels it correctly (`analysis/make_publication_figures.py:241` → `"Ag colloid SERS"`); there is **no Au mislabel to fix downstream**.

## Facts
| Field | Value |
| --- | --- |
| Substrate | **Ag citrate colloid (Lee–Meisel), NaBr-aggregated** |
| Excitation | **633 nm** HeNe (~600 µW, 20×) |
| Analytes | **63** (not 64) — 63 intensity columns in supplement-2.xlsx; prose says 63 |
| Spectra/analyte | **1** (average of 3 scans, background-subtracted) |
| Points | 716 per analyte, ~500–2000 cm⁻¹ (8 per-group wavenumber vectors) |
| Matrix | pure standard solutions (~15 µL analyte + 250 µL Ag colloid), **not serum** |
| Raw vs fit | processed averaged continuous spectra (negative values = bg-subtracted); the `.fit/.peaks` sibling is separate |
| In DuckDB | grounding (64 class summaries) |

## Discrepancy to preserve (do not silently reconcile)
The xlsx delivers **N-acetyl-D-tryptophan** (col 44) but omits **deoxyadenosine monophosphate (dAMP)** which the prose lists. The **xlsx (63 columns) is authoritative** for delivered spectra; treat dAMP as claimed-but-not-delivered.

## Role
**Direct molecular grounding — Ag-SERS observation domain (633 nm).** 63 pure-analyte reference spectra. Substrate metadata (Ag, 633 nm) must be preserved and NOT merged numerically with Raman (RamanBioLib) or other-substrate SERS without modality tags.
