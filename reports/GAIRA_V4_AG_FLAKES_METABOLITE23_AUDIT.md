# GAIRA V4 — Ag-flake Metabolite-"23" Audit  →  CORRECTION: 24 metabolites, ORC-Ag, peak tables only

**Date:** 2026-07-18 · Source: `raw/ag_flakes_metabolites_23/1-s2.0-S1386142523012726-mmc1.docx` (Zhang et al., *Spectrochim. Acta A*, Tsinghua). Registries: `data_audit/v4_ag_flakes_metabolite24_peak_registry.csv` (454 rows), `v4_ag_flakes_metabolite24_analytes.csv`.

## Corrections
- **24 metabolites, not 23** (DOCX title: "Surface-enhanced Raman Database of **24** Metabolites"; Tables S2/S3 both have 24 rows).
- **Substrate = electrochemically ORC-roughened silver** (AgNP-roughened film, 50–200 nm, R6G-validated) — **not colloidal "flakes"** (folder name is a local label).
- **Excitation wavelength and analyte concentration are NOT in the SI** (only spot size 10×10 µm swift-mapping). Do not assume.

## What the DOCX contains — PEAK TABLES ONLY (no reconstructable spectra)
Three tables (python-docx): S1 reagents (supplier/purity/CAS), S2 metabolites (class/formula/structure), **S3 = Raman peaks as `peak(code)` strings**. **No intensity vectors, no wavenumber axis, no continuous spectra.** Only peak positions with qualitative relative-intensity codes:
- **vs** very strong (>3), **s** strong (1–3), **w** weak (0.5–1), **vw** very weak (<0.5); **`*`** = exclusive/characteristic peak.

## Structured extraction (defensible)
`v4_ag_flakes_metabolite24_peak_registry.csv` = **454 peak rows across 24 metabolites** (`metabolite, peak_cm1, intensity_code, exclusive_characteristic`). Intensity codes: s 218, w 110, vs 87, vw 39. Per-metabolite peaks 8–27. Assignments are only narrative (DFT) in the body → no clean per-peak assignment column.
24 analytes span 5 classes: 10 glycerophospholipids (PC/LPC), 5 sphingolipids (SM/Cer/sphinganine), + amino-acid derivatives/dipeptides (Leu-Leu, gamma-Glu-Leu…), taurine, hypaphorine, 1-hydroxy-2-naphthoic acid, phenylacetylglutamine.

## Recommended role
**Direct PEAK-LEVEL SERS grounding (ORC-Ag substrate)** — usable for **MSS band construction, collision analysis, and physics-atlas peak evidence**, NOT full-spectrum grounding, retrieval, or numeric BSV. Do **not** fabricate spectra. It adds lipid/sphingolipid peak evidence (a class thin in the current grounding).
