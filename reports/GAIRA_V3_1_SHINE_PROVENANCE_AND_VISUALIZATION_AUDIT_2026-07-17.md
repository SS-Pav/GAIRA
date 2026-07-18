# GAIRA V3.1 — SHINE Provenance & Visualization Audit

**Date:** 2026-07-17

## Search for usable SHINE spectra
Re-searched SHINE raw + processed sources:
- Raw: `/Volumes/SSD_Rad/GAIRA_DATA/raw/shine_ev_sers/SERS-Hepatotoxicity_DATA_CODE_FIGURE(.zip)` — ~15,027 deeply-nested per-scan `s_N` files; **no per-sample or per-condition mean-spectra file**.
- Processed: `pilot3_shine_*` tables carry only the **8-axis autoresearch BSV** (protein_peptide, lipid_membrane, nucleic_acid, carbohydrate_glycan, small_molecule_metabolite, matrix_background, substrate_adsorption_bias, protocol_sensitive_signal). **No table has `wn_`/`intensity_json` columns.**

**Conclusion: SHINE spectra cannot be defensibly reconstructed at demo build time** (averaging 15,027 nested raw scans is out of scope and unvalidated). Recomputation through the demo's 11-axis engine is therefore not possible.

## Disposition (per decision rule)
The full **11-axis radar is REMOVED** for SHINE. Replaced with a **Legacy reduced-dimensional SHINE response**:
- heatmap of the **actually-active autoresearch source axes** across dose × time,
- a dose × time trajectory on the dominant active axis,
- explicit provenance note: legacy cached autoresearch BSV, collapsed upstream (raw nonzero 11-axis values ≈ 2 per cohort),
- **no** title implying position in the full 11-axis reference space.

## Why
The autoresearch SHINE BSV is a **collapsed low-dimensional projection** (only ~2–3 axes carry signal; 100% of 15,027 spectra fire on the same handful). Presenting it as an independent 11-axis projection would imply 11 measured dimensions that do not exist.

## Tests (`tests/test_shine_projection_provenance.py`)
- The SHINE renderer is the reduced-dimensional one and draws **no** `radar_figure`.
- SHINE 11-axis BSV is confirmed collapsed (≤3 active axes per cohort).
