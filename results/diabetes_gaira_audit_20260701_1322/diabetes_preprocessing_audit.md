# Diabetes EV-SERS — preprocessing audit

## Raw data
- Source: `/Volumes/SSD_Rad/GAIRA_DATA/raw/diabetes_plasma_ev_sers/extracted/`
- MATLAB files:
    - `RawDataImpact.mat` — `smoothed_spectra` object array, one entry per
        Impact-cohort patient. Each entry is a (737 × N_scans) matrix.
    - `RawDataStrong.mat` — same shape, Strong-D cohort.
- Metadata: `patient_data.csv` (64 rows, 13 columns).

## Wavenumber calibration
- Method: cubic polyfit of pixel index → wavenumber against 8 known Raman
    peaks (Phe / Tyr / lipid / amide anchors). Source: `Figure3.m` of the
    original manuscript. Pixel range used = 162–898 (inclusive) → 737 rows,
    matching the .mat data.
- Peaks used: 620.9, 795.8, 1001.4, 1031.8, 1155.3, 1450.5, 1583.1, 1602.3 cm⁻¹.
- Coverage: ~484–1642 cm⁻¹ across the 737 pixels.

## GAIRA preprocessing (applied in this audit, via `build_report`)
- Per patient: mean of technical SERS scans → interpolate to the demo's
    canonical grid (400–1800 cm⁻¹ at 1 cm⁻¹).
- `build_report` internally applies:
    - Savitzky–Golay smoothing (window 11, polynomial 3)
    - ASLS baseline (λ=1e5, p=0.01, 8 iterations)
    - L2 normalization
- Peak detection: SciPy `find_peaks` with prominence floor 5e-3,
    minimum distance 6 cm⁻¹.
- Motif scoring: 11 curated class-level motifs (`gaira_core/motif_scoring.py`)
    with anchor-first-then-support geometric mean over co-firing bands.
- MSS: 11 curated analyte anchor/support/anti sets
    (`gaira_core/data_loader.py:MOLECULES`).
- Substrate physics: `Ag colloid SERS` — dampens purine 720–740 cm⁻¹
    (×0.65) and mildly boosts thiol/thione 490–510 cm⁻¹ (×1.20).
- BSV projection: motif fires → noisy-OR aggregation over the 11 axes.

## What we do NOT do here (compared to the prior GAIRA_BUILD audit)
- We do **not** apply an additional sum-to-one normalization or CLR transform
    on the BSV. The prior audit's per-axis Cohen's d values are computed on
    CLR-transformed spectrum-level BSVs; ours are computed on the demo
    pipeline's raw motif-based BSVs at the patient-mean level. Both are
    valid decompositions of the same biology; they will produce different
    absolute magnitudes but qualitatively similar directions.
- We do **not** subtract a background or blank spectrum. The .mat rows
    already carry the study's own smoothed spectra (per the field name);
    additional background handling would be double-processing.
- We do **not** remove cosmic rays or perform outlier rejection at this
    stage. The mean-of-scans step provides implicit robustness.

## Replicates
- Each patient contributes 1 BSV row = mean over N_scans SERS scans.
- N_scans ranges 441–1089 depending on the acquisition map size.
- Statistical tests are performed **at the subject level** (1 patient = 1 n),
    not at the spectrum level. This avoids the pseudoreplication issue
    flagged in the audit brief.
