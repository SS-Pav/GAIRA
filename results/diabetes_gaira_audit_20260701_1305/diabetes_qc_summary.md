# Diabetes EV-SERS — QC summary

- Patients in metadata: 64
- Patients with spectra loaded: 63
- Patients missing from .mat: 1 (2151-0465)
- Wavenumber grid: 447.9 – 1619.3 cm⁻¹ (737 pixels native)
- Interpolated to demo grid: 400–1800 cm⁻¹ at 1 cm⁻¹ (1401 points)

## 2-group counts
- OWD: 39 patients
- NWD: 24 patients

## 4-subgroup counts
- White Impact: 20 patients
- Asian Strong-D: 17 patients
- Asian Impact: 8 patients
- White Strong-D: 5 patients

## Non-zero-axis distribution (11-axis BSV, patient-level)
- min: 11 / max: 11 / median: 11
- % of patients with ≥8 axes populated: 100.0%

## Potential batch effects to watch
- Impact vs Strong-D come from two different site/protocol codes (`2151-*` vs `32113-*`). Any BSV difference may partially reflect the site/protocol difference. Downstream analyses should stratify.
- BMI, HbA1c, and age distributions differ by group (see label audit).
