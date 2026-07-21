# Biological Data Audit (Part 6)

Every biological dataset found under `/Volumes/SSD_Rad/GAIRA_DATA/raw/`, how it loads,
its counts, labels, status, and the analysis unit. REAL datasets are projected live
through the frozen V6 engine and committed as sanitized artifacts in
`biological_artifacts/` (so a fresh checkout runs without the volume).

## Wired — genuine V6 (committed)

| dataset | source path | format | spectra / units | groups | wavenumber | unit | status |
|---|---|---|---|---|---|---|---|
| covid_serum_raman | raw/covid_serum_raman | txt matrices + wave_number.txt | 465 spectra | COVID 159 / Healthy 150 / Suspected 156 | file (900, 400–2112) | spectrum | REAL |
| hcc_serum | raw/hcc_serum/data.csv | wide CSV | 144 spectra | HCC 72 / control 72 | column headers (crop ≥200) | spectrum | REAL |
| diabetes_plasma_ev_sers | raw/diabetes_plasma_ev_sers/extracted | .mat object arrays | 63 patients | Impact 39 / Strong-D 24 | Fig3 cubic (737, 484–1642) | **patient** | REAL |
| shine_ev_sers | raw/shine_ev_sers/.../Figure4/data | consolidated .mat `clustered` | 720 (subsampled) | D0 360 / D2 360 × dose C0/10/20/40 | Fig4D cubic (737, 448–1619) | spectrum (cohort) | REAL |
| small2023_ev | raw/small2023_ev | NormedProbe1.mat + SI xlsx axis | 600 (subsampled) | c00…c100 (probe conc.) | Fig_S7 col A (1131, 670–1800) | spectrum | REAL |

Counts are measured from the files, not remembered. SHINE and small2023 are
deterministically stratified-subsampled (90/cohort, 100/level) for artifact size; the
subsampling is logged and reproducible.

## Discrepancies found & resolved

- **SHINE spectra count.** The historical "15,027 SHINE spectra" figure matches neither
  the raw scan total (23,646) nor the QC'd consolidated totals (RawDataSet91 = 3,531).
  Resolution: use the **consolidated, QC'd 737-ch `clustered` matrices** (RawDataSet91);
  the demo subsamples 720 from these with documented counts.
- **SHINE Day 1.** `D1_*` folders have no per-sample structure and are **empty in the
  consolidated matrices** → Day 1 is omitted; only D0/D2 are used (which is what the
  paired analysis needs).
- **SHINE pairing.** Cell-culture secreted EV → replicate *i* at D0 is not the same
  physical specimen as *i* at D2. Pairing is therefore **cohort-level (day-vs-day within
  a dose)**, not per-well; stated on the page and in the analysis.
- **small2023 wavenumber.** Not in the `.mat`; recovered from the SI `Fig_S7 (1).xlsx`
  (col A = Probe-1, 1131 ch, 670–1800 cm⁻¹, exact channel-count match). The two probes
  have different axes (1131 vs 1400) and must not be concatenated → Probe-1 only.
- **small2023 labels.** `c00…c100` are probe CONCENTRATION levels, not disease → marked
  **characterization-only** (no biological contrast claimed).
- **diabetes 40-vs-39.** Metadata lists 40 Impact rows but the `.mat` has 39 objects.
  Resolution: labels from the `.mat` file (not the demographic CSV), patient-level, 39+24.

## Sanitization

Records contain only: anonymised `id`, `group`, optional `strata`, and numeric V6
outputs (`coord`, `themes`, `motifs`, `ood`, `confidence`, `background`). **No
demographics** are read or stored (verified by test + grep). SHINE/small2023 are
cell-culture (no PII); diabetes demographics are never touched.

## UNAVAILABLE (registry shows honestly; no output fabricated)

- Liver serum SERS (CCA/HCC/LM/NC) — 4-class per-sample txt, not yet wired for V6.
- Ovarian plasma Raman+SERS — paired zips, not yet wired.
- Others on disk (coeliac faecal, covid variants, cspp, nature_serum, ucla saliva,
  stroke urine `.rar`, single-vesicle `.rar`) — not wired; some carry sensitive
  filename metadata (coeliac age/gender) and are deliberately not surfaced.

## Reproduce

`python tools/build_biological_v6.py` (with the volume mounted) regenerates all five
artifacts + `manifest.json`, each fingerprint-locked to atlas `09ed804a…`.
