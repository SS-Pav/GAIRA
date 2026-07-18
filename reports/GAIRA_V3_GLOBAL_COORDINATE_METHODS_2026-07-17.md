# GAIRA V3 — Global Coordinate Methods

**Date:** 2026-07-17 · **Calibration:** `global_coordinate_calibration_v1.json` (frozen)
**Build:** `tools/build_global_coordinate_reference.py` (deterministic) · **Apply:** `gaira_core/global_coordinates.py`

---

## What defines biochemical meaning
Axis meaning is fixed by the **unchanged V2 engine**: 11 spectral motifs (`motif_scoring.py`), 11 curated MSS reference analytes (`mss_scoring.py`), substrate rules, and the noisy-OR projection — plus the 202-molecule RamanBioLib reference family mapping. **Meaning is NOT defined by any biological population or by disease labels.** The global calibration only sets *scale*, not meaning.

## What defines scale
A **frozen robust per-axis standardization**, fit label-free on a documented biological reference population:

```
global_j (unbounded) = (raw_bsv_j − center_j) / scale_j
global_j (display)   = clip(global_j, −4, +4)          # unbounded value always preserved
  center_j = median(raw_bsv_j)                          over the frozen fit population
  scale_j  = max( 1.4826 · MAD(raw_bsv_j), 0.02 )       robust ~sigma, floored
```

- **Fit population:** 275 biological Ag-SERS spectra — serum-liver (212 patients) + EV-diabetes (63 samples), pooled **label-free**. This is the "normal reference range."
- **Scale floor = 0.02 (2% BSV):** axes whose reference spread is below this (thinly-grounded G01/G03/G04) are treated as noise-level, preventing near-zero spread from exploding global variance. Without it, Pyrimidine/Purine-nuc global dynamic range inflated to ~140/81; with it, ~8.
- **Robust statistics** (median/MAD, not mean/SD) resist the skewed biological mixture distribution.
- **Deterministic:** raw projection + robust stats are deterministic; two builds produce an identical content hash (`content_sha256`), verified. `build_timestamp` is stored separately and excluded from the hash.

## What biological datasets contribute (and their sanctioned role)
- **Role A — anchors (meaning):** 202 reference analytes + motif/MSS definitions. Coverage/ontology only; **not** used for center/scale.
- **Role B — calibration behaviour:** adenine (6 live Ag-SERS concentrations) + ergothioneine (55 live Ag-SERS spectra). Projected through the frozen transform but **not** used to set the reference range — by design their extremes may exceed it.
- **Role C — biological range:** serum + EV. Used **only** to estimate center/scale/quantiles (ranges, matrix distributions), never to define axis meaning.

## Why the >180,000 biological spectra are NOT molecular grounding
The corpus contains >180k EV/serum/mixture spectra. These are **mixtures**, not pure molecular references. Using them to define analyte identity or axis meaning would (a) conflate matrix/acquisition identity with biochemistry and (b) violate the anti-circularity rule (disease datasets must not define axes). V3 uses biological spectra strictly for **range and nuisance estimation** (Role C). Axis meaning stays with the 202 references + curated motif/MSS definitions. Only 275 biological spectra were used for the fit (a documented, label-free subset); scaling to all 180k is deferred (it would not change the *meaning*, only tighten range estimates, and is unnecessary for a prototype).

## How redox dominance is addressed
Per-axis robust scaling expresses every axis in units of its own biological reference spread, so no axis dominates the radar purely by numeric scale. Result (over the 336-sample projection):
- Redox (G10) **raw** variance rank **2** → **global** variance rank **2** (comparability preserved; not forced to rank 11 — *comparability, not cosmetic equality*).
- Ergothioneine titration in **global** redox coordinates rises monotonically **0.68 → 3.9** (0 → 2 µM) and **exceeds the biological reference range** [−2.15, +6.53] — i.e. biologically meaningful extreme redox states still exceed normal, exactly as required.
Because the fit population is biological-only, calibration titrations are not allowed to inflate the "normal" scale.

## How cohort invariance is tested
Global coordinates are a pure function of the frozen calibration, so a sample's coordinates cannot depend on its comparison set. `tests/test_global_coordinate_invariance.py` projects representative samples alone, with their own cohort, with a different disease cohort, and with a mixed EV+serum set: **max global deviation ≤ 1e-9** (exact). Cohort-relative z-scores DO change across sets (expected). Raw BSV is unchanged from V2 (≤1e-9, `tests/test_v2_raw_bsv_regression.py`).

## Why V3 is not yet a learned foundation model
No neural network, encoder, autoencoder, generative model, or disease classifier is trained. The transform is a transparent, inspectable, deterministic robust standardization. Global coordinates are a **target-definition and calibration prototype**; a learned encoder is deferred to a later release after these coordinates and validation targets are stable.

## Artifacts
- `data/generated/global_coordinate_calibration_v1.json` — center/scale/quantiles, transform rules, counts, composition, limitations, content hash.
- `data/generated/global_coordinate_reference_samples_v1.csv` — 336 per-sample raw + global coords (+ labels, stored for post-fit comparison, never used in fit).
- `data/generated/global_coordinate_build_manifest_v1.json` — source-file SHA-256s, population composition, determinism note.

## Known limitations
Fit population is 100% Ag-SERS (no Raman); Raman-regime samples project off-distribution. Six axes are not independently grounded (see ontology audit). Dataset identity remains a moderate separator (see nuisance report).
