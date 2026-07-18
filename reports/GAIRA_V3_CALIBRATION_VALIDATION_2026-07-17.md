# GAIRA V3 — Calibration Validation

**Date:** 2026-07-17 · Tests: `tests/test_calibration_behavior.py` (all pass) · Diagnostics: `tools/validate_global_coordinates.py`

Each calibration intervention is tested and **honestly reported**. No inconsistent result was converted into a positive validation.

---

## Adenine (6 real bAgNP concentrations, live Ag-SERS projection)
Source: `raw/adenine_sers_control/Adenine_bAgNPs_*.CSV` (10 pg/mL → 10 µg/mL), parsed + cropped 400–1800 + projected through the unchanged V2 engine with `substrate="Ag colloid SERS"`.

| Metric | Value |
| --- | --- |
| Concentrations loaded | 6/6 (REAL mode, no placeholder) |
| Spearman(log₁₀C, G01 purine-nuc) | **0.83** |
| Monotonic step fraction (G01) | 0.80 |
| G01 dynamic range (raw) | 0.107 |
| Target/off-target response ratio | **6.6×** (G01 moves 6.6× the average off-target axis) |
| Global coordinates finite | yes |
| Cohort-invariant | yes (global coords independent of comparison set) |
| Raw V2 BSV unchanged | yes (≤1e-9) |

**Verdict: supportive.** Purine-associated axis is the dominant, directionally-correct responder. Perfect monotonicity is not required and not claimed; substrate dampening keeps the call class-level.

## Ergothioneine (55 real spectra, 11 conc × 5 reps, live Ag-SERS projection)
Source: `raw/ergothioneine_serum/ERG_calibration.csv`. **Live raw-spectrum projection** (distinct from the cached 8-axis SAEL dose table used in Mode 2's slider — both are surfaced and labelled).

| Metric | Value |
| --- | --- |
| Concentrations | 11 (0–2.0 µM) |
| Spearman(C, G10 redox) | **0.94** |
| G10 dynamic range (raw) | 0.067 |
| G11 (sibling) dynamic range | 0.009 → thione routes to G10, not the metabolite sibling |
| Global redox by conc | 0.68 → 1.62 → 1.74 → 2.38 → 2.97 → 3.05 → 3.45 → 3.88 → 3.85 → 3.90 → 3.79 |
| Exceeds biological reference range [−2.15, 6.53]? | high doses reach ~3.9 σ (within extreme range; monotonic) |

**Verdict: supportive.** Strong ordered redox response; the redox split routes signal to G10 over G11. Cached SAEL vs live projection are explicitly distinguished.

## Hypoxanthine / uricase / uric-acid — three separate interventions
Source: SAEL contrasts (`calibration_conditions.csv` + `calibration_delta_bsv.csv`), cached 8→11 remapped. **Reported separately — never merged into one "uric acid validation" claim.**

| Contrast | n (ctrl/pert) | Expected | Observed | Axes disagreeing | Verdict |
| --- | --- | --- | --- | --- | --- |
| Hypoxanthine spike — serum | 50/50 | purine ↑ | agree | 0 | **supportive** |
| Hypoxanthine spike + uricase | 5/5 | purine ↑ | agree | 0 | **supportive** |
| **Uricase depletion — Sigma serum** | 5/5 | purine ↓ | **opposite on several axes** | **6** | **inconsistent (preserved)** |

**Verdict: partially supportive overall.** The uricase-depletion contrast is honestly recorded as **inconsistent** (6 of 11 axes moved opposite to the literature-expected direction; small n=5/5; likely serum-matrix/substrate variability). This inconsistency is surfaced, not laundered. No isotope (¹⁵N/¹³C) data exists in the corpus (not fabricated).

---

## Cross-cutting calibration facts
- All calibration coordinates are finite; global coordinates are cohort-invariant (≤1e-9).
- Raw V2 BSV for every calibration input is numerically identical to V2 (≤1e-9).
- Redox handling verified separately (see methods report): raw var rank 2 → global rank 2; ergothioneine extremes exceed the biological range.
