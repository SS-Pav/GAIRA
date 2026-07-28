# PREPROCESSING_AUDIT
### Every step that turns a raw Raman spectrum into a corpus vector

*Part 3 of the GAIRA Foundation Model audit. The contract is simple and strict: **every
spectrum — reference or projected — passes through the identical deterministic
pipeline**, because a projection into a frozen basis is only meaningful if the query is
expressed in the same representation the basis was fit on. Source of truth:
`gaira.preprocessing.pipeline` and `gaira.foundation.dataset._prep`. Reproduced by
`foundation_audit/code/preprocessing_demo.py`.*

---

## 1. The pipeline (canonical config `P2 = asls + savgol + l2`)

```
raw (wavenumber, intensity)
  │  1. CROP        to the analysis window [450, 1800] cm⁻¹
  │  2. BASELINE    ASLS  (λ=1e5, p=0.01, 8 iterations) — subtract broad background
  │  3. SMOOTH      Savitzky-Golay (window=9, poly=3)   — denoise, preserve band shape
  │  4. RESAMPLE    linear interp onto a fixed 2 cm⁻¹ grid → 676 bins; NaN outside range
  │  5. NORMALIZE   L2 (unit vector) on the finite portion
  ▼
corpus vector  v ∈ ℝ⁶⁷⁶   (‖v‖₂ = 1)
  │  6. CLIP ≥ 0   applied by the NMF fit/projection only (non-negativity)
  ▼
NMF input / projection query
```

Figure `figures/preprocessing_stages.png` shows all six stages on a real adenine
spectrum — the 722 cm⁻¹ purine ring-breathing mode survives cleanly through to the
clipped vector.

---

## 2. Each step, audited

**1 · Crop — window [450, 1800] cm⁻¹.**
Chosen because *all three Raman sources cover it*. It is deliberately **wider** than the
520–1750 window used in the earlier Ag-SERS-constrained stages: with SERS removed from
the representation, there is no need to restrict to the Ag-SERS overlap region, so the
lipid/CH region up to 1800 and the low-shift region down to 450 are retained. Spectra
narrower than the window are NaN-padded outside their own range (mean finite fraction
across the corpus = **0.9988**, i.e. essentially every analyte spans the full window).

**2 · Baseline — asymmetric least squares (ASLS).**
`λ=1e5` (smoothness), `p=0.01` (asymmetry — points above the fit are weighted 0.01,
below 0.99, so the baseline hugs the valleys), `n_iter=8`. ASLS is the standard
fluorescence/background remover for biological Raman: it removes the slowly varying
continuum without carving into sharp bands. In the demo, the sloping 785 nm background
(≈54 k→17 k counts) is removed while every peak is preserved (stage 2).

**3 · Smooth — Savitzky-Golay (window=9 points ≈ 18 cm⁻¹, cubic).**
A polynomial-fit smoother that removes shot noise while preserving peak height and width
(unlike a moving average). Window 9 at 2 cm⁻¹ spacing is well below the width of a real
Raman band (~10–20 cm⁻¹), so band integrity is maintained.

**4 · Resample — fixed 2 cm⁻¹ grid, 676 bins, linear interpolation.**
This is what makes spectra *comparable*: every spectrum, regardless of native
instrument sampling or excitation, is expressed on the identical axis. Outside a
spectrum's measured range the grid is NaN (never extrapolated).

**5 · Normalize — L2 (unit vector).**
Removes absolute-intensity / concentration scale so the representation encodes spectral
*shape*, not how much sample was on the slide. L2 (not area, not SNV) was retained
because it is the natural norm for the cosine-similarity and NNLS geometry the atlas
uses downstream. Normalization is computed on the finite portion only.

**6 · Non-negativity clip (NMF only).**
The NMF fit and the frozen projection clip `v` at 0. After ASLS+L2 a spectrum has small
negative lobes (baseline-subtraction undershoot). Audited magnitude across the whole
corpus:

| quantity | value |
|---|---:|
| fraction of grid points negative before clip | **11.25 %** |
| fraction of absolute signal **mass** clipped to zero | **0.75 %** |

So although ~1 in 9 points is slightly negative, the clip removes **<1 % of the total
signal mass** — it is a numerical hygiene step, not a lossy transform. This matters
scientifically: the parts-based (non-negative) interpretation of the coordinates is
bought at negligible cost to fidelity.

---

## 3. Was this pipeline *chosen*, or assumed?

Chosen, and stress-tested twice:

- **Phase 1 (comparability):** `P2_asls_savgol_l2` is one of six named candidate
  pipelines (`PIPELINES` in `pipeline.py`); it was selected as the canonical
  comparability pipeline for reference Raman.
- **Preprocessing autoresearch (120-pipeline search):** a later, more aggressive search
  tried to *beat* the canonical pipeline on cross-modal Raman↔SERS retrieval. Outcome
  **"P4 — apparent improvement is caused by overprocessing"**
  (`results/v5_rebuild/preprocessing_autoresearch/tables/final_decision.json`):
  > *"Cross-modal retrieval rises only for pipelines that strip the broad shared
  > component, which collapses Ag-SERS replicate agreement (0.62 vs 0.95) without any
  > gain in matched-vs-mismatched peak specificity — appearance, not shared chemistry."*

  **No pipeline was frozen from the search** (`selected_pipeline.json: selected=null`).
  The conservative, integrity-preserving P2 stands precisely because the search proved
  that anything more aggressive damages real spectral structure. This is the correct
  scientific outcome: the preprocessing does the *minimum* required for comparability and
  no more.

---

## 4. Determinism & identical-treatment guarantee

- Every primitive (ASLS solve, Savitzky-Golay, interp, L2) is deterministic — no random
  seed, no data-dependent parameter fitting beyond the fixed ASLS/SG constants.
- The SAME `preprocess()` with the SAME `(PREPROC, GRID, WINDOW)` is used for (a) the
  reference corpus, (b) held-out reference analytes, (c) external serum Raman, and (d)
  every projected SERS/serum/biological query in the validation and demo layers. There
  is no train/test preprocessing skew.
- The audit re-ran the full pipeline and confirmed the corpus vectors reproduce (Part 4:
  the resulting NMF components are byte-identical to the frozen atlas).

**Verdict.** The preprocessing is a standard, conservative, fully deterministic Raman
pipeline (crop → ASLS → Savitzky-Golay → 2 cm⁻¹ resample → L2 → non-negative clip). It
is applied identically to every spectrum, it discards <1 % of signal mass to buy
non-negativity, and an explicit search confirmed that nothing more aggressive can be
justified without harming spectral integrity. No changes recommended.
