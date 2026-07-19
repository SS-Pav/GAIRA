# GAIRA V5 — Serum Spike-in Projection Validation

**Validation study, not model optimisation.** The Raman Reference Atlas v0.1 (NMF k=24) was frozen throughout: its fingerprint `09ed804a40836f4a05a91ba10900cded` was verified on load and re-verified after every analysis. No retraining, no rotation, no ontology change, no preprocessing tuned to improve trajectories. Branch `gaira-v5-rebuild-plan`; nothing pushed.

> **Principal finding: concentration is registered; chemical identity mostly is not.**
> Pure-analyte dose series move monotonically and saturate in **7/7** arms (Spearman ρ 0.93–1.00 vs a permutation null of ≈0.24, **p = 0.002** in every arm), and enzymatic urate depletion moves in the chemically correct direction (cos **−0.61**). But serum spikes at physiological concentrations show **no overall directional agreement** with their own pure-analyte reference (median cos **−0.013** vs mismatched null **−0.036**) — except for strong silver adsorbers: hypoxanthine (0.89), xanthine (0.74), ergothioneine (0.53), guanine (0.41).

---

## Reproduce

```bash
cd results/v5_rebuild/spike_validation/code
python run_phase1_2.py     # dataset audit, preprocessing audit, replicate QC   (~26 s)
python run_phase3_11.py    # projection, trajectories, reproducibility, Phase 7, controls (~8 s)
python make_report.py      # figures + 10-page PDF
```

---

## A constraint that frames everything (Phase 10)

**Every controlled perturbation dataset in GAIRA is Ag- or Au-SERS**, while the atlas is built from pure **Raman** references. All projections are therefore out of domain *by construction*:

| dataset | n | median OOD |
| --- | --- | --- |
| pure Ag-SERS metabolites | 265 | 0.153 |
| isotopic (urate) | 73 | 0.231 |
| ILS adenine | 3381 | 0.242 |
| uricase depletion | 20 | 0.241 |
| spiked serum | 265 | 0.276 |
| ergothioneine calibration | 55 | 0.282 |
| unspiked serum | 15 | 0.282 |

Per the study design, rising OOD is not itself failure. The question asked throughout is whether motion remains *internally coherent* despite being off-domain.

---

## Phase 1–2 — datasets and preprocessing

**Datasets:** ILS adenine (3381 spectra, 15 laboratories, 4 substrates, 2 lasers, 14 concentrations, 3 batches, 225 blanks), 51-analyte serum spike panel (265 spectra, 5 replicates each), unspiked serum (15), pure Ag-SERS references (265), uricase depletion (20), isotopic urate (73), ergothioneine calibration (55).

**Preprocessing contract.** A projection into a frozen NMF basis is only meaningful in the representation the basis was fitted in, so the atlas-native pipeline (ASLS → Savitzky–Golay → L2 on 450–1800 cm⁻¹ @ 2 cm⁻¹) is **mandatory and was not tuned**. Only the pre-steps were evaluated.

**A preprocessing error caught and corrected.** My first cosmic-ray filter used a large-residual rule and flagged **13% of all points** on the 3 cm⁻¹ ILS axis — 73 "cosmic rays" per 534-point spectrum. At that sampling a real 12 cm⁻¹ band spans only ~4 points, so the rule was deleting genuine bands. It was replaced with a sharpness test that is applied **only where the axis oversamples the narrowest plausible band (≤2 cm⁻¹/point)**. On the ILS and other coarse axes despiking is now *declined and recorded as skipped*; on the 1.7 cm⁻¹ B&WTek axis it removes 0.09% of points.

**Replicate QC** (median within-condition cosine): unspiked serum 0.998, spiked serum 0.998, uricase 0.999, ergothioneine 0.998, pure SERS 0.948, isotopic 0.945, ILS adenine 0.831 (multi-laboratory, expected lower). 455 replicate spectra were flagged as robust-z outliers; all are documented in `phase2_exclusions.csv` and were **retained** — none was removed for being inconvenient.

---

## Phase 4/5/8 — dose-response (positive result)

| arm | levels | ρ | perm p | null ρ | straightness | step-cosine | best model | linear R² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ILS cAg@532 | 14 | 0.952 | 0.002 | 0.243 | 0.666 | +0.048 | saturating | 0.810 |
| ILS cAg@785 | 14 | 0.996 | 0.002 | 0.230 | 0.837 | +0.426 | saturating | 0.890 |
| ILS cAu@785 | 14 | 0.996 | 0.002 | 0.224 | 0.721 | +0.181 | saturating | 0.944 |
| ILS sAg@532 | 14 | 0.987 | 0.002 | 0.250 | 0.611 | −0.167 | saturating | 0.932 |
| ILS sAg@785 | 14 | 0.978 | 0.002 | 0.241 | 0.614 | −0.243 | saturating | 0.773 |
| ILS sAu@785 | 15 | 0.946 | 0.002 | 0.235 | 0.487 | −0.256 | saturating | 0.853 |
| ergothioneine | 11 | 0.927 | 0.002 | 0.289 | 0.581 | +0.128 | saturating | 0.903 |

- **Does the trajectory move?** Yes, in every arm, far above the label-permutation null.
- **Is it monotonic?** Yes (ρ ≥ 0.93 throughout).
- **Is it smooth?** *Partly.* Mean consecutive-step cosine ranges +0.43 → **−0.26**: several arms advance monotonically in distance while zig-zagging in direction. Monotonic magnitude is not a coherent path.
- **Does it plateau?** Yes — a saturating (Langmuir-type) model beat linear and logarithmic fits in **7/7** arms, the expected behaviour for analytes competing for finite colloid adsorption sites.
- **Reproducibility:** median replicate displacement-direction cosine 0.50–0.88 by arm (worst: sAg@532).

---

## Phase 7 — serum spike vs pure analyte (the decisive test)

| quantity | value |
| --- | --- |
| matched cos (median) | **−0.013** |
| mismatched-analyte null (median) | −0.036 |
| median angle to pure direction | 90.7° |
| analytes beating the 95th-percentile null | **7 / 53** |
| median replicate direction cosine | 0.865 |

Spike displacements are **reproducible** (0.87) but **not aligned** with the analyte's own pure-analyte direction. The exceptions are chemically specific:

| analyte | family | conc (µM) | cos | angle |
| --- | --- | --- | --- | --- |
| hypoxanthine | purine | 10 | **0.888** | 27° |
| xanthine | purine | 50 | **0.736** | 43° |
| creatinine | small N | 80 | 0.574 | 55° |
| ergothioneine | cofactor | 5 | **0.534** | 58° |
| ascorbate | organic acid | 60 | 0.460 | 63° |
| guanine | purine | 10 | **0.405** | 66° |

**3 of 5 purines** succeed — purines chemisorb to silver through ring-nitrogen lone pairs — as does ergothioneine, whose thione sulfur binds silver strongly. The failures are dominated by amino acids (15/43), saccharides and lipids, all weak Ag adsorbers. **Displacement magnitude predicts direction agreement (r = +0.54).**

---

## Phase 6 — component activation

Per-analyte activation deltas (spike − unspiked serum) across all 24 components are in `phase6_component_activation.csv`, with effect sizes relative to replicate scatter. Because Phase 7 shows most spikes do not move in an analyte-specific direction, component-level activations for non-responding analytes should **not** be read as biochemical evidence; they are reported for completeness. Themes remain interpretive overlays.

---

## Phase 9 — mixtures: not testable

Each analyte was spiked individually at a single concentration; no combinatorial A+B spike exists in this corpus, so Δ(A+B) ≈ Δ(A)+Δ(B) cannot be evaluated. The uricase experiment is a depletion, not a mixture.

---

## Phase 11 — controls (all pass)

- **Unspiked serum:** replicate cosine 0.9988 — the control condition is highly stable.
- **ILS blanks (n=225):** batch drift from the grand mean 0.009 / 0.009 / 0.024 → **no systematic drift**.
- **Uricase depletion:** displacement 0.250 with cos **−0.61** versus the urate direction. Enzymatic removal of urate should move *away* from urate, so the negative sign is the chemically expected result — a genuine positive control.
- **Isotopic specificity:** urate vs ¹³C/¹⁵N-labelled urate sit at distance 0.196, cosine 0.868 — isotopologues remain close, as they should.

---

## What the data support / do not support

**Support:** the manifold registers *concentration* coherently and reproducibly across 15 laboratories, 4 substrates and 2 lasers; the dose relationship is saturating, consistent with surface-site competition; depletion moves in the correct direction; controls show no drift.

**Do not support:** that a physiological-level spike in serum can be *identified* from its position in the manifold. Median directional agreement is indistinguishable from a mismatched-analyte null. This study provides no evidence for chemical-identity recovery from serum spikes, and none is claimed.

**Interpretation (separated from observation):** the atlas registers a perturbation when the analyte generates enough Ag-SERS signal to rise above the colloid background, and is blind to it otherwise. This is consistent with prior GAIRA findings that these Ag-SERS spectra are background-dominated, and is a property of the *measurement*, not of the Raman atlas.

**Speculation (explicitly labelled):** if surface competition rather than the manifold is limiting, spikes at higher effective surface coverage or with background-suppressed acquisition might recover directional agreement for weak adsorbers. This study cannot test that.

---

## Limitations

- Every perturbation dataset is Ag/Au-SERS projected into a Raman atlas; **nothing here validates in-domain Raman behaviour**.
- Single concentration per serum spike → no in-serum dose-response, no detection threshold, no mixture test.
- Distance-from-control can rise for reasons unrelated to the analyte (colloid aggregation, laser power, fouling). The permutation null controls concentration-*label* assignment but not a confound that varies monotonically with concentration by design.
- Trajectory smoothness is substrate-dependent and in three arms is negative; "monotonic" should not be read as "smooth".

## Recommended next experiments

1. **An in-domain Raman dose-response** (pure analyte, Raman, several concentrations). This is the single most informative missing measurement — it would separate "the atlas cannot track concentration" from "Ag-SERS is the limiting factor". No such dataset exists in GAIRA today.
2. **Serum spikes as a concentration series** rather than one level, enabling in-serum dose-response and per-analyte detection thresholds.
3. **Blank-colloid difference acquisition**, to test directly whether background suppression restores directional agreement for weak adsorbers.

---

## Outputs

`GAIRA_V5_Serum_Spike_Projection_Validation.pdf` (10 pages) · `tables/` (phase1 audit, phase2 QC + exclusions, phase3 projections ×7, phase4_8 trajectories, phase5 reproducibility, phase6 activation, phase7 serum-vs-pure + summary, phase9 mixture, phase10 OOD, phase11 controls) · `figures/` (5) · `artifacts/` (processed spectra, per-dataset metadata, study manifest with fingerprint verification).
