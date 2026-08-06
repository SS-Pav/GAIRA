# Phase 06 — Chemistry Evidence Layer

**Status:** Not started — **next approved step**
**Canonical numbering** (adopted 2026-08-06). Supersedes the phase previously specified in
`phase_06_in_domain_raman_validation/`, which is retained and marked SUPERSEDED.
**Architectural decision under test:** A-19 (`GAIRA_V7_ARCHITECTURE_STATUS_AFTER_PHASE05.md`).

---

## Purpose

Build and validate the **16-dimensional Chemistry Evidence** layer: a frozen, calibrated,
probabilistic map from the 49-dimensional CSM activation vector onto the sixteen classes of the
frozen `v7_fine_16` ontology.

This is the fourth object to occupy the architectural slot above the CSM representation. The
first three lost information (themes 0.405, Meta Components 0.392, declared axes 0.664, against
the CSM's 0.855 class top-1 on unseen molecules). Phase 06 exists to find out whether a layer
whose label space *matches the frozen evaluation ontology* behaves differently.

## Why sixteen

Not a tuned hyperparameter. Sixteen is the size of the evaluation ontology frozen in Phase 00,
which is also the label space of the frozen Tier-1 success criteria S-01 and S-03. Any other
number makes V7 unmeasurable against its own bar.

## Inputs — all frozen, none refitted

| Input | Fingerprint / source |
|---|---|
| CSM dictionary and registry | `0b4aa550ccefed3edabdbde5bae11c8d` |
| LSM registry | `208482d6f7178b5b8f16cace91be55b0` |
| balanced references, canonical IDs, class partition, CV folds | Phase 00 / 01 |
| frozen V5 control baseline | `phase00_baseline_metrics.csv`, harness `v7_harness_v1` |
| Phase 05 engine and the 11-axis profile | `20d8bd99ce71f45a125c6a2b1d719e51` — the layer to beat |

## Objectives

1. Learn the frozen map `E ∈ ℝ₊^{49×16}` offline, on training folds only, under
   molecule-grouped CV.
2. Calibrate it. Report ECE, Brier, sharpness and discrimination — **all four** (P-18).
3. Report the **unassigned mass** per spectrum. Evidence that supports no class is reported,
   never redistributed.
4. Measure, on the frozen folds: top-1, top-3, macro F1, balanced accuracy, per-class precision
   and recall, confusion matrix.
5. Measure replicate consistency, cross-fold stability and noise robustness under the same
   seven perturbations × five levels used in Phase 05.
6. Establish provenance: every Chemistry Evidence coordinate must resolve to CSMs → LSMs →
   molecules → spectra, verified chain by chain.
7. **Run the R-01 control** (see below).
8. Measure V7 against the **frozen Tier-1 criteria** under `v7_harness_v1` — the first time this
   has been done (U-06).

## The R-01 control — mandatory, not optional

The Chemistry Evidence layer predicts the same sixteen classes that partitioned the Phase-01
local decompositions. Part of its accuracy may therefore be an imprint of the partition rather
than a property of the representation (risk R-01, unknown U-02).

The control: fit a **class-agnostic** decomposition of the same balanced references at matched
total dictionary size, build the same evidence map over it, and evaluate on the same folds. The
gap between the two is the portion of the result attributable to the class prior. Report it
whatever it is.

## Deliverables

- frozen `E`, calibrator, and manifest with a fingerprint covering both
- validation tables: retrieval, calibration, per-class, robustness, stability
- **head-to-head against the archived 11-axis profile** on identical folds
- R-01 control result
- frozen-criteria measurement under `v7_harness_v1`
- radar examples, confusion matrices, per-class analyses, provenance chains
- publication-quality figures (PNG, 200 dpi) + a PDF figure report
- scientific audit at Nature-family standard
- regression tests, including adversarial tests for every defect found

## Decision Gate DG-06

| Check | Requirement |
|---|---|
| **Scientific** | Chemistry Evidence **clearly exceeds** the archived 11-axis profile (0.664 class top-1 on unseen molecules) on identical folds, with the difference significant at α = 0.05 after correction |
| **Scientific** | the layer retains ≥ 0.50 of the CSM layer's held-out class information (P-18 informativeness floor, pre-registered) |
| **Scientific** | calibration is *informative*: discrimination ≥ 0.75 and sharpness > 0.05, reported alongside ECE (P-18) |
| **Scientific** | the R-01 control is reported, whatever it shows |
| **Engineering** | deterministic, bit-for-bit; no fitting at inference; batch-independent; fingerprint verified |
| **Architecture** | non-negativity holds; unassigned mass reported; provenance chains 100% intact; no upstream artefact modified |
| **Decision** | **Proceed** to Phase 07 · **Repeat** with a corrected map · **Redesign** — reinstate A-16 (the 11-axis profile) as the interpretable layer and re-plan |

**A negative result is a valid outcome.** If Chemistry Evidence does not clearly exceed the
11-axis profile, the honest conclusion is that no layer above the CSM representation improves on
it, and the architecture should say so rather than proceed.
