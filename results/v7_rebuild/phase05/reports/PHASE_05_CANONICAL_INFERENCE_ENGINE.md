# GAIRA V7 — Phase 05: The Canonical CSM Inference Engine

**Status** GATE_FAILED — 15 of 16 gates pass; G6 (calibration ECE ≤ 0.10) fails, and §6 argues
the failure is the finding rather than a defect to be patched.
**Scope** Raman only. No SERS, no cross-modality experiment.
**Engine fingerprint** `cf82bc8d5d0519ab5221ee9104a7223b`
**Frozen inputs verified** LSM `208482d6f7178b5b8f16cace91be55b0` · CSM `0b4aa550ccefed3edabdbde5bae11c8d`

---

## 0. What this phase changed, and why

Phase 05 replaces the inference path built in Phase 04. Two things moved.

**The canonical representation is now the 49-dimensional CSM activation vector.** Phase 04
measured the cost of going higher: chemistry-class generalisation to molecules the atlas had
never seen ran 0.608 (raw spectrum) → 0.855 (CSM) → 0.405 (theme). Phase 04.5 then tested
whether a second-order factorisation of the CSM activations recovered anything and found it did
not — Meta Components retained 0.185 of CSM information. Two independent attempts at a higher
abstraction have now failed the same way. The stack pays until the CSM layer and not after it,
so the CSM layer is where inference happens.

**The interpretable layer is declared, not discovered.** The Phase 03 themes were found by NMF
over CSM co-activation and named afterwards; they lost half the class signal and still needed a
human to name them. The Biochemical Evidence Profile here is eleven axes defined *a priori* from
Raman band assignments, with a fixed sparse map from each frozen CSM onto them. Nothing is
fitted. The profile sits *beside* CSM inference rather than replacing it, so interpretation costs
nothing that retrieval needs.

Everything from Phases 00–04 is read and never written. The run aborts on a fingerprint
mismatch before any computation.

### The pipeline

```
unknown Raman spectrum
  → canonical preprocessing (450–1800 cm⁻¹, 2.0 step, 676 bins, asLS → SG → L2)
  → non-negative projection onto 49 frozen CSMs   [NNLS, nothing fitted]
  → 49-d activation vector
  → ┬ 1. analyte retrieval        154 reference vectors, cosine, calibrated
    ├ 2. chemistry-class inference
    ├ 3. Biochemical Evidence Profile  11 declared axes
    ├ 4. provenance                axis → CSM → LSM → molecule → spectra
    └ 5. uncertainty               residual · margin · entropy · open-set rejection
```

Geometry is read for visualisation and appears nowhere in the inference path (gate G14).

---

## 1. Direct CSM projection (Step 1)

NNLS onto the frozen dictionary — no regularisation parameter, therefore no quantity that could
be tuned per spectrum, and no negative component, which a physical mixture cannot have.

| diagnostic | mean | min |
|---|---:|---:|
| explained variance | 0.821 | 0.206 |
| active CSMs (of 49) | 9.6 | — |
| Hoyer sparsity | 0.876 | — |
| activation entropy (normalised) | 0.338 | — |

A mean EV of 0.821 with 9.6 of 49 motifs active is the shape a mixture model should have. The
minimum of 0.206 is not noise to be smoothed away: those are the spectra the atlas genuinely
cannot explain, and §4 shows the residual channel is what lets the engine say so.

## 2. Reference retrieval and metric selection (Step 2)

One reference vector per canonical molecule (the mean activation over its spectra). Seven
similarity metrics benchmarked under **molecule-grouped** CV, so a metric is never credited for
retrieving a molecule whose reference vector contains the query.

| metric | grouped-CV class top-1 | top-3 |
|---|---:|---:|
| **cosine** | **0.844** | 0.970 |
| angular | 0.844 | 0.970 |
| pearson / centered cosine / correlation | 0.841 | 0.962 |
| mahalanobis | 0.816 | 0.927 |
| spearman | 0.421 | 0.655 |

Selection is **nested**: the metric applied to each outer fold is chosen on inner folds of that
fold's training set, so no outer test spectrum influences the choice. Per-fold picks were cosine,
pearson, cosine, cosine, mahalanobis → **cosine** by majority.

Spearman's collapse to 0.421 is informative. Rank correlation discards activation *magnitude*,
and in a sparse non-negative code most of the vector is zero — ranking ties on 40 of 49
coordinates. Magnitude is the signal here, not an artefact of scaling.

## 3. Two splits, and one number that does not exist (Step 10)

**Split A** — leave one spectrum out; the molecule's other replicates stay in the bank.
**Split B** — molecule-grouped; the molecule is absent from the bank entirely.

| | Split A | Split B |
|---|---:|---:|
| molecule top-1 | 0.605 | *undefined* |
| molecule top-3 | 0.763 | *undefined* |
| molecule top-5 | 0.795 | *undefined* |
| class top-1 | 0.941 | 0.845 |
| class top-3 | — | 0.971 |
| macro F1 | — | 0.807 |
| balanced accuracy | — | 0.797 |

Molecule top-k under Split B is **undefined, not zero**. The correct answer is not among the
candidates, so a score of 0.000 would describe the experimental design rather than the engine.
Phase 04 reported 0.000 at every abstraction level before this distinction was drawn.

Class inference on molecules the atlas has never seen is the phase's strongest result: 0.845
top-1 across 16 chemistry classes, 0.971 within three. The confusion matrix (Figure 3) puts the
failures where a spectroscopist would expect them — `small_nitrogenous` (F1 0.400, n=7) and
`phospholipid_sphingolipid` (F1 0.556, n=8) are small classes whose members are spectrally
dominated by chains and rings they share with larger classes. `sterol_steroid` and
`nucleic_acid_polymer` are perfect.

## 4. Open-set rejection (Step 4) — and a channel that runs backwards

Negatives are synthetic: extreme corruption (60% Gaussian noise, 40 cm⁻¹ broadening, 95% peak
dropout, 10% stretch) plus structured non-Raman signals (white noise, fluorescence-only
polynomials). No SERS spectrum appears anywhere in this phase.

| channel | AUROC |
|---|---:|
| nearest reference distance | 0.953 |
| **joint (all channels, z-scored against in-domain)** | **0.921** |
| activation sparsity | 0.880 |
| activation entropy | 0.876 |
| residual fraction / explained variance | 0.714 |
| top1–top2 margin | 0.547 |
| centroid distance | 0.297 |
| OOD Mahalanobis | 0.203 |

At an operating point that accepts 95% of in-domain Raman, the joint score rejects **79.9%** of
synthetic negatives. By negative kind: white noise 0.996, fluorescence-only 0.992, extreme
broadening 0.974, extreme stretch 0.881, extreme dropout 0.688. Peak dropout is the hard case,
and reasonably so — a spectrum missing most of its peaks still *is* a Raman spectrum.

**Two channels score below chance, and the reason is structural.** Both measure distance from the
centre of activation space, and in a sparse non-negative code the centre is not where the real
data live. A pure compound activates a few motifs strongly and sits far out; a degraded or
structureless spectrum spreads weak activation across many motifs and lands near the population
mean. Density-based OOD detection is therefore *actively misleading* in this representation —
not merely uninformative. The signs were declared before the run and are reported unflipped;
flipping them post hoc would convert a finding into a fitted parameter. The channels that work
all measure *evidence quality* (how well the atlas explains the spectrum, how concentrated the
explanation is) rather than position.

## 5. Chemistry-class inference (Step 5)

Covered in §3. Grouped CV throughout; the full confusion matrix, per-class precision/recall and
F1 are in `class_confusion_matrix_v1.csv` and `class_precision_recall_v1.csv`.

## 6. Calibration (Step 3) — the gate that fails, and why it should

Five methods, fitted on training folds and scored on held-out folds only:

| method | ECE | Brier | sharpness | discrimination |
|---|---:|---:|---:|---:|
| **dirichlet (selected)** | 0.130 | **0.145** | 0.275 | **0.891** |
| temperature | 0.159 | 0.195 | 0.250 | 0.769 |
| isotonic | 0.160 | 0.228 | 0.188 | 0.681 |
| platt | **0.080** | 0.242 | **0.000** | 0.707 |
| uncalibrated | 0.590 | 0.586 | 0.001 | 0.707 |

The first pass of this phase selected on ECE and chose Platt scaling. Inspecting the resulting
reports showed why that was wrong: **every spectrum received a confidence of 0.605** — exactly
the Split A top-1 accuracy. A constant predictor at the base rate is perfectly calibrated by
construction and carries no information whatsoever, and ECE cannot see the difference. Its
sharpness is 0.000 and its Brier is the worst in the table.

This is the same failure mode caught in Phase 03 (softmax theme mode), Phase 04 (theme-mode
evidence leakage) and Phase 04.5 (Meta Component stability). Four times now, a reproducibility-
or calibration-flavoured metric has been maximised by a degenerate low-information output. The
fix each time has been an explicit informativeness constraint, and the pattern is worth stating
as a design rule rather than rediscovering it a fifth time.

Selection was changed to **Brier** — a strictly proper scoring rule, so it decomposes into
calibration *and* refinement and cannot be won by flattening — restricted to methods that
discriminate (AUROC of confidence vs correctness > 0.55) and are sharp (σ > 0.02). Temperature
scaling's internal objective was changed from ECE to Brier for the same reason. Dirichlet
calibration wins on the corrected rule.

**G6 was left exactly as pre-declared, and it fails.** The selected calibrator's ECE is 0.130,
not ≤ 0.10. Relaxing the threshold after seeing the number would be moving the goalpost; the
honest statement is that *on this corpus, ECE ≤ 0.10 is reachable only by a calibrator that
reports the same confidence for every spectrum*. A companion gate G6b — the confidence must
actually discriminate — was added and passes at 0.891, and the two together describe the
engine's calibration more truthfully than either alone.

The practical consequence is visible in Figure 7b: thresholding on confidence trades coverage
for accuracy smoothly and monotonically, which is what a usable confidence has to do.

## 7. The Biochemical Evidence Profile (Steps 6–7)

Eleven axes, each defined by Raman band windows with weights. A CSM loads on an axis only when
**both** conditions hold: one of its diagnostic bands (from the frozen Phase 02 registry) falls
in an axis window, *and* the prominence it carries there is at least 10% of its total diagnostic
band strength. The result is sparse — mean 3.8 axes per CSM — and every loading is traceable to
a named band.

Windows overlap wherever the chemistry overlaps, and the rows therefore do not sum to one. The
largest overlaps are heterocyclic/purine (Jaccard 0.307), unsaturation/amide (0.223) and
aliphatic/carbohydrate (0.210). This is the honest state of Raman band assignment — 1650 cm⁻¹ is
amide I *and* cis C=C, and no weighting scheme resolves that from one band — so it is reported in
`evidence_axis_window_overlap_v1.csv` rather than hidden by forcing a partition.

### Are the axes real?

A falsifiability test: AUROC of each axis's magnitude separating the chemistry classes it claims
to be about from all others. **Chemistry labels are used only here, as evaluation.** Nothing in
the axis definitions or the loading matrix ever saw a label.

| axis | supporting CSMs | specificity | AUROC | verdict |
|---|---:|---:|---:|---|
| chromophore / conjugated | 5 | 3.28 | 0.894 | grounded |
| purine | 9 | 2.69 | 0.934 | grounded |
| aliphatic chain | 33 | 1.40 | 0.909 | grounded |
| heterocyclic ring | 19 | 1.95 | 0.848 | grounded |
| aromatic residue | 12 | 2.41 | 0.784 | grounded |
| carbohydrate skeletal | 45 | 1.09 | 0.748 | grounded |
| phosphate / nucleic | 7 | 2.95 | 0.735 | grounded |
| sulfur / thiol | 20 | 1.90 | 0.663 | weak |
| amide / protein | 23 | 1.76 | 0.663 | weak |
| carbonyl / ester | 3 | 3.79 | 0.626 | weak |
| **unsaturation** | 11 | 2.49 | **0.534** | **not discriminative** |

Seven of eleven axes are grounded; four are not, and they are reported as failures rather than
softened.

### Does the threshold make the axes?

`SUPPORT_FLOOR` (the share of a CSM's diagnostic band strength an axis must claim before that CSM
counts as supporting it) and the prominence window were both chosen once. Sweeping them:

| parameter | value | supporting CSMs per axis | axes per CSM | grounded | mean AUROC |
|---|---:|---:|---:|---:|---:|
| support floor | 0.05 | 25.7 | 5.78 | 7 | 0.758 |
| support floor | **0.10** | **17.0** | **3.82** | **7** | **0.758** |
| support floor | 0.15 | 12.1 | 2.71 | 7 | 0.758 |
| support floor | 0.20 | 9.1 | 2.04 | 7 | 0.758 |
| prominence window | 20 cm⁻¹ | 16.8 | 3.78 | 6 | 0.720 |
| prominence window | **40 cm⁻¹** | **17.0** | **3.82** | **7** | **0.758** |
| prominence window | 80 cm⁻¹ | 17.3 | 3.88 | 8 | 0.774 |

`SUPPORT_FLOOR` has **no effect at all** on the grounding verdicts, and that is structural rather
than lucky: the floor governs which CSMs are *counted* as supporting an axis — the specificity
weight, the confidence term, the provenance chains — while the axis magnitudes that the AUROC test
consumes are the raw loadings. Sparsity and grounding are independent knobs, and only one of them
was tuned.

The prominence window does move the result, from 6 to 8 grounded axes across a 4× range. The
chosen 40 cm⁻¹ is the middle value and was fixed before the sweep; a wider window would have
flattered the phase by one axis. Gate G8b requires ≥ 6 grounded axes at *every* setting tested,
which holds.

**The unsaturation result is a label failure, not an axis failure.** The primary test scores the
axis against `fatty_acid` + `acylglycerol`, but those classes contain saturated members, so a
perfect unsaturation axis would still score near chance. The sharp test — unsaturated vs
saturated *within* the lipids, using the C=C-bearing molecule names — was declared alongside the
primary one and gives **AUROC 1.000** (25 vs 25 spectra). The axis separates the two groups
perfectly. This refines the evaluation label, never the axis: the windows and the loading matrix
were fixed before the secondary test ran.

`amide_protein` at 0.663 is a genuine weakness and traces to the same overlap: proteins,
triglycerides and unsaturated lipids all put intensity near 1650 and 1265 cm⁻¹. A triglyceride's
profile therefore shows a real amide spoke (visible in Figure 12, top row). This is a limitation
of one-band-window reasoning, and resolving it needs band *shape* — width and asymmetry — which
this phase does not model.

`carbonyl_ester` rests on 3 CSMs. That is honest sparsity, not a bug: the corpus's 450–1800 cm⁻¹
window barely reaches the C=O stretch, which sits at 1720–1780 at the very edge.

### The radar (Step 7)

Every spoke carries four quantities: magnitude, confidence, coverage and the number of supporting
CSMs. Confidence is the product of coverage, normalised specificity, a support term saturating at
three CSMs, and the spectrum's reconstruction quality — all in [0, 1] and multiplied, so it is
deliberately pessimistic. Spoke thickness and marker size encode it, so an axis inferred from one
motif in a poorly reconstructed spectrum *looks* thin (Figure 9).

## 8. Provenance (Step 8)

3,133 active-axis chains were built and verified link by link against the frozen registries.
**Zero broken chains.** Every LSM id resolves in the Phase 01 registry, every molecule name in
the Phase 00 canonical list, and every chain terminates in measured reference spectra. The
contributions listed in a waterfall are the actual additive terms `a_m · M_ma`, so they sum
exactly to the axis value being explained (Figure 10).

## 9. Geometry (Step 9)

Read for visualisation only. Gate G14 asserts, and the code enforces, that no inference output
depends on it.

## 10. Noise robustness (Step 11)

Seven perturbations × 5 severity levels × 4 representations. Perturbed queries are scored against
**clean, molecule-grouped** reference banks, so the class numbers below are held-out throughout.

| representation | clean class top-1 (unseen molecule) | mean perturbed | retention |
|---|---:|---:|---:|
| raw spectrum | 0.592 | 0.530 | 0.895 |
| LSM | 0.848 | 0.782 | 0.923 |
| **CSM** | 0.845 | **0.790** | **0.935** |
| evidence profile | 0.664 | 0.612 | 0.921 |

**The hypothesis holds, on both halves.** CSM projection is more robust than the raw spectrum
(retention 0.935 vs 0.895) *and* far more discriminative on molecules the atlas has not seen
(0.845 vs 0.592). There is no trade here — the abstraction is not buying robustness with
accuracy.

An earlier version of gate G11 compared *in-sample* clean accuracy, where raw spectra win 0.992
to 0.973 by matching themselves, and the gate failed. That comparison is risk R-10 and says
nothing about discrimination; it was replaced with the molecule-grouped one, which is the
comparison the hypothesis is actually about. Both numbers are reported in
`robustness_summary_v1.csv`.

Two secondary observations. **Intensity scaling is exactly invariant** for every representation —
a useful sanity check that L2 normalisation is doing what it claims. **Fluorescence at severity
0.8 destroys everything** (CSM 0.295, raw 0.160): a background four-fifths the height of the
signal is not a solvable problem at the representation layer, and the honest response is
rejection, which §4 shows the engine performs at AUROC 0.992 on fluorescence-only negatives.

The evidence profile retains 0.921 — nearly as robust as the CSM layer it is computed from — but
carries 0.664 clean class accuracy against the CSM's 0.845. Eleven interpretable axes cost about
0.18 of class top-1 relative to 49 motifs. That is the price of interpretability, stated plainly,
and it is roughly half what the Phase 03 theme layer cost (0.405).

## 11. Comparison against the previous Theme/BSV pipeline

| | Phase 03/04 themes | Phase 05 evidence profile |
|---|---|---|
| origin | NMF over CSM co-activation, named afterwards | declared from Raman band assignments |
| dimensionality | 4 accepted themes | 11 axes |
| class top-1, unseen molecule | 0.405 | 0.664 |
| relation to CSM inference | replaced it | sits beside it — CSM retrieval is unaffected |
| naming | required human interpretation; every theme initially came out "aliphatic chain + …" | named by construction |
| grounding test | none | per-axis AUROC vs declared chemistry; 7 of 11 pass |
| provenance | theme → CSM → LSM | axis → CSM → LSM → molecule → spectra, verified |

The declared layer is better on every axis that matters, and the reason is structural rather than
clever: a discovered layer must simultaneously find structure, be namable, and preserve
information, and on 375 spectra it cannot do all three.

## 12. Limitations

1. **Molecule identification is not solved.** Split A top-1 is 0.605 and top-5 is 0.795. Five
   candidates get you to four in five; one candidate does not. The engine should be read as a
   *chemistry-class* instrument with a molecule shortlist, not a molecule identifier.
2. **Calibration is imperfect** (ECE 0.130) and the phase does not hide it. The informative
   calibrators available within the four declared families all sit near 0.13.
3. **Open-set negatives are synthetic.** They are corruption and structureless signal, not
   genuinely novel chemistry. Whether the engine rejects a real molecule absent from the atlas is
   *untested*, and the corpus contains no such holdout by construction.
4. **Four axes are not grounded** at AUROC ≥ 0.70 on their primary test. One (unsaturation) is a
   label artefact with a perfect secondary test; three (amide, sulfur, carbonyl) are real
   weaknesses, the first two driven by band overlap and the third by the corpus window barely
   reaching the C=O stretch.
5. **Band-window reasoning ignores band shape.** Width, asymmetry and splitting carry assignment
   information this phase discards, and that is the principal cause of the amide/unsaturation
   confusion.
6. **154 molecules is a small reference bank** for a 154-way retrieval problem, and 375 spectra
   is a small corpus for calibration with 10 confidence bins (≈37 spectra per bin).
7. **Raman only.** Nothing here licenses any claim about SERS. Phase 04's attempt found the atlas
   could not detect real Ag-SERS as out-of-domain (AUROC 0.548); that experiment is out of scope
   now, and its negative result stands unrefuted rather than resolved.

## 13. Future directions

- **Band-shape features** to break the 1650 cm⁻¹ degeneracy — the single highest-value addition.
- **A genuine open-set holdout**: withhold a chemistry class entirely from the atlas and ask
  whether the engine rejects it. This is the experiment §12.3 says is missing.
- **A richer calibration feature set** (margin, entropy and residual jointly, not just the top
  softmax value), which is the obvious route to ECE < 0.10 without losing sharpness.
- **Per-class conformal prediction** to replace top-k with a set that carries a coverage
  guarantee.
- Expanding the reference bank, which raises the ceiling on §12.6 more directly than any
  modelling change.

## 14. Gates

| gate | status |
|---|---|
| G1 frozen fingerprints verified | PASS |
| G2 nothing upstream refitted | PASS |
| G3 mean explained variance ≥ 0.80 | PASS (0.821) |
| G4 Split A molecule top-1 ≥ 0.60 | PASS (0.605) |
| G5 Split B class top-1 ≥ 0.60 | PASS (0.845) |
| G6 calibration ECE ≤ 0.10 | **FAIL (0.130)** — see §6 |
| G6b calibration is informative | PASS (discrimination 0.891, sharpness 0.275) |
| G7 open-set joint AUROC ≥ 0.80 | PASS (0.921) |
| G8 ≥ 6 evidence axes grounded | PASS (7 of 11) |
| G8b axis grounding stable across threshold choices | PASS (≥6 at every setting) |
| G9 no broken provenance chains | PASS (0 of 3,133) |
| G10 CSM more robust than raw | PASS (0.935 vs 0.895) |
| G11 CSM preserves discrimination on unseen molecules | PASS (0.845 vs 0.592) |
| G12 engine deterministic on repeat | PASS |
| G13 no cross-modality experiment | PASS |
| G14 geometry not used in inference | PASS |

## 15. Reproduction

```bash
PYTHONPATH=src python results/v7_rebuild/phase05/code/run_phase05.py    # ~55 s
PYTHONPATH=src python results/v7_rebuild/phase05/code/make_figures.py
PYTHONPATH=src python results/v7_rebuild/phase05/code/make_pdf.py
PYTHONPATH=src python -m pytest tests/test_v7_phase05.py -q
```

Deterministic: `SEED = 0`, no unseeded randomness, and the engine reproduces its outputs
bit-for-bit on repeat (G12). Output root resolves through `GAIRA_V7_OUTPUT_ROOT`; no path is
hardcoded.
