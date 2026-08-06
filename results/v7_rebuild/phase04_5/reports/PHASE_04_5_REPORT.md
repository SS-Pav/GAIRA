# GAIRA V7 — Phase 04.5: Hierarchical NMF over Frozen CSM Activations

**Frozen inputs verified before any computation:** atlas `09ed804a…`, LSM registry
`208482d6…`, CSM dictionary `0b4aa550…`, theme registry `f54d4835…`. Nothing was refitted.

**Verdict: HIERARCHICAL ABSTRACTION DOES NOT IMPROVE OVER CSM. Recommendation: discard.**
11 of 11 gates pass; the negative result is the result.

---

## 1. Executive summary

Meta Components were fitted by NMF on `A ∈ ℝ₊^{375×49}` — spectra × frozen CSM activations, not
spectra, not a similarity matrix, not a graph. A Meta Component is therefore a pattern of motif
*usage*. Eight values of `K` and two variants (plain, geometry-regularised) were swept and
selected on a pre-registered Pareto frontier in which reconstruction carries only 0.14 of the
weight, as the brief requires.

**Selected: plain NMF, K = 3.** Geometry regularisation changed nothing material — same `K`,
identical explained variance to three decimals, marginally lower Pareto score.

**The result against the CSM layer, on identical frozen splits:**

| representation | dim | molecule top-1 (A) | class top-1 (B) | macro F1 | replicate consistency | info retained vs CSM |
|---|---:|---:|---:|---:|---:|---:|
| raw spectrum | 676 | 0.790 | 0.608 | 0.522 | 0.904 | — |
| LSM | 50 | **0.806** | 0.851 | 0.801 | 0.891 | 0.992 |
| **CSM** | 49 | 0.799 | **0.856** | **0.810** | 0.893 | 1.000 |
| Meta Components | 3 | 0.353 | 0.392 | 0.196 | **0.980** | **0.185** |

Meta Components retain **18.5%** of the CSM layer's information and **45.8%** of its class
retrieval. Macro F1 falls from 0.810 to 0.196.

**They do win on every stability metric — and that is exactly the problem.** Replicate
consistency 0.980 vs 0.893; activation-stability robustness AUC 0.975 vs 0.928; class-retrieval
robustness AUC 0.944 vs 0.936. These are the metrics a representation that says nearly the same
thing about everything maximises for free. This is the same trap the Phase 03 softmax theme
mode fell into, where a constant flat vector scored best on reproducibility while carrying no
information.

**An informativeness floor therefore gates the stability gains**: a layer can only *augment*
the CSM layer if a downstream user computing both would gain something, which requires it to
retain usable information. At 0.185 information and 0.458 class retrieval it fails a 0.50 floor
on both counts, and the stability gains do not count.

**No `K` in the sweep rescues it.** The downstream diagnostic — computed but deliberately not
used for selection, since selecting `K` on the metric the layer is judged by would be circular
— shows the best achievable class retrieval over all 16 (variant, `K`) combinations is **0.677
at K = 12**, still well below the CSM layer's 0.856.

---

## 2. Method

**What was factorised.** `A = 375 × 49`, the frozen CSM activations from the Phase 04 engine.
9.3 CSMs are active per spectrum on average.

**The two variants.**
- *plain*: `A ≈ WH`, `W ∈ ℝ₊^{375×K}`, `H ∈ ℝ₊^{K×49}`.
- *geometry-regularised*: `+ λ·tr(H L Hᵀ)` with `L` the normalised Laplacian of the frozen
  Phase 02.5 CSM k-NN graph (158 edges). The penalty is **one-sided by construction**:
  `tr(H L Hᵀ) = ½ Σ w_ij ‖h_·i − h_·j‖²` rewards nearby CSMs for loading similarly and contains
  no term that grows when distant CSMs do. It cannot push anything apart or manufacture a
  cluster. `L` is the **combinatorial** Laplacian `D − W`, not the symmetric-normalised one:
  the normalised form's null space is `D^½1` rather than `1`, so it would penalise a uniform
  loading in proportion to degree — a hidden preference the design never asked for. Its own
  unit test caught that during development.

**Model selection.** Eleven metrics computed at every `(variant, K)`; eight enter the Pareto
composite with pre-registered weights — bootstrap stability 0.22, consensus stability 0.18,
explained variance 0.14, interpretability 0.14, component sparsity 0.12, redundancy 0.10
(minimised), mutual coherence 0.06 (minimised), activation entropy 0.04 (minimised).

**Inference.** Frozen `H`; a new spectrum's Meta vector is the NNLS projection of its 49 CSM
activations onto it. No fitting.

## 3. Interpretability evidence

Evidence first, no manual naming. Full records in `tables/meta_component_evidence_v1.csv`.

| | spectra dominant | top CSMs | bridge CSMs | dominant classes |
|---|---:|---:|---:|---|
| MC-01 | 69 | 6 | **5 of 6** | peptide_protein (6) |
| MC-02 | 73 | 6 | 0 | acylglycerol (3), fatty_acid (3), phospholipid (…) |
| MC-03 | 233 | 6 | 0 | free_amino_acid (5), mono_oligosaccharide (1) |

Two observations that matter more than any name.

**MC-03 dominates 233 of 375 spectra** — 62% of the corpus runs the same programme. That is not
a biochemical programme; it is a background.

**MC-01 loads almost entirely on bridge CSMs** (5 of its 6 top CSMs are Phase 02.5 bridges).
The one component with a clean single-class signature is built from the CSMs the geometry
already flagged as ambiguous. Read generously, it is the "shared skeletal chemistry" programme;
read sceptically, the factorisation has isolated the part of the dictionary that was least
well-defined to begin with.

MC-02 is the only component with a chemically clean and non-bridge reading: acylglycerol,
fatty acid and phospholipid CSMs co-activating — the lipid programme, which every previous
phase has also found. It is corroboration, not new information.

## 4. Validation — twelve axes, four representations, identical splits

Splits are the frozen Phase 04 ones: A = leave-one-spectrum-out over 309 replicated spectra;
B = molecule-grouped folds, where molecule top-k is undefined and only class metrics are
reported. Full table in `tables/representation_comparison_v1.csv`.

Beyond §1: cross-fold reproducibility, activation sparsity, effective rank, participation
ratio, biochemical coherence, redundancy and calibration are all reported. **Calibration is
where the collapse is starkest: ECE 0.647 for Meta Components against 0.169 for CSM** — the
retrieval score means almost nothing at the Meta level.

## 5. The noise robustness study

Twelve physically-motivated perturbations — Gaussian and shot noise, baseline drift,
polynomial fluorescence, cosmic spikes, global and per-band intensity scaling, wavelength
shift, spectral stretch, peak dropout, band broadening, and all of them combined — each swept
over five levels, each applied to all 375 spectra, each re-projected through all four
representations. 240 measured conditions.

Queries are perturbed; the reference set stays clean, which is the realistic setting.

**Mean area under the robustness curve, as a fraction of each representation's own clean
performance:**

| representation | molecule (A) | class (B) | activation stability | worst-case retained |
|---|---:|---:|---:|---:|
| raw | 0.925 | 0.868 | 0.905 | 0.814 |
| LSM | 0.891 | 0.924 | 0.920 | 0.762 |
| CSM | 0.902 | 0.936 | 0.928 | 0.788 |
| Meta | **0.489** | **0.944** | **0.975** | **0.307** |

**The hypothesis was that Meta Components would trade a little accuracy for a lot of
robustness. They trade most of the accuracy for almost none.** The class-retrieval robustness
gain over CSM is +0.008 AURC; the clean class-retrieval cost is −0.464 top-1. On molecule
identity they are not more robust at all — they retain 31% of clean performance at the worst
perturbation level against the CSM layer's 79%.

## 6. Why it fails

**The CSM activation matrix is already sparse and already low-rank in the way that matters.**
9.3 of 49 CSMs active per spectrum, and a rank-3 factorisation captures 18.5% of it. The
information distinguishing molecules lives in *which* few CSMs activate, not in a shared
co-activation pattern across many. NMF at low rank averages exactly that away.

**The stability metrics reward the failure.** Every criterion in the Pareto composite except
explained variance and interpretability is maximised by a smoother, lower-information
representation. With stability weighted 0.40 and reconstruction 0.14 — the weighting the brief
asked for — the frontier walks toward `K = 3`, and `K = 3` is where the information is gone.
That is not an argument for reweighting: at the best `K` the layer still loses to CSM by 0.18
top-1.

**The abstraction has nothing left to abstract.** The CSM layer is already an abstraction of
the LSM layer, which is already an abstraction of the spectrum. Phase 04 measured this: class
generalisation improves 0.608 → 0.855 from raw to CSM, then falls to 0.405 at the theme layer.
Phase 04.5 adds a fourth abstraction to a stack that stopped paying at the second.

## 7. Limitations

1. **One corpus**, 375 spectra, 154 molecules — as with every previous phase.
2. **NMF only.** Sparse PCA, ICA and archetypal analysis over the activation matrix were not
   tried; a different factorisation family might behave differently, though the informativeness
   ceiling implied by 9.3 active CSMs per spectrum applies to all of them.
3. **λ = 0.1 was not swept.** Geometry regularisation changed nothing at that value; a much
   larger λ was not tested, though a stronger prior would reduce information further.
4. **The perturbations are synthetic.** They are physically motivated but not measured — real
   instrument-to-instrument variation may differ.
5. **The informativeness floor of 0.50 is a judgement.** It is stated in advance of the verdict
   logic rather than derived; at any floor above 0.46 the conclusion is unchanged, and at a
   floor below 0.185 the verdict would flip to "augment".

## 8. Recommendation

**Discard.** The CSM layer remains the canonical inference representation. Meta Components
should not be added to the inference path, should not enter the BSV, and should not be carried
into Phase 05.

The one thing worth keeping is the **finding**: GAIRA's abstraction stack pays until the CSM
layer and not after it. Phase 03 showed themes cost identity; Phase 04.5 shows a second-order
factorisation of the same activations costs more and buys less. Two independent attempts at a
higher abstraction have now failed the same way, which is a stronger statement about the
architecture than either alone.

## 9. Decision gate

See the gate returned with this report.
