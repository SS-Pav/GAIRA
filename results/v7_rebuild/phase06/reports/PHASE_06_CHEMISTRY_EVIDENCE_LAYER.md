# GAIRA V7 — Phase 06: The Validated Chemistry Evidence Layer

**Status** COMPLETE — 18 of 18 gates pass.
**Scope** Raman only. No SERS, Ag-SERS, serum, plasma, EV, mixture, perturbation-dataset or
DART-Met data is loaded, benchmarked, or cited as validation anywhere in this phase.
**Implements** decision A-19 (16-dimensional Chemistry Evidence). **Does not implement** BSV2
(Phase 07) or hierarchical molecular retrieval (Phase 08).
**Frozen inputs verified** LSM `208482d6f7178b5b8f16cace91be55b0` ·
CSM `0b4aa550ccefed3edabdbde5bae11c8d` · Phase 05 engine `20d8bd99ce71f45a125c6a2b1d719e51`

---

## 1. Executive summary

The 49-dimensional CSM activation vector can be turned into a calibrated, continuous,
interpretable 16-dimensional chemistry evidence vector **at a cost of 0.010 in class top-1**.

| | value | 95% CI (molecule bootstrap) |
|---|---:|---|
| fine-class top-1 | **0.835** | [0.775, 0.889] |
| fine-class top-3 | **0.976** | [0.947, 0.997] |
| macro-F1 | **0.793** | [0.706, 0.860] |
| macro-F1, classes with ≥5 molecules | 0.826 | — |
| balanced accuracy | **0.830** | [0.760, 0.908] |
| mean reciprocal class rank | 0.903 | [0.864, 0.936] |

Against the layer it replaces and the layers it must beat, on identical outer folds:

| representation | dim | top-1 | top-3 | macro-F1 |
|---|---:|---:|---:|---:|
| raw spectrum | 676 | 0.592 | 0.768 | 0.479 |
| legacy Theme/BSV (A-13/A-14) | 5 | 0.491 | 0.797 | 0.361 |
| legacy 11-axis profile (A-16) | 11 | 0.664 | 0.853 | 0.576 |
| **Chemistry Evidence (this phase)** | **16** | **0.835** | **0.976** | **0.793** |
| CSM activation | 49 | 0.845 | 0.971 | 0.807 |
| LSM activation | 50 | 0.848 | 0.960 | 0.815 |

**The layer clears every pre-registered gate**, including the two that matter most: it stays
within 0.02 of the CSM layer it is computed from (G7: |0.835 − 0.845| = 0.010) and it exceeds
the legacy Theme/BSV layer by 0.344 (G8 required ≥ 0.20). It also delivers the *best top-3 in
the entire stack* — 0.976, above the 49-dimensional layer — which is what a well-behaved
evidence vector should do: when it is wrong about first place it is almost never wrong about
the neighbourhood.

**Three findings are more interesting than the headline.**

1. **Errors are chemically reasonable at 4.2× chance.** 45.2% of the 62 errors fall on a
   *pre-declared* chemically adjacent pair against a chance rate of 10.9%. The declaration was
   made before any confusion matrix was inspected.
2. **Novelty detection works, except where it matters most.** Withholding an entire chemistry
   class gives a mean AUROC of 0.863 over six classes — but **acylglycerol scores 0.489, no
   better than chance**, because fatty acids remain in the atlas and a triacylglycerol's Raman
   spectrum is dominated by the same acyl-chain motifs. §12 states this plainly.
3. **The curated ontology is only 60% recoverable from the spectra alone** (ARI 0.595 against an
   out-of-fold unsupervised clustering). The 40% disagreement is where the curated layer earns
   its place — and it is also why the unsupervised comparator's apparently higher accuracy is a
   structural artefact rather than a result (§14).

---

## 2. Architecture continuity

Phase 06 consumes frozen artefacts and writes only under its own tree. Verified before any
computation:

| input | fingerprint | role |
|---|---|---|
| V5 atlas | `09ed804a…` | control, untouched |
| LSM registry | `208482d6…` | read |
| CSM dictionary | `0b4aa550…` | **the projection basis** |
| Phase 05 engine | `20d8bd99…` | reproduced, then extended |
| CV folds | split fingerprint recorded in every artifact | 5 folds grouped by `canonical_id` |

Nothing was refitted: not preprocessing, not the class-local NMF, not the LSMs, not the CSMs,
not the canonical identities, not the chemistry ontology. The legacy 11-axis band map (A-16) is
**absent from canonical Phase 06 inference** and appears only as a comparator in §13 (gate G14).

## 3. The exact chemistry-class inference method

### 3.1 What Phase 05 did — traced to source, not to prose

`src/gaira/v7/inference/engine.py:86-89` and `run_phase05.py::topk_class`:

$$e_c(x) \;=\; \max_{i \in c}\; \cos\!\left(a(x),\, r_i\right)$$

One nearest reference molecule decides the class. Top-k ranks *distinct classes* by that
per-class maximum. There is no class-size correction, no calibration and no probability — the
output is a label, not a vector.

**Reproduced bit-for-bit** before anything was changed:

| metric | Phase 05 | reproduced |
|---|---:|---:|
| class top-1 | 0.845333 | 0.845333 |
| class top-3 | 0.970667 | 0.970667 |
| macro-F1 | 0.806783 | 0.806783 |
| balanced accuracy | 0.796612 | 0.796612 |

### 3.2 What Phase 06 computes

Selected by nested CV from 37 candidates: **`D:A_max_idf:lam0.5`**

$$e_c(x) \;=\; w_c \cdot \max_{i \in c} \cos\!\left(a(x), r_i\right) \cdot b_{\beta(c)}(x)^{\lambda},
\qquad \lambda = 0.5$$

- `w_c` — inverse-frequency class-size correction, normalised so the mean weight is 1. It
  changes the *distribution* of evidence across classes, never its overall scale. Without it, an
  80-spectrum class outscores a 7-spectrum class simply by having more chances to contain a near
  neighbour.
- `b_{β(c)}` — evidence for the frozen **broad superclass** of class `c`, from the Phase 00
  `broad_class` column (six superclasses, curated before any V7 model existed).
- **Routing is soft.** The broad term multiplies and is strictly positive, so a fine class stays
  reachable when its superclass is not top-1. A hard filter would make a broad error
  unrecoverable; at six-way accuracy that would permanently lose a non-trivial share of queries.

## 4. Definition of the 16-dimensional evidence vector

$$e(x) = [e_1, \ldots, e_{16}] \in \mathbb{R}_+^{16}$$

Each coordinate means: **"support that the query occupies the reference region represented by
chemistry class c."**

It does **not** mean molar fraction, concentration, percent composition, proven presence of the
class in a mixture, or — absent the calibrator — a probability. The corpus contains only pure
Raman references; nothing in this phase licenses a mixture reading.

### The 16 classes, in the frozen `CLASS_ORDER`

| # | class | molecules | spectra | broad superclass |
|---:|---|---:|---:|---|
| 0 | acylglycerol | 17 | 23 | lipid |
| 1 | carboxylic_acid_metabolite | 8 | 23 | energy_metabolism |
| 2 | chromophore_pigment | 4 | 10 | redox_cofactor |
| 3 | fatty_acid | 17 | 27 | lipid |
| 4 | free_amino_acid | 18 | 75 | protein_amino_acid |
| 5 | mono_oligosaccharide | 20 | 43 | carbohydrate |
| 6 | nucleic_acid_polymer | 3 | 3 | nucleic |
| 7 | peptide_protein | 30 | 80 | protein_amino_acid |
| 8 | phosphate_metabolite | 3 | 11 | energy_metabolism |
| 9 | phospholipid_sphingolipid | 5 | 8 | lipid |
| 10 | polysaccharide | 5 | 10 | carbohydrate |
| 11 | purine | 5 | 17 | nucleic |
| 12 | pyrimidine | 3 | 9 | nucleic |
| 13 | small_nitrogenous | 2 | 7 | energy_metabolism |
| 14 | sterol_steroid | 10 | 13 | lipid |
| 15 | sulfur_thiol_cofactor | 4 | 16 | redox_cofactor |

**Imbalance is severe: 80 spectra in the largest class against 3 in the smallest — 26.7×.** §9
shows the consequences and §10 of the audit asks whether the headline is inflated by it.

### Which normalisation is canonical

| view | top-1 | macro-F1 | replicate consistency | mean entropy | spread of total mass |
|---|---:|---:|---:|---:|---:|
| **raw evidence** | 0.835 | 0.793 | **0.947** | 0.603 | **0.864** |
| L1-normalised | 0.835 | 0.793 | 0.947 | 0.603 | 0.000 |
| calibrated probabilities | 0.835 | 0.793 | 0.920 | 0.289 | 0.000 |

The ranking is identical in all three — they are monotone transforms per spectrum — so the
choice is not about accuracy. It is about what is destroyed. **Raw evidence is canonical
internally** because it is the only view in which total mass varies across spectra: a query the
atlas supports weakly has low evidence *everywhere*, and both normalised views erase that by
construction (spread 0.000). **Calibrated probabilities are the classification output.** The L1
view exists only for the radar, where there is no absolute scale anyway. The prior stated in the
brief is confirmed, and it was tested rather than assumed.

## 5. Candidate models

37 candidates across four families, all benchmarked:

| family | variants | best macro-F1 |
|---|---|---:|
| A — similarity-weighted molecule evidence | 6 aggregations × 4 size corrections | 0.838 (`topk_mean`, no correction) |
| B — class prototypes | mean, median, medoid, 2-means, shrinkage | 0.828 (`multi2`) |
| C — transparent probabilistic | logreg, shrinkage LDA, nearest centroid, prototype likelihood, class-conditional distance | 0.810 (nearest centroid) |
| D — hierarchical broad→fine, soft routing | λ ∈ {0.5, 1.0, 2.0} | 0.814 (λ = 1.0) |

No neural network, random forest, boosted tree or opaque embedding was used — the brief forbids
them, and so does provenance: a chemistry score that cannot be decomposed cannot be traced.

## 6. Nested validation

Inner folds select the model, the aggregation, the size correction and every hyperparameter;
outer folds evaluate. The held-out molecule is absent from the reference bank, the prototypes,
the calibrator and the selection loop.

**Three different models won across five folds** — `D:A_max_idf:lam0.5` twice,
`A:logsumexp:idf`, `D:A_max_idf:lam2.0`, `A:topk_mean:none`. The modal choice is canonical; the
spread is reported because it is a real property of a 154-molecule corpus, not a nuisance.

**Is the modal choice a poor summary?** No, and this was tested rather than argued. A fully
nested ensemble — each outer fold averaging the top-4 candidates by *its own* inner score —
gains **+0.016 macro-F1**, below the 0.02 threshold pre-declared before the check ran. An
earlier construction that averaged the four models winning *somewhere* across the folds scored
+0.032, but that member set is informed by inner loops which saw other folds' test molecules;
**half the apparent advantage was that leak.** The modal single model stands, and it can be
stated in one line, which an averaged ensemble cannot.

**Selection bias is measurable and it is large.** The best flat candidate — selected on the same
folds it is scored on — reaches macro-F1 0.838. The honest nested number is 0.793. **The
difference of 0.045 is exactly what nested CV exists to remove**, and reporting the flat number
would have overstated the layer by more than the entire gap between it and the CSM layer.

## 7. Calibration

Phase 05 established that ECE alone selects a constant predictor. Selection here minimises
**log loss** — strictly proper — subject to floors declared in the module before the run:
sharpness > 0.05 and confidence discrimination > 0.60.

| method | log loss | Brier | ECE | classwise ECE | sharpness | discrimination |
|---|---:|---:|---:|---:|---:|---:|
| **temperature (selected)** | **0.826** | **0.320** | 0.179 | 0.038 | 0.184 | 0.735 |
| uncalibrated | 1.050 | 0.455 | 0.369 | 0.061 | 0.142 | 0.698 |
| platt | 1.511 | 0.586 | 0.344 | 0.069 | 0.136 | **0.884** |
| dirichlet | 1.991 | 0.662 | 0.217 | 0.056 | 0.177 | 0.776 |
| vector scaling | 2.410 | 0.720 | 0.296 | 0.056 | 0.173 | 0.763 |
| isotonic | 3.604 | 0.365 | **0.116** | **0.030** | 0.176 | 0.780 |

**Isotonic regression wins ECE and loses log loss by a factor of four.** The reason is
structural: it calibrates each class independently and the result is then renormalised, which
fixes the marginals and destroys the joint likelihood. Selecting on ECE would have chosen it.
This is the same lesson as Phase 05 in a different disguise, and it is why the selection rule
is a proper scoring rule.

Final calibrated performance: **ECE 0.125, classwise ECE 0.026, Brier 0.320, log loss 0.826,
sharpness 0.225, discrimination 0.668.** The classwise ECE of 0.026 is the more reassuring
number — it says small classes are not being systematically over- or under-confident — and it is
reported because top-label ECE would hide exactly that failure.

**One honest weakness.** Discrimination of 0.668 is modest, and Figure 16 shows a concrete
consequence: urea is misclassified as a free amino acid at confidence 0.99. Calibration reduces
over-confidence on average; it does not catch a confidently wrong answer on a molecule whose
chemistry genuinely resembles its neighbour's.

## 8. Per-class results

Macro-F1 0.793 spans a wide range. The four weakest classes are the four smallest:

| class | n | precision | recall | F1 |
|---|---:|---:|---:|---:|
| small_nitrogenous | 7 | 0.333 | 0.429 | 0.375 |
| phospholipid_sphingolipid | 8 | 0.455 | 0.625 | 0.526 |
| chromophore_pigment | 10 | 0.615 | 0.800 | 0.696 |
| sulfur_thiol_cofactor | 16 | 0.786 | 0.688 | 0.733 |

Restricting the macro average to the 10 classes with ≥ 5 molecules gives **0.826** against
0.793 over all 16 — the difference is the weight carried by classes that are barely evaluable
under molecule-grouped CV. Both are reported; neither replaces the other.

Full precision, recall and F1 for all 16 classes are in `chemistry_per_class_v1.csv`; the
confusion matrix is in `chemistry_confusion_matrix_v1.csv` and Figure 6.

**Errors are chemically reasonable.** Of 62 errors, 28 (45.2%) fall on a pre-declared adjacent
pair — fatty acid ↔ acylglycerol, purine ↔ nucleic-acid polymer, protein ↔ free amino acid, and
the rest listed in `registry.ADJACENT` — against a chance rate of 10.9%, a **4.2× lift**. These
are reported separately and are **not** scored as correct.

## 9. Soft-evidence validation

The vector is validated, not only its argmax:

| property | value |
|---|---:|
| mean rank of the true class | 1.39 |
| true class in the top 3 | 0.976 |
| mean true-class evidence share | 0.431 |
| mean top1−top2 margin | 0.386 |
| mean normalised entropy | 0.603 |
| within-class cosine | 0.760 |
| between-class cosine | 0.264 |
| **separation** | **0.497** |
| replicate consistency | 0.947 |
| effective rank | 12.12 of 16 |
| broad-superclass top-1 recovered from the same 16-vector | 0.808 |

Entropy of 0.603 is the number that says this is genuinely a soft representation: a one-hot
vector would score 0, a flat vector 1. **Effective rank 12.12 of a nominal 16** says the axes
are not 16 independent directions — the adjacent chemistries share variance, which is expected
and is reported rather than hidden (risk R-12, one level up).

## 10. Radar interpretation

The radar has exactly 16 axes, one per frozen class, built from the validated evidence vector
alone. No manual band-based axis appears (G13, G14). Each spoke carries evidence magnitude
(radius), calibrated confidence (spoke thickness and marker size), rank, margin and the number
of supporting reference molecules.

Because 16 spokes are cluttered, an **ordered-bar alternative is supplied** (Figure 17) showing
the same numbers without angular distortion. Seven representative cases are generated by rule
rather than by eye: clear single class, ambiguous adjacent classes, low-EV, misclassified,
high-confidence correct, low-confidence correct, and cross-source.

**The radius is evidence relative to the strongest axis. It is not a concentration and not a
composition**, and the figure says so on its face.

## 11. Provenance

1,125 chains verified, **0 broken**. The decomposition is mathematically exact for the selected
family: class evidence is an explicit function of named per-molecule similarities, and each
similarity is an inner product of the query's CSM activation with a named reference activation.
Every link — molecule, CSM, LSM — is checked against the frozen registries, and every chain
terminates in measured Raman spectra.

## 12. Held-out chemistry novelty

**Not cross-modality OOD.** An entire chemistry class is removed from the reference bank, the
prototypes and all model fitting; its spectra are then queried.

| withheld class | n | AUROC | abstain rate | max evidence (novel vs in-domain) | nearest represented |
|---|---:|---:|---:|---|---|
| sterol_steroid | 13 | **0.984** | 1.000 | 0.242 vs 0.670 | carboxylic acid metabolite |
| sulfur_thiol_cofactor | 16 | 0.973 | 0.625 | 0.273 vs 0.683 | free amino acid |
| mono_oligosaccharide | 43 | 0.960 | 0.953 | 0.284 vs 0.690 | carboxylic acid metabolite |
| purine | 17 | 0.909 | 0.765 | 0.264 vs 0.669 | sulfur/thiol cofactor |
| pyrimidine | 9 | 0.865 | 0.000 | 0.440 vs 0.682 | carboxylic acid metabolite |
| **acylglycerol** | 23 | **0.489** | **0.000** | **0.614 vs 0.665** | **fatty acid** |

Mean AUROC **0.863**; abstain rate 0.557 at a 5% in-domain false-abstain budget.

**Acylglycerol fails completely and the reason is chemical.** With acylglycerols withheld, fatty
acids remain, and a triacylglycerol's Raman spectrum is dominated by the same acyl-chain motifs
— CH₂ scissoring at 1440, C–C skeletal at 1080, cis C=C at 1650. Its evidence barely falls
(0.614 against an in-domain 0.665) and it abstains on nothing. The engine is not wrong that the
chemistry is *present*; it is wrong that the chemistry is *represented*.

**The honest statement of this capability is therefore conditional: novelty detection works when
the withheld chemistry has no close represented neighbour, and fails when it does.** Pyrimidine
shows an intermediate case — AUROC 0.865 but a 0% abstain rate, meaning the ranking separates
novel from in-domain while the threshold set for a 5% false-abstain budget does not.

## 13. Failure analysis

**Low-EV tail.** 16 spectra (4.3%) reconstruct at EV < 0.50. Accuracy on them is 0.625 against
0.844 elsewhere — degraded but far from random, so a poor reconstruction is a warning rather
than a disqualification. All 16 are listed by name in `low_ev_cases_v1.csv`.

**By source.** RamanBioLib 0.842 (n=202, mean EV 0.890), Gobbato metabolites 0.850 (n=153, EV
0.749), amino-acid grounding set 0.650 (n=20, EV 0.675). The weakest source is also the smallest
and the one with the lowest reconstruction quality; the three effects are confounded and are
reported as such.

**By error type.** 313 correct, 28 adjacent-class errors, 34 distant-class errors. Adjacent
errors have *lower* mean EV (0.786) than distant errors (0.855) — distant errors are not
reconstruction failures, which points at ontology ambiguity or genuinely unusual spectra rather
than at a preprocessing problem.

No engine tuning was performed against these cases.

## 14. Comparison with legacy semantic layers and unsupervised grouping

§1 gives the layer comparison. The semantic comparator (Part 13 of the brief):

| semantic layer | K | top-1 | chance-adjusted | interpretable | comparable? |
|---|---:|---:|---:|---|---|
| curated fine-16 | 16 | 0.845 | 0.835 | yes | yes |
| unsupervised 16 (out-of-fold) | 16 | 0.907 | 0.900 | no | **no** |
| frozen broad-6 | 6 | 0.875 | 0.850 | yes | yes |

**The unsupervised number must not be quoted as a result.** An unsupervised grouping is defined
by proximity in CSM space, and retrieval predicts by proximity in CSM space: predicting it is
close to self-prediction, and it would score highly even if the clusters were chemically
meaningless. Fitting the clustering out-of-fold removed a bookkeeping leak (0.931 → 0.907) but
cannot remove the structural advantage.

**The question the comparator can actually answer** is whether the curated ontology encodes
chemistry the data alone do not reveal. Agreement indices: **ARI 0.595, AMI 0.725** against the
curated ontology. Substantially recoverable, not fully — and the ~40% disagreement is where a
curated layer earns its place: distinctions that are chemical rather than spectral, such as
acylglycerol vs fatty acid and purine vs pyrimidine. The curated ontology is retained. It is
nameable, frozen, the label space of the frozen Tier-1 success criteria, and the only one of the
three a spectroscopist can argue with.

## 15. Robustness

11 perturbations × 5 levels, Raman only:

| representation | clean top-1 | mean perturbed | retention |
|---|---:|---:|---:|
| raw spectrum | 0.592 | 0.532 | 0.898 |
| CSM 49 | 0.845 | 0.797 | **0.943** |
| **Chemistry Evidence 16** | 0.835 | 0.782 | **0.937** |
| legacy 11-axis | 0.664 | 0.609 | 0.917 |
| legacy Theme/BSV | 0.491 | 0.433 | 0.882 |

The semantic layer tracks the CSM layer it is computed from (0.937 vs 0.943) and clearly beats
the raw spectrum on both accuracy and retention. It does not *improve* on the CSM layer's
stability — abstraction is not buying robustness here — and that is reported rather than
claimed.

## 16. Implications for Phase 07 (BSV2)

Phase 07 will factorise the Chemistry Evidence matrix. Three properties measured here bear
directly on whether that can work:

1. **Effective rank is 12.12 of 16.** There is redundancy to compress, but not much — a K far
   below 12 will lose real structure, and Phase 04.5 showed what that costs.
2. **Mean entropy is 0.603.** The evidence is genuinely distributed, so chemistry co-occurrence
   is a real object to factorise rather than a near-one-hot matrix with nothing to find.
3. **Replicate consistency is 0.947.** The input to Phase 07 is stable at the molecule level,
   so programme instability there would be a property of the factorisation rather than of noise
   in its input.

The Phase 07 informativeness floor should be measured against **0.835 top-1 and 0.793 macro-F1**,
which are the numbers BSV2 must retain at least half of (S-28).

## 17. Limitations

1. **Class imbalance is 26.7×** and the four weakest classes are the four smallest. Macro-F1 and
   balanced accuracy are reported precisely because top-1 hides this.
2. **Three classes have ≤ 3 molecules** (nucleic_acid_polymer, phosphate_metabolite, pyrimidine,
   and small_nitrogenous has 2). Under molecule-grouped CV, a 2-molecule class is evaluated on
   one molecule at a time with the other as its only reference — a genuinely hard regime that no
   modelling choice fixes.
3. **Novelty detection is conditional** (§12) and fails outright for chemistry with a close
   represented neighbour.
4. **Discrimination is 0.668** and confidently wrong answers exist (urea at 0.99).
5. **Selection is unstable across folds** — three models won across five folds — which is a
   corpus-size symptom. The fully nested ensemble check bounds the cost of picking the modal
   model at +0.016 macro-F1.
6. **Effective rank 12.12 of 16** means the axes are correlated; treating them as 16 independent
   chemistries would overstate the resolution.
7. **The 0.010 accuracy cost against the CSM layer is real**, even if it clears G7. The layer is
   justified by interpretability, calibration and top-3, not by accuracy.
8. **Everything is pure-Raman reference spectra.** No mixture, no biological matrix, no
   concentration series. Nothing here licenses a mixture or composition reading.

## 18. Decision gate

All 18 gates pass. See the final decision-gate block in the phase summary and
`phase06_gates_v1.csv`.

## 19. Reproduction

```bash
PYTHONPATH=src python results/v7_rebuild/phase06/code/run_phase06.py     # ~7 min
PYTHONPATH=src python results/v7_rebuild/phase06/code/make_figures.py
PYTHONPATH=src python results/v7_rebuild/phase06/code/make_pdf.py
PYTHONPATH=src python -m pytest tests/test_v7_phase06.py -q
```

Deterministic: `SEED = 0`, no unseeded randomness, output root resolves through
`GAIRA_V7_OUTPUT_ROOT`, and no path is hardcoded.
