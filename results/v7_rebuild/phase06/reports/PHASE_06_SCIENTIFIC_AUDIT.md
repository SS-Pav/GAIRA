# GAIRA V7 — Phase 06 Scientific Audit

Adversarial. The question is not *did the gates pass* — they did, all 18 — but *would a Nature
Methods referee believe the layer is ready to become the input to Phase 07*, and where would
they push first.

The brief names seven questions. Each is answered directly in §A, with the evidence, before the
general critique in §B.

---

## A. The seven questions the brief demands

### A1. Are the 16 classes scientifically defensible?

**Mostly, with two soft spots.**

They are a curated chemical partition frozen in Phase 00, they are the label space of the frozen
Tier-1 success criteria, and 14 of the 16 are unambiguous to any spectroscopist. Two are
weaker. **`small_nitrogenous` (2 molecules, 7 spectra)** is a residual category rather than a
chemistry — it is defined partly by what it is not, and it scores the lowest F1 in the phase
(0.375). **`carboxylic_acid_metabolite`** overlaps `phosphate_metabolite` conceptually (both are
energy-metabolism small molecules) and empirically (it is the modal "nearest represented class"
for three of the six withheld classes, which is the signature of a broad attractor rather than a
tight chemistry).

Crucially, **the ontology was not changed** despite both observations. Ontology changes require
a separate architecture decision (Part 19 of the brief), and confusing two classes is not
grounds for merging them.

The out-of-fold agreement result is the strongest defence and the strongest caveat at once: ARI
0.595 / AMI 0.725 says the curated partition is substantially — but only substantially —
recoverable from the spectra without labels. A referee could reasonably read 0.595 as "40% of
this ontology is chemical knowledge the spectra do not carry", which is either the point of a
curated layer or an indictment of it depending on prior.

### A2. Is the high accuracy inflated by class imbalance?

**Partly, and the phase measures by how much.**

Top-1 of 0.835 is spectrum-weighted, and the two largest classes (peptide_protein 80,
free_amino_acid 75) are 41% of the corpus. The imbalance-robust numbers are lower:
**balanced accuracy 0.830, macro-F1 0.793.** The gap between top-1 and macro-F1 is 0.042, which
is the honest size of the inflation.

Two further points in the layer's favour. First, the selected model carries an **inverse-frequency
size correction**, chosen by nested CV over the uncorrected alternative — the correction was
selected, not imposed. Second, the *inner-loop selection metric was macro-F1, not top-1*,
specifically so the selection could not ignore the small classes.

A referee would still note that F1 tracks class size almost monotonically (Figure 22b), and that
the four weakest classes are the four smallest. That is a corpus property, not a modelling
defect, and §17.2 of the report says so.

### A3. Is the evidence vector merely a disguised hard classifier?

**No, and this was tested rather than asserted.**

- Mean normalised entropy **0.603** (one-hot = 0, flat = 1).
- Mean true-class evidence share **0.431** — the winning class holds under half the mass.
- Mean top1−top2 margin **0.386** on a [0,1] scale.
- True class in the top 3 for **97.6%** of spectra while top-1 is 83.5%: the 14-point gap is
  information the argmax discards.
- **Effective rank 12.12 of 16** — the vector spans a genuinely multi-dimensional space.
- Broad-superclass accuracy of **0.808 recovered from the 16-vector by summation**, which is
  only possible if the sub-class mass is distributed sensibly.

Figure 16's `glucose oxidase` case is the clearest single demonstration: correct, but at
confidence 0.22 with mass spread over eight classes. A hard classifier cannot express that.

### A4. Does calibration fail on small classes?

**No — and this is the phase's most reassuring number.**

**Classwise ECE is 0.026** against a top-label ECE of 0.125. Classwise ECE is the one-vs-rest
average and is the metric that exposes small-class miscalibration; top-label ECE would hide it
entirely. A referee should be told that both are reported and that the small-class-sensitive one
is five times better.

The residual concern is different and real: **discrimination is 0.668**, which is modest, and
Figure 16 shows urea misclassified as a free amino acid at confidence **0.99**. Calibration
fixes average over-confidence. It does not fix a confidently wrong answer on a molecule whose
chemistry genuinely resembles its neighbour's, and no calibrator can.

### A5. Does the radar imply mixture composition incorrectly?

**It could, and three specific guards were built against it.**

1. The canonical internal representation is **raw evidence, not L1-normalised** — precisely
   because L1 normalisation makes the vector *look* like a composition. The comparison table in
   §4 of the report shows the L1 and calibrated views have total-mass spread exactly 0.000,
   which is the property that invites the misreading.
2. The radar plots radius **relative to the strongest axis**, and the figure states on its face
   that this is not a concentration and not a composition.
3. An **ordered-bar alternative** is supplied, which carries no implication of a filled area at
   all.

The honest residual risk: a filled polygon is a filled polygon, and a reader who ignores the
caption will read area as amount. For pure references the question does not arise; if this layer
is ever shown on a mixture spectrum, the radar should be replaced by the bar view. That
recommendation belongs in the phase that first handles mixtures, and this phase handles none.

### A6. Does held-out chemistry novelty actually work?

**Conditionally, and one of six cases fails outright.**

Mean AUROC 0.863 across six withheld classes, with five between 0.865 and 0.984. **Acylglycerol
scores 0.489 — chance — and abstains on 0% of its spectra**, because fatty acids remain in the
atlas and share the acyl-chain motifs that dominate a triacylglycerol spectrum.

This is the correct experiment and it produced a genuine negative. The report states it in the
executive summary rather than in a limitations appendix, which is the right placement.

Two caveats a referee would add. First, **pyrimidine has AUROC 0.865 but a 0% abstain rate** —
the ranking separates the populations while the threshold calibrated to a 5% in-domain
false-abstain budget does not. AUROC and operational abstention are not the same claim, and the
report gives both. Second, six classes is a small sample of the sixteen; the four not tested
(peptide_protein, free_amino_acid, fatty_acid, carboxylic_acid_metabolite, and others) include
the two largest, and withholding those would be a much harder test that was not run.

### A7. Is the semantic layer stable enough to become Phase 07 input?

**Yes at the molecule level; less so at the model-selection level.**

For: replicate consistency **0.947**, within/between-class separation **0.497**, robustness
retention **0.937** across 11 perturbations, provenance complete, deterministic on rerun.

Against: **three different models won across five outer folds**, and the flat-vs-nested gap of
0.045 in macro-F1 shows the corpus is small enough that selection is genuinely unstable. Phase
07 will consume the output of *one* of those models. If a different fold assignment would have
selected a different model, the Phase 07 input is to that extent arbitrary.

**This was tested, not left as a recommendation** — see B2. A fully nested ensemble gains
+0.016 macro-F1 over the modal choice, below the 0.02 threshold declared before the check ran.
The instability is real; its cost is bounded and small.

---

## B. Where a referee would push, hardest first

**B1. The 0.010 accuracy loss is not free, and the justification is not accuracy.**
The layer scores 0.835 against the CSM layer's 0.845. G7 was pre-registered at ±0.02 so this
passes, but the honest framing is that Phase 06 *pays* for interpretability rather than gaining
from it. The payment is small and the return is real (top-3 0.976 — the best in the stack —
calibration, a 16-axis radar, exact provenance), but the report must never describe the layer as
*more accurate* than the representation it compresses. It is not.

**B2. Selection instability — raised, then tested, and the test found a leak of its own.**
Three models across five folds, and the canonical model is a *modal* choice with 2 of 5 votes.
This audit's first draft asked for an ensemble as a robustness check. It was run, and the first
construction averaged the four models that won *somewhere* across the five folds — which scored
macro-F1 **+0.032** over the modal choice and looked decisive.

That set is informed by inner loops that saw other folds' test molecules: a second-order leak.
Rebuilt so each outer fold ensembles the top-4 candidates by **its own** inner score, the gain
falls to **+0.016** — below the 0.02 threshold pre-declared before either version ran. **Half
the apparent ensemble advantage was the leak.**

The modal single model therefore stands, and it stands on a measurement rather than on
convenience. What remains true is that selection is genuinely unstable at this corpus size, and
a 154-molecule corpus cannot settle which of four near-equivalent aggregations is best.

**B3. The novelty holdout omits the largest classes.**
Withholding `peptide_protein` (80 spectra, 30 molecules) or `free_amino_acid` (75, 18) would be
the most informative test in the phase, and neither was run. The six chosen classes span
small/large and distinctive/overlapping as the brief asked, but the largest two were excluded by
the choice of which cells to fill. That was a judgement call and it should be stated as one.

**B4. `carboxylic_acid_metabolite` behaves like an attractor.**
It is the modal nearest-represented class for three of the six withheld chemistries — sterol,
mono/oligosaccharide and pyrimidine — which have very little in common. A class that absorbs
unrelated novelty is either very broad or poorly bounded. This is worth investigating before it
propagates into BSV2, where an attractor class would anchor a programme.

**B5. The unsupervised comparator cannot be made fair, and the phase should say so more loudly.**
Out-of-fold fitting reduced 0.931 → 0.907 but the residual advantage is structural. The report
explains this in §14 and the figure marks it amber, but the number is in a results table where a
casual reader will find it. It would be safer to remove the accuracy column from the comparator
table entirely and report only the agreement indices.

**B6. Effective rank 12.12 of 16 is a warning for Phase 07, not a reassurance.**
The report frames it as "there is redundancy to compress". It equally says the 16 axes carry
about 12 directions of information, so a BSV2 with K ≪ 12 is compressing something real. Given
Phase 04.5's history, the Phase 07 pre-registration should treat 12.12 as an upper bound on what
can be discarded cheaply.

**B7. Two classes have fewer than 4 molecules under molecule-grouped CV.**
`small_nitrogenous` has 2 molecules. When one is held out, the class has exactly one reference.
Its F1 of 0.375 is close to the ceiling of what that regime allows, and reporting it alongside
80-molecule classes in a macro average gives it equal weight. The macro-F1 is therefore dragged
by a class that is arguably not evaluable, and an alternative macro over classes with ≥ 5
molecules would be worth reporting alongside.

---

## C. Errors found and fixed during the phase

Both were found by inspecting outputs, not by tests, and both changed reported numbers.

| # | defect | consequence had it stood | fix |
|---|---|---|---|
| 1 | **Three required methods raised silently on sklearn 1.8** — `C:logreg`, `vector_scaling`, `dirichlet` (`multi_class` removed from the API). The benchmark caught the exception per-method and continued. | The phase would have reported a "complete" benchmark that silently omitted one of four model families and two of six mandated calibrators. | API call corrected; **gate G18 added** so no pre-registered candidate or calibrator can fail silently again. G18 detects a new failure mode and replaces no pre-registered gate. |
| 2 | **The unsupervised comparator was fitted on all 154 molecules**, including the test fold, in the same space used for retrieval. | The comparator scored 0.931 against the curated ontology's 0.845, which reads as "the curated ontology is worse". | Refitted inside training folds only (0.931 → 0.907), and the residual structural bias is now stated rather than reported as a result. |
| 3 | **The selection-stability ensemble leaked its member set across folds** — the four models averaged were those that won *somewhere*, a set informed by inner loops that saw other folds' test molecules. | The ensemble would have appeared to beat the modal model by +0.032 macro-F1 and would have been shipped as canonical. | Rebuilt fully nested: each fold ensembles the top-4 by its own inner score. Gain falls to +0.016, below the pre-declared threshold. |

A third issue was caught before it reached a result: the per-spectrum metadata join is asserted
rather than assumed — the run aborts if `spectrum_quality_v1.csv` does not align row-for-row
with the balanced references.

---

## D. Claims checked against what was computed

| claim | verdict |
|---|---|
| "Raman only; no SERS, serum, plasma, EV, mixture or DART-Met data" | **holds** — no such path is reachable in the phase's code; asserted by test |
| "nothing upstream refitted" | **holds** — reads only; fingerprint gate aborts on mismatch |
| "Phase 05 reproduced bit-for-bit" | **holds** — all four metrics agree to < 1e-9 |
| "nested molecule-grouped validation" | **holds** — inner folds select, outer folds evaluate, no molecule crosses |
| "16 axes in a fixed order" | **holds** — `CLASS_ORDER` is a module constant, asserted by test |
| "the legacy 11-axis map is absent from canonical inference" | **holds** — it appears only in the Part 6 comparator |
| "errors are chemically adjacent at 4.2× chance" | **holds** — adjacency declared before any confusion matrix was inspected |
| "0 broken provenance chains" | **holds** — 1,125 chains, every link checked against the frozen registries |
| "the evidence is exactly decomposable" | **holds for the selected family** — and the code refuses to claim it for families where it is false |
| "calibration is informative" | **holds** at the declared floors (sharpness 0.225 > 0.05, discrimination 0.668 > 0.60); the floors are lower than Phase 05's and that is visible |

---

## E. What would change the conclusions

- **An ensemble over the five fold-selected models materially beating the modal choice** would
  mean the canonical model is a poor summary of the selection (B2).
- **Withholding `peptide_protein` and finding novelty AUROC near chance** would generalise the
  acylglycerol failure from "chemistry with a close neighbour" to "any large class", which would
  substantially weaken the novelty claim (B3).
- **A `carboxylic_acid_metabolite` audit showing it is spectrally heterogeneous** would argue for
  an ontology review — a separate architecture decision, not a Phase 06 change (B4).

---

## F. Recommendations, in priority order

1. ~~Run the ensemble check.~~ **Done** — see B2. Fully nested, the gain is +0.016 macro-F1,
   below the pre-declared 0.02 threshold; the modal model stands.
2. **Extend the novelty holdout to `peptide_protein` and `free_amino_acid`** (B3). The two
   largest classes are the two most informative omissions.
3. **Audit `carboxylic_acid_metabolite` for internal heterogeneity** (B4) — report only; any
   ontology change is a separate decision.
4. **Report a macro-F1 restricted to classes with ≥ 5 molecules** alongside the full macro (B7),
   so the headline is not dominated by a 2-molecule class.
5. **Remove the accuracy column from the unsupervised comparator table** and keep only the
   agreement indices (B5).
6. **Carry effective rank 12.12 into the Phase 07 pre-registration as an upper bound** on cheap
   compression (B6).

---

## G. Overall

The phase does what it claimed, and its negative results are reported at full strength: the
layer costs 0.010 in accuracy, novelty detection fails outright for one of six withheld classes,
selection is unstable across folds, and the four weakest classes are the four smallest. Two real
defects were found and fixed mid-phase, one of which would have let a third of the model families
silently vanish from a "complete" benchmark.

The strongest results — 0.976 top-3, classwise ECE 0.026, 4.2× chemically-reasonable error lift,
0 broken provenance chains from 1,125 — are well-evidenced and honestly bounded.

**Scientific confidence that this layer is a sound input to Phase 07: 8 / 10.** One point is
deducted for the untested large-class novelty case (B3) and one for the acylglycerol failure
itself, which is a real capability boundary rather than a fixable defect. Selection instability
(B2) is now measured rather than suspected, and it does not change the canonical model.
