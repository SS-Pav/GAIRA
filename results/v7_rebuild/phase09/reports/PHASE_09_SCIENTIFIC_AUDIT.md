# GAIRA V7 — Phase 09 Scientific Audit

Every conclusion in this phase sorted by how much weight it can bear, followed by the defects
found while producing it and what a hostile referee would ask next.

---

## A. Strongly supported

These would survive review.

**A1. The engine reproduces the frozen retrieval baseline exactly.**
Top-1 0.6053, top-3 0.7627, top-5 0.7947, top-10 0.8107, MRR 0.6870, nDCG@5 0.7112 — all six
identical to Phase 05/08 to the digit, computed by a separate implementation on the same frozen
artefacts. This is a binary claim and it is met. It establishes that integrating eight phases into
one object introduced no behavioural drift.

**A2. The engine is deterministic and stateless.**
Verified by repeat invocation, and structurally guaranteed: no attribute is mutated after
construction, both output dataclasses are frozen, and no random number is drawn at inference.

**A3. Every retrieval score reconciles.**
$|\sum_j \hat q_j \hat R_{mj} - S_m| < 10^{-9}$ for all 375 × 10 candidates. There is no hidden
term. This is a property of the cosine, not a measurement, but it is asserted rather than assumed.

**A4. Chemistry-class accuracy on unseen molecules is 0.851 ± the molecule-grouped fold spread.**
Computed with the chemistry model refitted inside each of five molecule-grouped folds. Top-3 is
0.976 and macro F1 0.811. This is the phase's headline performance number and it is measured the
hard way.

**A5. The radar is the most robust output the engine produces.**
Across 35 perturbation conditions the ordering never reverses: radar cosine 0.965 > chemistry
top-1 0.889 > molecule top-1 0.811. This is the behaviour the architecture was designed for —
degrade the specific claim before the general one — and it is now demonstrated rather than
asserted. Baseline drift at 0.80 is the worst case and even there the radar holds at 0.867.

**A6. The information ceiling sits at the CSM layer.**
0.608 raw → 0.850 LSM → 0.855 CSM → 0.664 axes → 0.405 themes → 0.392 Meta. Six representations,
one protocol, one corpus, measured across five phases. The claim is not that the CSM is optimal;
it is that four independent attempts to build above it each lost information, which is a
considerably stronger statement than any single comparison.

**A7. The low-reconstruction tail is chemically coherent.**
Pyruvate 0.209, thymine 0.258–0.286, malic acid 0.262–0.312, urea 0.345 — small, high-symmetry
molecules with few strong bands in 450–1800 cm⁻¹. A motif dictionary built largely from
biopolymers should struggle exactly here, and the engine flags them rather than answering
confidently. The prediction and the observation agree.

---

## B. Weakly supported

True as measured, but any of these could move.

**B1. "Calibration ECE 0.0534 for chemistry."**
Computed in-sample, from the shipped fit. The held-out calibration was measured in Phase 06 at
ECE 0.1247, and *that* is the number an external user should expect. The 0.0534 describes the
object being shipped, which is a legitimate thing to report, but it must never be quoted as
calibration quality on new molecules.

**B2. "Retrieval ECE 0.1205."**
Better than the pinned-temperature 0.2529 it replaced, but retrieval confidence is derived from
the score margin, and the margin is a weak signal — discrimination is 0.6914, only moderately
above chance. The risk–coverage curve is the more useful object and is published for that reason.
An operator should read the curve, not the ECE.

**B3. "Molecule top-1 is 0.605."**
This is the honest figure under the singleton convention, and 68 spectra (18.1%) are structurally
unretrievable. Whether 0.605 or the exclusion figure of ≈0.74 is the "real" number depends on
what question is being asked, and neither is wrong. The report quotes the conservative one, but a
reader comparing GAIRA to a published system whose benchmark has no singletons is not comparing
like with like.

**B4. "Warnings fired on 86 unknown and 18 outlier spectra."**
The thresholds are inherited from Phase 05 and were not swept in Phase 09 — deliberately, since
tuning them here would be tuning on the evaluation set. But that means the counts describe the
thresholds as much as the corpus. The *direction* is sound (the flagged spectra are the poorly
reconstructed ones); the counts are not robust.

**B5. "Radar reproducibility 0.960."**
Measured across replicate spectra of the same molecule within the same corpus, so it captures
within-library variation. It does not establish reproducibility across instruments, laboratories
or acquisition conditions, which is the reproducibility a user would actually care about.

---

## C. Unsupported — do not claim these

**C1. Any performance claim outside pure Raman reference spectra.**
No SERS, serum, plasma, EV, tissue or pathogen spectrum entered this phase. Nothing here predicts
behaviour in those regimes, and the robustness study — which perturbs clean spectra — is not a
substitute for measuring transfer.

**C2. "The engine achieves 0.955 chemistry accuracy."**
That is the in-sample figure. It is reported with an explicit `IN_SAMPLE_WARNING` in the artefact
and captioned as such on the figures. Quote 0.851.

**C3. "Macro AUC 0.999" as a performance claim.**
Same reason, compounded: AUC is optimistic under class imbalance, and this one is in-sample as
well. It is a sanity check that the model is not broken, nothing more.

**C4. Any reading of the radar as a concentration, abundance or mixture fraction.**
The L2 normalisation destroyed absolute scale before anything else happened, and a similarity is
not a quantity of material. This is the single most likely misreading of the engine's output and
it is prohibited by gate G12.

**C5. "The engine detects unknown molecules."**
It does not. The `unknown` and `outlier` warnings detect **spectra the atlas cannot explain**. A
molecule absent from the 154-molecule bank whose spectrum the atlas explains well will produce a
confident, wrong, unflagged identification. This is a real limitation and it is stated in the
spec rather than softened.

**C5b. "The warnings catch structureless input."** — *measured during test authoring.*
A regression test was written asserting that white noise triggers a warning. It failed, so the
behaviour was measured instead of assumed. Over 20 seeds, white noise reconstructs at CSM
explained variance ≈ 0.61 — **above** the 0.50 `unknown` floor — and a warning fires on only
**1 of 20**. Confidence does separate cleanly (noise maximum 0.495 against a corpus mean of
0.803, and only 5.3% of real spectra fall below 0.495), so the *confidence* is informative even
where the *warning* is not. The reason is structural: 49 non-negative basis spectra spanning
450–1800 cm⁻¹ can assemble a broad positive vector reasonably well, so "explains the spectrum" is
a weaker test than it looks against unstructured input. **Read the confidence, not the flag.**
The test now encodes the measured separation rather than the assumption that failed.

**C6. "Robustness at level σ implies robustness on a real degraded instrument."**
The perturbations are parametric models applied to clean spectra. Real degradation is correlated
across mechanisms and is not drawn from these families.

---

## D. Defects found and fixed during this phase

| # | defect | consequence had it stood | fix |
|---|---|---|---|
| 1 | **Validation 4 was computed in-sample.** Chemistry accuracy was reported as 0.9547 with the model fitted on all 375 spectra. | The phase would have claimed 0.955 chemistry accuracy — an eleven-point overstatement — as its headline, in a document intended to be the final word on V7. This is R-10, the same failure the rebuild has been guarding against since Phase 03. | Held-out molecule-grouped computation added (0.8507), the in-sample figures retained but carrying an explicit `IN_SAMPLE_WARNING` string in the artefact itself, and gates **G15** and **G16** added. |
| 2 | **Retrieval calibration temperature was pinned at 0.02**, a value carried over rather than fitted. | ECE 0.2529 would have been reported as the engine's retrieval calibration — twice its true value — making the engine look badly overconfident when the defect was in the evaluation. | Temperature fitted in-fold; ECE 0.1205. |
| 3 | **A local variable `cls` shadowed the classmethod's own `cls` parameter** inside `GAIRAEngine.load()`. | The constructor call at the end of `load()` would have failed, or — worse under a different ordering — bound the wrong object. It worked by accident of line ordering, which is the most dangerous kind of working. | Renamed to `cls_arr`; the redundant `cls_` alias removed. |
| 4 | **Retrieval confidence was initially derived from `1/rank`** and scored against correctness defined as `rank ≤ 1`. | Discrimination would have been exactly 1.000 — the confidence and the outcome were the same quantity twice. A perfect score that is perfect because the question was asked of itself. | Replaced with the score margin, which is available at inference time without knowing the answer. Discrimination 0.6914. |

Defects 1, 2 and 4 are the same shape: **an evaluation that flatters the system because it was
given information it will not have in use.** Defect 4 is the purest form — a metric computed
against itself.

### The pattern this phase confirms

Across V7 the recurring failure has been *a selection or evaluation metric maximised by something
other than the thing being measured*. Phase 03 softmax themes, Phase 04 theme-mode leakage,
Phase 04.5 Meta stability, Phase 05's ECE-optimal constant calibrator, Phase 06.5's K = 4 by
bootstrap ARI, Phase 07's K = 16 identity solution — and now Phase 09's self-referential
discrimination. **Seven occurrences.** The countermeasure that has worked every time is the same:
name what a degenerate answer would look like *before* running the selection, and add a floor
that disqualifies it.

A second, narrower pattern also recurred: *a comparison whose two arms were not given the same
problem* — Phase 06's unsupervised comparator, Phase 06.5's ensemble, Phase 08's retrieval bank.
It did not recur here, because Phase 09 made no comparisons; it reproduced them.

---

## E. What a referee would ask next

1. **Report the held-out chemistry calibration alongside the in-sample one in the engine's own
   artefacts**, rather than requiring a reader to find it in Phase 06. Currently ECE 0.0534
   (in-sample, Phase 09) and ECE 0.1247 (held-out, Phase 06) live in different documents.
2. **Sweep the `unknown` and `outlier` thresholds and publish the sensitivity**, so B4's counts
   come with an error bar. This must be done in a phase that can declare the sweep in advance,
   not retroactively.
3. **Measure open-set behaviour directly** by holding out whole molecules from the bank and
   asking whether the engine's confidence drops. C5 is currently an argued limitation, not a
   measured one, and it is the limitation most likely to matter in use.
4. **Report cross-source reproducibility**, splitting the radar-reproducibility measurement by
   source library rather than pooling it (B5).
5. **State the fold-level spread on 0.851**, not just the point estimate.

None of these would change the decision. All five would make it easier to defend.

---

## F. Assessment of the phase itself

Phase 09 did what a packaging phase should: it took eight phases of decisions and showed they
compose into one object without contradiction. The strongest evidence for the rebuild is not any
individual number but the fact that **integration surfaced no inconsistency between phases** —
retrieval reproduced to the digit, the chemistry model loaded and predicted as specified, the CSM
projection matched. A rebuild that had been quietly overfitting its way forward would not survive
that test.

The phase's own weakness is that it is *entirely* in-corpus. Every number here comes from the 375
spectra that also built the atlas. The robustness study is the closest thing to an external test
and it is synthetic. That is not a defect in the phase — it is the honest boundary of what V7's
data allows — but it means the correct reading of Phase 09 is **"the architecture is internally
sound and faithfully implemented"**, not "the architecture works". The second claim requires data
V7 does not have.

**Confidence that the engine is a faithful implementation of the frozen architecture: 10 / 10.**
This is checkable and was checked.

**Confidence that freezing V7 here is the right call: 9 / 10.** The deduction is for the untested
open-set behaviour (C5, E3). If the engine turns out to be confidently wrong on molecules outside
its bank, that is an architecture-level gap rather than a corpus-level one, and it would be
better to have found it before declaring the architecture final. Everything else points to a
freeze: four measured failures above the CSM layer, exact reproduction of every baseline, and a
clear, corpus-shaped path forward that needs no architectural change.
