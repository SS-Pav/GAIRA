# GAIRA V7 Phase 03 — Scientific Audit

**Reviewer-style assessment**, written as if for *Nature Biomedical Engineering* / *Nature
Machine Intelligence* / *Nature Methods*. Adversarial by design: the job is to find what a
referee would attack, not to defend the work.

**Overall recommendation to the authors: major revision before any external submission.** The
methodology is unusually careful. The evidence base is too thin to support a methods paper on
its own, and two claims in the current draft are not supported by the data behind them.

---

## 1. Strengths

**S1. The label firewall is real and is enforced in code, not in prose.** Discovery uses no
chemistry label; labels are revealed once, after `K` is fixed and validated. The tests assert
that no primary representation *accepts* a class label as a parameter — a signature-level check
that cannot be satisfied by a comment. Reviewers routinely doubt such claims; here it is
falsifiable.

**S2. Six method defects were found, demonstrated, and corrected mid-run, and all six are
recorded.** Two of them (an admissibility criterion that could not fail; a degeneracy check
that passed a run where one theme claimed all 49 CSMs) would have produced a publishable-looking
but wrong result. The most striking is defect 4: every theme came out named "aliphatic chain +
…" because CH₂ scissoring dominates prominence in nearly every biological Raman spectrum. This
is the canonical over-interpretation trap in Raman biology, it arose spontaneously, and it was
caught.

**S3. Independent corroboration across three layers.** Theme-02's bands (1064, 1298, 1442,
1658) reproduce the single Phase 02 equivalence, reached by an unrelated objective. The
top-level hierarchy split reproduces the Phase 02.5 hydrophobic/polar bipartition and PCA's one
reproducible component. Three layers agreeing from three objectives is stronger evidence than
any single validation.

**S4. Negative results are reported at full strength.** One theme rejected on a stability floor
despite having the most members; nine CSMs left poorly explained; the theme basis losing 45% of
reconstruction; value over the CSM layer reported as small. None of this is buried.

**S5. Bridges are first-class.** Fifteen of 49 CSMs keep genuinely split membership. Most theme
layers in this literature hard-assign and never report how many assignments were marginal.

**S6. Full determinism and provenance.** Fixed seeds, fingerprint gates on every frozen input,
a manifest listing every output with its SHA-256, and a redirectable output root that
deliberately does not redirect frozen inputs.

---

## 2. Weaknesses

**W1. `n = 49` is too small for `K = 5` with this validation scheme.** Roughly ten CSMs per
theme. Bootstrap resampling at 85% removes seven CSMs; leave-one-out removes one. Neither
probes the regime a reviewer cares about. **This is the single most serious weakness** and it
is not fixable by analysis.

**W2. Two accepted themes sit barely above the rejection floor.** Theme-05 at 0.62 and Theme-01
at 0.69, against a floor of 0.60 and a rejected theme at 0.59. The line between accepted and
rejected is 0.03 of bootstrap recovery. A referee will ask what happens at floor 0.65, and the
honest answer is that Theme-05 would also fail.

**W3. Six method corrections during a single run invites the charge of tuning.** Each is
documented and each was demonstrated first — but a reviewer sees six changes between the first
result and the final one, and the final `K` moved 9 → 4 → 6 → 5 across them. The defence is
that every change was forced by a demonstrable defect and never by a preferred outcome; that
defence requires the reader to check six arguments.

**W4. `archetypal` beat `diffusion_gmm` on a composite of seven criteria with pre-registered
weights.** Different defensible weights could select a different model. No sensitivity analysis
over the criterion weights was run.

**W5. Post-hoc naming is mechanical.** Names come from a lookup of vibrational windows and a
two-family rule. No spectroscopist reviewed the assignments independently. Defect 5 — a protein
theme named "phosphate" from its 1004 cm⁻¹ band — shows exactly how this fails, and was caught
by inspection rather than by the method.

**W6. Agreement with curated chemistry is weak** (AMI 0.157). Presented as expected for an
unsupervised layer, which is fair — but it also means the themes cannot be validated against
the one external reference available.

**W7. Half the dictionary is not cleanly placed** (15 bridges + 9 poorly explained = 24 of 49).
Honest, and a real limit on the abstraction's coverage.

---

## 3. Unsupported or over-stated claims

**U1. "The theme layer adds value over the CSM layer."** Retrieval 0.237 vs 0.155 on a chance
of 0.101. No confidence interval, no significance test, `n = 49`. The direction is right; the
claim as stated is not supported at the strength implied. **Recommend: report a bootstrap CI or
downgrade to "a small improvement, not established at this n."**

**U2. "Theme-02 independently recovers the Phase 02 equivalence."** Both derive from the same
49 CSM spectra, so the objectives differ but the data do not. Independent *objective*, not
independent *evidence*. **Recommend: restate precisely.**

**U3. "Four levels emerge" in the hierarchy.** The procedure returns one level per distinct
group count between 2 and 5; with `K = 5` that is four levels almost by construction. No test
distinguishes a genuine four-level hierarchy from an agglomeration artefact. **Recommend:
report the top-level split, which is corroborated, and drop the claim of four levels.**

**U4. Theme names imply more specificity than the evidence carries.** "carboxyl / ester
carbonyl + amide backbone" describes bond systems, correctly, but reads as a molecular class.
The registry's `chemical_definition` is careful; the `name` is not always.

**U5. Confidence scores are a weighted sum of four validation numbers**, not calibrated
probabilities. They are used as though comparable across themes.

---

## 4. Likely reviewer criticisms

**R1.** *"You corrected your admissibility criterion twice and your degeneracy criterion twice.
How do I know the final K is not the one that produced the nicest themes?"* — The strongest
attack. Answer: every change was triggered by a demonstrable defect (a criterion that returned
1.000 for every candidate; a run where one theme claimed all 49 CSMs), and the intermediate
results are all recorded. It still requires the reader to audit six arguments.

**R2.** *"Five themes from 49 motifs from 154 molecules. What is the effective sample size?"* —
No good answer exists at this corpus size.

**R3.** *"Your reconstruction drops from 1.00 to 0.55. What is the theme layer for?"* — Answer:
compression with a small retrieval gain, and semantic axes for the BSV. A referee may find that
insufficient.

**R4.** *"Nine CSMs are poorly explained, including the one CSM your own Phase 02 accepted.
Does that not indicate K is too small?"* — A fair question that the current analysis does not
settle. `K = 3` was also admissible; larger `K` was not.

**R5.** *"Mode-family specificity is an IDF weighting estimated on the same sweep it scores.
Is that not circular?"* — Partly. The specificity table is estimated across all models and all
K before any theme is judged, but it is the same data. An external band-frequency reference
would be cleaner.

**R6.** *"Archetypal analysis returns extremes, not clusters. Are your themes poles rather than
groups?"* — Yes, deliberately, and that suits a continuum — but it changes what "membership"
means and the paper should say so plainly.

---

## 5. Recommended experiments

**E1 — highest value: an independent corpus.** Rebuild themes on a held-out Raman corpus and
measure whether the same five emerge. Nothing else addresses W1.

**E2 — criterion-weight sensitivity.** Sweep the seven composite weights over a Dirichlet
simplex and report how often each (model, K) wins. Directly answers W4 and R1.

**E3 — stability-floor sensitivity.** Report the accepted set at floors 0.55/0.60/0.65/0.70.
Answers W2 honestly rather than defending a threshold.

**E4 — independent spectroscopic review.** A Raman spectroscopist blind to the pipeline
assigns each theme's bands; compare with the automated naming. Answers W5 and U4.

**E5 — bootstrap CI on the value-over-CSM comparison**, plus a paired permutation test.
Converts U1 from a claim into a result.

**E6 — external band-frequency prior** for specificity, from a published Raman band
compilation rather than from this sweep. Answers R5.

**E7 — targeted expansion at the nine poorly-explained CSMs.** Tests whether they are
unrepresented chemistry or a `K` that is too small.

---

## 6. Remaining risks

| risk | severity | status |
|---|---|---|
| corpus too small for the claims (W1, R2) | **high** | not mitigable by analysis; E1 required |
| accepted/rejected boundary is 0.03 of bootstrap recovery (W2) | **high** | E3 would quantify it |
| six mid-run method corrections read as tuning (W3, R1) | medium | fully documented; E2 would settle it |
| naming is mechanical and has failed once already (W5, U4) | medium | E4 |
| half the dictionary unplaced (W7) | medium | honest; E7 |
| specificity weighting is estimated in-sample (R5) | low | E6 |
| confidence scores are uncalibrated (U5) | low | rename, or calibrate |

---

## 7. Verdict

The **method** is defensible and unusually well-audited; the corrections found during this run
are the sort most pipelines never surface. The **result** — five themes, four accepted, from 49
motifs — is plausible and internally corroborated across three layers.

The **evidence base is too small** for the claims a Nature-family methods paper would need, and
two claims (U1, U3) should be weakened before the work goes anywhere external. As an internal
foundation for Phase 04, it is sound provided the four constraints in the report's §8 are
carried forward.

**Approved for Phase 04. Not ready for external submission without E1, E2, E3 and E5.**
