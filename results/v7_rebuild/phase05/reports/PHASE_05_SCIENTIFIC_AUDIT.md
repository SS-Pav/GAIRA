# GAIRA V7 — Phase 05 Scientific Audit

An adversarial read of the canonical inference engine. The question is not *did the numbers come
out well* but *would a referee at a Nature-family journal believe them*, and where would they push
back first.

---

## A. Where the phase is strong

**A1. The two-split design closes the leakage hole Phase 04 left open.** Split B rebuilds the
reference bank inside every fold from training molecules only, so no reported class number is
contaminated by the query's own molecule. The claim "0.845 class top-1 on molecules the atlas has
never seen" is the claim the data support, and it is stated at that strength and no higher.

**A2. Metric selection is genuinely nested.** The metric applied to each outer fold was chosen on
inner folds of that fold's training set. This is more work than the phase needed — cosine wins
under any reasonable rule — but it means the selection cannot be the reason the number is good.

**A3. The evidence layer is falsifiable, and four axes fail.** Declaring eleven axes and then
testing each against chemistry it never saw is the right shape for an interpretability claim.
Reporting `unsaturation` at 0.534, `amide_protein` and `sulfur_thiol` at 0.663 and
`carbonyl_ester` at 0.626 as failures — rather than quietly widening windows until they passed —
is what makes the seven successes worth anything.

**A4. Provenance is verified, not asserted.** 3,133 chains, every link checked against the frozen
registries, zero broken. The waterfall's contributions are the actual additive terms, so a reader
can add them up and get the number back.

**A5. G6 was not relaxed.** The pre-declared ECE gate fails at 0.130 and is reported as failing.
The companion gate was *added*, not substituted.

## B. Where a referee would push, hardest first

**B1. "Your axes were declared a priori — but by whom, and when?"** The eleven axes and their band
windows were written by the same author who saw the corpus's chemistry-class distribution in
Phases 00–04. There is no timestamp separating the declaration from that knowledge. The windows
themselves are standard textbook assignments and would be recognisable to any Raman
spectroscopist, which is the honest defence — but it is a defence by appeal to convention, not by
pre-registration. **A referee is entitled to treat the axis definitions as weakly
label-informed.** The strongest available counter-evidence is that four axes fail their test,
which a label-fitted design would not produce; the strongest remaining fix is for someone else to
declare the windows.

**B2. The unsaturation secondary test looks like a rescue.** The primary test failed at 0.534;
a second test was then run and returned 1.000. The sequence is exactly what motivated reasoning
looks like from outside, even though the axis and its loading matrix were fixed before either
test ran and only the *evaluation label* changed. Two things make the rescue defensible and both
should be stated whenever the number is quoted: the label refinement is chemically forced (the
class `fatty_acid` demonstrably contains saturated members, so the primary test cannot reach 1.0
even in principle), and the molecule-name rule is mechanical rather than hand-picked. It remains
the weakest-looking result in the phase and should always be reported *with* the failing primary
number, never instead of it.

**B3. Threshold sensitivity — raised, then resolved, and only half reassuringly.** This audit's
first draft called `SUPPORT_FLOOR = 0.10` an unmotivated constant and asked for a sweep. The sweep
now runs in the phase (`evidence_axis_sensitivity_v1.csv`) and splits the question in two.
`SUPPORT_FLOOR` turns out to be irrelevant to the grounding verdicts — 7 of 11 at every value
from 0.05 to 0.20 — because it gates *support counting*, not the magnitudes the AUROC test reads.
The prominence window is not irrelevant: 20 / 40 / 80 cm⁻¹ give 6 / 7 / 8 grounded axes. The
chosen value is the middle of the three and was fixed before the sweep, and a referee can see that
the wider setting would have helped the phase. **The axis count is a soft number with a ±1
dependence on a measurement choice, and it should be quoted as such.**

**B4. Open-set validation is circular in a way the report understates.** Four of the six negative
kinds are perturbations from the *same module* used in Step 11's robustness study, at more extreme
levels. So the engine is shown to reject spectra corrupted by processes it was separately shown to
be robust against — an internally consistent story, but not evidence about novel chemistry.
§12.3 says this; the headline AUROC 0.921 does not, and the two travel separately once anyone
quotes the number.

**B5. The Split A / Split B framing hides that molecule identification is weak.** Top-1 of 0.605
on a 154-way problem is well above chance, but it is not identification, and the phase's headline
numbers (0.845, 0.921, 7/11) are all from other tasks. The report says this in §12.1. It should
probably say it in §0.

**B6. LSM and CSM are nearly indistinguishable.** Clean class top-1: 0.848 vs 0.845. Retention:
0.923 vs 0.935. The CSM layer's advantage over the layer beneath it is ~0.012 in retention and
*negative* in clean accuracy. Phase 02 built the CSM layer to merge redundant LSMs, and 48 of 49
CSMs are single LSMs, so the two representations are nearly the same object — which the numbers
confirm. **The phase's real finding is that the *motif* layer helps, not that the *consensus*
step does.** Nothing in Phase 05 justifies the CSM layer over the LSM layer; the justification, if
there is one, is Phase 02's and rests on interpretability rather than on these numbers.

**B7. Confidence is calibrated on Split A and applied to Split B.** The engine reports one
confidence, fitted to molecule-level correctness where the molecule is present. Applied to a
spectrum whose molecule is absent from the bank, that number is calibrated for the wrong event.
The Split B calibration column exists in the benchmark but the engine does not switch between
them, and it has no way to know which regime it is in.

**B8. `min explained variance = 0.206`.** Some corpus spectra are barely explained by the atlas
that was built from them. The report notes it and moves on. Which molecules those are, and
whether they are systematically one class, is not investigated — and if they cluster, that is a
statement about the atlas rather than about inference.

## C. Claims checked against what was actually computed

| claim in the report | verdict |
|---|---|
| "nothing upstream is refitted" | **holds** — only reads of frozen artifacts; fingerprint gate aborts on mismatch |
| "no cross-modality experiment" | **holds** — no SERS path is reachable in the phase's code |
| "geometry not used in inference" | **holds** — the geometry file is loaded and never consumed |
| "engine deterministic bit-for-bit" | **holds** — verified in-run (G12); NNLS is deterministic and no RNG touches the inference path |
| "molecule top-k undefined under Split B" | **holds** — the molecule is absent from every fold bank by construction |
| "the profile sits beside CSM inference rather than replacing it" | **holds** — retrieval reads the activation vector, never the profile |
| "7 of 11 axes grounded" | **holds** at the declared AUROC ≥ 0.70 threshold, which was set before the run; ±1 under the prominence-window sweep (B3) |
| "CSM more robust *and* more discriminative than raw" | **holds** on grouped CV; reverses in-sample, and both are reported |
| "mean 3.8 axes per CSM" | **holds** at `SUPPORT_FLOOR = 0.10`; see B3 |

## D. Errors found and fixed during the phase

These were caught inside the phase, mostly by inspecting outputs rather than by tests, and each
changed a reported number.

1. **Platt calibration reported 0.605 for every spectrum.** Selecting on ECE handed the phase to
   a constant predictor at the base rate. Found by reading Figure 12 and noticing three different
   molecules with identical confidence. Selection moved to Brier with discrimination and sharpness
   floors; temperature scaling's internal objective moved from ECE to Brier for the same reason.
2. **The Mahalanobis OOD channel centred negatives on their own mean**, asking how unusual each
   negative was among negatives. AUROC 0.176 → 0.203 after passing the in-domain mean. Still
   inverted, for the structural reason in §4 — the fix did not rescue the channel, it made the
   inversion interpretable.
3. **The first CSM → axis map loaded every axis on nearly every CSM** (8 of 11 axes had all 49),
   driving every specificity weight to ~1.00. Local prominence is non-zero almost everywhere in a
   676-bin window. Fixed by requiring a *diagnostic band* from the frozen registry inside the
   window before an axis can load at all.
4. **G11 originally compared in-sample accuracy**, where raw spectra win by self-matching. That is
   risk R-10 and the gate failed for the wrong reason. Replaced with the molecule-grouped
   comparison.
5. The report claimed **axis rows sum to at most one**; overlapping windows make the true mean
   1.068. Claim corrected and an overlap table added rather than forcing a partition.

## E. What would change the conclusions

- **A prominence-window choice outside 20–80 cm⁻¹ that moved the grounded count further** would
  weaken §7 (B3); inside that range the count is 7 ± 1.
- **A real held-out chemistry class that the engine failed to reject** would falsify the open-set
  claim as a claim about novelty (B4).
- **An LSM-vs-CSM comparison favouring LSM on interpretability metrics** would remove the last
  reason to keep the consensus layer in the inference path (B6).

## F. Recommendations, in priority order

1. ~~Sweep `SUPPORT_FLOOR`.~~ **Done** — see B3. It changes nothing; the prominence window is the
   parameter that matters, and it moves the grounded count by ±1.
2. **Add a held-out-class open-set experiment.** Withhold e.g. `sterol_steroid` from the atlas
   entirely and test rejection. This is the missing experiment, not a refinement.
3. **Quote the unsaturation pair together, always** — 0.534 primary, 1.000 secondary — and state
   the label defect in the same sentence (B2).
4. **Narrow the phase's headline claim** from "the canonical inference engine works" to "frozen
   *motif* projection supports calibrated chemistry-class inference with verified provenance;
   molecule identification remains a shortlist" (B5, B6).
5. **Fit a second calibrator on Split B outcomes** and have the engine report both, or report the
   Split A confidence explicitly as *conditional on the molecule being represented* (B7).
6. **Investigate the low-EV tail** (B8) — one table, and it belongs to the atlas rather than to
   this phase.

## G. Overall

The engine does what the brief asked and the negative results are reported at full strength: one
gate fails, four axes fail their primary test, two rejection channels run backwards, and molecule
identification is a shortlist rather than an answer. The strongest result — 0.845 chemistry-class top-1 on molecules
the atlas has never seen, with 0.935 robustness retention and zero broken provenance chains — is
well-evidenced and honestly bounded.

B3 is now closed. The thing a referee would still make the authors do is B4 — a real novelty
holdout — and it is not expensive. Until it exists, "open-set rejection" in this
phase means *rejection of degraded and structureless spectra*, and the report should use that
phrase.
