# GAIRA V7 — Phase 07 Scientific Audit

Harder than the implementation. Every claim below is attacked before it is allowed to stand.

---

## A. Attempting to reject each claim

### A1. "Nine biochemical programmes naturally emerge."  → **survives, weakened**

*Attack.* Nine is a tie-break, not a peak. The objective is flat across K = 8–11 (0.858, 0.859,
0.839, 0.859) and the rule picked the smallest within 0.01. Change the tolerance to 0.005 and
K = 11 wins; change it to 0.02 and K = 8 wins.

*Verdict.* The claim survives only as *"nine, under a declared tie-break over a flat region"*.
The stronger claim — that the data pick nine — is **unsupported**. What the data do support is
that eligible K spans 4–12 and that everything below 4 fails the informativeness floors.

*What would settle it.* Nothing available. This is a 154-molecule corpus and the objective
differences across K = 8–11 are far smaller than the bootstrap spread.

### A2. "The programmes are stable."  → **survives**

*Attack.* Bootstrap stability is inflated by resampling 85% of spectra from a corpus where 66
molecules have a single spectrum and the rest have two or three — resamples overlap heavily.

*Counter.* Three independent stability measures agree: bootstrap 0.972, seed refit 0.957, and
**molecule-grouped fold withholding 0.979**. The third is immune to the objection — an entire
fold of molecules is removed. And 0 of 9 programmes falls below 0.70, so the mean is not hiding
a fragile programme.

*Verdict.* **Strongly supported.**

### A3. "The programme names mean something."  → **survives, with a named weakness**

*Attack 1.* The descriptions come from a template that consults the frozen broad ontology. Give
it any loading vector and it will produce chemistry words. That is naming, not discovery.

*Counter.* The template refuses: a programme whose top axis falls below 15% is described as
*diffuse*. None did here, but the branch exists and is unit-tested. More importantly the
descriptions were checked against the *evidence* — P3's three axes really are fatty acid,
acylglycerol and phospholipid, which really are the same broad superclass.

*Attack 2.* **Two pairs of programmes get the same description.** P4 and P6 are both "protein and
amino-acid"; P1 and P7 are both "nucleic". If the naming cannot distinguish them, it is not
carrying information.

*Counter.* Their loadings are distinct (free amino acid vs peptide protein; purine vs pyrimidine)
and their overlap is below the redundancy floor. The template's vocabulary is coarser than the
programmes.

*Verdict.* **Supported for the loadings, weak for the names.** The report should quote programme
loadings, not programme names, wherever precision matters.

### A4. "Seven of nine programmes are genuine multi-chemistry compressions."  → **survives, but the threshold is arbitrary**

*Attack.* "Genuine" is defined as top axis < 50% of the loading. At a 40% threshold only three
would qualify (P0 35%, P3 30%, P8 33%); at 60% all nine would.

*Verdict.* **Weakly supported as stated.** The defensible version is the distribution itself —
top-axis shares run 30% to 55% — and the observation that the three *most* composite programmes
(lipid, energy metabolism, carbohydrate) are exactly the groupings the brief predicted. That
coincidence is the real evidence, and it was not engineered.

### A5. "BSV2 compresses 1.78× while retaining 88% of held-out chemistry prediction."  → **survives**

*Attack.* 0.667 vs 0.755 is a loss of 0.088 — 33 spectra. Calling that "retention 0.883" flatters
it.

*Counter.* Both numbers are reported. The floor was 0.50, declared before the run, and 0.883
clears it by a wide margin. The compression is real: 16 → 9.

*Verdict.* **Supported**, provided the absolute numbers travel with the ratio.

### A6. "BSV2 explains chemistry better than PCA."  → **REJECTED**

*Attack.* Held-out top-1 0.667 vs 0.661 is **two spectra**. No significance test would call that
a difference. And normalised mutual information is **2.075 for BSV2 against 2.649 for PCA** — the
control wins, decisively, on the information measure.

*Verdict.* **Unsupported. The claim must not be made.** BSV2 does not out-explain PCA. Its case
is non-negativity, additivity and nameable axes — properties PCA cannot have — and the report
now says exactly that.

### A7. "Programmes are not redundant."  → **survives**

Max pairwise overlap 0.400, mean 0.195, floor 0.90. The overlap matrix is in the artifacts.
**Supported.**

### A8. "No programme dominates."  → **survives**

Top programme usage 0.221 against a 0.60 ceiling; Phase 04.5's discarded MC-03 dominated 0.62.
**Supported.**

### A9. "The factorisation saw only Chemistry Evidence."  → **survives**

Enforced by a test that greps the model-fitting section of the run script for seven forbidden
symbols. Spectra appear later, for noise propagation and explanation only. **Supported.**

### A10. "Information retention is 0.818."  → **survives, with a caveat that matters**

*Attack.* Global EV averages over axes with wildly different evidence mass. One axis —
`nucleic_acid_polymer` — reconstructs at EV **0.064**, and two more below 0.20.

*Counter.* Per-axis reconstruction is reported in full, and the pattern is interpretable: the
badly reconstructed axes are the ones with 3, 7 and 10 spectra. BSV2 declines to fit noise.

*Verdict.* **Supported globally, with a per-axis caveat that must accompany it.** A downstream
user who cares about nucleic-acid polymers should not use BSV2 for that purpose.

---

## B. Defects found and fixed during the phase

| # | defect | consequence had it stood | fix |
|---|---|---|---|
| 1 | **The rule as first written selected K = 16 over a 16-dimensional input.** NMF learns a permutation of the identity: EV 1.000, bootstrap 0.997, every programme equal to one chemistry class. | The phase would have reported a "perfect" BSV2 that was the Chemistry Evidence layer with the columns shuffled, and gate G10 would have been the only thing standing between that and adoption. | Two floors added — K ≤ the input's effective rank, and no programme above 0.90 single-axis share — both restatements of constraints the brief states explicitly. Recorded as a specification bug, not a threshold change. |
| 2 | **Sparse NMF scored EV −0.401 at every K** on a single untuned α = 0.2 that drives the loadings to zero. | One of six required candidate families would have been excluded from the benchmark for a reason that is a hyperparameter, and the report would have implied sparse NMF cannot fit this data. | The penalty is swept over four values; the family is now eligible at two of them. |
| 3 | **The rule's winner has signed loadings** and P-02 says non-negativity is not optional. | A silently-adopted semi-NMF would have put a subtractive programme layer into the architecture without the question ever being asked. | The fully non-negative constrained optimum is computed and reported alongside, with its cost as a number, and the question is put to the decision gate. |

Defect 1 is the serious one. It is the **sixth** appearance in V7 of a selection rule maximised by
a degenerate answer, and the fifth was two phases ago.

---

## C. Conclusions by strength

**Strongly supported** — programme stability (three independent measures, 0 of 9 unstable);
low redundancy; no dominance; generalisation to held-out molecules; the factorisation's input
isolation; non-negativity of activations; that the lipid and energy-metabolism programmes are
genuine multi-chemistry compressions.

**Weakly supported** — that the number is specifically nine (a tie-break over a flat region);
that seven of nine are "genuinely composite" (threshold-dependent); the programme *names* (two
pairs collide).

**Unsupported / rejected** — that BSV2 explains chemistry better than PCA (rejected: −0.574 on
mutual information, +0.006 on top-1); that nucleic-acid-polymer chemistry survives the
compression (EV 0.064).

---

## D. What a referee would ask next

1. **Test K = 9 against K = 11 with a paired significance test**, not a tie-break. The objective
   differences are inside the bootstrap spread and the phase does not say so with a p-value.
2. **Report BSV2 against PCA with a confidence interval.** The +0.006 should be shown to cross
   zero explicitly rather than left to the reader.
3. **Decide P-02 at the loading level before Phase 08 builds on BSV2.** Shipping a subtractive
   programme layer and then discovering it is inadmissible would be expensive.
4. **State the nucleic-acid-polymer loss in the decision gate**, not only in the report.

---

## E. Overall

The phase is honest about a result that is good but not as good as the headline suggests. Nine
programmes, stable and interpretable, reconstructing 82% of the chemistry evidence at 1.78×
compression, is a real finding — and it sits alongside a rejected claim (BSV2 vs PCA), an
arbitrary K within a flat region, and one chemistry axis that does not survive.

**Confidence that BSV2 is a sound foundation for Phase 08: 7 / 10.** Deductions: the P-02
loading-sign question is open (−1), K = 9 is a tie-break rather than a peak (−1), and the PCA
control is uncomfortably close (−1).
