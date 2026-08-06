# GAIRA V7 Phase 04.5 — Scientific Audit

Adversarial review of a **negative** result. A negative result is easier to reach carelessly
than a positive one, so the audit's job is the reverse of usual: to check the layer was given a
fair chance before being discarded.

**Verdict: the negative result is sound and the layer was fairly tested — with two caveats
that a referee would press, and one place where the conclusion is stronger than the experiment
strictly supports.**

---

## 1. Was Meta Components given a fair chance?

| challenge | answer |
|---|---|
| *Was the right thing factorised?* | Yes — `A` is spectra × frozen CSM activations, exactly as specified. Not spectra, not a similarity matrix, not a graph. |
| *Was `K` searched widely enough?* | Eight values, 2–12. The downstream diagnostic shows the best achievable class retrieval at **any** `K` is 0.677 vs CSM's 0.856, so the conclusion does not depend on the `K` chosen. **Fair.** |
| *Was the geometry prior implemented correctly?* | The Laplacian penalty is one-sided and acts on `H`'s columns, which are the CSMs. Correct. But see W2. |
| *Was the selection rule rigged against it?* | The opposite. Stability carries 0.40 of the Pareto weight and reconstruction 0.14 — a weighting that *favours* exactly the smooth low-information solutions Meta Components produce. It still lost. **Fair.** |
| *Were the splits identical across representations?* | Yes — same frozen folds, same query sets, same metrics. |
| *Was the robustness study capable of showing a benefit?* | Yes. 12 perturbations × 5 levels × 4 representations = 240 conditions, curves normalised by each representation's own clean performance so a lower-starting-but-flatter representation can win. Meta Components **did** win on two of three AURC measures. |

**On the central question, the layer was tested fairly and lost.**

## 2. Strengths

**S1. The informativeness floor is the right correction and it was applied before the verdict.**
Without it the run would have concluded "partial benefit → augment" on 3 of 8 axes — and it did,
on the first pass, before the floor was added. Replicate consistency, activation stability and
robustness AUC are all maximised by a representation that says the same thing about everything.

**S2. The `K` diagnostic closes the obvious escape route** and was explicitly excluded from
selection, which would have been circular.

**S3. Both variants were run at every `K`**, so "the geometry prior might have helped" is
answered rather than assumed.

**S4. The failure mechanism is diagnosed, not just observed** (§6 of the report): 9.3 active
CSMs per spectrum, information in *which* CSMs fire rather than in shared co-activation.

**S5. A defect in the geometry prior was found by its own unit test and fixed.** The Laplacian
was symmetric-normalised, whose null space is `D^½1` rather than `1` — so a uniform loading was
penalised in proportion to each CSM's degree, a hidden preference for loading on low-degree CSMs
that nothing in the design called for. Replaced with the combinatorial Laplacian, which has
exactly the claimed semantics. The conclusion did not change (Pareto 0.7211 → 0.7193 for the
regularised variant, same `K`), but the prior now is what the report says it is.

**S6. This is the second independent failure of the same kind.** Phase 03 themes and Phase 04.5
Meta Components fail identically — both trade identity for stability, both are selected by
criteria that reward smoothness. Two failures with one mechanism is a finding about the
architecture.

## 3. Weaknesses

**W1. NMF is the only factorisation tried.** The brief specified hierarchical NMF, so this is
compliance rather than oversight — but the conclusion is stated as "hierarchical abstraction
does not improve over CSM" when what was tested is "non-negative rank-K factorisation of the
activation matrix does not". Those are not the same claim.

**W2. `λ = 0.1` was not swept.** Geometry regularisation changed nothing at that value, and the
report treats that as "the prior doesn't help". It may simply be too weak to do anything. A
sweep would cost little and would close the question.

**W3. The informativeness floor at 0.50 is a judgement call made after seeing that the layer
retains 0.185.** The report says the conclusion is unchanged for any floor above 0.46 — true,
but the floor was still chosen in the presence of the number.

**W4. Perturbations are synthetic.** Physically motivated, but no real multi-instrument data
was used.

**W5. The robustness advantage is real and is dismissed rather than explained.** Meta
Components genuinely retain more activation stability under every perturbation. The report
attributes this entirely to low information, which is consistent — but no experiment separates
"stable because uninformative" from "stable and uninformative", and a controlled test exists
(E3 below).

## 4. Where the conclusion outruns the experiment

**U1. "Hierarchical abstraction does not improve over CSM."** What was shown: *this* NMF, on
*this* activation matrix, at *these* ranks, on *this* corpus. A referee would ask for the
weaker phrasing. **Recommend: "second-order non-negative factorisation of the CSM activation
matrix does not improve over the CSM layer on this corpus."**

**U2. "The geometry prior changed nothing material."** Supported at λ = 0.1 only.

**U3. MC-01's bridge loading is described as possibly meaningful.** Five of its six top CSMs
being Phase 02.5 bridges is at least as consistent with the factorisation collecting
poorly-defined dictionary elements as with a real "shared skeletal chemistry" programme, and
nothing here distinguishes those.

## 5. Likely reviewer criticisms

**R1.** *"You chose an informativeness floor after seeing the result."* — The strongest attack.
Answer: the floor's justification is independent (a layer that augments must add usable
information), the same trap was documented in Phase 03 before this phase ran, and the
conclusion holds for any floor above 0.46. It remains a post-hoc constant.

**R2.** *"Your Pareto weights drove K to 3, where information is gone. Isn't the negative
result an artefact of your own weighting?"* — Answered by the K diagnostic: no K beats CSM.

**R3.** *"NMF on a matrix with 9.3 non-zeros per row of 49 — was rank-3 ever plausible?"* — A
fair point that argues the experiment was underpowered by design rather than that the layer
failed. The honest answer is that the sparsity of `A` was known before the phase ran and should
have been flagged as a prior expectation.

**R4.** *"Meta Components beat CSM on class-retrieval robustness. Why is that not a result?"* —
+0.008 AURC against −0.464 clean top-1. The trade is not close.

## 6. Recommended experiments

**E1.** Sweep `λ` over three orders of magnitude to close W2 and U2. Cheap.

**E2.** Try sparse PCA, ICA and archetypal analysis on the same `A`, to convert U1's weak claim
into the strong one. Moderate cost.

**E3.** The controlled test of W5: build a deliberately uninformative 3-dimensional
representation (random projection of `A`, or `A`'s top-3 PCs) and measure its robustness AUC.
If it matches Meta Components', "stable because uninformative" is established rather than
inferred.

**E4.** Repeat on a second corpus.

## 7. Remaining risks

| risk | severity | status |
|---|---|---|
| conclusion stated more broadly than tested (U1, W1) | **medium** | fix by rephrasing; E2 to earn the broad claim |
| informativeness floor is post-hoc (W3, R1) | medium | justification independent; conclusion robust above 0.46 |
| λ untested beyond one value (W2, U2) | low | E1 |
| robustness advantage dismissed rather than explained (W5) | low | E3 |
| single corpus | medium | E4 — inherited from every phase |

## 8. Verdict

The experiment is well-constructed, the selection rule was if anything biased in the layer's
favour, and the escape routes were closed. **The recommendation to discard is sound.**

Two corrections before this is written up anywhere external: narrow the claim from
"hierarchical abstraction" to "second-order non-negative factorisation of the CSM activation
matrix", and either sweep `λ` or drop the statement that the geometry prior does not help.

**Approved as a negative result. The CSM layer remains canonical.**
