# GAIRA V7 — Phase 06.5 Scientific Audit

Every conclusion in this phase, sorted by how much weight it can bear. The brief asks for
exactly this: which conclusions are **strongly supported**, **weakly supported**, or
**unsupported**.

---

## A. Strongly supported

These would survive a hostile referee.

**A1. No preferred cluster count exists in the CSM manifold.**
Zero of seven internal indices has an interior optimum across K = 2…30 and four algorithms.
Silhouette is perfectly monotone increasing (Spearman +1.000), neighbour preservation perfectly
monotone decreasing (−1.000), membership entropy perfectly monotone increasing (+1.000). This is
not a marginal call; it is as clean a negative as this kind of analysis produces. **K = 16 is not
special**, and any future claim that the geometry "recovers 16 classes" is unsupported by this
phase.

**A2. Chemistry dominates acquisition in the global distance structure.**
PERMANOVA R²: fine chemistry 0.452, excitation 0.118, source 0.040 — a 3.8× and 11× margin, with
999 permutations. Chance-corrected AMI tells the same story with a different statistic (0.703 vs
0.292 and 0.187). Two independent methods, same ordering, large margin.

**A3. Continuous coordinates carry more information than hard cluster ids.**
k-NN preservation 0.446 vs 0.237 and effective rank 10.56 vs 9.01. Both are label-free, both are
computed on the same partition, and the margin is large. This directly answers the brief's
Section 5 question.

**A4. The coordinates do NOT significantly improve retrieval.**
Molecule Δ+0.016, 95% CI [−0.005, +0.039], McNemar p = 0.180. Chemistry Δ+0.003, p = 1.000. The
fusion-weight sweep independently shows the effect is a shallow bump of ~0.006 that collapses to
−0.160 at w = 1. Three lines of evidence, one conclusion. **This is the finding the architecture
recommendation rests on, and it is the best-evidenced negative in the phase.**

**A5. The graph is modular against a proper null.**
Modularity 0.718 vs a degree-preserving configuration null of 0.347 ± 0.009 — z = 40. Not
marginal.

**A6. Five clusters are chemically pure without any label having been used.**
Sterols, saccharides, proteins and nucleic-acid polymers emerge at 100% fine-class purity from a
label-free procedure. This is a real result and it is the strongest positive evidence that the
CSM representation encodes chemistry.

---

## B. Weakly supported

True as far as measured, but any of these could move with a different corpus, linkage, or K.

**B1. "The space is modular and tree-like."**
Cophenetic correlation 0.870 supports tree-likeness, but average linkage is *known* to produce
high cophenetic correlation by chaining, and average linkage is also what the label-free rule
selected. The claim is partly an artefact of the linkage that was chosen. Ward's cophenetic
correlation is materially lower. **Quote the verdict as "consistent with a hierarchy", not as
"the space is a tree".**

**B2. "Four clusters are acquisition-confounded."**
This depends entirely on the classification thresholds — source purity exceeding chemistry purity
by more than 0.25 over baseline. The thresholds were declared before inspection, which is what
makes the claim admissible, but they were not swept. A different threshold gives a different
count. The *direction* is safe; the number 4 is not.

**B3. "The emergent partition agrees with the ontology at AMI 0.703."** — *raised here, then
measured.*
The concern was that this is conditional on K = 16, which A1 establishes is arbitrary. The
agreement curve was then computed across all 14 values of K and it confirms the concern:
**AMI rises monotonically and peaks at K = 24 (0.607), not at 16 (0.578, mean over four
algorithms).** There is no sense in which the geometry "recovers" sixteen classes; it agrees
better the more clusters it is allowed, exactly as a finer cut through a continuum would.
The headline 0.703 belongs to the canonical partition (average linkage, K = 16) and is one point
on a curve with no peak. **It must never be quoted without that condition**, and the promotion of
this item from "weakly supported" to "measured and conditional" is the single most useful thing
the audit did.

**B4. "35% of molecules bridge communities."**
Depends on k = 5 in the k-NN graph and on the greedy-modularity community assignment, neither of
which was swept. The qualitative claim — that boundaries are soft — is supported by the bimodal
distance distribution independently; the percentage is not robust.

**B5. Cluster composition statements generally.**
Seven of sixteen clusters have ≤ 2 molecules under average linkage. The composition table is
therefore describing nine real clusters and seven fragments, and any statement about "16
emergent clusters" overstates what was found.

---

## C. Unsupported — stated, but the evidence does not carry them

**C1. Intrinsic dimension.**
Levina–Bickel gives 1.40, the correlation dimension gives 4.68 — a 3.3× disagreement. The phase
reports both and declares them in disagreement, which is the honest handling, but **no intrinsic
dimension should be quoted from this phase at all**.

**C2. "The coordinates are interpretable."**
The Section 9 criterion failed: 5 of 16 clusters are chemically nameable. Calling a 16-coordinate
system interpretable when two-thirds of its axes are either acquisition-confounded or too small
to characterise would not survive review.

**C3. Any per-cluster chemical reading of the four acquisition-confounded clusters.**
The global verdict (A2) does not license a local one. C2 is 92% one source library and 45% one
chemistry; whatever it is describing, it is not purely chemistry.

**C4. That the geometry "validates" the curated ontology.**
It does not, and the phase must not be read that way. AMI 0.703 at an arbitrary K, with
completeness (0.812) exceeding homogeneity (0.732), says the two partitions overlap
substantially and disagree systematically. That is a comparison, not a validation.

---

## D. Errors found and fixed during the phase

| # | defect | consequence had it stood | fix |
|---|---|---|---|
| 1 | **Silhouette was NaN across the entire 56-row sweep.** Floating-point residue of ~1e-11 on the distance-matrix diagonal made scikit-learn reject it, and a bare `except` turned the rejection into a NaN. | Section 1's central question — does any K stand out? — would have been answered without its most important index. The monotonicity finding (A1) would not have been made. | Diagonal forced to exactly zero; the bare `except` removed so an uncomputable index raises instead of returning NaN. Gate G6b added. |
| 2 | **The canonical partition was selected as argmax bootstrap ARI over all K, and chose K = 4.** | Sections 2 and 5 would have described a 4-cluster partition, and the phase would have concluded the manifold has four biochemical regions. | Replaced with a declared convention (K = 16 for comparability), and the monotonicity test added to show that *no* K is distinguished. This is the **fifth** appearance of the P-18 stability-without-informativeness trap in V7. |
| 3 | **The Section 9 retrieval criterion required only that B > A**, with no significance test. | The phase would have recommended **Option C** — an architecture change — on +0.016 molecule top-1, i.e. six spectra, with a confidence interval crossing zero. | McNemar plus a molecule-level bootstrap CI added, and the criterion now requires significance. The recommendation flipped to Option A. |
| 4 | Molecules with a single spectrum crashed the Split A ranking (66 of 154 leave the bank entirely when their only spectrum is held out). | An IndexError, or — worse — a silent drop from the denominator that would have inflated MRR. | Counted as misses at rank *n*+1, matching Phase 05, with the count reported. |

Defect 3 is the one that matters. **Without it this phase would have recommended changing the
GAIRA architecture on a non-significant difference.**

---

## E. What a referee would ask next

1. ~~Report the agreement curve across K.~~ **Done** — see B3. It undermines the headline: AMI
   is monotone in K and peaks at 24.
2. **Sweep the classification thresholds** in Section 2 (B2), so the count of
   acquisition-confounded clusters comes with an error bar.
3. **Repeat the composition analysis under Ward linkage** (B1, B5) as a sensitivity check —
   average linkage's seven near-singleton clusters are a linkage artefact as much as a finding.
4. **Estimate the corpus size at which +0.016 would become significant.** If it is 4× the current
   corpus, that is an argument for Phase 11 corpus expansion; if it is 40×, the coordinates are
   not worth revisiting.

None of these would change the recommendation. All four would make it easier to defend.

---

## F. Assessment of the recommendation itself

**Option A is the right call and it is well-evidenced.** The decisive fact — that the coordinates
do not significantly improve retrieval — rests on three independent lines (paired McNemar, a
molecule-level bootstrap CI, and a fusion-weight sweep), all agreeing.

The recommendation is also *conservative in the right direction*. Adding a coordinate layer would
add a clustering algorithm, a cluster count, a kernel, a temperature and a fusion weight to a
frozen inference engine — five new degrees of freedom — in exchange for an effect that cannot be
distinguished from zero. The asymmetry between what is risked and what is gained is large, and it
points the same way as the significance test.

**What the phase genuinely established** is more interesting than the architecture decision: the
CSM manifold is organised by chemistry (R² 0.452), is modular against a proper null (z = 40),
contains five label-free chemically pure neighbourhoods — **and has no natural scale at which to
cut it.** That last point is the scientifically novel one, it is strongly supported, and it
should shape how GAIRA talks about spectral "classes" in future: as cuts through a continuum, not
as discoveries.

**Confidence that Option A is correct: 9 / 10.** B3 has now been measured and the conditional is
explicit, which removes the original deduction; the remaining one is for B1/B5 — the composition
table depends on a linkage choice that produces seven near-singleton clusters, and a Ward-linkage
sensitivity check has not been run. That would not change the recommendation, which rests on the
significance test, not on the composition.
