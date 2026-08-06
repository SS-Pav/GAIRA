# GAIRA V7 — Phase 03: Emergent Biochemical Theme Discovery

**Frozen inputs verified:** atlas `09ed804a40836f4a05a91ba10900cded`, LSM registry
`208482d6f7178b5b8f16cace91be55b0`, CSM dictionary `0b4aa550ccefed3edabdbde5bae11c8d`,
Phase 02.5 geometry (Wasserstein primary metric). Nothing upstream was refitted.

**Result:** `K = 5`, archetypal analysis. **4 themes accepted, 1 rejected.** 25 member CSMs,
15 bridges, 9 poorly explained. Theme fingerprint in `PHASE_STATE.json`.

---

## 1. Executive summary

Five soft-membership models were swept over `K = 2–15` on the 49 frozen Consensus Spectral
Motifs, with **no chemistry label visible at any point** during discovery. Selection used
label-free criteria and a band-based admissibility veto. Labels were revealed afterwards, once,
for interpretation and agreement.

**Archetypal analysis at K = 5 was selected.** Four themes survive validation:

| | theme | dominant bands (cm⁻¹) | CSMs | bootstrap | confidence |
|---|---|---|---:|---:|---:|
| Theme-01 | carboxyl / ester carbonyl + amide backbone | 790, 1156, 1236, 1366, 1584 | 16 | 0.69 | 0.76 |
| Theme-02 | **aliphatic chain + unsaturated chain** | 1064, 1130, 1298, 1442, 1658 | 17 | **0.96** | **0.90** |
| ~~Theme-03~~ | ~~aliphatic chain + amide backbone~~ | 650, 938, 1332, 1446, 1658 | 23 | 0.59 | **REJECTED** |
| Theme-04 | aliphatic chain + polar skeletal backbone | 526, 850, 1118, 1328, 1458 | 19 | 0.77 | 0.79 |
| Theme-05 | heterocyclic / conjugated ring + sulfur / thiol | 640, 674, 828, 1006, 1206 | 16 | 0.62 | 0.71 |

**Theme-02 independently recovers the one thing Phase 02 established.** Its bands are
1064 (C–C trans chain), 1130, 1298 (=C–H in-plane bend), 1442 (CH₂ scissoring), 1658 (cis C=C)
— the acyl-chain-plus-cis-unsaturation signature that was the single accepted CSM merge. Two
layers built on different objectives found the same chemistry, which is the strongest internal
corroboration in the phase. It is also the most stable theme by a wide margin (bootstrap 0.96,
leave-one-out 0.93).

**Theme-03 was rejected** for bootstrap recovery 0.59, below the pre-registered floor of 0.60 —
despite having the *most* supporting CSMs (23). Membership breadth is not evidence.

**The theme layer adds value, but not much.** Coarse-chemistry retrieval from theme
coordinates is 0.237 against 0.155 from raw CSM activations, on a chance rate of 0.101. That is
a real improvement (risk R-11 is not realised) and it is small.

---

## 2. Scientific motivation

Phase 02 asked which motifs are interchangeable and found one pair. Phase 02.5 asked how they
are related and found a low-dimensional continuum with one top-level hydrophobic/polar
bipartition, four validated neighbourhoods, seven bridges and five isolates. Neither produced a
coordinate system.

Phase 03 asks the remaining question: **what small set of latent chemistries explains that
geometry?** The answer becomes the semantic axes of the Biochemical State Vector, so it has to
be small, stable, chemically nameable, and honest about what it cannot place.

The objective is emphatically not to maximise the number of themes. If five is the answer,
five is what is reported; if a CSM cannot be placed, it stays unplaced.

## 3. Methods

### 3.1 What was frozen

49 CSM spectra (676 bins) · the CSM registry with its provenance · the Phase 02.5 primary
metric geometry lifted to CSM level · diffusion coordinates · the k-NN adjacency · bridge and
isolate annotations · the 10 geometry priors, used as **evidence, never as truth** — no prior
was imposed on the fit, and the themes were free to contradict them.

### 3.2 Five candidate models

| model | what it assumes | why it is a candidate |
|---|---|---|
| **archetypal** | themes are convex extremes of the CSM cloud | memberships are convex by construction; archetypes are spectra that could exist; poles are the right object for a continuum |
| sparse NMF | themes are non-negative spectral parts | the classical decomposition |
| relational fuzzy c-means | themes are regions of the frozen geometry | the only candidate whose objective is the geometry |
| diffusion-space GMM | themes are regions of the manifold | models a continuum rather than balls in 676-D |
| graph-regularised NMF | spectra + neighbourhood structure | the hybrid |

### 3.3 The label firewall, and how it was reconciled with the architecture

`VALIDATION_AND_DECISION_RULES.md` §4 lists *mutual information with chemistry* among the `K`
criteria; the brief forbids human ontology during discovery. Both were honoured by splitting
the list rather than ignoring either: **`K` is decided on label-free criteria**
(information retained, held-out reconstruction, stability, spectral coherence, compression,
calibration, membership sparsity) plus a **band-based admissibility veto**, and mutual
information is computed *post hoc* as evidence. Admissibility is judged from the themes' own
bands, so it needs no labels.

### 3.4 Admissibility and distinctness

A theme is admissible when the share of its dominant-band prominence carried by its two
strongest vibrational mode families reaches 0.60 — the question a spectroscopist actually asks,
"is most of this spectrum one chemistry?" A theme *set* is admissible only if no two themes
carry the same dominant-family pair.

## 4. Results

### 4.1 Selection

`archetypal` is admissible and distinct at `K = 3` and `K = 5`; `diffusion_gmm` at `K = 3, 4`;
`sparse_nmf` and `graph_regularised_nmf` are degenerate at every `K`. The composite selects
`archetypal, K = 5`.

### 4.2 Validation

| check | result |
|---|---|
| bootstrap stability | mean **0.727**, min 0.625 |
| leave-one-out | mean **0.738** |
| nearest-neighbour consistency | 0.535 against a chance of 0.211 — **2.5×** |
| graph modularity | 0.334 against a degree-preserving null of −0.022, **p < 0.001** |
| reconstruction | theme basis EV 0.549 vs CSM basis 1.000 at 9.8× compression |
| value over the CSM layer | retrieval 0.237 vs 0.155, chance 0.101 — **adds value** |
| source / excitation robustness | 0.657–0.863 across seven fair holdouts |
| ontology agreement (post hoc) | **AMI 0.157**, p < 0.0001 |
| membership roles | 25 member · 15 bridge · 9 poorly explained |

**On the ontology statistic.** NMI is 0.432 against a permutation null of 0.326 — inflated by
having 16 curated classes over 49 items. The **adjusted** mutual information, 0.157, is the
honest number, and it is what this report leads with. The themes agree with curated chemistry
better than chance and nowhere near perfectly, which is the expected result for an unsupervised
layer built on bands rather than labels.

### 4.3 Hierarchy — inferred, not assumed

Four levels emerge, at 2, 3, 4 and 5 groups. The two-group level splits the themes into an
**aliphatic/unsaturated** group and a **polar/ring/carboxyl** group — the same hydrophobic/polar
bipartition Phase 02.5 found independently in the CSM geometry, and the same axis PCA found as
its one reproducible component. Three layers arriving at the same top-level split from three
different objectives is the phase's second-strongest internal corroboration.

### 4.4 Continuous gradients

Five of fifteen theme × diffusion-coordinate pairs are significant gradients
(p < 0.05 against permutation): Theme-02 along DC1 (ρ = −0.58), Theme-04 along DC1 (ρ = +0.57),
Theme-01 along DC3 (ρ = −0.51), Theme-03 along DC3 (ρ = −0.44), Theme-05 along DC1 (ρ = +0.33).
Membership varies smoothly along the manifold rather than switching — the theme layer respects
the continuum Phase 02.5 established rather than discretising it.

### 4.5 Bridges — 15, left as bridges

Fifteen CSMs carry genuinely split membership. They were not forced into one theme. This is
where soft membership earns its place: a hard assignment would have to put each of them
somewhere and would be wrong about a third of the dictionary.

### 4.6 Poorly explained — 9, left unplaced

Nine CSMs are claimed by a theme whose spectrum reconstructs less than a third of them. They
are recorded, not absorbed. Inventing a theme to hold them would be the L-03 failure mode —
a motif borrowing foreign mass.

Note `csm00`, the one accepted Phase 02 equivalence, holds membership **1.000** in Theme-02 and
is still poorly explained by it: the theme captures its chemistry but not its exact shape. That
is a real limitation of a 5-theme basis, recorded rather than smoothed over.

## 5. Theme catalogue

Full records — members, memberships, bridges, evidence, counter-evidence, alternative
explanations, limitations — are in `tables/theme_catalogue_v1.csv` and
`artifacts/theme_registry_v1.json`. Every accepted theme carries recorded counter-evidence; a
theme with none has not been examined, and the registry invariant enforces this.

**Every theme's alternative explanation includes "shared Raman physics rather than shared
biochemistry."** For themes touching the aliphatic family the record states explicitly that
CH₂/CH₃ scissoring is the most common band in biological Raman and that aliphatic membership
alone is weak evidence of lipid biology; for ring-touching themes, that aromatic and
heterocyclic modes are shared by purines, pyrimidines, aromatic amino acids and pigments alike.

## 6. Where the method was corrected

Six defects were found during the run. Each changed the answer, each was demonstrated before
being changed, and each is recorded at the point of change in the code.

| # | defect | evidence | correction |
|---|---|---|---|
| 1 | admissibility gated on assignable band fraction | the window table tiles 450–1800 cm⁻¹, so the fraction was **1.000 for every theme of every model at every K** | prominence-weighted family concentration |
| 2 | mode families counted with equal weight | a fifth-ranked minor band vetoed themes its first two bands define; **no K was admissible under any model** | prominence weighting |
| 3 | degeneracy detected on the fit, not the membership | a run with one theme dominant for all 49 CSMs scored 0.497 information and 0.964 stability and **was selected**; it gave a single community, kNN agreement 1.000 against a chance of 1.000, and AMI exactly 0 | degeneracy tested on `S`, including the fraction of CSMs no theme explains |
| 4 | every theme named "aliphatic chain + …" | CH₂ scissoring near 1440 dominates prominence in nearly every biological Raman spectrum — **the exact "shared CH stretching ≠ lipid biology" trap** | mode families weighted by specificity across the sweep: carboxyl 3.89 > phosphate 2.28 > sulfur 2.19 > unsaturation 1.93 > amide 1.68 > ring 1.40 > skeletal 1.23 > **aliphatic 1.21** |
| 5 | 960–1010 cm⁻¹ was one window mapped to phosphate | the protein theme was named **"phosphate"** on the strength of its 1004 cm⁻¹ band, which in a protein is phenylalanine | split into PO₄ (960–995) and phenyl ring breathing (995–1010) — the same context-free assignment error the Phase 01 investigation caught at 702 cm⁻¹ |
| 6 | source robustness failed for all five themes | driven entirely by the RamanBioLib holdout, which removes **37 of 49 CSMs**; the other seven holdouts recover at 0.66–0.86 | holdouts removing more than half the corpus marked untestable, as Phase 02.5 does for single-source motifs |

Defect 4 is the one worth dwelling on: it is exactly the failure the brief warned about, it
appeared spontaneously, and it was invisible until the theme names were read.

## 7. Limitations

1. **49 CSMs is a small corpus for five themes.** Roughly ten CSMs per theme; every stability
   estimate rests on resampling that.
2. **Bootstrap stability is modest** — 0.727 mean, 0.625 minimum. Two accepted themes sit
   between 0.62 and 0.69, only just above the rejection floor.
3. **The theme basis loses 45% of the reconstruction** (0.549 vs 1.000) at 9.8× compression.
   That is the price of abstraction and it is a real cost, not a rounding error.
4. **Value over the CSM layer is small** (+0.082 retrieval on a chance of 0.101). Real, but not
   a transformation.
5. **Nine CSMs are poorly explained and fifteen are bridges** — half the dictionary is not
   cleanly placed. Honest, and a constraint on what Phase 04 can claim.
6. **Post-hoc naming is band-based and coarse.** Every name is a two-family compound; none
   asserts a molecular class.
7. **`archetypal` was selected over `diffusion_gmm` on a composite**, and both were admissible
   at more than one K. The selection is defensible but not overwhelming.
8. **AMI 0.157 is weak agreement** with curated chemistry. The themes are not a re-derivation
   of the ontology, which is the point — but it also means they cannot be validated against it.

## 8. Implications for Phase 04

Phase 04 (continuous BSV) consumes:

- **`S` (49 × 5)** — the soft membership matrix, rows summing to 1. **BSV dimension = K = 5.**
- **the 5-theme basis** (5 × 676), non-negative, for NNLS projection at inference.
- **`theme_registry_v1.json`** — names, confidences, counter-evidence, gradients, bridge
  annotations, and the poorly-explained list.
- **the hierarchy** — four levels, whose top level is the hydrophobic/polar split.

Four things Phase 04 must carry rather than discard:

1. **Theme-03 is rejected.** The BSV has five axes only if Phase 04 chooses to include a
   rejected one; the default is four accepted themes plus an explicit record of the fifth.
2. **Bridges must keep split membership.** A BSV that hard-assigns 15 of 49 CSMs discards
   exactly what soft membership was retained for.
3. **Poorly-explained CSMs need an uncertainty channel**, not a forced coordinate.
4. **Confidence per theme is heterogeneous** — 0.90 for Theme-02, 0.71 for Theme-05. A BSV that
   treats its axes as equally trustworthy overstates four of five of them.

## 9. Future work

Targeted corpus expansion aimed at the nine poorly-explained CSMs would test whether they are
unrepresented chemistry or corpus limits — the same question Phase 02 left open, now with
specific targets. A second corpus would also make bootstrap stability meaningful at K = 5.

## 10. Decision gate

See the gate returned with this report. All eleven gates PASS.
