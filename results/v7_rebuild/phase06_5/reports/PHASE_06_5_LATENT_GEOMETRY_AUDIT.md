# GAIRA V7 — Phase 06.5: Latent Spectral Geometry Audit

**Status** COMPLETE — 16 of 16 gates pass. **Audit only: no architecture was changed, no
inference layer was created, and Phase 07 was not begun.**
**Scope** Raman only. No SERS, Ag-SERS, serum, plasma, EV, mixture, DART or perturbation dataset
is loaded or cited as validation.
**Frozen inputs verified** LSM `208482d6…` · CSM `0b4aa550…` · engine `20d8bd99…`
**Recommendation** **Option A** — the current architecture is retained. The Continuous Spectral
Coordinates are a scientific instrument, not an inference layer.

---

## 1. Executive summary

The question was: *what biochemical organisation naturally emerges from the validated CSM
representation, independent of the curated ontology?* Four answers, in decreasing order of how
comfortable they are.

**1. The manifold is genuinely organised by chemistry.** PERMANOVA on molecule-level cosine
distances gives fine chemistry **R² = 0.452** against excitation 0.118 and source library 0.040
— chemistry explains **3.8× more** of the distance structure than the strongest acquisition
factor and **11× more** than the source. The emergent partition's adjusted mutual information
with the curated ontology is **0.703**. This is not a weak signal.

**2. But there is no preferred number of clusters. None.** Across 14 values of K and four
algorithms, **zero of seven internal indices has an interior optimum.** Silhouette rises
monotonically to K = 30 (Spearman +1.00), neighbour preservation falls monotonically (−1.00),
membership entropy rises monotonically (+1.00), and bootstrap stability is U-shaped — high at
K = 2–4 because a coarse partition is trivially reproducible, and high again at K = 24–30
because near-singleton clusters reproduce themselves. **K = 16 is not stable in any special
sense**; 2 of 4 algorithms call it stable, with a mean bootstrap ARI of 0.673, which is
unremarkable against its neighbours.

**3. The space is modular and tree-like, yet has no preferred cut height.** Modularity of the
5-NN graph is **0.718 against a degree-preserving null of 0.347 ± 0.009 (z = 40)**; the pairwise
distance distribution is bimodal with a valley depth of 0.80; the average-linkage cophenetic
correlation is 0.870. All three say *structure*. And 35% of molecules have nearest neighbours in
more than one community, which says *the structure has no sharp boundaries*. These are not in
conflict: it is a hierarchy with continuous branch lengths, and any single K is a cut through it
rather than a discovery of it.

**4. The Continuous Spectral Coordinates do not earn a place in inference.** They are
reproducible (0.963 across replicates), robust (mean coordinate cosine 0.958 under six Raman
perturbations), and they beat hard cluster ids on every count that matters — k-NN preservation
0.446 vs 0.237, effective rank 10.6 vs 9.0. **But adding them to CSM retrieval changes molecule
top-1 by +0.016 (95% CI [−0.005, +0.039], McNemar p = 0.180) and chemistry top-1 by +0.003
(p = 1.000). Neither is significant.** Six spectra is not an architecture.

---

## 2. Method, and the one thing that determines whether any of this is meaningful

**No chemistry label enters the construction of any geometry.** Clustering, kernel selection,
temperature selection and coordinate construction are all label-free. Labels are revealed
afterwards, in Sections 2, 4 and 7, as an external validation target.

**The unit of analysis is the canonical molecule (154), not the spectrum (375).** Replicates of
one molecule are near-duplicates in CSM space; clustering them would manufacture stability —
three spectra of glucose always co-assign, which says nothing about whether glucose belongs with
fructose. This is principle P-11 applied to geometry, and it changes the conclusions: a
spectrum-level analysis would report substantially higher stability for the same structure.

---

## 3. Section 1 — cluster stability audit

Six algorithms (Ward, average, complete, spectral, HDBSCAN, affinity propagation) × 14 values of
K, with 40 bootstrap resamples each for both partition-level ARI and per-cluster Jaccard survival.

### Does any K stand out?

| index | Spearman vs K | monotone | interior optimum | best K |
|---|---:|---|---|---:|
| silhouette | **+1.000** | yes | **no** | 30 |
| neighbour preservation | **−1.000** | yes | **no** | 2 |
| membership entropy | **+1.000** | yes | **no** | 30 |
| Davies–Bouldin | −0.723 | no | **no** | 30 |
| bootstrap ARI | +0.288 | no | **no** | 30 |
| mean cluster survival | −0.108 | no | **no** | 30 |
| Calinski–Harabasz | +0.090 | no | **no** | 30 |

**Zero of seven indices peaks in the interior.** Every index is tracking granularity rather than
structure. The pre-registered stability rule (bootstrap ARI ≥ 0.60 **and** minimum cluster
Jaccard ≥ 0.50, with membership entropy ≥ 0.50 to exclude degenerate partitions) marks *some*
algorithm stable at every K from 2 to 30 — which is another way of saying the criterion does not
discriminate.

### Is K = 16 genuinely stable, or merely convenient?

**Merely convenient.** 2 of 4 fixed-K algorithms call it stable; mean bootstrap ARI 0.673 sits
below K = 3 (0.733), K = 24 (0.747) and K = 30 (0.753). The free-K algorithms do not choose 16
either: HDBSCAN returns 3–11 clusters with 30–90 molecules left unassigned, and affinity
propagation returns 9–24 depending on its preference parameter.

**K = 16 is therefore adopted in this report as a reporting convention**, so that Section 7's
comparison with the 16-class ontology is like-for-like. It is not a discovered optimum, and the
report never treats it as one.

> **A methodological note that changed the analysis.** The first version of this phase selected
> the canonical partition as the argmax of bootstrap ARI over all K. It chose **K = 4**, because
> a coarse partition is trivially reproducible. That is the stability-without-informativeness
> trap that principle **P-18** exists to catch, appearing in a fifth place. The rule was replaced
> and the monotonicity test above was added.

---

## 4. Section 2 — cluster composition

Sixteen emergent clusters (average linkage, chosen label-free by bootstrap stability at K = 16).
Classified by a rule whose thresholds were declared before any cluster was inspected:

| kind | count | examples |
|---|---:|---|
| chemically coherent | **5** | C3 sterol/steroid (100% pure, 5 molecules), C13 mono/oligosaccharide (100%, 16), C14 peptide protein (100%, 24), C11 nucleic acid polymer (100%, 3), C12 sterol/steroid (100%, 5) |
| acquisition confounded | **4** | C1 (source purity 0.77, excitation 0.87, chemistry 0.53), C2 (source 0.92 vs chemistry 0.45), C4, C5 |
| unresolved | **7** | mostly n ≤ 2 — too small to characterise |

**Five clusters are 100% chemically pure.** They are real biochemical neighbourhoods that no
label was used to find: sterols, saccharides, proteins, nucleic-acid polymers.

**Four are acquisition-confounded**, and this is the uncomfortable finding. Their source or
excitation purity exceeds their chemistry purity and the corpus baseline by more than 0.25 —
C2 in particular is 92% one source library while only 45% one chemistry. **A global verdict that
chemistry dominates does not license a per-cluster chemical reading**, and Section 4's R² of
0.452 must be quoted alongside this.

Full composition — members, sources, excitations, dominant CSMs, dominant LSMs, dominant bands,
within-cluster variance, nearest clusters, bridges and outliers — is in
`cluster_composition_v1.json`, with the written justification for every classification.

---

## 5. Section 4 — confounding analysis

PERMANOVA on molecule-level cosine distances, 999 permutations:

| factor | levels | marginal R² | pseudo-F | p |
|---|---:|---:|---:|---:|
| **fine chemistry** | 16 | **0.452** | 7.60 | 0.001 |
| broad chemistry | 6 | 0.202 | 7.48 | 0.001 |
| excitation | 7 | 0.118 | 3.26 | 0.001 |
| replicate count | 7 | 0.113 | 3.12 | 0.001 |
| reconstruction EV tertile | 3 | 0.073 | 5.95 | 0.001 |
| source library | 3 | 0.040 | 3.14 | 0.001 |
| intensity tertile | 3 | 0.040 | 3.12 | 0.001 |

**Chemistry dominates.** Every factor is significant at p = 0.001, which at n = 154 means
*detectable*, not *important*; the R² column is what matters. Chance-corrected AMI between the
emergent partition and each factor tells the same story: fine chemistry 0.703, broad chemistry
0.596, then a gap to reconstruction-EV tertile 0.300, excitation 0.292, replicate count 0.237
and source 0.187.

Two caveats stated in full. **Reconstruction EV differs strongly across clusters (ANOVA η² =
0.597, p < 1e-4)** — some clusters are genuinely better explained by the atlas than others, and
that is partly a property of the atlas rather than of the chemistry. And **replicate count has
R² 0.113**, close to excitation: molecules with more spectra sit differently in the space, which
is a residual of the balanced-reference construction rather than a chemical fact.

---

## 6. Section 8 — hierarchical structure

| measure | value | reading |
|---|---:|---|
| modularity (5-NN graph) | **0.718** | vs degree-preserving null 0.347 ± 0.009, **z = 40**, p = 0.005 |
| communities | 9 | sizes 39, 27, 24, 16, 15, 11, … |
| distance-distribution valley depth | 0.80 | **bimodal** — there are genuinely near and far regimes |
| cophenetic correlation (average linkage) | **0.870** | **tree-like** |
| fraction of molecules bridging communities | **35.1%** | the boundaries are soft |
| intrinsic dimension | Levina–Bickel 1.40 vs correlation 4.68 | **the estimators disagree; neither is quoted as the answer** |

**Verdict: modular.** But the modularity coexists with 35% bridging and with the complete
absence of a preferred cut height from Section 1. The consistent picture is a **hierarchy with
continuous branch lengths** — real communities, real nesting, no natural scale.

This extends Phase 02.5's finding one level up. There, the 49 *motifs* formed a continuum with
one defensible bipartition. Here, the 154 *molecules* in motif space form a modular hierarchy
with no defensible cut. The two are consistent: motifs are the vocabulary, molecules are what is
written with it, and the writing has more structure than the vocabulary.

---

## 7. Section 5 — Continuous Spectral Coordinates

Five kernels × six temperatures, selected on **label-free neighbour preservation** among
non-degenerate settings. Selected: **`cosine_power` at T = 1.0**.

| property | continuous coordinates | hard cluster ids |
|---|---:|---:|
| k-NN preservation vs 49-d CSM space | **0.446** | 0.237 |
| effective rank | **10.56** of 16 | 9.01 |
| mean entropy | 0.520 | — |
| replicate reproducibility | **0.963** | — |
| mean bridge score | 0.390 | — |
| mean coordinate cosine under 6 perturbations × 5 levels | **0.958** | — |
| argmax stability under perturbation | 0.949 | — |

**Continuous coordinates carry substantially more information than hard cluster ids** — nearly
double the neighbour preservation — which answers the brief's question in Section 5 affirmatively.

But 0.446 neighbour preservation also means **more than half of each molecule's ten nearest
neighbours in the 49-dimensional space are not its neighbours in the 16-dimensional coordinate
space.** The coordinates are a lossy summary, and that loss is the reason Section 6 comes out as
it does.

---

## 8. Section 6 — retrieval benchmark

Molecule-grouped throughout, with the **clustering refitted inside every training fold** so no
cluster definition ever sees a test molecule.

| arm | mol top-1 | top-3 | top-5 | MRR | chem top-1 | chem top-3 | macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **A** CSM only | 0.605 | 0.763 | 0.795 | 0.687 | **0.845** | 0.971 | **0.807** |
| **B** CSM + coordinate prior | **0.621** | 0.769 | 0.797 | **0.694** | **0.848** | 0.971 | 0.798 |
| **C** coordinates only | 0.560 | 0.634 | 0.659 | 0.612 | 0.685 | 0.897 | 0.578 |

**Is B better than A?**

| task | Δ | 95% CI (molecule bootstrap) | McNemar | p | significant |
|---|---:|---|---|---:|---|
| molecule | +0.0160 | [−0.0054, +0.0390] | 10 / 4 | 0.180 | **no** |
| chemistry | +0.0027 | [−0.0115, +0.0193] | 5 / 4 | 1.000 | **no** |

**Neither improvement is significant.** The fusion-weight sweep confirms it: chemistry top-1 is
0.845 at w = 0, peaks at 0.851 for w ∈ {0.1, 0.25}, and collapses to 0.685 at w = 1. There is a
shallow, non-significant optimum near w = 0.1–0.25 — consistent with a small amount of
complementary information, and far too small to justify a coordinate layer in the inference path.

**A note on Split A.** 66 of 154 molecules have a single spectrum. Under leave-one-spectrum-out
they leave the reference bank entirely and can never be retrieved. Phase 05 counted these as
misses and this phase does the same, for comparability; the reported 0.605 therefore has a
structural ceiling well below 1.0.

---

## 9. Section 7 — geometry versus the curated ontology

| measure | value |
|---|---:|
| ARI | 0.564 |
| **AMI** | **0.703** |
| NMI | 0.749 |
| VI | 1.078 |
| homogeneity | 0.732 |
| **completeness** | **0.812** |

**And this number is conditional on K = 16, which Section 1 shows is arbitrary.** The agreement
curve across K, averaged over the four fixed-K algorithms, is:

| K | 2 | 3 | 4 | 6 | 8 | 10 | 12 | **16** | 20 | **24** | 30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AMI | 0.10 | 0.21 | 0.24 | 0.35 | 0.46 | 0.51 | 0.55 | **0.58** | 0.59 | **0.61** | 0.59 |

**Agreement is monotone in K too**, peaking at K = 24 rather than at the ontology's 16. There is
no sense in which the geometry "recovers" sixteen classes: it agrees with the ontology
progressively better as it is allowed more clusters, exactly as a finer cut through a continuum
would. The headline 0.703 is the *canonical partition* (average linkage at K = 16); the 0.578 in
the table above is the mean over four algorithms at the same K. Both are one point on a curve
with no peak.

**Completeness exceeds homogeneity, and that asymmetry is the finding.** Curated classes tend to
stay *together*; emergent clusters *merge* several of them. The geometry is coarser than the
ontology where chemistries are spectrally similar, and finer where a single class spans
different acquisition regimes.

Disagreements, explained rather than scored:

- **`peptide_protein` splits across 4 clusters.** Proteins differ by size and by acquisition, and
  the amide backbone that defines them chemically is shared with everything else that has a C=O
  near 1650 cm⁻¹.
- **`mono_oligosaccharide` splits across 3.** A tight sugar core plus a scattered tail.
- **`sulfur_thiol_cofactor` splits across 3.** Four molecules, three of them chemically unlike
  each other apart from carrying sulfur.
- **`fatty_acid` and `acylglycerol` merge**, as Phase 06's adjacency analysis predicted: a
  triacylglycerol's spectrum is dominated by its acyl chains.

Neither partition is wrong. They answer different questions — the ontology asks what a molecule
*is*, the geometry asks what its spectrum *resembles*.

---

## 10. Section 9 — does the coordinate system earn a place in inference?

| criterion | verdict | evidence |
|---|---|---|
| reproducibility | **PASS** | 0.963 across replicates, floor 0.90 |
| robustness | **PASS** | mean coordinate cosine 0.958 over 6 × 5 perturbations |
| stability | **PASS** | bootstrap ARI 0.773, min cluster survival 0.505 |
| generalisation | **PASS** | coordinates alone reach chemistry top-1 0.685 |
| biochemical meaning | **PASS** | chemistry dominates; AMI 0.703 |
| **retrieval improvement** | **FAIL** | molecule Δ+0.016 CI[−0.005, +0.039] p = 0.180; chemistry Δ+0.003 p = 1.000 |
| **interpretability** | **FAIL** | 5 of 16 clusters chemically nameable; 4 acquisition-confounded, 7 unresolved |

**Five of seven pass. The two that fail are the two that would justify an architecture change.**

---

## 11. Section 10 — architecture recommendation

> ### Option A — retain the current architecture
> ```
> CSM  →  Chemistry Evidence  →  BSV2
> ```
> The Continuous Spectral Coordinates are retained as a **scientific instrument**: an auxiliary
> analysis for exploring the manifold, generating hypotheses about spectral neighbourhoods, and
> flagging molecules that sit between chemistries. They do **not** enter the inference path.

**Why not Option B** (coordinates *before* Chemistry Evidence): the coordinates lose more than
half the local neighbourhood structure (k-NN preservation 0.446), and chemistry inference from
coordinates alone falls to 0.685 from the CSM layer's 0.845. Inserting them upstream would
discard information the next layer needs.

**Why not Option C** (parallel, then fusion): this is the option the evidence came closest to
supporting, and it fails on significance. +0.016 molecule top-1 with a CI crossing zero, and
+0.003 chemistry top-1 at p = 1.000, is not a basis for adding a layer, a kernel, a temperature,
a cluster count and a fusion weight to a frozen inference engine. The fusion-weight sweep shows
a shallow optimum near w = 0.1–0.25 worth ~0.006 chemistry top-1 — real, perhaps, and far below
the noise floor of a 154-molecule corpus.

**Why not Option D:** nothing in the evidence points at a different architecture. The manifold is
organised by chemistry, the CSM layer already exposes that organisation, and the Chemistry
Evidence layer already reads it at 0.845.

**What would change this recommendation.** A corpus large enough for +0.016 to become
significant — roughly 4× the current molecule count at the observed effect size — or a
demonstration that the coordinates carry information the Chemistry Evidence layer cannot, on a
task other than retrieval. Neither is available now, and neither is a reason to adopt the layer
now.

---

## 12. Limitations

1. **154 molecules is small for a geometry audit.** Every stability curve is noisy and the
   retrieval difference sits inside the noise.
2. **Four of sixteen clusters are acquisition-confounded**, and the source/excitation confound
   cannot be removed by design — the corpus is what it is.
3. **The intrinsic-dimension estimators disagree by more than 3×** (1.40 vs 4.68). Neither is
   quoted as the answer.
4. **K = 16 is a convention.** Every composition statement in Section 2 is conditional on it,
   and a different K would give different clusters — because there is no right K. The agreement
   with the ontology is likewise conditional, and rises monotonically to K = 24.
5. **Average linkage produces unbalanced clusters** (7 of 16 have ≤ 2 molecules). It was chosen
   label-free by bootstrap stability at K = 16, but a different linkage would give a different
   composition table.
6. **Split A has a structural ceiling** — 66 singleton molecules can never be retrieved.
7. **All embeddings are visualisation only.** UMAP distances are not quantitative and are not
   used for any claim.

## 13. Reproduction

```bash
PYTHONPATH=src python results/v7_rebuild/phase06_5/code/run_phase06_5.py   # ~9 min
PYTHONPATH=src python results/v7_rebuild/phase06_5/code/make_figures.py
PYTHONPATH=src python results/v7_rebuild/phase06_5/code/make_pdf.py
PYTHONPATH=src python -m pytest tests/test_v7_phase06_5.py -q
```

`SEED = 0` throughout; output root resolves through `GAIRA_V7_OUTPUT_ROOT`.
