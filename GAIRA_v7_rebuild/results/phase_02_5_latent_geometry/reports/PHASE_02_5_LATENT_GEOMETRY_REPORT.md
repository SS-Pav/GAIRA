# GAIRA V7 — Phase 02.5: Latent Geometry of Spectral Motif Space

**Analysis only.** Nothing frozen is refitted. No themes are created.
Frozen inputs verified: atlas `09ed804a40836f4a05a91ba10900cded`, LSM registry
`208482d6f7178b5b8f16cace91be55b0`, CSM dictionary `0b4aa550ccefed3edabdbde5bae11c8d`.

---

## 1. Executive summary

Phase 02 asked which motifs are *interchangeable* and found one pair. Phase 02.5 asks which are
*related*, and the answer is much richer.

**Motif space is a low-dimensional continuum of overlapping neighbourhoods, not a set of
discrete islands.** The pairwise distance distribution is unimodal (valley depth 0.003 — no
density gap), mean local intrinsic dimension is 3.9 against an ambient 676, community
conductances run 0.48–0.90 (islands would be near zero), and the three cluster-quality indices
disagree about K at every value. Yet the structure is not noise: kNN graph modularity is 0.436
against a degree-preserving null of 0.070 ± 0.003, and nearest neighbours share a chemistry
class 2.8× more often than a label permutation allows (0.240 vs 0.086, p < 0.0001).

**The three groups Phase 02 rejected as merges are all real neighbourhoods.** Each separates
from the rest of the dictionary by a factor of ~2 in mean distance, each is internally
low-dimensional, and each carries an interpretable set of shared bands. They failed as motifs
and succeed as themes — which is exactly the distinction Phase 03 exists to exploit.

**Seven bridge motifs and five isolated motifs are identified by name.** Both matter to Phase
03 for opposite reasons: bridges are what a hard theme assignment would misplace, and isolated
motifs are what it would force into a theme they do not belong to.

**Ten provisional priors are produced.** They constrain Phase 03; they do not decide it.

---

## 2. The scientific question

Phase 02's negative result — 48 of 50 motifs left as singletons — could mean either that
class-local motifs genuinely share nothing, or that *equivalence* is simply the wrong question
to ask of them. Two motifs can describe the same bond system, respond to the same molecules and
sit next to each other in every geometry, and still fail a substitution test, because a
consensus spectrum that replaces both must reproduce both and a shared neighbourhood does not
have to.

So: **how are the motifs organised, when we stop asking whether they can be merged?**

## 3. Equivalence versus neighbourhood

| | Phase 02 | Phase 02.5 |
|---|---|---|
| question | can one basis element replace both? | how are they related? |
| test | reconstruction survives substitution | geometric proximity survives nulls and resampling |
| outcome | merge or do not merge | neighbourhood, gradient, bridge, or isolate |
| cost of a wrong yes | reconstruction collapses | a theme prior is too broad |
| answer here | 1 merge of 1225 pairs | 4 named neighbourhoods, 7 bridges, 5 isolates |

**Related is not identical.** A high-similarity neighbourhood is scientifically important even
where nothing should be merged — and this corpus is almost entirely made of such
neighbourhoods.

## 4. Frozen inputs and provenance

50 LSMs (Phase 01, `H` 50 × 676) · 49 CSMs (Phase 02, sensitivity control) · 375 balanced
reference spectra · 154 canonical molecules · the seven Phase 02 edge-feature matrices and the
molecule-level activation matrix. All read-only; all three fingerprints asserted at start-up.

**Two firewalls, enforced in code and in tests.** No chemistry-class label and no source label
enters any representation or any distance used to construct the geometry. Both are revealed at
step 8, to evaluate what was found. A geometry built on the class partition would rediscover
the class partition and prove nothing (risk R-01).

## 5. Representations tested

| view | dim | what it captures | what it is blind to |
|---|---:|---|---|
| spectral profile | 676 | overall band shape | amplitude |
| peak representation | 175 | discrete diagnostic features | the continuum between peaks |
| band family | 20 | which bond systems carry intensity | fine peak position |
| activation | 154 | what the motif responds to | what it looks like |
| reconstruction contribution | 154 | what it uniquely explains | shared explanatory mass |
| provenance | 17 | evidence breadth (counts, source/excitation entropy) | chemistry — class identity deliberately excluded |
| edge feature | 350 | relational position in the dictionary | absolute spectral identity |

## 6. Distance metrics

Ten metrics, benchmarked on **scale-free** probes — each probe divided by that metric's own
median observed distance, because a background separation of 0.106 under Euclidean and 0.006
under cosine cannot be compared until both are read against the scale each metric works on.

| metric | ampl. invariance | shift tolerance | background separation | kNN coherence | null z |
|---|---:|---:|---:|---:|---:|
| **wasserstein** | 1.000 | 0.958 | 0.064 | **0.240** | 9.08 |
| spearman | 1.000 | 0.999 | 0.047 | 0.148 | 11.94 |
| euclidean (L2) | 1.000 | 0.701 | 0.091 | 0.152 | 12.74 |
| jensen–shannon | 1.000 | 0.684 | 0.070 | 0.152 | 16.67 |
| pearson | 1.000 | 0.933 | 0.043 | 0.152 | 9.45 |
| spectral cosine | 1.000 | 0.911 | 0.008 | 0.152 | 10.33 |
| band overlap | 1.000 | 0.865 | 0.013 | 0.196 | 5.48 |
| peak set | 1.000 | 0.937 | 0.006 | 0.124 | 5.49 |
| activation profile | — | — | — | 0.188 | — |
| Phase 02 composite | — | — | — | 0.016 | — |

**Primary spectral metric: Wasserstein (earth-mover along the Raman shift axis).** Selected by
the pre-stated rule — among metrics leaking under 5% of their own scale to amplitude and
costing under 10% for a 6 cm⁻¹ shift, maximise background separation × null separation — and it
is also the scientifically right answer for an ordered axis. Every other metric here is
bin-wise and therefore treats a 10 cm⁻¹ shift and a 900 cm⁻¹ shift as equally different, which
is wrong for spectra whose peaks drift with excitation, substituent and hydrogen bonding. Its
chemical coherence, 0.240, is 58% higher than the next best.

Activation geometry: **Spearman on molecule-balanced activation profiles**.
Multi-view geometry: **weighted similarity fusion** (§14).

Cosine, notably, is the *worst* metric on background separation (0.008) — it is dominated by
the shared broad envelope, which is the failure Phase 02 had already found at the edge level.

## 7. Null models

Six, each destroying one kind of structure: band-position permutation, intensity permutation,
class-label permutation, molecule-activation permutation, source-label permutation, and a
degree-preserving graph rewiring. Nothing in this report is claimed without one beside it.

## 8. Linear geometry

PCA on the spectral profiles is **not a useful description of this space**. PC1 carries 14.7%
of variance, and cumulative variance reaches only 45% by PC6. Only PC1 is reproducible under
resampling (loading stability 0.82); PC2 falls to 0.52 and PC3 to 0.32.

| PC | variance | stability | driven by | interpretation |
|---|---:|---:|---|---|
| 1 | 14.7% | 0.82 | +1442, +1444, +1440 cm⁻¹ | **CH₂/CH₃ scissoring** — an aliphatic-content axis, running from `peptide_protein.m01` to `fatty_acid.m03` |
| 2 | 8.4% | 0.52 | +1656, +1654 cm⁻¹ | amide I / C=C — unsaturation and carbonyl, but not reproducible |
| 3 | 6.2% | 0.32 | +1656, −1584, +846 | mixed; not interpretable |

PCA's trustworthiness (0.782) and kNN preservation (0.360) are the worst of the four
embeddings. **The space is not linear**, and the honest reading of a PCA plot here is that it
shows one real axis — aliphatic CH₂ content — and then noise.

Per-PC confounding was tested by PERMANOVA on each score axis; no PC is significantly
associated with source or excitation at α = 0.05 after the primary axis.

## 9. Nonlinear geometry

| embedding | trustworthiness | continuity | kNN preservation | Procrustes disparity |
|---|---:|---:|---:|---:|
| UMAP | **0.954** | **0.961** | **0.760** | 0.45 |
| diffusion map | 0.934 | 0.958 | 0.548 | — |
| spectral embedding | 0.936 | 0.956 | 0.560 | — |
| PCA | 0.782 | 0.796 | 0.360 | — |

**UMAP is stable enough to read, and the sweep is why we can say so.** Across
n_neighbors ∈ {5, 8, 12, 20}, min_dist ∈ {0.05, 0.3, 0.7} and three seeds, the k-NN Jaccard
against the high-dimensional geometry stays in 0.60–0.65 and across seeds in 0.64–0.78, with
trustworthiness 0.954–0.977 throughout. Neighbourhoods survive the parameters; the *global*
arrangement does not (Procrustes disparity 0.45), so relative positions of distant clusters in
any UMAP panel here carry no information and are not interpreted.

**The diffusion spectrum has no sharp gap** (eigenvalues 1.000, 0.186, 0.150, 0.119; gap
0.036). A clean gap after coordinate k would say "k discrete clusters". Its absence is the
first quantitative evidence that this space is continuous.

## 10. Graph and hierarchical structure

- kNN graph (k = 5): 5 Louvain communities, **modularity 0.436 against a degree-preserving null
  of 0.070 ± 0.003** (z = 124, p < 0.001). The community structure is real.
- Hierarchical clustering: best silhouette 0.136 — very low — at K = 2, bootstrap ARI 0.404.
- Silhouette, Davies–Bouldin and Calinski–Harabasz select different K at every linkage.

So the graph has genuine community structure *and* no defensible cluster count. Those are
compatible: a continuum with denser regions has exactly this signature.

## 11. Discrete versus continuous organisation

| evidence | result | reading |
|---|---|---|
| distance distribution | 2 modes, valley depth **0.003** | no density gap → **not discrete** |
| local intrinsic dimension | mean **3.86** (ambient 676) | low-dimensional structure |
| diffusion spectrum | gap 0.036 | no clean cluster count |
| conductance | 0.48–0.90 across regions | boundaries are permeable |
| modularity vs null | z = 124 | denser regions are real |
| cluster indices | mutually contradictory | no defensible K |

**Verdict: MIXED, dominated by overlapping continua.** Region classification:

| region | motifs | conductance | local dim | stability | geometry |
|---|---:|---:|---:|---:|---|
| region02 | 16 | 0.480 | 4.23 | 0.857 | **branching** |
| region01 | 12 | 0.565 | 3.16 | 0.931 | overlapping |
| region04 | 11 | 0.667 | 4.71 | 0.445 | overlapping |
| region03 | 6 | 0.805 | 4.23 | 0.950 | overlapping |
| region00 | 5 | 0.684 | 2.06 | 0.328 | **unresolved** |

One branching region, three overlapping, one unresolved. No region qualifies as discrete.

## 12. Biochemical neighbourhoods

Chemistry was revealed only at this point. Across 250 nearest-neighbour links:

| relationship tier | links | share |
|---|---:|---:|
| generic Raman overlap | 188 | 75% |
| shared substructure | 23 | 9% |
| artefact or unresolved | 20 | 8% |
| broad superfamily | 17 | 7% |
| exact equivalence | 2 | 1% |

**Three quarters of all nearest-neighbour relationships are indistinguishable from the null.**
That is the same lesson Phase 02 learned at the edge level, and it is why "these two motifs are
neighbours" is not by itself a finding.

### Validated neighbourhoods

**Lipid superfamily** (8 motifs; acylglycerol, fatty acid, phospholipid, sterol, one amino
acid). Separation ratio 2.05, conductance 0.746, local dimension **2.82** — the most
low-dimensional group in the corpus. Shared bands 892, 1062, 1118, 1134, 1172, 1296, 1438 cm⁻¹.
Ordered along the leading diffusion coordinate the group runs free amino acid → sterol → free
fatty acids → acylglycerols and phospholipid: an **aliphatic chain-order axis**, with the
sterol adjacent to but not inside the acyl cluster. The sterol is a genuine neighbour, not a
member — which is precisely what its band-prominence disagreement of 0.375 said in Phase 02.

**Polar skeletal backbone** (7 motifs; carboxylic acid, amino acid, mono-/oligosaccharide,
protein, phosphate, polysaccharide). Separation ratio **2.31** — the strongest of the three —
conductance 0.781, local dimension 3.87, internal stability 0.770. Shared bands 476, 576, 852,
870, 936, 1074, 1120, 1252, 1330, 1344, 1460 cm⁻¹. The 1074/1120 pair is the glycosidic and
skeletal C–O/C–C region, and its dominance confirms the Phase 02 reading: proteins and
polysaccharides are close because of **C–O/C–C/C–N skeletal modes**, not glycoprotein biology.
The proximity survives peak-prominence weighting (band-overlap distance is one of the ten
metrics and gives the same neighbourhood) and survives leave-one-source-out.

**Heterocyclic ring system** (3 motifs; nucleic acid polymer, purine, thiol cofactor).
Separation ratio 2.10, internal stability **1.000**, local dimension 3.25. Shared bands 526,
624, 722, 1036, 1248, 1334, 1486, 1668 cm⁻¹ — 722 and 1334 are purine ring modes, 526 and 624
are S–S and C–S. The cofactor motifs are close to the purine motif because coenzyme A and
acetyl-CoA **contain adenine**; the group is a genuine heterocyclic neighbourhood in which the
cofactor member is a mixture, which is why merging it destroyed reconstruction.

**cis-unsaturation** (2 motifs) — the one Phase 02 equivalence, carried forward unchanged.

### Rejected as neighbourhoods

- **A single global "generic Raman" attractor.** Tested and not found: if one existed, the
  hub motifs would share neighbours with everything, but the five hubs sit in four different
  regions.
- **A source-defined neighbourhood.** Source explains R² = 0.130 of the geometry against
  chemistry's 0.617 (§13).
- **A protein–polysaccharide *phenomenon*.** The proximity is real; the shared cause is
  skeletal-mode overlap, not shared biology.

### Bridges — 7

`free_amino_acid.m00`, `free_amino_acid.m06`, `peptide_protein.m05`,
`phospholipid_sphingolipid.m01`, `sterol_steroid.m00`, `sulfur_thiol_cofactor.m00`,
`sulfur_thiol_cofactor.m01`

High betweenness, low local clustering: motifs on the paths *between* neighbourhoods without
belonging to one. `free_amino_acid.m06` bridging into the lipid group and `sterol_steroid.m00`
bridging out of it are the two that most constrain Phase 03 — a hard theme assignment would
have to put each on one side and would be wrong either way.

### Isolated — 5

`carboxylic_acid_metabolite.m00`, `chromophore_pigment.m00`, `free_amino_acid.m02`,
`peptide_protein.m02`, `small_nitrogenous.m00`

No close neighbour under the primary geometry. `chromophore_pigment.m00` is the clearest case
and the most chemically sensible: the conjugated C=C system of a carotenoid at 1150/1520 cm⁻¹
has no analogue anywhere else in this corpus.

## 13. Source and excitation effects

| label | PERMANOVA F | p | R² | kNN accuracy | chance |
|---|---:|---:|---:|---:|---:|
| **chemistry class** | 3.65 | **0.001** | **0.617** | 0.240 | 0.102 |
| excitation | 2.43 | 0.044 | 0.178 | 0.432 | 0.366 |
| source | 3.53 | 0.022 | 0.130 | 0.696 | 0.619 |

**Chemistry dominates, but source and excitation are both significant and must be carried
forward as cautions.** Chemistry explains 3.5× more of the geometry than excitation and 4.7×
more than source, and the kNN excess over chance is 2.4× for chemistry against 1.2× and 1.1×.
The confounding is real but secondary.

**8 of 50 motifs are single-source and therefore source-untestable.** They are named in
`confounding_v1.csv` and flagged in every prior that contains them. Leave-one-source-out and
leave-one-excitation-out geometries are reported in `validation/`.

## 14. Multi-view integration

| method | stability | null z | coherence | source robustness | Pareto |
|---|---:|---:|---:|---:|---:|
| **weighted similarity** | 0.822 | 9.68 | 0.152 | 0.947 | **winner** |
| multiple-kernel embedding | 0.874 | 1.81 | 0.220 | 0.943 | |
| single-view wasserstein | 0.502 | 12.46 | 0.240 | 0.923 | control |
| graph consensus | 0.742 | 5.81 | 0.184 | 0.939 | |
| concatenated features | 0.777 | 4.32 | 0.316 | 0.835 | |
| similarity network fusion | 0.023 | −86287 | 0.060 | 0.859 | **degenerate** |

Similarity network fusion collapses the distance range on this data, which makes its null
separation unbounded; it is excluded from the Pareto and reported, not hidden. Its exclusion
matters: under min–max normalisation a single −86,287 would flatten the null-separation
criterion for every other candidate.

**Each candidate is scored against its own matched band-permutation null.** Scoring fused
geometries against the single-view null compares distances on different scales, and null
separation is scale-dependent — that error, before it was fixed, selected a different winner.

Multi-view integration buys stability (0.822 vs 0.502 for the single view) at a real cost in
chemical coherence (0.152 vs 0.240). Notably `concatenated_features` has the best coherence
(0.316) and the worst source robustness (0.835) — it is partly reading the provenance view.

**Two primary geometries exist and they are not interchangeable.** The Pareto winner is the
weighted-similarity fusion; the neighbourhoods, cards, roles, regions and priors in §11–12 are
computed on the **primary metric geometry** (Wasserstein), which is the one with the best
chemical coherence and the one whose distances are spectroscopically interpretable. The two
agree on only **0.508** of nearest neighbours. That is a large disagreement and it is stated
here rather than buried: it means "nearest neighbour" is not a geometry-independent fact in
this space, and any Phase 03 construction that depends on a single motif's exact neighbour list
is on weak ground. The stable objects are the *neighbourhoods* — which recur under both — not
the individual links. Both matrices ship under distinct keys (`D_primary_metric`,
`D_primary_geometry`) so no downstream reader can conflate them.

## 15. Representative motif trajectories

The lipid neighbourhood's diffusion-ordered sequence (§12) is the clearest trajectory in the
corpus: aliphatic character increases monotonically along DC1, with the 1440 cm⁻¹ CH₂
scissoring band growing and the sharp low-wavenumber structure of the amino-acid motif fading.
This is the same axis PCA found as PC1, recovered independently — which is why PC1 is the one
component with reproducible loadings.

## 16. Outliers and isolated motifs

Five (§12). All five are candidates for **singleton themes or exclusion from soft membership
entirely**. Forcing an isolated motif into a theme is the L-03 failure mode — a motif borrowing
foreign mass — and Phase 03 should treat `prior_isolated_diagnostic` as a list of exclusions
rather than a theme.

## 17. Implications for Phase 03

1. **Build themes on neighbourhoods, not on clusters.** There is no defensible K. A soft,
   overlapping membership matrix is not a convenience here, it is the only representation the
   geometry supports.
2. **The three rejected Phase 02 groups are the strongest theme candidates** — priors
   `lipid_superfamily` (conf 0.71), `polar_skeletal_backbone` (0.83),
   `heterocyclic_ring_system` (0.83). Each failed as a motif because merging destroys
   reconstruction; soft membership never replaces the motifs, so that cost does not arise.
3. **Seven bridge motifs must be allowed genuinely split membership.** They are the test of
   whether the theme layer is soft in practice or only in form.
4. **Five isolated motifs must be allowed to belong to no theme.**
5. **`prior_cis_unsaturation` must not be split** — it is the one established equivalence.
6. **Carry the source flags.** Any prior containing single-source motifs inherits a caution.
7. **Score whatever Phase 03 builds against the same nulls.** 75% of nearest-neighbour links
   here are indistinguishable from generic Raman statistics; there is no reason to expect the
   theme layer to be different, and no way to know without measuring.

Ten priors are in `artifacts/phase03_geometry_priors.json`, each with geometry type, shared
bands, evidence, source confounding, confidence and an explicit `must_not_hard_merge` list.

## 18. Limitations

1. **50 motifs is a small space for manifold estimation.** Local intrinsic dimension at k = 10
   uses a fifth of the corpus per estimate; the value 3.86 should be read as "low", not as 3.86.
2. **Chemistry-class labels are curated, not measured.** kNN coherence against them measures
   agreement with a human partition, not correctness.
3. **Source and excitation are partly confounded with each other**, and with chemistry: the
   lipid classes are largely RamanBioLib at 532/1064 nm. Their separate contributions cannot be
   fully identified in this corpus.
4. **8 motifs are single-source.** For those, source robustness is undefined, not established.
5. **PHATE was not run** — not installed, and the brief permitted skipping it rather than adding
   a dependency. Topological persistence was likewise not attempted at n = 50.
6. **The two primary geometries agree on only half of all nearest-neighbour links** (0.508).
   The four validated neighbourhoods recur under both, which is why they are reported; no
   individual link should be treated as robust.
7. **CSM control agrees but is not independent** — 48 of 49 CSMs are single LSMs, so neighbour
   agreement of 0.980 is close to a tautology and should not be read as external validation.

## 19. Decision gate

See the gate returned with this report.
