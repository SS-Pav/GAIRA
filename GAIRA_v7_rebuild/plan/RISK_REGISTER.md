# GAIRA V7 — Risk Register

Every risk carries: probability, severity, detection method, mitigation, and phase owner.

Scales — **Probability:** Low (<25%) · Medium (25–60%) · High (>60%).
**Severity:** Low (report it) · Medium (rework a phase) · High (rework the architecture) ·
Critical (invalidates results, or ships a wrong answer).

---

## R-01 — Class labels bias the local decompositions

| | |
|---|---|
| **Probability** | **High** |
| **Severity** | Medium |
| **Description** | The chemical-family partition is a human prior. Fitting within it can imprint it on the representation, so Phase 03 "discovers" cross-class structure that merely re-encodes the classes it started from. The `unknown` class (6 analytes) is a guaranteed instance — it is not a chemistry, so any LSM fitted over it has no chemical meaning. |
| **Detection** | Compare the CSM community structure to the class partition (adjusted Rand index). Test whether LSMs from different classes ever merge — if almost never, the partition has been reproduced rather than transcended. Check the `provenance_overlap` edge feature for circularity. |
| **Mitigation** | Discount within-class overlap in the provenance edge feature. Resolve `unknown` in Phase 00 (assign or exclude). Run a sensitivity arm with a perturbed partition. Report ARI(CSM communities, classes) as a standing diagnostic. |
| **Owner** | Phase 00 (partition), Phase 02 (test), Phase 03 (measure) |

---

## R-02 — Rare classes have too few analytes to support NMF

| | |
|---|---|
| **Probability** | **High** — certain, actually |
| **Severity** | Medium |
| **Description** | polyol has 1 analyte; phospholipid, carotenoid, small_nitrogenous have 2; nucleic_acid and pyrimidine have 3. NMF over `n=1` is meaningless; over `n=2–3` it is a memorisation. |
| **Detection** | Known in advance from the family census. Also detected by stability failure — no LSM clears the threshold. |
| **Mitigation** | Hard rule `k_c ≤ ⌊n_analytes/2⌋`, so `n<2` gets no fit. Route to the Strategy-F anchor mechanism with quality gate, novelty gate, written chemical justification, and a permanent `is_anchored` flag with widened uncertainty. **Never duplicate spectra** (P-11). Feed the gap to Phase 09. |
| **Owner** | Phase 02, Phase 03 |

---

## R-03 — Local dictionaries become incomparable

| | |
|---|---|
| **Probability** | Medium |
| **Severity** | **High** |
| **Description** | **The central bet of V7.** Partitioning buys fair capacity allocation at the cost of comparability: an LSM from the sterol fit and one from the fatty-acid fit are two local descriptions with no shared coordinate. If Phase 03 cannot reintegrate them, Strategy D has traded one problem for a worse one. |
| **Detection** | Phase 03 integration quality: low within-CSM cohesion, high singleton fraction, poor held-out recovery, no stable community structure at any threshold. |
| **Mitigation** | All LSMs live in the same `ℝ₊^676` grid, so they are directly comparable *as spectra* — this is the structural reason to expect integration to work. Six independent edge features rather than one. Five integration methods benchmarked. Threshold swept. **Fallback:** if integration fails, Phase 01's control arm A remains a working architecture, and the failure is documented as a negative result about partitioning. |
| **Owner** | Phase 03 (with fallback to Phase 01) |

---

## R-04 — Excessive motif proliferation

| | |
|---|---|
| **Probability** | Medium |
| **Severity** | Medium |
| **Description** | `Σ_c k_c` across ~15 classes could reach several hundred LSMs. Too many CSMs hurts interpretability, inflates live projection cost, and produces redundant axes. |
| **Detection** | Count LSMs and CSMs against pre-declared budgets; redundancy matrix; effective rank far below nominal count. |
| **Mitigation** | `k_c ≤ ⌊n_analytes/2⌋`. Stability threshold on every LSM. Redundancy penalty in `M` selection. Smallest-on-Pareto-plateau rather than argmax. Report effective rank alongside nominal count. |
| **Owner** | Phase 02, Phase 03 |

---

## R-05 — Consensus clustering becomes arbitrary

| | |
|---|---|
| **Probability** | Medium |
| **Severity** | Medium |
| **Description** | Consensus clustering has many free choices — linkage, cut height, resampling scheme, consensus matrix threshold. Different defensible choices give different CSM sets, and the choice is easy to make post-hoc. |
| **Detection** | Sensitivity analysis across choices; low consensus stability; results that shift materially under small parameter changes. |
| **Mitigation** | Pre-register the rule (P-12). Sweep the sensitive parameters and report the full surface. Require a stable region, not a single point. Benchmark against methods with different failure modes (graph communities, spectral clustering) — agreement across method families is stronger evidence than any single method's internal criterion. |
| **Owner** | Phase 03 |

---

## R-06 — A second NMF removes class-specific detail

| | |
|---|---|
| **Probability** | Medium |
| **Severity** | **High** |
| **Description** | If meta-factorisation is selected, its compression can erase exactly the molecule-discriminating residual LSMs Phase 02 worked to isolate — destroying the fine-resolution gain that is V7's entire purpose. |
| **Detection** | Track molecule-discriminating LSMs specifically: do they survive into distinguishable CSMs? Measure within-class retrieval before and after integration. |
| **Mitigation** | Meta-NMF is **not presumed to win** — it is one of five candidates, and the stated prior favours graph/hybrid routes because meta-NMF sees only one of six edge features and its equal row weighting reintroduces the L-01 bias. If it is selected anyway, survival verification is a mandatory additional gate. |
| **Owner** | Phase 03 |

---

## R-07 — Graph communities reflect threshold choices

| | |
|---|---|
| **Probability** | **High** |
| **Severity** | Medium |
| **Description** | Community structure in a thresholded similarity graph is notoriously threshold-dependent. A crisp community structure at one cut can vanish at the next, and reporting only the lucky cut is a well-known way to manufacture structure. |
| **Detection** | Threshold sweep with community stability at each point. If stability has no plateau, the structure is threshold-artefactual. |
| **Mitigation** | Mandatory sweep. Select only from a stable region. Cross-check with a threshold-free method (hierarchical consensus on the full similarity matrix). If no stable region exists, the graph construction is inadequate — revise it, and report that as a finding. |
| **Owner** | Phase 03 |

---

## R-08 — Anchored atoms duplicate learned motifs

| | |
|---|---|
| **Probability** | Medium |
| **Severity** | Medium |
| **Description** | A rare-chemistry anchor admitted directly into the CSM dictionary may span a direction the learned motifs already cover, creating a redundant, collinear axis. Collinear dictionary atoms make NNLS solutions unstable — small input changes flip mass between them — which would break determinism-in-spirit even while remaining deterministic in code. |
| **Detection** | Novelty gate: residual of the anchor after projection onto the existing stable CSM set. Post-admission redundancy matrix. Dictionary coherence / condition number. |
| **Mitigation** | Novelty gate is a hard admission requirement with a pre-declared threshold. Post-admission redundancy check. Anchors permanently flagged `is_anchored`, `n_analytes=1`, with inflated downstream uncertainty. |
| **Owner** | Phase 03 |

---

## R-09 — Leakage through aliases or replicates

| | |
|---|---|
| **Probability** | Medium |
| **Severity** | **Critical** |
| **Description** | The same molecule appearing in train and test under two spellings inflates every metric invisibly. Real hazards already observed: `riboflavin`/`riboﬂavin` (U+FB02 ligature — these appear as *separate* entries in existing tables), `acetyl coenzyme a`/`acetyl-coa` (which is *also* assigned to two different families), `urea`/`ure`, missing-space fatty-acid names. Replicate leakage does the same thing more subtly. |
| **Detection** | Phase-00 leakage checks; fuzzy-match audit over canonical names; InChIKey/SMILES cross-check where available; the three `cv_splits_v1.json` checks. |
| **Mitigation** | NFKC + whitespace + case normalisation before matching. Manual review of every merge **and** every near-miss non-merge. Group CV by `canonical_id`. Preserve enantiomers/anomers as genuinely distinct. **All three leakage checks must read `false` or Phase 00 does not pass.** |
| **Owner** | Phase 00 |

---

## R-10 — Evaluation remains in-sample

| | |
|---|---|
| **Probability** | Medium |
| **Severity** | **Critical** |
| **Description** | If model selection sees held-out analytes — through `k_c` sweeps, thresholds, `M`, `K`, or the quality score `q` — every Phase-07 number is inflated and the replacement decision is made on fiction. |
| **Detection** | Audit every sweep for which folds it used. Compare in-sample and held-out performance: a large gap is the signature. |
| **Mitigation** | All sweeps and thresholds fitted on training folds only. `q` frozen in Phase 00 before Phase 01. Phase 07 uses the Phase-00 splits untouched. Every manifest records which folds each decision saw. |
| **Owner** | Phase 00 (splits), all phases (discipline), Phase 07 (audit) |

---

## R-11 — Theme abstraction becomes decorative

| | |
|---|---|
| **Probability** | Medium |
| **Severity** | Medium |
| **Description** | Themes that merely relabel CSMs add complexity and no information. **This has already happened once**: at V6.2, `theme_raw` and `theme_posterior` were numerically identical at every metric on every ontology — the Bayesian posterior refinement changed no decisions. |
| **Detection** | Compare CSM-level and theme-level performance directly. If the theme layer never changes a decision, it is decorative. |
| **Mitigation** | Phase 04 must demonstrate value over the CSM layer or record that it does not. If it does not, ship the CSM-level BSV with `K = M` and say so plainly. |
| **Owner** | Phase 04 |

---

## R-12 — BSV axes become correlated

| | |
|---|---|
| **Probability** | Medium |
| **Severity** | Medium |
| **Description** | Highly correlated themes mean `K` axes carry far fewer than `K` degrees of freedom, so the BSV overstates its own resolution. Downstream users reading `K` independent numbers would be reading redundant ones. **Precedent:** the V5 24-component space had participation ratio 15.2 — a 38% gap, visible only because it was measured. |
| **Detection** | Participation ratio, effective entropy rank, axes-for-90%-variance on the BSV covariance; pairwise axis correlation matrix. |
| **Mitigation** | Sparsity in `S`. Redundancy penalty in `M` selection. **Report effective rank alongside `K` in the atlas manifest**, so the gap is disclosed rather than discovered. If the gap is large, reduce `K`. |
| **Owner** | Phase 04, Phase 05 |

---

## R-13 — Runtime projection becomes too complex

| | |
|---|---|
| **Probability** | Low |
| **Severity** | Medium |
| **Description** | A two-level dictionary with hundreds of atoms could make live projection slow or numerically unstable, undermining live and DART use. Coherent (near-collinear) dictionary atoms make NNLS ill-conditioned. |
| **Detection** | Latency benchmark; NNLS conditioning and iteration counts; sensitivity of activations to small input perturbations. |
| **Mitigation** | Only the CSM dictionary is on the projection path — the LSM layer is evidence and is optional at runtime. `M` bounded by the redundancy penalty. Dictionary coherence monitored. Latency is a Phase-06 reported metric. |
| **Owner** | Phase 06 |

---

## R-14 — Live inference becomes nondeterministic

| | |
|---|---|
| **Probability** | Low |
| **Severity** | **Critical** |
| **Description** | Any fitting, RNG use, batch statistic, or iteration-order dependence at inference destroys comparability — the property the entire architecture exists to provide. It can enter innocently: a PCA re-fitted "just for the plot", a normalisation computed over the batch. |
| **Detection** | Static check for `fit`/`fit_transform`/`partial_fit`/RNG in the inference path. Repeat-run byte comparison. Single-spectrum vs batch-of-N comparison. Two-machine comparison. |
| **Mitigation** | Architectural prohibition with a closed permitted-operations list. All four checks are Phase-06 gates. Frozen PCA is *applied*, never fitted, and its artefact carries the disclaimer in the file itself. |
| **Owner** | Phase 06 |

---

## R-15 — SERS assumptions contaminate the Raman foundation

| | |
|---|---|
| **Probability** | Low |
| **Severity** | **Critical** |
| **Description** | Any SERS data, SERS-motivated band weighting, or SERS-derived prior entering the Raman foundation would encode substrate preference as chemistry. GAIRA's own evidence: on Ag colloid 50 of 51 analytes homogenise onto a purine-like attractor, and raw theme cosine 0.92 is a baseline artefact requiring null correction. |
| **Detection** | Audit every fitting input against the corpus card's `excluded_domains`. Audit band weightings for SERS-motivated choices. |
| **Mitigation** | Corpus card exclusions enforced and re-verified in Phase 00. Raman-only fitting is an architectural rule (P-10). SERS is modelled later as an explicit observation model over the Raman latent state. Biological and SERS material may be *projected* through a frozen atlas, never fitted. |
| **Owner** | Phase 00, Phase 01, Phase 02 |

---

## R-16 — Source/excitation confounding within a class

| | |
|---|---|
| **Probability** | Medium |
| **Severity** | Medium |
| **Description** | Source and excitation are partially confounded: `gobbato` is entirely 785 nm while `RamanBioLib` spans nine excitations, and 785 nm is 62% of the corpus. If a class is drawn overwhelmingly from one source, its local decomposition may model instrument response rather than chemistry — and Strategy D's partitioning makes this *more* likely, not less, because it removes the diluting effect of the rest of the corpus. |
| **Detection** | Per-class source and excitation composition table (mandatory Phase-02 output). Test whether LSMs correlate with source rather than chemistry. |
| **Mitigation** | Report composition per class. Where a class is source-dominated, flag every LSM from it and consider excluding the class from CSM integration. Excitation tracked as a nuisance factor throughout, as in V5. |
| **Owner** | Phase 00 (measure), Phase 02 (report and act) |

---

## R-17 — The corpus, not the architecture, is the binding constraint

| | |
|---|---|
| **Probability** | Medium |
| **Severity** | **High** |
| **Description** | 167 analytes across 18 classes, with several classes at 1–3 analytes and sphingolipids absent entirely. It is genuinely possible that no architecture recovers fine-family chemistry from this corpus, and V7 fails for reasons no rebuild could fix. |
| **Detection** | Phase 07 fails Tier-1 criteria while Phase 02/03 diagnostics look healthy — stable LSMs, coherent CSMs, complete provenance, but flat retrieval. |
| **Mitigation** | This is an acceptable and useful outcome (P-13), not a defeat to be argued around. Phase 09 exists for it, driven by V7's own residual analysis. Diagnose it explicitly rather than by elimination: if per-class LSM quality is good and integration is clean but retrieval does not move, the constraint is data. |
| **Owner** | Phase 07, Phase 09 |

---

## Summary

| ID | Risk | P | S | Owner |
|---|---|---|---|---|
| R-01 | Class labels bias local decompositions | High | Med | 00/02/03 |
| R-02 | Rare classes too small for NMF | High | Med | 02/03 |
| R-03 | Local dictionaries incomparable | Med | **High** | 03 |
| R-04 | Motif proliferation | Med | Med | 02/03 |
| R-05 | Consensus clustering arbitrary | Med | Med | 03 |
| R-06 | Second NMF removes detail | Med | **High** | 03 |
| R-07 | Communities threshold-artefactual | High | Med | 03 |
| R-08 | Anchors duplicate learned motifs | Med | Med | 03 |
| R-09 | Alias / replicate leakage | Med | **Critical** | 00 |
| R-10 | In-sample evaluation | Med | **Critical** | all |
| R-11 | Themes decorative | Med | Med | 04 |
| R-12 | BSV axes correlated | Med | Med | 04/05 |
| R-13 | Runtime too complex | Low | Med | 06 |
| R-14 | Inference nondeterministic | Low | **Critical** | 06 |
| R-15 | SERS contamination | Low | **Critical** | 00/01/02 |
| R-16 | Source/excitation confounding | Med | Med | 00/02 |
| R-17 | Corpus is the binding constraint | Med | **High** | 07/09 |

**The four Critical risks — R-09, R-10, R-14, R-15 — are all detectable by mechanical checks
and all have Phase gates.** None of them should ever surprise this project; the register
exists so that they cannot.
