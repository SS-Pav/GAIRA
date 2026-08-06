# GAIRA V7 — Prior Architecture Limitations

A consolidated, evidence-cited account of where the V5/V6/V6.2/V6.3 architecture reached its
limit. Every claim names its source table. This document exists so that no V7 phase has to
re-argue *why* the rebuild is happening, and so that any claim of V7 improvement has a
precise baseline to improve on.

Companion documents: `GAIRA_V7_CONTEXT.md` (narrative), `SCIENTIFIC_DESIGN_PRINCIPLES.md`
(what V7 does about it), `../plan/SUCCESS_CRITERIA.md` (what counts as fixed).

---

## L-01 — The reconstruction objective counts spectra, not molecules

**Mechanism.** Global NMF minimises `‖X − WH‖²_F`, a sum over rows of `X`. Each row is one
spectrum. The gradient contribution of a chemistry is therefore proportional to how many
spectra carry it, not to how chemically distinct it is.

**Evidence.** Corpus card (`assets/foundation/manifold.json`): 375 spectra, 167 analytes,
272 replicate groups, 87 analytes with replicates, max replicate group size 3. Family census
(`results/v6_rebuild/tables/p2_family_census.csv`): 18 families over a 32→1 analyte range.

**Magnitude.** Five families (protein 32, saccharide 27, amino_acid 17, triglyceride 15,
organic_acid 15) hold 106 of 167 analytes — **63%** of the objective. Four families
(nucleic_acid 3, phospholipid 2, carotenoid 2, polyol 1) hold **8 analytes, 4.8%**.

**Why it is structural, not incidental.** Adding more spectra of the same molecules does not
fix it. The imbalance is in *chemical coverage*, which reflects what reference libraries
actually contain — protein and sugar references are abundant, sterol and flavin references
are not. Any global objective will inherit that skew.

---

## L-02 — Components are stable but chemically impure

**Evidence.** `assets/foundation/component_registry_v1.json`, 24 components:

| Metric | Value |
|---|---|
| Components with purity ≥ 0.5 | **3 of 24** |
| Purity values ≥ 0.5 | 0.803, 0.544, 0.521 |
| Median purity | 0.328 |
| Minimum purity | 0.187 |
| Median bootstrap stability | 0.799 |
| Mean bootstrap stability | 0.812 |
| p10 bootstrap stability | 0.517 |
| Minimum bootstrap stability | 0.653 |

**The diagnosis.** High stability with low purity is the signature of a *reproducibly mixed*
basis. The fit reliably converges to the same components; those components reliably describe
several chemistries at once. This is not a convergence problem or a seed problem — those
would show as low stability. It is a capacity-allocation problem.

**Corroboration.** Intrinsic dimensionality of the 24-component space
(`manifold.json → stats.intrinsic_dimensionality`): participation ratio **15.2**, effective
entropy rank **18.4**, and 16 components carry 90% of latent variance. Twenty-four slots
encode roughly 15–18 effective directions — so some capacity is spent on duplication while
rare chemistry gets none.

---

## L-03 — The curated motif layer borrows unrelated component mass

Because the motif layer was laid *over* an already-mixed basis, motifs could only score by
combining components that carried foreign chemistry.

**Evidence.** `results/v6_rebuild/tables/p2_motif_audit.csv` (v1 motif set, 13 motifs):

| Motif | Spectral purity | Band fidelity | Discriminative AUC | Corpus coverage | Top-activating family |
|---|---:|---:|---:|---:|---|
| `sterol_ring_system` | 0.244 | **0.018** | **0.683** | 3 analytes (1.8%) | **fatty_acid** ✗ |
| `porphyrin_macrocycle` | 0.328 | 0.099 | 0.995 | 4 analytes (2.4%) | protein (top-4 includes **thymine**) ✗ |
| `flavin_redox_cofactor` | 0.432 | 0.126 | 1.000 | **2 analytes (1.2%)** | cofactor ✓ |
| `glycan_co_network` | 0.201 | **0.009** | 0.911 | 12 analytes (7.2%) | saccharide ✓ |
| `colloid_matrix_background` | 0.279 | 0.030 | 0.941 | 6 analytes (3.6%) | organic_acid (non-biochemical motif) |
| `protein_amide_backbone` | 0.399 | 0.087 | 0.934 | 9 analytes (5.4%) | protein ✓ |

`sterol_ring_system` is the clearest failure: a motif nominally about the sterol ring system,
whose top four activating analytes are *arachidonic acid, trilinolenin, trilinolein, adenine*.
It was reading acyl-chain and ring-breathing mass because no component isolated sterol
chemistry for it to read.

**Motif redundancy** (`p2_motif_redundancy.csv`, activation correlation / support cosine):

| Pair | Activation corr | Support cosine |
|---|---:|---:|
| `porphyrin_macrocycle` ↔ `flavin_redox_cofactor` | 0.693 | 0.699 |
| `carboxylate_organic_acid` ↔ `colloid_matrix_background` | 0.678 | 0.687 |
| `purine_ring_breathing` ↔ `sterol_ring_system` | 0.243 | **0.679** |
| `lipid_acyl_chain` ↔ `sterol_ring_system` | 0.582 | 0.501 |

The third row is the tell: purine and sterol motifs share 0.68 component support while their
activations barely correlate. They are built from overlapping component mass that means
different things in different places.

**Coverage.** The v1 motif set named exemplars for only 35.9% of the 167 corpus analytes;
**107 of 167 (64.1%)** were unclaimed by any motif — including 100% of nucleic_acid,
phospholipid, and carotenoid, and 93.3% of triglyceride.

**What V6 fixed and what it did not.** V6 rebuilt the layer from spectroscopy alone
(`results/v6_rebuild/artifacts/mss_motifs_v6.yaml`): dropped the circular `parent_theme`
weight that had fed the theme ontology back into the motif score, split `lipid_acyl_chain`
into `fatty_acyl_chain` + `triglyceride_ester` (the 15 triglycerides and 12 free fatty acids
separate on the ester carbonyl ~1745 and C-O-C ~1160, neither of which v1 had), rebuilt
`sterol_ring_system`, and expanded exemplar coverage — 13 → 18 motifs. That improved the
overlay. **It could not change the components underneath.**

---

## L-04 — Fine-family retrieval plateaued at ~0.65–0.68

**Evidence.** `results/v6_rebuild/v63_ontology_revalidation/tables/v63_metrics_by_ontology.csv`
(n = 167 analytes, analyte-grouped evaluation):

| Level | Dim | Old ontology (18) | V6.3 fine (16) | V6.3 broad (6) | Random control |
|---|---:|---:|---:|---:|---:|
| coord | 24 | 0.6587 | 0.6467 | **0.8204** | 0.0963 |
| **MSS** | 17 | **0.6766** | **0.6707** | **0.8084** | 0.0983 |
| theme_raw | 6 | 0.6048 | 0.6228 | 0.7665 | 0.1063 |
| theme_posterior | 6 | 0.6048 | 0.6228 | 0.7665 | 0.1063 |
| system_raw | 4 | 0.5090 | 0.5689 | 0.6946 | 0.1133 |

95% CI at MSS/old: [0.605, 0.749]. MRR 0.758. Mean first rank 6.98.

**Reading it.** The fine-family ceiling sits at 0.65–0.68 regardless of which fine ontology is
used. The MSS layer is the best layer at every ontology — confirming the motif abstraction
helps — but it helps by about +0.02 over raw components, not by breaking the ceiling.

Note also that `theme_raw` and `theme_posterior` are **numerically identical at every
metric**. The Bayesian posterior refinement over themes changed no decisions. That is worth
knowing: added machinery on top of an insufficient representation bought nothing.

---

## L-05 — Ontology cleanup was not the fix

V6.3 tested the hypothesis that the ceiling was a *labelling* artefact. It rebuilt the
evaluation ontology (18 → 16 fine classes, plus a 6-class broad level) and re-ran everything.

**Evidence.** `v63_statistics.csv` (McNemar + permutation, n = 167):

| Level | Comparison | Δ accuracy | 95% CI | McNemar p | Significant |
|---|---|---:|---|---:|---|
| coord | old vs fine | **−0.0120** | [−0.060, 0.042] | 0.824 | **No** |
| MSS | old vs fine | **−0.0060** | [−0.054, 0.042] | 1.000 | **No** |
| theme | old vs fine | +0.0180 | [−0.030, 0.066] | 0.629 | **No** |
| **system** | old vs fine | **+0.0599** | [0.012, 0.114] | **0.041** | **Yes** |
| coord | old vs broad | +0.1617 | [0.096, 0.228] | 3.5e−06 | Yes |
| MSS | old vs broad | +0.1317 | [0.078, 0.186] | 3.0e−06 | Yes |
| system | old vs broad | +0.1856 | [0.120, 0.252] | 3.7e−08 | Yes |

**Conclusion.** The cleaned fine ontology produced **one** significant improvement, at the
four-class system level (+0.060, p = 0.041) — the coarsest level, where labelling noise
mattered most. At coord and MSS the change was negative and non-significant.

The broad-level gains are real but partly mechanical (fewer classes is an easier task). The
V6.3 design controlled for this with twelve size-matched *random* ontologies: random 6-class
grouping scores 0.096–0.113. The `gain_beyond_mechanical` column of `v63_comparison.csv`
records **+0.550** at coord and **+0.572** at MSS. So coarse chemistry is genuinely present —
but the fine resolution problem is untouched.

**Semantic hierarchy cleanup did not solve fine-resolution representation.**

---

## L-06 — The majority of failures are genuine representation failures

**Evidence.** `v63_waterfall.csv` decomposes each old-ontology failure into four causes:

| Level | Old failures | Resolved by better labels | Near-miss (right broad class) | **True representation errors** | New failures introduced |
|---|---:|---:|---:|---:|---:|
| coord | 57 | 9 (15.8%) | 22 (38.6%) | **26 (45.6%)** | 11 |
| **MSS** | **54** | **7 (13.0%)** | **16 (29.6%)** | **31 (57.4%)** | **8** |
| theme_raw | 66 | 10 (15.2%) | 21 (31.8%) | 35 (53.0%) | 7 |
| system_raw | 82 | 15 (18.3%) | 18 (22.0%) | 49 (59.8%) | 5 |

**57.4% of MSS failures are true representation errors.** For those spectra the coordinate
system places chemically distinct molecules in the same neighbourhood, and no relabelling,
no re-mapping, and no posterior refinement can separate them. Only a different
representation can.

Note the last column: the cleanup also *introduced* 8 new MSS failures. Ontology surgery is
not free.

---

## L-07 — Thin and poorly isolated chemistry

Chemistries that remained thin, poorly isolated, or both — each with its measured deficit:

| Chemistry | Corpus support | Observed deficit |
|---|---|---|
| **Sterol / steroid** | 9 analytes; 7 uncovered (77.8%) | motif AUC 0.683, band fidelity 0.018, top-activated by fatty acids |
| **Porphyrin / heme** | motif coverage 4 analytes (2.4%) | 0.69 redundancy with flavin; thymine in top-4 activators |
| **Flavin** | motif coverage **2 analytes (1.2%)** | 0.69 redundancy with porphyrin |
| **Phosphate** | no dedicated v1 motif (added in V6 as `nucleic_backbone_phosphate`) | nucleic_acid family 100% uncovered in v1 |
| **Phospholipid** | **2 analytes; 100% uncovered** | no motif, no component |
| **Sphingolipid** | not represented | — |
| **Carboxylate / organic acid** | 15 analytes; 8 uncovered (53.3%) | 0.678 activation correlation with the non-biochemical colloid/matrix motif |
| **Carotenoid** | **2 analytes; 100% uncovered** | — |
| **Nucleic acid** | **3 analytes; 100% uncovered** | — |

These are precisely the chemistries a serum/EV/tissue interpretation engine most needs to
resolve, and precisely the ones a global objective has least incentive to model.

---

## L-08 — The structural root cause

All of L-01 through L-07 reduce to one statement:

> **A flat global decomposition asks one basis to simultaneously represent broad shared
> structure and rare molecule-specific structure, under a single reconstruction loss, with
> capacity allocated by spectrum count.**

Broad shared structure (amide I/III, CH₂/CH₃ deformation, ring breathing, C–O–C stretch)
appears in most of a 375-spectrum corpus. Rare molecule-specific structure (ester carbonyl,
sterol ring, isoalloxazine, phosphate backbone) appears in a handful of spectra. Under one
squared-error loss, modelling the former is worth vastly more than modelling the latter.

The observed consequences line up exactly:

| Prediction of the diagnosis | Observed |
|---|---|
| Broad chemistry recovered well | 0.820 coord / 0.808 MSS broad-6 retrieval, +0.55 over random control |
| Fine chemistry recovered poorly | 0.647–0.671 fine retrieval, flat across ontologies |
| Rare chemistry gets no dedicated capacity | 3/24 components pure; sterol, flavin, phospholipid, carotenoid unisolated |
| Overlay layers inherit the mixing | motifs borrow foreign mass; AUC 0.68 on sterol |
| Adding abstraction on top does not help | theme_posterior ≡ theme_raw at every metric |
| Relabelling does not help | one significant fine-level gain out of four |

---

## What V7 must therefore change

Not "more components". Not "a better ontology". Not "a better overlay". The change must be
to **where capacity is allocated and what the statistical unit is**:

1. **Unit change** — one canonical molecule is one reference unit, not one spectrum
   (addresses L-01).
2. **Capacity change** — decompose *within* chemical classes so a 32-analyte family cannot
   consume the capacity a 2-analyte family needs (addresses L-01, L-02, L-07).
3. **Adaptive capacity** — `k_c` chosen per class on evidence, not a single global `k`
   (addresses L-02, L-07).
4. **Bottom-up motifs** — CSMs derived from cross-class consensus of local fits, not curated
   onto a fixed basis, so a motif with no support is *visible as a singleton* rather than
   silently borrowing mass (addresses L-03).
5. **Provenance as a first-class field** — every CSM records its LSMs, classes, analytes,
   and bands, so "which chemistry actually supports this axis" is always answerable
   (addresses L-03, L-07).
6. **Pre-registered evaluation** — analyte-grouped splits, permutation nulls, and failure
   waterfalls fixed *before* fitting, so V7 cannot repeat the V6.3 experience of measuring
   the wrong thing well (addresses L-04, L-05, L-06).

Whether these changes actually raise fine-family retrieval is an **open empirical question**,
answered in Phase 07 against the criteria in `../plan/SUCCESS_CRITERIA.md`. This document
establishes the problem, not the outcome.
