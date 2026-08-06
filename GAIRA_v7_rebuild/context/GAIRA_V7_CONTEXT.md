# GAIRA V7 — Canonical Scientific Context

This is the reference context document for every V7 implementation prompt. Any later V7
phase prompt may assume this document and nothing else about the project's history.

> **Status updated 2026-08-06.** Phases 00, 01, 02, 02.5, 03, 04, 04.5 and 05 are **COMPLETE**.
> The V5 atlas `09ed804a40836f4a05a91ba10900cded` is unmodified and remains in production.
> §6 of this document describes the **original** target representation; three of its layers were
> built, measured and archived on evidence. The current and planned architecture is in
> `architecture/GAIRA_V7_TARGET_ARCHITECTURE.md`, and the evidence ledger is in
> `context/GAIRA_V7_ARCHITECTURE_STATUS_AFTER_PHASE05.md`. This document is retained as the
> record of the scientific motivation, which is unchanged.
>
> **Corpus counts.** This document says "167 analytes" throughout, which was the pre-audit
> normalised-name count. The canonical figure established by the Phase 01 corpus identity audit
> is **154 canonical molecules** from **375 spectra** in **16 fine chemistry classes**. The V5
> baseline tables are still reported at n = 167 because that is how they were frozen.

---

## 1. What GAIRA is trying to be

GAIRA is a domain-aware Raman/SERS evidence and interpretation engine. Its scientific
commitments predate and constrain V7:

1. Spectra are **mixtures**, not fingerprints.
2. **Peak ≠ molecule.** A single peak never licenses a molecular identity claim.
3. Prefer **biochemical themes, subfamilies, and motifs** over definitive molecule assignment.
4. Weighting is **domain-aware** (serum vs EV vs plasma vs tissue vs pathogen).
5. Interpretation is **uncertainty-aware**: region-based mapping, multi-assignment,
   ambiguity tracking, confidence tiers.
6. Literature assignments are **not ground truth**.
7. Source-backed, assignment-grade evidence outranks vague mention.

V7 changes the *representation*, not these commitments.

---

## 2. The original architecture (V5, frozen and still in production)

```
All Raman grounding spectra  (375 spectra, 167 analytes)
        ↓  canonical preprocessing (asls → savgol → L2, 450–1800 cm⁻¹, 2 cm⁻¹, 676 bins)
single global NMF  (one basis for the whole corpus, seed 0)
        ↓
24 components   H ∈ ℝ^{24×676},  fingerprint 09ed804a40836f4a05a91ba10900cded
        ↓  band-and-exemplar overlay
MSS motifs   (13 in v1, rebuilt to 18 in V6)
        ↓  soft membership
themes       (13 in the v2 ontology; 6 at the selected V6.2 medium level)
        ↓
BSV
```

Inference is a fixed-dictionary NNLS projection onto `H`, then the overlay and mapping
layers, then reference normalisation and OOD scoring. Nothing is fitted at inference time.

---

## 3. What held up — and must be preserved in V7

These are not incidental; they are why the engine is trustworthy at all. **V7 inherits every
one of them.**

| Property | Why it matters | Evidence |
|---|---|---|
| **Deterministic preprocessing** | Identical input → identical coordinates, on any machine, with no lab volume mounted | `tools/reproduce_gaira_foundation.py` reproduces the basis to a byte-identical fingerprint |
| **Non-negative spectral projection** | Activations are additive contributions, not signed abstractions; a negative "amount of protein chemistry" is meaningless | NNLS throughout `src/gaira/engine/` |
| **Frozen coordinate system** | Two spectra measured years apart are comparable because the axes did not move | fingerprint pinned and checked on every engine load |
| **Transparent reconstruction** | Every activation can be pushed back to a spectrum and residual-checked | explained variance 0.712 at k=24 |
| **Continuous evidence, not hard classification** | The output is a coordinate, not a label; downstream users decide thresholds | BSV is a vector, not an argmax |
| **MSS as the strongest interpretation layer** | The motif layer beat both the raw component layer and the theme layer on retrieval | MSS top-1 **0.677** vs coord 0.659 vs theme 0.605 (V6.2 ontology, n=167) |
| **Broad-superclass chemistry recovery** | The representation genuinely carries coarse chemistry | broad-6 retrieval: coord **0.820**, MSS **0.808**, themes 0.767 — against a size-matched random-ontology control of ~0.10 |
| **BSV as a coordinate, not a label** | Enables ΔBSV, trajectories, cohort comparison, and DART | `src/gaira/engine/bsv.py` |
| **Domain context kept separate from spectral evidence** | Serum priors never contaminate the universal representation | `src/gaira/engine/domain.py` is downstream of the BSV |

**A note on the broad-superclass result.** The 0.82/0.81 figures are not artefacts of an
easier task. The V6.3 revalidation ran twelve size-matched *random* ontologies as a control:
random 6-class grouping gives 0.096–0.113 top-1. The gain beyond mechanical class-count
reduction is **+0.55** at coord level and **+0.57** at MSS level. The coarse chemistry is
really in the representation.

---

## 4. What failed, or stayed limited

All numbers below come from committed or on-disk tables; the source is named for each.

### 4.1 Global NMF weights spectra, not molecules

The objective `min ‖X − WH‖²` sums over **rows of X**. A row is one spectrum. Therefore:

- an analyte with 3 replicate spectra exerts 3× the pull of an analyte with 1;
- a chemical class with 32 analytes exerts ~32× the pull of a class with 1.

The corpus is severely unbalanced at the class level
(`results/v6_rebuild/tables/p2_family_census.csv`, 167 analytes across 18 families):

| Family | Analytes | Uncovered by any v1 MSS motif | % uncovered |
|---|---:|---:|---:|
| protein | 32 | 17 | 53.1 |
| saccharide | 27 | 17 | 63.0 |
| amino_acid | 17 | 11 | 64.7 |
| triglyceride | 15 | 14 | **93.3** |
| organic_acid | 15 | 8 | 53.3 |
| fatty_acid | 12 | 10 | **83.3** |
| sterol | 9 | 7 | **77.8** |
| cofactor | 6 | 2 | 33.3 |
| unknown | 6 | 4 | 66.7 |
| purine | 5 | 0 | 0.0 |
| polysaccharide | 5 | 4 | 80.0 |
| lipid | 5 | 4 | 80.0 |
| nucleic_acid | 3 | 3 | **100.0** |
| pyrimidine | 3 | 0 | 0.0 |
| phospholipid | 2 | 2 | **100.0** |
| small_nitrogenous | 2 | 1 | 50.0 |
| carotenoid | 2 | 2 | **100.0** |
| polyol | 1 | 1 | 100.0 |
| **Total** | **167** | **107** | **64.1** |

Protein + saccharide + amino_acid + triglyceride + organic_acid = 106 of 167 analytes
(63%). Those five families set the reconstruction objective. Nucleic acid, phospholipid,
and carotenoid together contribute 7 analytes and were entirely unclaimed by the v1 motif
set. Replicate structure adds a second, smaller layer of the same problem: 87 of 167
analytes have replicates, 272 replicate groups, max group size 3.

### 4.2 Components mix shared and molecule-specific chemistry

Component purity (dominant-family fraction, `assets/foundation/component_registry_v1.json`):

- **Only 3 of 24 components reach purity ≥ 0.5** (0.803, 0.544, 0.521).
- Median purity **0.328**; minimum 0.187.
- Bootstrap stability is by contrast healthy: median 0.799, mean 0.812, min 0.653, p10 0.517.

So the components are *stable* but *not chemically resolved*. A stable mixture of two
chemistries is still a mixture. This is the signature of a basis asked to reconstruct
everything with a shared vocabulary.

Supporting structure: intrinsic dimensionality of the 24-component space is participation
ratio **15.2**, effective entropy rank **18.4**, 16 components for 90% of latent variance.
The basis is not 24 independent chemistries; it is roughly 15–18 effective directions,
some of which are duplicated.

### 4.3 Motifs borrow unrelated component mass

From `results/v6_rebuild/tables/p2_motif_audit.csv` (v1 motif set audited against the
frozen components). Two examples, both damning and both fixed only partially in V6:

- **`sterol_ring_system`** — spectral purity 0.244, band fidelity 0.018, discriminative AUC
  **0.683** (the worst in the set). Its top activating analytes are *arachidonic acid,
  trilinolenin, trilinolein, adenine* — the top activating family is **fatty_acid**, not
  sterol. The motif was reading acyl-chain mass.
- **`porphyrin_macrocycle`** — top activating analytes *melanin, **thymine**, myoglobin,
  hemoglobin*. A pyrimidine ranks second.
- **`flavin_redox_cofactor`** — corpus coverage 2 analytes (**1.2%**).
- **`carboxylate_organic_acid`** vs **`colloid_matrix_background`** — activation
  correlation 0.678, component-support cosine 0.687. A biochemical motif and a
  non-biochemical matrix motif were nearly collinear.
- **`porphyrin_macrocycle`** vs **`flavin_redox_cofactor`** — activation correlation 0.693,
  support cosine 0.699.

V6 rebuilt the motif layer from spectroscopy alone (dropping the circular `parent_theme`
weight, splitting `lipid_acyl_chain` into `fatty_acyl_chain` + `triglyceride_ester`, and
rebuilding `sterol_ring_system`), taking the set from 13 to 18 motifs. That helped. It did
not change the underlying components the motifs must read from.

### 4.4 Fine-family retrieval plateaued; the ontology was not the bottleneck

V6.3 rebuilt the evaluation ontology to test whether the fine-family retrieval ceiling was a
*labelling* problem rather than a *representation* problem. It was not.

From `results/v6_rebuild/v63_ontology_revalidation/tables/`:

| Level | Old ontology (18 classes) | V6.3 fine (16) | V6.3 broad (6) | Random control |
|---|---:|---:|---:|---:|
| coord (24 components) | 0.659 | 0.647 | **0.820** | 0.096 |
| **MSS (17 motifs)** | **0.677** | **0.671** | **0.808** | 0.098 |
| themes (6 continuous) | 0.605 | 0.623 | 0.767 | 0.106 |
| systems (4 coarse) | 0.509 | 0.569 | 0.695 | 0.113 |

Significance (`v63_statistics.csv`, n = 167, McNemar + permutation):

- coord, old vs fine: Δ = **−0.012**, p = 0.83 — **not significant**
- MSS, old vs fine: Δ = **−0.006**, p = 1.00 — **not significant**
- theme, old vs fine: Δ = +0.018, p = 0.63 — **not significant**
- **system, old vs fine: Δ = +0.060, p = 0.041 — the only significant fine-level gain**
- every level, old vs broad: Δ = +0.11 to +0.19, p < 1e-5 — significant, but that is the
  mechanical effect of coarsening plus real coarse chemistry

**Conclusion: the cleaned ontology materially improved only the four-class system level.**
Semantic hierarchy cleanup did not solve fine-resolution representation. Fine-family
retrieval remains ~0.65–0.68.

### 4.5 The failures are genuine representation failures

The failure waterfall (`v63_waterfall.csv`) decomposes every old-ontology failure:

| Level | Old failures | Resolved by better fine labels | Near-miss (right broad class) | **True representation errors** | New failures introduced |
|---|---:|---:|---:|---:|---:|
| coord | 57 | 9 | 22 | **26 (45.6%)** | 11 |
| **MSS** | **54** | **7** | **16** | **31 (57.4%)** | **8** |
| themes | 66 | 10 | 21 | 35 (53.0%) | 7 |
| systems | 82 | 15 | 18 | 49 (59.8%) | 5 |

**57% of MSS failures remained genuine representation failures after ontology cleanup.**
These spectra are not mislabelled and not near-misses; the coordinate system does not
separate them. No amount of relabelling fixes that.

### 4.6 The structural diagnosis

The four findings above have one cause.

> A flat global decomposition asks a single basis to simultaneously represent
> **broad shared structure** (amide I, CH₂ deformation, ring breathing — present in most
> of the corpus) and **rare molecule-specific structure** (the ester carbonyl that
> separates a triglyceride from a free fatty acid, the sterol ring system, the flavin
> isoalloxazine). These are competing objectives under one reconstruction loss, and the
> dense classes win.

Consequences, all observed:

- broad chemistry is recovered well (0.81–0.82) because it is what the dense classes share;
- fine chemistry is recovered poorly (0.65–0.68) because the discriminating structure is
  rare and cheap to sacrifice;
- rare high-quality chemistries never receive dedicated components at all — sterol,
  porphyrin, flavin, phosphate, phospholipid, and carboxylate chemistry remained thin or
  poorly isolated;
- motifs built on top inherit the mixing and borrow mass from unrelated components.

---

## 5. The V7 change of statistical unit

This is the single most important idea in V7.

**From:**

> one spectrum = one vote

**To:**

> one canonical molecule = one scientific reference unit

Replicates remain scientifically essential — but for the *right* purposes:

| Replicates are used for | Replicates are **not** used for |
|---|---|
| quality estimation (SNR, baseline artefacts, cosmic rays) | increasing a molecule's weight in the fit |
| uncertainty on the analyte prototype | increasing a chemical class's share of basis capacity |
| stability and bootstrap resampling | manufacturing apparent statistical support |
| reproducibility checks across excitation and instrument | — |

And the same logic is applied one level up: **each chemical class gets its own local
decomposition**, so a 32-analyte protein family cannot consume the capacity that sterol
chemistry needs.

Two guardrails, stated now so no later phase can quietly violate them:

1. **Balancing is not oversampling.** Rare classes must never be bootstrapped by
   duplicating identical spectra. Duplicating a spectrum adds zero information; it only
   moves the loss surface while inflating apparent support. A class with 2 analytes has
   2 analytes' worth of evidence and every downstream artefact must say so.
2. **Chemical class is an organisational prior, not a target.** Classes are used to
   *partition the decomposition*, never as the inference output. V7 does not predict class.

---

## 6. The V7 target representation — **AS ORIGINALLY SPECIFIED; see the status note above**

```
Raw Raman grounding spectra
        ↓
canonical preprocessing                    (unchanged from V5 — asls / savgol / L2)
        ↓
analyte-balanced reference construction    (Phase 01 — one molecule, one unit)
        ↓
class-specific local latent decompositions (Phase 02 — per-class NMF, adaptive k_c)
        ↓
stable Local Spectral Motifs (LSMs)        (Phase 02 — stability-selected)
        ↓
cross-class motif integration              (Phase 03 — similarity graph over all LSMs)
        ↓
Consensus Spectral Motifs (CSMs)           (Phase 03 — the canonical evidence unit)
        ↓
soft biochemical themes                    (Phase 04 — ARCHIVED A-13, evidence F-01)
        ↓
absolute continuous Biochemical State Vector (BSV)   (Phase 05 — ARCHIVED A-14, evidence F-02)
        ↓
context-aware interpretation               (domain layer, downstream, unchanged in kind)
```

Every stage must remain:

- **non-negative** — additive contributions only;
- **interpretable** — every axis traceable to bands, LSMs, analytes, and sources;
- **deterministic** — fixed seeds, fixed order, byte-reproducible artefacts;
- **provenance-preserving** — every CSM knows which LSMs, classes, analytes, and datasets support it;
- **suitable for live projection** — inference is projection and arithmetic, never fitting;
- **suitable for a future Raman→SERS observation model** — the Raman representation is the
  latent state; SERS is a *measurement channel* applied to it, never a training domain for it;
- **suitable for future DART trajectories** — BSV(E, t) must be well-defined and comparable
  across time points in a fixed coordinate system.

---

## 7. What V7 explicitly does not assume

- **Not assumed: NMF-on-NMF wins.** A second factorisation over the LSM activation matrix
  is *one candidate* for Phase 03, benchmarked against graph communities, consensus
  clustering, and hierarchical dictionaries. See `architecture/LEARNING_MODE_ARCHITECTURE.md`.
- **Not assumed: V7 beats the current atlas.** Replacement requires meeting pre-registered
  criteria in `plan/SUCCESS_CRITERIA.md`, frozen during Phase 00, measured in Phase 07.
- **Not assumed: more components is better.** The current basis already has ~15–18
  effective directions in 24 slots. Motif proliferation is a tracked risk.
- **Not assumed: the class prior is correct.** Phase 02 must test whether the chemical-family
  partition is biasing the local decompositions, and must report classes where it does.

---

## 8. Relationship to the existing frozen atlas

The V5 atlas (`09ed804a40836f4a05a91ba10900cded`) **remains the production coordinate system
throughout V7 development.** It is the control arm. It is not modified, not re-fitted, and
not deprecated until Phase 07 delivers evidence that V7 clears the pre-registered bar.

If V7 fails to clear the bar, the correct outcome is a documented negative result and a
retained V5 atlas — not a lowered bar.

---

## 9. Terminology pointer

`MSS` is **legacy terminology**. The V7 canonical term for the cross-class evidence unit is
**Consensus Spectral Motif (CSM)**. Full definitions, including the mapping
`legacy MSS → V7 CSM`, are in `context/TERMINOLOGY_AND_DEFINITIONS.md`.
