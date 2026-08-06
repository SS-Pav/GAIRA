# GAIRA V7 — Phase 01 Scientific Investigation

## Are the Local Spectral Motifs a defensible biochemical dictionary?

**Role:** Principal Investigator, adversarial review · **Date:** 2026-08-06 ·
**Branch** `gaira-v7-rebuild`

**Premise of this document.** Phase 01 passed every engineering gate and every architecture
compliance check. That is not evidence that its motifs are scientifically valid. The purpose
here was to **falsify** the Phase 01 conclusions, not to confirm them, and to approve the
phase only if the evidence survived.

**It did not survive intact.** A defect in the `k_c` selection criterion was found, diagnosed,
corrected and validated. The corrected dictionary then survived all ten investigations.

---

## 0. Executive summary

| Investigation | Verdict |
|---|---|
| 1 Uniqueness | ✅ 0 duplicate pairs; max within-class cosine 0.841; nothing to merge |
| 2 Per-molecule reconstruction | ⚠→✅ **falsified the original result**; corrected; 6 of 154 molecules remain below EV 0.5, all corpus-driven |
| 3 `k_c` robustness | ✅ 0 of 16 classes on a knife edge |
| 4 Source consistency | ⚠ 1 of 4 testable classes source-dependent; **the 4 confounded classes are untestable** |
| 5 Spectroscopic interpretability | ⚠→✅ assignment method was context-free and wrong; corrected; motifs now chemically diagnostic |
| 6 Coverage | ✅ 148 of 154 molecules well represented; 6 orphans diagnosed |
| 7 Hidden redundancy | ✅ 26 cross-class pairs identified as Phase 02 hypotheses; none merged |
| 8 Sensitivity | ✅ seed 1.000, noise 0.999, molecule bootstrap 0.932 |
| 9 Corpus vs algorithm | ✅ after the fix, **0 classes algorithm-limited**; 9 corpus-limited |
| 10 Readiness | **Approved with documented risks** |

**The single most important finding:** the reported class-average explained variance of 0.757
concealed individual molecules reconstructed at **EV 0.12**. Class averages are the wrong
statistic for a dictionary whose purpose is to represent molecules.

---

## 1. The falsification that succeeded, and the fix

### 1.1 What was wrong

Phase 01 reported per-class explained variance and nothing finer. Computing per-molecule
reconstruction inside each class's own basis gave:

| Molecule | Class | EV (before) |
|---|---|---:|
| urea | small_nitrogenous | **0.123** |
| thymine | pyrimidine | **0.130** |
| tubulin | peptide_protein | **0.180** |
| trilinolenin | acylglycerol | **0.288** |
| malic acid | carboxylic_acid_metabolite | 0.322 |
| xylanase | peptide_protein | 0.387 |

A dictionary that reconstructs a protein at EV 0.18 is not representing that protein.

### 1.2 Diagnosis

Decomposing the composite term-by-term across the `k` sweep isolated a single cause. For
`acylglycerol`, moving k=2→3:

| Term | Δ (weighted) |
|---|---:|
| held-out reconstruction | **+0.082** |
| **redundancy** | **−0.361** |
| activation sparsity | +0.021 |
| **net composite** | **−0.048** |

The `redundancy` criterion — defined as the **maximum pairwise cosine** between components,
weighted −1.0 — was cancelling genuine reconstruction gains. At k=3 the max cosine was 0.807:
two acylglycerol motifs sharing acyl-chain bands. **That is chemically expected and correct,
not duplication.**

Three separate errors compounded in that one term:

1. **It measured the wrong quantity.** The criterion's stated purpose is "is it a duplicate?".
   Duplicates are cosine ≈ 1. Penalising 0.807 penalises shared chemistry.
2. **It is a `max` statistic.** One overlapping pair poisons the score for the whole `k`.
3. **It double-counted.** The rejection stage already removes duplicates at cosine ≥ 0.95.
   Redundancy was penalised once in selection and again in rejection.

I first hypothesised that cosine between non-negative Raman vectors has an inherently high
floor, making the metric meaningless. **That hypothesis was wrong and I discarded it:** the V5
atlas components have median pairwise cosine 0.098 and max 0.521, so near-orthogonal
non-negative bases are achievable. The problem was the criterion's definition, not the metric's
floor.

### 1.3 The fix

`redundancy` = **fraction of component pairs at cosine ≥ 0.95** (the duplication threshold
already used by the rejection stage), replacing max pairwise cosine.

### 1.4 Validation — held-out, not in-sample

The decisive question is whether extra components **generalise or memorise**. In-sample EV
always improves with `k`; held-out EV does not.

| Class | `k_c` before→after | worst in-sample EV | **held-out EV** | stability | duplicate pairs |
|---|---|---|---|---|---:|
| peptide_protein | 2 → **10** | 0.180 → 0.938 | **0.645 → 0.718** | 1.000 → 0.950 | 0 → 0 |
| acylglycerol | 1 → **3** | 0.288 → 0.872 | **0.580 → 0.800** | 1.000 → 1.000 | 0 → 0 |
| fatty_acid | 2 → **5** | 0.595 → 0.902 | **0.579 → 0.674** | 1.000 → 1.000 | 0 → 0 |
| sterol_steroid | 2 → **3** | 0.575 → 0.816 | **0.508 → 0.585** | 1.000 → 1.000 | 0 → 0 |
| free_amino_acid | 5 → **7** | 0.488 → 0.633 | **0.266 → 0.281** | 0.950 → 0.893 | 0 → 0 |
| polysaccharide | 1 → **2** | 0.585 → 0.869 | 0.520 → 0.560 | 1.000 → 1.000 | 0 → 0 |
| carboxylic_acid_metab. | 3 → **2** | 0.322 → 0.307 | 0.144 → 0.133 | 0.972 → 1.000 | 0 → 0 |

**Held-out EV improved in every class where `k_c` rose.** Stability stayed ≥0.89. Zero
duplicate pairs at any `k`, confirming the old metric was never detecting duplication.

One class regressed slightly (`carboxylic_acid_metabolite`, k 3→2, held-out 0.144→0.133).
Investigated separately: at **every** k from 1 to its ceiling of 4, held-out EV stays in
0.12–0.16. That class is corpus-limited, not selection-limited (§9).

### 1.5 Consequences

| | before | after |
|---|---:|---:|
| LSMs | 33 | **50** |
| `k_c` values | {1, 2, 3, 5} | **{1, 2, 3, 5, 6, 7, 10}** |
| Corpus mean per-molecule EV | 0.757 | **0.853** |
| Molecules below EV 0.5 | 14 | **6** |
| Typing: shared / subfamily / discriminating | 26 / 7 / **0** | 21 / 26 / **3** |
| Classes flagged prior-dominated (R-01) | 6 | **5** |

The typing change matters most for what comes next. Before the fix the
`molecule_discriminating` type was **empty**; Phase 02's consensus step depends on that typing
to know which motifs may be merged across classes and which must not. Under-decomposition had
silently removed the distinction Phase 02 needs.

It also revises an earlier conclusion: `peptide_protein`, `fatty_acid` and `sterol_steroid`
were flagged prior-dominated before the fix and are not flagged after. **Prior-domination was
partly an artefact of under-decomposition, not a property of the partition.**

---

## 2. Investigation 1 — Are the LSMs genuinely unique?

Pairwise cosine, Pearson and Spearman between every retained LSM within each class
(`inv1_uniqueness_v1.csv`; figures 2 and 3).

| | |
|---|---:|
| Maximum within-class cosine | **0.841** |
| Pairs ≥ 0.95 (duplication threshold) | **0** |
| Pairs ≥ 0.90 | **0** |
| Mean within-class cosine | 0.30 |

**No motif is a duplicate of another and nothing should be merged.** The dendrograms (fig. 3)
show no branch merging below the 0.95 threshold in any class, so no lower `k_c` would produce
the same representation — the converse question the brief asked.

Pearson and Spearman agree with cosine on ordering; because all vectors are non-negative,
Pearson is systematically lower (it subtracts the mean) but never reorders the top pairs.

---

## 3. Investigation 2 — Per-molecule reconstruction

Post-fix (`inv2_per_molecule_reconstruction_v1.csv`; figures 4 and 5).

| Class | n | `k_c` | mean EV | **worst EV** | band EV |
|---|---:|---:|---:|---:|---:|
| peptide_protein | 30 | 10 | 0.975 | **0.938** | 0.983 |
| acylglycerol | 17 | 3 | 0.968 | 0.872 | 0.978 |
| fatty_acid | 17 | 5 | 0.955 | 0.902 | 0.972 |
| phospholipid_sphingolipid | 5 | 2 | 0.941 | 0.910 | 0.959 |
| polysaccharide | 5 | 2 | 0.920 | 0.869 | 0.942 |
| nucleic_acid_polymer | 3 | 1 | 0.918 | 0.897 | 0.955 |
| sterol_steroid | 10 | 3 | 0.902 | 0.816 | 0.927 |
| sulfur_thiol_cofactor | 4 | 2 | 0.872 | 0.617 | 0.899 |
| chromophore_pigment | 4 | 2 | 0.805 | 0.625 | 0.889 |
| mono_oligosaccharide | 20 | 6 | 0.790 | 0.614 | 0.841 |
| free_amino_acid | 18 | 7 | 0.774 | 0.633 | 0.839 |
| **purine** | 5 | 2 | 0.671 | **0.495** | 0.802 |
| **phosphate_metabolite** | 3 | 1 | 0.573 | **0.491** | 0.669 |
| **pyrimidine** | 3 | 1 | 0.540 | **0.130** | 0.643 |
| **carboxylic_acid_metabolite** | 8 | 2 | 0.527 | **0.307** | 0.599 |
| **small_nitrogenous** | 2 | 1 | 0.517 | **0.123** | 0.598 |

**Corpus mean EV 0.853; 6 of 154 molecules below 0.5.**

### Answering the brief's specific questions

**Are proteins really represented well?** Yes, now. Worst-case protein EV is 0.938 and the mean
is 0.975 — the best of any class. Before the fix the worst was 0.180.

**Why did peptide_protein need only `k_c = 2` despite 30 molecules?** It did not. That was the
defect. At adequate capacity it takes `k_c = 10`, which is chemically sensible: 30 proteins
spanning globular, fibrous, enzymic and transport families are not one substructure.

**Are acylglycerols genuinely homogeneous, or is the optimisation collapsing diversity?** It was
collapsing diversity. At `k_c = 1` the worst acylglycerol reconstructed at EV 0.288; at
`k_c = 3` it reaches 0.872. They are *relatively* homogeneous — 3 motifs for 17 molecules
against protein's 10 for 30 — which fits their chemistry: all are triacylglycerols differing
mainly in acyl-chain length and unsaturation.

**Purines, pyrimidines, small nitrogenous compounds** remain weak, and the diagnosis is
different (§9): all three are ceiling-bound at `k_c = 1` because ⌊n/2⌋ = 1 for n = 2–3. Thymine
at EV 0.130 shares a single basis vector with uracil and cytosine. This is a corpus limit the
specification's own ceiling enforces, not an optimisation failure.

---

## 4. Investigation 3 — Robustness of `k_c`

Every class refitted at `k_c − 1`, `k_c`, `k_c + 1`, comparing held-out reconstruction,
stability, activation sparsity, duplicate fraction, and Hungarian basis match
(`inv3_kc_robustness_v1.csv`; figure 6).

**0 of 16 classes sit on a knife edge** — defined in advance as a neighbouring `k` gaining more
than 0.05 held-out EV. The largest neighbouring gain anywhere is +0.034
(`phospholipid_sphingolipid`).

Several classes show large *negative* neighbour deltas (`nucleic_acid_polymer` −0.580,
`chromophore_pigment` −0.260, `phosphate_metabolite` −0.101), meaning `k_c + 1` is markedly
worse. Those are firmly determined, not marginal.

Hungarian alignment shows the selected basis is largely contained in the `k_c + 1` basis
(mean matched cosine > 0.9 in most classes): raising `k` splits an existing motif rather than
reorganising the dictionary, which is the expected behaviour of a well-conditioned
factorisation.

**`k_c` is robust.** The one caveat is that the ceiling ⌊n/2⌋ is never tested — for the four
smallest classes `k_c` equals the ceiling, so we cannot know whether more capacity would help
without relaxing a constraint the specification imposes for good reason.

---

## 5. Investigation 4 — Source consistency

For each class, every spectrum was projected into that class's own basis and activation
coefficients compared across acquisition sources (Mann–Whitney, Bonferroni-corrected;
`inv4_source_consistency_v1.csv`; figure 7).

| Class | sources compared | spectra | motifs differing | verdict |
|---|---|---:|---:|---|
| free_amino_acid | RamanBioLib vs amino-acid grounding | 75 | 0 / 7 | indistinguishable |
| mono_oligosaccharide | RamanBioLib vs Gobbato | 43 | 0 / 6 | indistinguishable |
| fatty_acid | RamanBioLib vs Gobbato | 27 | 0 / 5 | indistinguishable |
| **carboxylic_acid_metabolite** | RamanBioLib vs Gobbato | 23 | **2 / 2** | **source-dependent** |

**This is the most serious unresolved risk in Phase 01, and not for the reason it appears.**

Only 4 of 16 classes are testable. The other 12 draw essentially all their molecules from one
library. And **the four classes flagged source-confounded in Phase 00 — `peptide_protein`
(94% RamanBioLib), `acylglycerol` (94%), `sterol_steroid` (91%), `nucleic_acid_polymer`
(100%) — are precisely the ones that cannot be tested**, because there is no second source to
compare against.

So: source confounding in the classes where it is most likely is **untested, not disproven**.
Where it *could* be tested, three of four classes are clean. The one failure
(`carboxylic_acid_metabolite`, both motifs source-dependent, p = 0.016) is also the class with
the worst held-out reconstruction — consistent with a class whose 8 molecules are too diverse
and too source-split to support a coherent basis.

**No source correction was applied.** Any correction would have to be fitted on the classes
that *can* be tested and extrapolated to those that cannot, which would import an untested
assumption into the foundation. The honest action is to carry the risk forward explicitly:
Phase 02 must test cross-source agreement of any consensus motif built from these four classes.

---

## 6. Investigation 5 — Spectroscopic interpretability

### 6.1 A methodological error, found and corrected

The first pass used a context-free band-assignment table. It reported the 702 cm⁻¹ band of
`sterol_steroid.m00` as **"purine ring breathing"**. In a sterol, 702 cm⁻¹ is the classic
cholesterol ring skeletal mode. A referee would reject that assignment immediately.

Raman band assignment is inherently context-dependent: the same wavenumber means different
things in different chemistry. The assignment layer was rebuilt to be **conditioned on the
chemistry class the motif was fitted in**, with a generic fallback where no class-specific
assignment applies. 44% of listed bands now receive a class-specific diagnostic assignment;
the rest are honestly labelled generic.

### 6.2 The motifs are chemically real

Figure 8 shows the corrected assignments. Selected examples:

**`acylglycerol.m00`** (class-shared, n=12, stability 1.00)
`1068` all-trans acyl C–C · `1304` CH₂ twist · `1440` CH₂ scissoring ·
**`1744` ESTER C=O** — the band that distinguishes an acylglycerol from a free fatty acid.

**`fatty_acid.m00`** (subfamily, n=10, stability 1.00)
`1064` all-trans acyl C–C · `1128` all-trans · `1296` CH₂ twist · `1438` CH₂ scissoring —
the same acyl-chain architecture **without the ester carbonyl**.

That contrast is the strongest single piece of evidence in this investigation. Two classes,
fitted completely independently with no shared information, learned the same acyl-chain
chemistry and differed exactly where the chemistry differs. Nothing told the algorithm that
triacylglycerols have an ester linkage.

**`sterol_steroid.m00`** — `702` cholesterol ring skeletal · `1442` CH₂ scissoring ·
`1672` C=C stretch (Δ5 unsaturation). Textbook sterol.

**`peptide_protein.m00`** (n=18) — `520` S–S disulfide · `856` tyrosine Fermi doublet ·
`1246` amide III · `1452` CH₂/CH₃ · `1676` amide I. Textbook protein backbone.

**`purine.m00`** — `722` purine ring breathing (adenine 723 / guanine 730) · `1334` purine ring
stretch · `1488` imidazole ring stretch.

**`chromophore_pigment.m00`** — `1006` C–CH₃ rocking · `1156` C–C polyene ·
`1518` C=C polyene. The canonical resonance-enhanced carotenoid triplet, recovered from 4
molecules.

### 6.3 Why these motifs exist

The class-shared motifs encode the **structural invariant** of their class: the amide backbone
for proteins, the all-trans acyl chain for lipids, the fused ring for sterols, the conjugated
polyene for carotenoids. The subfamily motifs encode **the axis along which the class varies** —
for fatty acids, the balance of all-trans (1128) versus gauche (1096) conformers and the
presence of cis unsaturation (1265); for proteins, secondary-structure differences in the
amide I/III envelope.

That is exactly the decomposition the architecture intends: a shared chemistry motif plus the
axes of variation within it.

---

## 7. Investigation 6 — Coverage

**148 of 154 molecules (96.1%)** reconstruct above EV 0.5. Six orphans
(`inv6_orphan_diagnosis_v1.csv`):

| Molecule | Class | EV | n in class | `k_c` | ceiling | Diagnosis |
|---|---|---:|---:|---:|---:|---|
| urea | small_nitrogenous | 0.123 | 2 | 1 | 1 | corpus — class at the size floor |
| thymine | pyrimidine | 0.130 | 3 | 1 | 1 | corpus — ceiling-bound |
| malic acid | carboxylic_acid_metabolite | 0.307 | 8 | 2 | 4 | corpus — chemical heterogeneity |
| fumarate | carboxylic_acid_metabolite | 0.388 | 8 | 2 | 4 | corpus — chemical heterogeneity |
| phosphoenolpyruvate | phosphate_metabolite | 0.491 | 3 | 1 | 1 | corpus — ceiling-bound |
| urate | purine | 0.495 | 5 | 2 | 2 | corpus — ceiling-bound |

**None is a preprocessing failure** — mean quality scores 0.655–0.902, all above the QC floor.
Urea's 0.655 is the lowest and is worth noting, but a quality issue would not produce EV 0.123
on its own.

Urate is chemically interesting rather than defective: it is the only oxopurine with a fully
oxidised ring in a 5-molecule class dominated by adenine/guanine-type purines. Its residual
sits at the C=O stretch, exactly where an oxopurine differs from an aminopurine.

---

## 8. Investigation 7 — Hidden cross-class redundancy

26 cross-class motif pairs at cosine ≥ 0.70, spanning 14 class pairs
(`inv7_class_pair_hypotheses_v1.csv`; figure 9). **Nothing was merged** — these are hypotheses
for Phase 02.

| Class pair | max cosine | Reading |
|---|---:|---|
| **peptide_protein ~ polysaccharide** | **0.970** | ⚠ needs scrutiny — see below |
| acylglycerol ~ fatty_acid | 0.948 | expected: shared acyl chain |
| acylglycerol ~ phospholipid_sphingolipid | 0.899 | expected: shared acyl chain |
| fatty_acid ~ sterol_steroid | 0.896 | expected: CH₂ envelope |
| fatty_acid ~ phospholipid_sphingolipid | 0.892 | expected |
| acylglycerol ~ sterol_steroid | 0.875 | expected |
| purine ~ sulfur_thiol_cofactor | 0.830 | plausible: both ring systems with heteroatoms |
| chromophore_pigment ~ peptide_protein | 0.818 | plausible: flavin/porphyrin cofactors are protein-associated |

The **lipid superfamily co-clusters exactly as it should**: acylglycerol, fatty_acid,
phospholipid_sphingolipid and sterol_steroid form a dense block. Four independently fitted
class bases converged on a shared acyl-chain chemistry without any shared information. That is
a strong positive signal for Phase 02 consensus.

**The one pair that concerns me is `peptide_protein.m03 ~ polysaccharide.m00` at 0.970.** Band
positions are near-identical (482/480, 864/862, 942/940, 1130/1130) and both are generic
C–C / C–O skeletal envelopes with no class-specific diagnostic band. Two readings: either
this is genuine shared carbohydrate chemistry (several corpus proteins are glycoproteins), or
one of the two is a low-information "background" motif that a consensus step would wrongly
promote. **Phase 02 must resolve this before merging.** It is the single highest-priority
item in the cross-class hypothesis set.

---

## 9. Investigation 8 — Sensitivity

Four perturbations, 8 repeats each, measured as Hungarian-matched basis similarity to the
reference fit (`inv8_sensitivity_v1.csv`; figure 10).

| Perturbation | mean similarity | worst class |
|---|---:|---:|
| random seed | **1.000** | 1.000 |
| 1% spectral noise | **1.000** | 1.000 |
| 5% spectral noise | **0.999** | 0.988 |
| molecule bootstrap (80%) | **0.932** | 0.815 |

Seed and noise invariance are essentially perfect — unsurprising given the deterministic seed
schedule and `nndsvda` initialisation, but worth confirming rather than assuming.

Molecule bootstrap is the meaningful test and the dictionary holds at 0.932 mean. The worst
class (0.815) is the expected behaviour of a small class where removing 20% of molecules
removes real chemistry rather than redundancy.

**Motif identity persists under every perturbation.**

---

## 10. Investigation 9 — Corpus versus algorithm

After the `k_c` correction (`inv9_limitations_v1.csv`; figure 11):

| Attribution | classes |
|---|---:|
| Adequate (worst-molecule EV ≥ 0.7) | 7 |
| **Corpus-limited** | **9** |
| **Algorithm-limited** | **0** |

Every remaining weakness is corpus-driven, in one of two ways:

**Ceiling-bound (4 classes).** `small_nitrogenous` (n=2), `pyrimidine` (3),
`phosphate_metabolite` (3), `nucleic_acid_polymer` (3) have ⌊n/2⌋ = 1, so a single basis vector
must represent every molecule. Thymine at EV 0.130 is the direct consequence. **Only corpus
expansion resolves this** — the ceiling exists precisely to stop motifs becoming memorised
molecules, and relaxing it for n=3 would defeat its purpose.

**Heterogeneity-limited (5 classes).** `carboxylic_acid_metabolite` is the clearest case: 8
molecules spanning mono-, di- and tri-carboxylic, hydroxy- and keto-acids. Held-out EV stays
at 0.12–0.16 at **every** k from 1 to the ceiling of 4. Eight molecules do not span that
chemistry at any capacity.

**What Phase 02 can and cannot fix.** Phase 02 pools motifs across classes, so it can help
where a class's chemistry is shared with a better-populated neighbour — the lipid superfamily
is the obvious case. It cannot help `pyrimidine` or `small_nitrogenous`, whose chemistry has no
well-populated neighbour in this corpus. Those require Phase 08 corpus expansion, and
Investigation 9 identifies exactly which chemistries: pyrimidines, small nitrogenous
metabolites, phosphate metabolites, and small organic acids.

**No datasets are recommended for addition here.** The requirement is a measured residual
direction, which is Phase 08's job, not this investigation's.

---

## 11. Formal scientific review

### Major strengths

1. **The motifs are chemically real, and demonstrably so.** The acylglycerol/fatty-acid
   contrast — same acyl chain, ester carbonyl present in one and absent in the other, from two
   completely independent fits — is the kind of evidence that cannot be produced by
   over-fitting. Nothing told the algorithm about ester linkages.
2. **The capacity reallocation works as designed.** Rare chemistry receives 0.411 decomposition
   units per molecule against 0.299 for dense chemistry; under the V5 global fit both received
   0.156.
3. **Statistical robustness is strong.** Zero duplicate motifs, zero knife-edge `k_c`, basis
   similarity 0.93–1.00 under every perturbation tested.
4. **The lipid superfamily emerges without being told.** Four independently fitted class bases
   converged on a shared acyl-chain motif — a genuine, unprompted discovery and the best
   available evidence that Phase 02's consensus premise is sound.
5. **Determinism and provenance are complete.** Identical registry across runs; every artefact
   hashed.

### Major weaknesses

1. **Source confounding is untested where it matters most.** The four classes flagged
   confounded are exactly the four that cannot be tested, because each draws ≥91% of its
   molecules from a single library. This is the most serious unresolved risk.
2. **Six molecules remain badly represented** (EV 0.12–0.50), four of them ceiling-bound in
   classes of 2–3 molecules.
3. **`carboxylic_acid_metabolite` fails at every capacity** — held-out EV 0.12–0.16 across the
   whole `k` range, with both motifs source-dependent. This class should be treated as
   provisional in Phase 02.
4. **The `k_c` ceiling is never tested.** Four classes sit exactly at ⌊n/2⌋; we cannot know
   whether more capacity would help them without relaxing a constraint that exists for a reason.
5. **No downstream benefit is demonstrated.** Phase 01 delivers a dictionary. Whether it
   improves retrieval or the BSV is unmeasured.

### Unexpected discoveries

1. **Under-decomposition masqueraded as a scientific finding.** Before the fix, three classes
   were flagged "prior-dominated" — all motifs class-shared, no internal structure — and the
   `molecule_discriminating` type was empty. Both looked like statements about the chemistry.
   Both were artefacts of a defective selection criterion. **A method defect can present as a
   biological conclusion, and every engineering gate will still pass.**
2. **The `peptide_protein ~ polysaccharide` pair at cosine 0.970.** Unexpected, unexplained,
   and the highest-priority item for Phase 02 scrutiny.
3. **Class averages actively concealed the problem.** Class-mean EV 0.757 looked acceptable
   while individual molecules sat at 0.12.

### Remaining risks

| Risk | Severity | Mitigation |
|---|---|---|
| Source confounding untestable in 4 single-source classes | **High** | Phase 02 must test cross-source agreement of consensus motifs built from them |
| `carboxylic_acid_metabolite` unrepresentable and source-dependent | Medium | treat as provisional; do not let it anchor a consensus motif |
| 6 orphan molecules, 4 ceiling-bound | Medium | Phase 08 corpus expansion; documented as corpus-limited |
| `peptide_protein.m03 ~ polysaccharide.m00` may be a background motif | Medium | resolve before any Phase 02 merge |
| `k_c` ceiling untested at the small-class boundary | Low | revisit only if the corpus grows |

### Confidence in the representation

**7.5 / 10.**

The dictionary is chemically interpretable, statistically robust, uniquely resolved and
reproducible. It loses points for the untestable source confounding in four classes (the
single largest unknown), for six molecules that remain badly represented, and for having
required a significant method correction discovered only by adversarial review — which raises
the reasonable question of what else has not yet been probed.

### Likelihood that Phase 02 improves the biochemical hierarchy

**High (~75%).** The strongest evidence is the lipid superfamily converging independently
across four class fits: that is precisely the structure consensus clustering exists to find,
and it is present in the data rather than assumed. The typing distribution (21 shared / 26
subfamily / 3 discriminating) gives Phase 02 the discrimination it needs, which it did not have
before the fix.

### Likelihood that Phase 02 merely propagates existing artefacts

**Low–moderate (~25%).** Two specific channels, both identified and both mitigable:

- The four single-source classes contribute 16 of 50 motifs. If those encode instrument
  response, consensus clustering will pool the artefact across classes and give it the
  appearance of cross-class support. Mitigation: source-stratified validation of any consensus
  motif drawing on them.
- The `peptide_protein ~ polysaccharide` pair at 0.970 could promote a low-information
  background motif into a consensus motif with apparently strong support. Mitigation: resolve
  before merging.

---

## 12. Modifications made, and their justification

| # | Change | Before → after | Justification |
|---|---|---|---|
| 1 | `redundancy` = duplicate-pair fraction, not max pairwise cosine | 33 → 50 LSMs; worst protein EV 0.180 → 0.938 | **Held-out** EV improved in every class where `k_c` rose; stability ≥0.89; zero duplicate pairs at any k. Validated on generalisation, not in-sample fit. |
| 2 | Class-conditioned Raman band assignment | 702 cm⁻¹ in a sterol: "purine ring breathing" → "cholesterol ring skeletal mode" | Raman assignment is context-dependent; the original was an assignment error a referee would reject. |

Both are in `src/gaira/v7/lsm/classlocal.py` and the investigation module respectively, with
the reason recorded in the code. No architecture was changed, no Phase 00 artefact touched, and
the frozen atlas fingerprint is unchanged.

---

## 13. Deliverables

**Tables** (16) — `inv1_uniqueness` · `inv2_per_molecule_reconstruction` ·
`inv2_bandwise_residuals` · `inv2_class_reconstruction_summary` · `inv3_kc_robustness` ·
`inv3_knife_edge` · `inv4_source_consistency` · `inv4_source_per_motif` ·
`inv5_spectroscopic_interpretation` · `inv6_coverage` · `inv6_orphan_diagnosis` ·
`inv7_intermotif_similarity` · `inv7_cross_class_candidates` · `inv7_class_pair_hypotheses` ·
`inv8_sensitivity` · `inv9_limitations`

**Figures** (11, SVG + PNG) — `phase01_investigation/figures/`

**Artefacts** — `investigation_summary_v1.json`, `inv1_similarity_matrices.npz`
