# GAIRA V7 — Phase 02: Consensus Spectral Motif construction

**Objective:** discover whether Local Spectral Motifs learned *independently* inside separate
chemistry classes converge onto shared biochemical spectral phenomena.

**Answer:** almost never. Of 1225 candidate pairs, four groups are proposed by the graph and
**one survives falsification** — a cis-unsaturation motif shared between free fatty acids and
their triacylglycerol esters. The other three describe overlapping skeletal vibrations, not
shared phenomena, and their members remain separate.

`M = 49` · 1 accepted merge · 48 singletons · 3 merges rejected and undone ·
CSM fingerprint `0b4aa550ccefed3edabdbde5bae11c8d` · all 8 gates PASS

---

## 1. Current V7 pipeline (P-17)

```
[✔ 00]  benchmark lock · 154 canonical molecules · frozen splits
   ↓
[✔ 01]  balanced references → 16 independent class-local NMF fits → 50 LSMs
   ↓                            registry 208482d6f7178b5b8f16cace91be55b0 (consumed read-only)
[✔ 02]  seven-feature Consensus Spectral Graph
   ↓    → null calibration → significance sweep → threshold consensus
   ↓    → 4 merge proposals → falsification → 1 accepted
   ↓    49 CSMs, full provenance                        THIS PHASE
   ↓
[  03]  CSM → soft biochemical themes                   ← next
   ↓    inputs: csm_dictionary_v1.npz, csm_registry_v1.json, activation matrix
   ↓    outputs: S ∈ ℝ₊^{M×K}, theme registry, K on a pre-registered Pareto frontier
   ↓
[  04]  continuous BSV  →  [05] engine  →  [06] Raman validation
```

**Frozen V5 atlas:** `09ed804a40836f4a05a91ba10900cded`, verified before and after. It was
loaded to check its fingerprint and for nothing else (P-15).

---

## 2. Hypotheses

| | |
|---|---|
| **H0** | LSMs from independent class-local fits do not converge; apparent cross-class similarity is explained by shared broad Raman structure (CH stretches, skeletal modes). |
| **H1** | Some LSMs describe one shared spectral phenomenon, recoverable as a CSM. |

The default is **not merged**. Every candidate merge was treated as a claim carrying the burden
of proof, and the pre-registration (`config/phase02_preregistration_v1.md`, committed before any
sweep ran) fixed the evidence, the weighting and the acceptance conditions in advance.

**H0 is rejected for exactly one pair and retained for every other.**

---

## 3. Method

### 3.1 Seven edge features

Contract C-06 requires six; the brief adds reconstruction substitutability, a strictly stronger
falsification channel. All seven are computed for all 1225 pairs and stored on every edge.

| # | feature | what it catches | α |
|---|---|---|---:|
| 1 | spectral cosine | overall shape | 0.20 |
| 2 | diagnostic-band prominence agreement | agreement *where it matters*, pedestal removed | 0.25 |
| 3 | peak-position agreement | excitation-invariant, intensity-free | 0.15 |
| 4 | bootstrap co-occurrence | property of the data or of one fit | 0.15 |
| 5 | activation co-occurrence | do they respond to the same molecules | 0.10 |
| 6 | provenance overlap, within-class discounted | are they describing the same evidence | 0.05 |
| 7 | reconstruction substitutability | can one actually do the other's job | 0.10 |

The edge weight is the **weighted geometric mean**. Under an arithmetic mean a cosine of 0.95
carries an edge whose every other channel is near zero; under a geometric mean any single
near-zero channel collapses the weight. This is the operational form of *never merge motifs
solely because spectral cosine is high*, and it is why no per-feature hard floor is imposed —
floors would add six unswept arbitrary cuts, leaving the geometric mean to enforce the same
requirement continuously with exactly one cut, which is swept.

### 3.2 Feature independence — measured, not asserted

"Seven independent lines of evidence" is a claim about a correlation matrix, so the matrix is
computed and published (figure 10). Largest off-diagonal ρ = **0.81**
(spectral cosine ↔ bootstrap co-occurrence), which is expected and stated: feature 4 contains a
resampled cosine term by construction. The next largest are 0.66 (cosine ↔ activation) and 0.45
(cosine ↔ band prominence). Substitutability and provenance are near-independent of everything
(|ρ| ≤ 0.24), which is what makes them useful vetoes.

### 3.3 Null calibration

An edge weight of 0.6 means nothing until you know what 0.6 looks like when there is no shared
chemistry left to find. Sixty band-permutation replicates (each motif circularly shifted, bands
re-detected with the same detector, five of seven channels recomputed) give
**73,500 null edge weights**.

> **Null mean 0.156 against an observed mean of 0.174.** Only 61 of 1225 pairs clear p < 0.01.
> The overwhelming majority of apparent LSM similarity is exactly what generic Raman band
> statistics already produce (figure 3).

This is the single most important number in the phase. It is also invisible without the null,
which is why the null is not an appendix.

---

## 4. Validation

| # | validation | result |
|---|---|---|
| 1 | reconstruction, CSMs vs LSMs | mean EV 0.8578 → 0.8570, **Δ −0.0009**; 0 of 154 molecules beyond the 0.05 tolerance |
| 2 | bootstrap stability of the grouping | graph-proposal ARI **1.000** (mean and min, 50 resamples) |
| 3 | leave-one-class-out | ARI **1.000** for all 16 held-out classes |
| 4 | source robustness | 1 CSM flagged; adjudicated in §7 |
| 5 | spectroscopic interpretability | class-conditioned assignment on every CSM |
| 6 | cross-CSM redundancy | max cosine 0.970; 3 pairs above 0.90, **all three explicitly adjudicated** |
| 7 | false-merge investigation | null model + four pre-declared suspects, §6 |

Reconstruction damage is concentrated exactly where the one merge acts, and stays small:
arachidonic acid −0.035, linoleic acid −0.017, trilinolein −0.016, α-linolenic acid −0.011.
A merge that costs a little on precisely the molecules it claims to describe is behaving as a
merge should.

**On the redundancy gate.** Three CSM pairs sit above cosine 0.90 — the highest is
`peptide_protein.m03 ↔ polysaccharide.m00` at **0.970**. They are *not* merged. Gating on
"no CSM pair above 0.90 cosine" would contradict this phase's own thesis, so the gate was
stated as **no *unexamined* redundant pair**: every such pair must carry a recorded
adjudication. All three do.

---

## 5. Integration method — chosen on evidence, table published regardless

| method | M | composite | stability | cohesion | separation | chem. coherence | retained LSM info | held-out recon. | hyperparam. sens. | singleton frac. | redundancy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **graph_community** | 34 | **0.898** | 1.000 | 0.589 | 1.000 | 0.926 | 0.919 | 0.764 | 0.000 | 0.882 | 0.841 |
| spectral | 20 | 0.610 | 0.181 | 0.486 | 1.000 | 0.901 | 0.732 | 0.649 | 0.088 | 0.850 | 0.841 |
| hybrid | 30 | 0.536 | 0.526 | 0.322 | 0.991 | 0.859 | 0.900 | 0.772 | 0.474 | 0.667 | 0.783 |
| meta_nmf | 8 | 0.379 | 0.233 | 0.178 | 0.993 | 0.475 | 0.685 | 0.601 | 0.092 | 0.000 | 0.772 |
| consensus_clustering | 1 | 0.300 | 0.000 | 0.022 | 1.000 | 0.200 | 0.408 | 0.399 | 0.000 | 0.000 | 0.000 |

The architecture's stated prior — that graph or hybrid routes look more promising because
meta-NMF sees one of seven evidence channels — is **borne out**, and for the predicted reason:
meta-NMF returns M = 8 with **no singletons at all**, fusing 50 motifs into 8 groups on
activation co-occurrence alone, and scores 0.475 on chemical coherence against 0.926 for the
graph. R-06 does not apply: meta-NMF did not win.

`consensus_clustering` collapsing to M = 1 is not a bug. Average linkage on `1 − w` over a graph
whose 1172 non-adjacent pairs all sit at distance 1.0 has nothing to separate; it is the
expected behaviour of a topology-blind method on a sparsified graph, and it is reported rather
than quietly dropped.

**Consensus operator:** `leading_direction` (composite 0.8754) over `stability_weighted_mean`
(0.8754) and `nonnegative_medoid` (0.8691). The first two are separated by 0.00007 — for
practical purposes they are equivalent on this data, and that is worth saying rather than
presenting a coin-flip as a finding.

---

## 6. The four pre-declared false-merge suspects

All four were named in advance. All four clear the null at p ≤ 0.0001. All four are proposed by
the graph. **One survives.**

| suspect pair | w | cosine | band prom. | peaks | bootstrap | activation | subst. | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| acylglycerol ↔ fatty_acid | 0.887 | 0.948 | 0.784 | 0.857 | 0.916 | 0.954 | 0.930 | **genuine shared chemistry** |
| peptide_protein ↔ polysaccharide | 0.862 | 0.970 | 0.780 | 0.875 | 0.800 | 0.975 | 0.788 | overlapping skeletal vibrations |
| purine ↔ sulfur-thiol cofactor | 0.697 | 0.830 | 0.607 | 0.500 | 0.663 | 0.941 | 0.763 | overlapping skeletal vibrations |
| phospholipid ↔ sterol | 0.662 | 0.860 | 0.375 | 0.500 | 0.847 | 0.959 | 0.957 | lipid superfamily convergence — **not** one motif |

### acylglycerol.m01 ↔ fatty_acid.m01 — ACCEPTED as `csm00`

Bands **1654** (cis C=C stretch), **1264** (=C–H in-plane bend), 1436 (CH₂ scissoring), 1082
(C–C skeletal trans), 970, 866 (glycerol C–C). Contributors agree at cosine 0.985; isolated
reconstruction cost −0.005 EV.

Its projected support is **arachidonic acid, linoleic acid, trilinolenin, α-linolenic acid** —
every one polyunsaturated, and nothing else in the corpus. The 1654/1264 pair is the textbook
cis-unsaturation signature, and the chemistry is unambiguous: **the C=C double bond is the same
bond whether the acyl chain is free or esterified to glycerol.** A free fatty acid and its
triacylglycerol differ at the carboxyl terminus, not along the chain. Two independent
decompositions — one fitted only on triglycerides, one fitted only on free fatty acids —
recovering the same band pair is exactly the convergence Phase 02 was built to detect.

**Verdict: genuine shared chemistry.**

### peptide_protein.m03 ↔ polysaccharide.m00 — REJECTED (within `proposal03`)

The highest spectral cosine in the entire graph (0.970) and the most seductive merge. It fails
on two counts: bootstrap co-occurrence for the group falls below 0.50, and merging costs
−0.070 EV. Chemically, both motifs are dominated by the 1000–1150 cm⁻¹ C–O/C–C skeletal region
— the glycosidic C–O–C of a polysaccharide and the C–N/C–C backbone of a protein produce
overlapping envelopes in that window without being the same vibration.

**Verdict: overlapping skeletal vibrations.** Not glycoprotein biology: the supporting molecules
are unglycosylated proteins and plant polysaccharides with no glycoprotein among them, so there
is no shared molecular species to carry a glycoprotein interpretation.

### purine ↔ sulfur-thiol cofactor — REJECTED (within `proposal16`)

Grouped with `nucleic_acid_polymer.m00`, which is the tell: the group is held together by the
ring-breathing region near 700–740 cm⁻¹ shared by purine rings and by the C–S stretch region of
thiol cofactors. Peak agreement is only 0.500 and band prominence 0.607 — the two motifs agree
on where broad intensity sits, not on which peaks they have. Merging costs −0.075 EV.

**Verdict: overlapping skeletal vibrations.** Note the chemically sensible sub-structure:
coenzyme A and acetyl-CoA *contain* adenine, so a genuine purine contribution to the cofactor
class is expected — but that makes the cofactor motif a *mixture*, not the same motif.

### phospholipid ↔ sterol — REJECTED (within `proposal00`)

Substitutability is 0.957 and activation co-occurrence 0.959 — but band prominence agreement is
**0.375**, the lowest of the four. The two agree on broad lipid envelope and on which molecules
they respond to, and disagree on their diagnostic peaks. This is the exact failure mode the
architecture predicted for feature 1 and the reason feature 2 exists. The enclosing group of
eight motifs spans acylglycerol, fatty acid, phospholipid, sterol and one amino-acid motif, and
merging it costs −0.076 EV.

**Verdict: lipid superfamily convergence — real, and too coarse to be one motif.** All lipids
share CH₂ scissoring near 1440, chain C–C near 1060–1130 and carbonyl near 1730. That common
envelope is genuine biology, but collapsing it into a single CSM would discard the ester/free
distinction, the sterol ring system and the phosphate head group at once. The right home for
this structure is Phase 03: it is a **theme**, not a motif.

---

## 7. The source-confounding flag

`csm00` is flagged: both contributing LSMs draw 87.3% of their supporting molecules from
RamanBioLib, and a cross-class merge whose members share one measurement source is a candidate
instrument artefact.

**The flag does not survive contact with the provenance.** The two sides of the merge were
measured at *different excitation wavelengths*:

| side | molecules | excitation |
|---|---|---|
| free fatty acids | arachidonic, linoleic, palmitoleic, α-linolenic | **532 nm** |
| triacylglycerols | tri-11-eicosenoin, trielaidin, trierucin, tripalmitolein, tripetroselinin | **1064 nm** |
| both | oleate, triolein (also in Gobbato), trilinolein, trilinolenin | 532/785/1064 nm |

An instrument artefact would have to survive a 532 → 1064 nm change in excitation, which no
substrate or detector artefact does. Two of the fourteen supporting molecules are additionally
cross-source (RamanBioLib + Gobbato). The flag is retained in the record as a caveat, and the
counter-evidence is recorded with it.

---

## 8. Where the method was corrected

Six construction defects were found during the run. Each was demonstrated before it was changed,
each change is recorded at the point of change, and none was made to reach a desired answer.

| # | defect | evidence | correction |
|---|---|---|---|
| 1 | `band_overlap` ≈ `spectral_cosine` | ρ = **0.978**; the two carried 0.45 of the edge weight between them | peak *prominence* agreement on a shared band vocabulary; ρ → 0.45 |
| 2 | `substitutability` had no power | median 0.971, min 0.108 — a 10-motif dictionary absorbs any single swap | marginal-gain ratio against the class dictionary *without* the motif |
| 3 | `MIN_ACTIVATION` applied on the wrong scale | peak-normalised → provenance median 1.000; pooled-normalised → mean 0.003 | class-normalised share, the quantity the constant was defined for |
| 4 | threshold rule anchored to peak perturbation stability | stability rose monotonically and peaked at 0.968 where 33 of 50 motifs were isolated | see §9 |
| 5 | bootstrap perturbation invented edges | noise on the full matrix creates ~1200 spurious edges on a 53-edge graph; reported ARI 0.100 for a structure LOCO reproduced at 1.000 | perturb only edges that exist |
| 6 | per-CSM reconstruction cost confounded | measured against the fully merged basis, charging each merge for its neighbours' losses | isolate: merge one group at a time |

Defect 4 is the same failure mode as the Phase 01 `k_c` composite, where two of six criteria
were maximal by definition at k = 1 and "do not decompose" won everywhere. Both times the cause
was a criterion that is not comparable across the thing being swept.

---

## 9. R-07 fired, and what was done about it

**No contiguous run of three cuts produces an invariant partition — under either sweep.** The
raw-weight sweep's partition changes at nearly every step (communities 2, 2, 3, 3, 4, 10, 20,
24, 31, 36, 42 …), and so does the significance sweep's. The pre-registered rule returns FAIL
for both, and its own text says what that means:

> *"If no stable region exists, the graph construction is inadequate and must be revised — that
> is a finding, not a nuisance (risk R-07)."*

**The finding: the LSM similarity structure is a continuum, not a set of separated communities.**
A few strongly-supported groups are embedded in a smooth background, and forcing a partition
algorithm to assign all 50 motifs makes most of the churn happen among motifs that carry no
significant edge at all.

The estimator was therefore revised from *pick one cut inside a stable region* to *take the
consensus across the whole sweep*: two LSMs join only if they are co-assigned at **every viable
significance level**. What the sweep was meant to certify — that the answer does not depend on
where the cut falls — is now true by construction, and the co-assignment matrix is published as
the evidence.

Two further decisions inside that estimator:

**Viability.** A level at which one community holds more than half the motifs has not found
communities, and neither has one at which more than half are isolated. Including α = 0.20–0.10,
where 2–3 communities covered all 50 motifs, let those levels dominate and fused 34 of 50 LSMs
into five groups at a cost of 0.148 EV. Averaging over levels where the algorithm has visibly
failed is not consensus.

**Unanimity, not majority.** The default is *not merged*, so the burden sits on the merge.

| co-assignment rule | groups | non-trivial | LSMs absorbed | largest group |
|---|---:|---:|---:|---:|
| **1.0 (unanimous)** | **34** | **4** | **20** | **8** |
| 0.8 | 28 | 6 | 28 | 10 |
| 0.6 | 21 | 5 | 34 | 15 |
| 0.5 (majority) | 12 | 4 | 42 | 20 |

The looser the rule, the more the continuum gets swept into giant groups. All four arms are
reported.

---

## 10. Accepted and rejected consensus motifs

### Accepted — 1

| CSM | LSMs | classes | molecules | bands (cm⁻¹) | cohesion | uncertainty | isolated EV cost | status |
|---|---:|---|---:|---|---:|---:|---:|---|
| `csm00` | 2 | acylglycerol, fatty_acid | 14 | 866, 970, 1082, 1264, 1436, 1654 | 0.987 | 0.015 | −0.005 | diagnostic |

### Rejected — 3, with reasons

| proposal | LSMs | classes | cohesion | mean edge w | isolated EV cost | rejection reason |
|---|---:|---:|---:|---:|---:|---|
| `proposal00` | 8 | 5 | 0.851 | 0.564 | −0.076 | reconstruction degrades beyond tolerance |
| `proposal03` | 7 | 6 | 0.800 | 0.396 | −0.070 | bootstrap co-occurrence < 0.50; reconstruction degrades |
| `proposal16` | 3 | 3 | 0.880 | 0.508 | −0.075 | reconstruction degrades beyond tolerance |

A rejected merge is a merge that **does not happen**: all 18 members returned to being separate
CSMs rather than staying fused under a "rejected" label.

### Singletons — 48

Not failures. A singleton CSM is a Local Spectral Motif that no other class-local decomposition
independently confirmed, and it is flagged, counted and kept visible exactly as the contract
requires. That 48 of 50 motifs are singletons is the honest description of this corpus.

---

## 11. Provenance

Every CSM resolves CSM → LSM → chemistry class → canonical molecule → original spectrum, and
every level is stored rather than recomputed. `tables/csm_provenance_chain_v1.csv` holds the
fully expanded chain; nine C-07 invariants are checked and all pass, including *every LSM is
assigned to exactly one CSM* (50 assignments over 50 LSMs, 0 duplicates, 0 missing).

---

## 12. Architecture compliance (P-16)

18 of 18 specification items PASS — pooled LSMs; all six contract features plus
substitutability; provenance within-class discounted (R-01); threshold swept with selection from
a region (R-07); five methods compared with the table published; `M` justified; R-06 not
applicable; resolvable provenance; singletons and anchors visible; C-07 invariants; frozen atlas
not an input (P-15); Phase 01 consumed read-only; feature independence measured; the four named
suspects investigated; null model; rules pre-registered (P-12); pipeline redrawn (P-17).

---

## 13. Remaining scientific risks

1. **The corpus, not the method, may be the limit.** 48 singletons out of 50 could mean class-local motifs genuinely do not converge, or that 154 molecules across 16 classes is too thin for convergence to be *detectable*. These are not distinguishable from within this corpus. Phase 08 (targeted expansion) is the test.
2. **The lipid superfamily is real and unresolved.** `proposal00` is not noise — eight motifs across five classes, cohesion 0.851, every pair unanimous across the sweep. It fails as a *motif* and should be revisited as a **theme** in Phase 03. Carrying it forward as an unmerged set is the correct Phase 02 outcome, not a closed question.
3. **`csm00` rests on 14 molecules from one library**, mitigated but not eliminated by the excitation-wavelength argument in §7.
4. **Feature 4 shares a cosine term with feature 1** (ρ = 0.81). Six of the seven channels are effectively independent; the report does not claim seven.
5. **A single accepted merge is a thin basis for Phase 03.** Themes will have to be built over a CSM layer that is very nearly the LSM layer. That is a real constraint on what Phase 03 can claim, and it should shape the theme design rather than be discovered inside it.

---

## 14. Implications for Phase 03 (Biochemical Themes)

- **The CSM layer is 49 units, not a compressed basis.** Phase 03 inherits essentially the full LSM resolution. `M ≈ 50` means the soft membership matrix `S ∈ ℝ₊^{M×K}` carries the entire burden of abstraction — none of it was done here.
- **The structure Phase 02 rejected is the structure Phase 03 needs.** `proposal00` (lipid superfamily), `proposal03` (polar C–O/C–C backbone) and `proposal16` (ring systems) all failed as motifs *for the same reason*: merging them destroys reconstruction. A soft, overlapping theme assignment does not have that problem, because it never replaces the motifs. Phase 03 should treat the three rejected groups as its **strongest theme candidates**, and the co-assignment matrix as a prior.
- **Do not expect themes to be class-disjoint.** Every rejected group spans 3–6 chemistry classes. A theme layer that reproduces the class partition would be re-encoding the Phase 00 prior (risk R-01) rather than learning anything.
- **The null-calibrated edge weight transfers.** Whatever Phase 03 builds, it should be scored against the same band-permutation null. The lesson of §3.3 — that most apparent similarity is generic — applies at least as strongly one level up.
