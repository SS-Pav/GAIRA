# GAIRA V7 — Architecture Status After Phase 05

**Written** 2026-08-06, after Phase 05 completed and was pushed.
**Scope** documentation only. No code, no algorithm, no artefact, no number in this document
was produced by anything run for it — every figure is quoted from a committed phase table with
its source named.

This document supersedes nothing. It *records* what the accumulated evidence now supports,
which architectural decisions survive it, and which are archived. Archived decisions are
preserved in full, labelled, and remain reproducible from their committed phase outputs. No
history is rewritten.

Companion documents updated in the same pass: `GAIRA_V7_TARGET_ARCHITECTURE.md`,
`TERMINOLOGY_AND_DEFINITIONS.md`, `plan/GAIRA_V7_REBUILD_PLAN.md`,
`plan/SUCCESS_CRITERIA.md`, `plan/VALIDATION_AND_DECISION_RULES.md`,
`plan/PHASE_DEPENDENCY_MAP.md`.

---

## 1. Executive summary

Six phases of V7 have run. The evidence they produced points in one direction, and it is not
the direction the original architecture pointed.

**Demonstrated.** Class-local decomposition followed by a consensus motif layer produces a
49-dimensional representation that generalises to molecules the atlas has never seen:
chemistry-class top-1 rises **0.608 → 0.855** from the raw spectrum to the CSM layer
(Phase 04, `hierarchy_retrieval_v1.csv`). The same layer is *more* robust than the raw spectrum
under seven physically-motivated corruptions, not less (Phase 05, retention 0.935 vs 0.895).
Provenance is complete and machine-verified: 3,133 chains, none broken.

**Falsified.** Every attempt to build an abstraction layer *above* the CSM layer has lost
information without returning anything measurable. Soft themes: class top-1 **0.855 → 0.405**
(Phase 04). Second-order NMF over CSM activations: 0.185 of the CSM layer's information
retained, class top-1 0.392 (Phase 04.5, discarded). A declared eleven-axis evidence profile,
built specifically to avoid the discovery step that was blamed for the theme layer's failure,
still lands at 0.664 (Phase 05). Three independent constructions, three losses, one direction.

**Unknown.** Whether a *chemistry-conditioned* layer — one that carries 16 class-evidence
coordinates rather than 4–11 discovered or declared abstractions — behaves differently. Nothing
in V7 has tested this. It is the hypothesis Phase 06 exists to test, and §6 states plainly why
it might fail for the same reason the others did.

**One inconsistency found while writing this document, and it is material.** The frozen success
criteria (`SUCCESS_CRITERIA.md`, frozen in Phase 00, P-13 forbids adjusting them) define the
V7 replacement bar as **fine-16 chemistry-class retrieval**: V5 MSS scores 0.6707 and V7 must
reach ≥ 0.7507. Meanwhile `TERMINOLOGY_AND_DEFINITIONS.md` states that chemical class "is not
the inference output — V7 never predicts class." The project has been measuring itself on a
task its own vocabulary forbids it to perform, since Phase 00. §7 resolves this. It is not a
new conflict introduced by the proposed architecture; the proposed architecture makes an
existing conflict visible.

---

## 2. What has been scientifically demonstrated

Each row cites the committed table it comes from. "Unseen molecule" means molecule-grouped
cross-validation on the frozen `v7_cv_v1` folds — the molecule is absent from the reference
bank, so the number is not self-matching.

| # | Demonstrated | Evidence | Source |
|---|---|---|---|
| **D-01** | The corpus, folds, metrics and V5 control are fixed and reproducible | atlas rebuilt from raw to a byte-identical fingerprint `09ed804a…`, max abs diff 0.0; 12/12 gates | Phase 00 |
| **D-02** | Class-local decomposition yields a usable motif dictionary | 16 independent class-local NMF fits → 50 stable LSMs | Phase 01 |
| **D-03** | The `k_c` selection rule had a real defect, and correcting it mattered | the redundancy criterion penalised *shared chemistry* rather than duplication; dictionary 33 → 50 LSMs after correction, validated on held-out generalisation | Phase 01 investigation |
| **D-04** | Independently-learned local motifs almost never describe the same phenomenon | of 1,225 candidate pairs, 4 merges proposed, **1 survived falsification**; 48 of 49 CSMs are singletons | Phase 02 |
| **D-05** | Motif space is a low-dimensional continuum with one real bipartition | kNN modularity 0.620 vs degree-preserving null 0.070 ± 0.003 (p < 0.001); mean local intrinsic dimension 3.9 of ambient 676; the only defensible cluster count is K = 2, hydrophobic/polar, bootstrap ARI 0.879 | Phase 02.5 |
| **D-06** | **The abstraction stack pays up to the CSM layer** | class top-1 on unseen molecules: raw 0.608 → LSM 0.850 → **CSM 0.855** | Phase 04 `hierarchy_retrieval_v1.csv` |
| **D-07** | Molecule identity is retrievable when the molecule is represented | Split A molecule top-1 0.605, top-5 0.795 (154-way, chance 0.006) | Phase 05 |
| **D-08** | Chemistry-class inference generalises to unseen molecules | Split B top-1 **0.845**, top-3 0.971, macro F1 0.807, balanced accuracy 0.797 over 16 fine classes | Phase 05 |
| **D-09** | The CSM layer is both more accurate *and* more robust than the raw spectrum | clean class top-1 0.845 vs 0.592; retention under 7 corruptions × 5 levels 0.935 vs 0.895 | Phase 05 `robustness_summary_v1.csv` |
| **D-10** | Intensity scaling is exactly invariant at every layer | max−min class top-1 < 1e-9 across the 5-level sweep | Phase 05 |
| **D-11** | Provenance is complete end to end | 3,133 axis → CSM → LSM → molecule → spectrum chains verified against the frozen registries, **0 broken** | Phase 05 |
| **D-12** | Inference is deterministic and fits nothing | bit-for-bit identical on repeat; static check finds no fit/RNG in the inference path | Phase 04, Phase 05 |
| **D-13** | Rejection of degraded and structureless spectra works | joint AUROC 0.921; 79.9% of synthetic negatives rejected at 95% in-domain acceptance | Phase 05 |
| **D-14** | Confidence can be made informative, but not to the pre-registered ECE | selected Dirichlet calibrator: ECE 0.130, discrimination 0.891, sharpness 0.275 | Phase 05 |

### D-06 restated, because it is the load-bearing result

The six-level hierarchy measured in Phase 04 on identical frozen splits:

| level | dim | molecule top-1 (Split A) | **class top-1, unseen molecule (Split B)** |
|---|---:|---:|---:|
| L1 raw spectrum | 676 | 0.790 | 0.608 |
| L2 LSM | 50 | 0.806 | 0.850 |
| **L3 CSM** | **49** | 0.799 | **0.855** |
| L4 theme | 4 | 0.553 | 0.405 |
| L5 BSV | 4 | 0.553 | 0.405 |
| L6 geometry | 5 | 0.495 | 0.541 |

The curve rises to L3 and falls after it. Everything in §3 is a consequence of that shape.

---

## 3. What has been falsified

"Falsified" here means: a specific architectural claim was tested on pre-registered criteria and
did not hold. It does not mean the object was worthless — it means it cannot occupy the place the
architecture assigned it.

| # | Claim, as the original architecture stated it | What was measured | Verdict |
|---|---|---|---|
| **F-01** | *Soft biochemical themes are the semantic axes of the BSV* | class top-1 on unseen molecules falls 0.855 (CSM) → **0.405** (theme); molecule top-1 falls 0.799 → 0.553 | **Falsified as an inference layer.** The theme layer is a lossy re-description, not an abstraction that buys anything. |
| **F-02** | *The BSV is the canonical output coordinate* | BSV ≡ theme coordinates under the selected variant, so it inherits F-01 exactly; effective rank 2.40 of nominal K = 4 (risk R-12 realised) | **Falsified as the canonical representation.** Retained as a derived reading (§5, A-04). |
| **F-03** | *A second-order factorisation over CSM activations recovers a coarser programme layer* | Meta Components (K = 3, plain NMF): information retained vs CSM **0.185**, class top-1 0.392 vs 0.856, macro F1 0.196 vs 0.810. Won every stability axis and failed the 0.50 informativeness floor. Best achievable over all 16 (variant, K) combinations: 0.677, still below CSM | **Falsified. Discarded**, Phase 04.5. |
| **F-04** | *Discovery was the problem; a declared, band-grounded evidence layer will not lose information* | the 11-axis Biochemical Evidence Profile scores 0.664 class top-1 on unseen molecules against the CSM's 0.845 | **Partially falsified.** Better than themes (0.405) and still a 0.18 loss. Declaring the axes helped; it did not close the gap. |
| **F-05** | *The frozen atlas can detect real Ag-SERS as out-of-domain* | AUROC 0.548 on the reconstruction, 0.548 after the residual-based fix on real SERS (the synthetic probe rose 0.670 → 0.946) | **Falsified.** A non-negative Raman-motif dictionary reconstructs SERS of the same metabolites comfortably. This caused Phase 04's `GATE_FAILED` and is now out of scope (§5, A-09). |
| **F-06** | *ECE ≤ 0.10 is achievable for the retrieval confidence* | the ECE-optimal calibrator (Platt) reports **0.605 for every spectrum** — exactly the base rate — with sharpness 0.000 and the worst Brier in the table. The best informative calibrator reaches 0.130 | **Falsified as stated.** Gate G6 fails and was not relaxed. |

### The pattern across F-01, F-03, F-04 — and why it is the most important finding in V7

Three abstraction layers were built above the CSM layer by three different mechanisms:

* **discovered** by archetypal analysis over CSM co-activation (Phase 03 themes),
* **discovered** by non-negative factorisation of the CSM activation matrix (Phase 04.5 Meta),
* **declared** from Raman band assignments with no fitting at all (Phase 05 evidence axes).

All three lost class information. Two of the three additionally scored *better* than the CSM
layer on stability or reproducibility metrics while doing so — the Phase 03 softmax theme mode
was the most reproducible option available, and Meta Components won every stability axis
outright. **Reproducibility metrics are maximised by representations that say nearly the same
thing about everything.** Each phase caught this only after adding an explicit informativeness
constraint, and Phase 05 caught a fourth instance in calibration, where ECE selected a constant
predictor.

This is now recorded as a standing methodological requirement, not a per-phase discovery:
see the proposed principle **P-18** in §7.2, and gate **DG-INFO** in
`plan/VALIDATION_AND_DECISION_RULES.md`.

---

## 4. What remains unknown

| # | Unknown | Why it matters | Where it is addressed |
|---|---|---|---|
| **U-01** | Whether a 16-dimensional chemistry-evidence layer preserves information where 4-, 11- and 3-dimensional layers did not | the entire proposed architecture rests on it | **Phase 06**, gate DG-06 |
| **U-02** | How much of the class signal is an artefact of the class partition itself | the LSM fits were *organised* by the same 16 classes the evidence layer predicts (risk R-01). If the partition imprinted itself, 0.845 is partly circular | **Phase 06** must run the pre-registered R-01 control |
| **U-03** | Whether the engine rejects genuinely novel chemistry | all Phase 05 negatives are corruption or structureless signal from the same module used in the robustness study. "Open-set rejection" currently means *rejection of degraded spectra* | **Phase 06**, held-out-class experiment |
| **U-04** | Whether BSV2 over chemistry evidence escapes the F-01/F-03/F-04 pattern | it is the fourth attempt at the same architectural position | **Phase 07**, gate DG-07 with the informativeness floor pre-registered |
| **U-05** | Whether hierarchical retrieval beats direct cosine on molecule identity | Split A top-1 is 0.605; the chemistry prior is the obvious lever and it is untested | **Phase 08**, gate DG-08 |
| **U-06** | Whether V7 clears the frozen Tier-1 replacement bar under `v7_harness_v1` | **it has never been measured.** Phase 05's 0.845 is a per-spectrum number on 5-fold grouped CV; the frozen bar is a per-analyte number at n = 167. They are not the same protocol | **Phase 06** (§7.1) |
| **U-07** | Whether band *shape* resolves the 1650 cm⁻¹ degeneracy | amide I and cis C=C are indistinguishable to window-based reasoning; this is the leading cause of the amide axis's failure | deferred; noted in Phase 05 §13 |
| **U-08** | Whether the CSM layer earns its place over the LSM layer | class top-1 0.845 vs 0.848, retention 0.935 vs 0.923. The consensus step merged **one** pair of 1,225 candidates | **Phase 06** must report both layers side by side |

U-08 deserves emphasis. Phase 02 accepted a single merge, so 48 of 49 CSMs *are* single LSMs.
The two layers are nearly the same object and the numbers confirm it. Nothing measured so far
justifies the CSM layer over the LSM layer on performance; the justification is provenance and
interpretability, and that should be stated rather than implied.

---

## 5. Architectural decision ledger

Every decision that defines V7, with its evidence, its current confidence, and whether it stays.
**Confidence** is one of *established* (measured, replicated across phases), *supported*
(measured once, not yet replicated), *provisional* (reasoned, not yet measured), *refuted*.

### 5.1 Decisions that remain in V7

| ID | Decision | Evidence | Reason it stays | Confidence | Status |
|---|---|---|---|---|---|
| **A-01** | Canonical preprocessing unchanged from V5 — asLS → SG → L2, 450–1800 cm⁻¹, 2.0 step, 676 bins | Phase 00 byte-identical reproduction | comparability across every atlas version depends on it | established | **ACTIVE** |
| **A-02** | One canonical molecule = one reference unit; replicates never inflate weight | Phase 01; P-11 | the V5 failure L-01 was the objective counting spectra | established | **ACTIVE** |
| **A-03** | Class-local decomposition with adaptive `k_c` | D-02, D-03 | prevents a 32-analyte family consuming sterol chemistry's capacity | established | **ACTIVE** |
| **A-04** | Non-negativity at every layer | P-02; NNLS throughout | a negative amount of a chemistry is meaningless; additivity is what makes provenance decomposable | established | **ACTIVE** |
| **A-05** | Learning offline, inference is projection only | P-09; D-12 | comparability; batch-independence | established | **ACTIVE** |
| **A-06** | Determinism and fingerprint verification on load | D-01, D-12 | non-negotiable | established | **ACTIVE** |
| **A-07** | Provenance as a first-class field | D-11 | P-04; it is what makes a claim auditable | established | **ACTIVE** |
| **A-08** | **The CSM activation vector is the canonical representation** | D-06, D-09 | the abstraction curve peaks here and falls after | established | **ACTIVE — promoted in Phase 05** |
| **A-09** | Raman-only scope; SERS is a measurement channel, never a training or validation domain in the core | F-05; P-10 | a Raman atlas evaluated on Ag-SERS was answering a question the project does not ask | supported | **ACTIVE — narrowed in Phase 05** |
| **A-10** | Geometry (Phase 02.5) is visualisation and prior, never inference | D-05; L6 scores 0.541 vs CSM 0.855 | it was never an inference path and measuring it confirmed it should not be | established | **ACTIVE** |
| **A-11** | Chemical class partitions the decomposition | A-03 | organisational prior | established | **ACTIVE** |
| **A-12** | Frozen V5 atlas is the control arm, never a foundation | P-15 | preserves an honest comparison | established | **ACTIVE** |

### 5.2 Decisions now archived — preserved, not deleted

Archived means: removed from the active inference path, retained on disk with all outputs,
reproducible, and citable. Every archived phase keeps its report, audit, figures, tables and
tests exactly as committed.

| ID | Archived decision | Evidence that retired it | What is preserved | Where |
|---|---|---|---|---|
| **A-13** | **Soft biochemical themes as the semantic axes** (original Phase 04 of the plan; run as Phase 03) | F-01 | K = 5 archetypal model, 4 accepted themes, 1 rejected on bootstrap 0.59 < 0.60, membership matrix `S`, 15 bridge CSMs, 10 reports, full audit | `results/v7_rebuild/phase03/` |
| **A-14** | **BSV = Sᵀc as the canonical output coordinate** | F-02 | frozen projection engine, 6 abstraction levels benchmarked, BSV reference frame, effective-rank analysis | `results/v7_rebuild/phase04/` |
| **A-15** | **Meta Components / hierarchical NMF over CSM activations** | F-03 | K sweep 2–12, two variants, 12 perturbations × 5 levels, 14 figures, informativeness floor, verdict *discard* | `results/v7_rebuild/phase04_5/` |
| **A-16** | **The 11-axis declared Biochemical Evidence Profile as the interpretable layer** | F-04; superseded by the 16-d chemistry evidence layer | axis definitions, grounding test (7/11 grounded), window-overlap table, sensitivity sweep, radars, provenance waterfalls | `results/v7_rebuild/phase05/` |
| **A-17** | **SERS out-of-domain detection as a V7 gate** | F-05; A-09 | the failing experiment and its diagnosis; Phase 04's `GATE_FAILED` record | `results/v7_rebuild/phase04/` |
| **A-18** | Legacy MSS curated overlay (V5) | pre-V7: L-03, `sterol_ring_system` AUC 0.683 top-activated by fatty acids | V5 atlas in production, unmodified | V5 assets |

**A-16 is archived after one phase and that deserves a note.** The 11-axis profile is not a
failure of execution — it is grounded (7 of 11 axes discriminate their declared chemistry at
AUROC ≥ 0.70), fully provenanced, and it beat the theme layer by 0.26. It is archived because
the proposed 16-d chemistry evidence layer occupies the same architectural slot with a
label space that matches the frozen evaluation ontology (`v7_fine_16`) and the frozen success
criteria. If Phase 06 finds the 16-d layer does *not* clearly exceed it, gate DG-06 requires
reinstating A-16 rather than proceeding.

### 5.3 Decisions proposed but not yet evidenced

| ID | Proposed | Status | Evidence required |
|---|---|---|---|
| **A-19** | 16-dimensional Chemistry Evidence as the interpretable layer | **PLANNED** | Phase 06 / DG-06 |
| **A-20** | BSV2 = learned biochemical programmes over Chemistry Evidence only | **PLANNED** | Phase 07 / DG-07 |
| **A-21** | Hierarchical molecular retrieval with a soft chemistry prior | **PLANNED** | Phase 08 / DG-08 |

---

## 6. The principal scientific risk in the proposed architecture

Stated plainly, because the plan should be able to reach a negative result.

**BSV2 is the fourth attempt to build a layer above the CSM representation.** The first three —
discovered themes, discovered meta-components, declared evidence axes — all lost class
information, and two of them scored *better* on stability while doing so. The proposed BSV2
differs in its input (16-d chemistry evidence rather than 49-d motif activations) and that is a
real difference: factorising class-evidence co-occurrence is not the same object as factorising
motif usage. But the architectural position is identical, and so is the failure mode to watch
for.

Three requirements follow, and they are written into the Phase 07 gate rather than left to
judgement:

1. **The informativeness floor is pre-registered, not added after the numbers arrive.** BSV2
   must retain ≥ 0.50 of the Chemistry Evidence layer's information *and* ≥ 0.50 of its
   held-out class prediction before any stability or robustness gain is allowed to count.
2. **K is chosen on a Pareto frontier, never on reconstruction alone** (risk R-12, and the
   Phase 04.5 finding that the best K on any single metric was not the best K overall).
3. **"BSV2 does not improve on Chemistry Evidence" is an acceptable and publishable outcome.**
   If it occurs, Chemistry Evidence remains the terminal interpretable layer and Phase 08
   proceeds from it directly. P-13.

A secondary risk, U-02, is sharper than it looks: the Chemistry Evidence layer predicts the same
16 classes that partitioned the LSM fits in Phase 01. Risk R-01 has been flagged since Phase 00
and has never been fully controlled. Phase 06 must run the control — a class-agnostic
decomposition of the same corpus, evaluated on the same folds — before 0.845 can be described
as a property of the representation rather than of the partition.

---

## 7. Inconsistencies found, and how they are resolved

### 7.1 The frozen success criteria measure a task the terminology forbids

`SUCCESS_CRITERIA.md` §6, frozen in Phase 00 at commit `b9f78ff5…`:

> S-01 CSM/MSS-equivalent fine top-1 — baseline 0.6707 — **frozen threshold ≥ 0.7507**

and `phase00_baseline_metrics.csv` shows the level that number comes from:
`v5_atlas::v7_fine_16`, i.e. **retrieval of the correct fine-16 chemistry class**.

`TERMINOLOGY_AND_DEFINITIONS.md`, *Class / chemical family*:

> It is **not the inference output**. V7 never predicts class.

Both statements have stood since Phase 00. They cannot both be right.

**Resolution.** The frozen thresholds stay exactly as they are — P-13 forbids adjusting them and
nothing here adjusts them. The *terminology* is what was wrong: it conflated two different
prohibitions. What the project actually needs to forbid is class as a **terminal claim** and
class as **supervision inside the decomposition**. Class as an *intermediate, probabilistic,
uncertainty-carrying evidence coordinate* was never the danger, and it is what the frozen bar
has always measured. The amended definition is in `TERMINOLOGY_AND_DEFINITIONS.md` under
*Chemistry Evidence*, and principle **P-06** is amended in §7.2.

This resolution does not make the 16-d layer safe. It makes the *conflict* resolved. U-02
remains the open scientific question, and DG-06 requires the R-01 control before the layer is
accepted.

### 7.2 Principle amendments

| Principle | Change | Reason |
|---|---|---|
| **P-06** *Chemistry organises; chemistry does not supervise* | **AMENDED.** Chemical class remains an organisational prior for decomposition and must never supervise a local fit. It is now admissible as an *intermediate probabilistic evidence coordinate* carrying uncertainty. It remains inadmissible as a terminal hard label. | §7.1; the frozen bar has always been fine-16 retrieval |
| **P-07** *Themes are chemistry; biology is downstream* | **RETAINED, scope narrowed.** The prohibition on biology in the representation is unchanged. "Theme" is now legacy vocabulary (A-13); the principle now governs Chemistry Evidence axes and BSV2 programmes. | A-13 archived |
| **P-18** *Stability without informativeness is not evidence* | **NEW.** No representation, mode, calibrator or model may be selected on a reproducibility, stability or calibration metric unless it first clears a pre-registered informativeness floor. | four independent instances: Phase 03 softmax themes, Phase 04 theme-mode leakage, Phase 04.5 Meta stability, Phase 05 ECE-optimal constant calibrator |

### 7.3 Smaller inconsistencies, now corrected in the documents

| Found | Correction |
|---|---|
| `GAIRA_V7_CONTEXT.md` and `GAIRA_V7_TARGET_ARCHITECTURE.md` both state "Nothing in V7 has been implemented" | corrected — six phases are complete |
| Both documents cite the corpus as "167 analytes" | the canonical count is **154 canonical molecules** from 375 spectra; 167 is the pre-audit normalised-name count. Resolved in the Phase 01 corpus identity audit and now used consistently |
| `GAIRA_V7_TARGET_ARCHITECTURE.md` §2 labels the theme map "PHASE 03 ✔ COMPLETE" and BSV "PHASE 04 ✔ COMPLETE" using *original* numbering while the plan uses *canonical* numbering | both documents now use canonical numbering with the mapping table restated |
| The dependency map still shows Phase 05 = BSV and Phase 06 = engine integration | rewritten |
| `SUCCESS_CRITERIA.md` §1 quotes the baseline at n = 167 | retained verbatim as the frozen record; the n = 154 note is added alongside without altering any threshold |

---

## 8. Provenance of this document

| Claim class | Source |
|---|---|
| all V7 numbers | committed phase tables under `results/v7_rebuild/<phase>/tables/` and `artifacts/`, named per row |
| all V5 baseline numbers | `results/v7_rebuild/phase00/tables/phase00_baseline_metrics.csv` (harness `v7_harness_v1`) |
| frozen fingerprints | atlas `09ed804a40836f4a05a91ba10900cded`; LSM `208482d6f7178b5b8f16cace91be55b0`; CSM `0b4aa550ccefed3edabdbde5bae11c8d`; theme `f54d4835ffdf8aa2d50a4a203da0e8f4`; Phase 05 engine `20d8bd99ce71f45a125c6a2b1d719e51` |
| principles and risks | `SCIENTIFIC_DESIGN_PRINCIPLES.md` (P-01…P-17, + P-18 proposed here), `plan/RISK_REGISTER.md` (R-01…R-17) |

Nothing was recomputed for this document. If a number here disagrees with a phase table, the
phase table is correct and this document is wrong.
