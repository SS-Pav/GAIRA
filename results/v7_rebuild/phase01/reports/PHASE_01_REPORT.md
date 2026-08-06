# GAIRA V7 — Phase 01 Report (CANONICAL)
## Balanced references → class-local NMF → Local Spectral Motifs

**Branch** `gaira-v7-rebuild` · **Status** COMPLETE · **Architecture compliance 18/18 PASS** ·
**Gates** 8/8 PASS · **Tests** 57 passed

> **Revised 2026-08-06 following the Phase 01 scientific investigation.** The `k_c` selection
> composite contained a defect — the duplicate/redundancy criterion penalised *shared
> chemistry* rather than duplication — which suppressed `k_c` across the corpus and left
> individual molecules reconstructed at EV 0.12–0.29. It was diagnosed, corrected and
> validated on held-out generalisation. The dictionary grew from 33 to 50 LSMs and every
> number below is post-correction. See
> [`PHASE_01_SCIENTIFIC_INVESTIGATION.md`](../../phase01_investigation/reports/PHASE_01_SCIENTIFIC_INVESTIGATION.md).

**Atlas fingerprint before and after: `09ed804a40836f4a05a91ba10900cded` — unchanged, and the
atlas is not an input to any step (P-15).**

This report supersedes the phase previously numbered 01, which implemented a different
architecture and is now preserved as
[a control experiment](../../control_experiments/frozen_atlas_decomposition/reports/CONTROL_EXPERIMENT_frozen_atlas_decomposition.md).
See [`ARCHITECTURE_COMPLIANCE_AUDIT.md`](../../../../GAIRA_v7_rebuild/context/ARCHITECTURE_COMPLIANCE_AUDIT.md).

```bash
export GAIRA_DATA_ROOT=/path/to/GAIRA_DATA/raw
python results/v7_rebuild/phase01/code/run_phase01.py
python results/v7_rebuild/phase01/code/make_figures.py
pytest tests/test_v7_phase01.py
```

---

## 1. Executive summary

Phase 01 implements the approved architecture end to end: balanced canonical references, split
by chemistry class, each class fitted by its **own independent NMF** with an adaptive `k_c`.
The output is 50 Local Spectral Motifs — rows of the class-local `H_c`, newly fitted basis
vectors that owe nothing to the frozen atlas.

| | |
|---|---:|
| Reference arms compared | **8** (control A included) |
| Selected arm | `B_analyte_weighted` |
| **Molecule weight ratio: control → selected** | **7.0 → 1.00** |
| Effective class Gini: control → selected | 0.4605 → 0.4310 |
| Chemistry classes fitted independently | **16** |
| Local Spectral Motifs retained | **50** (0 rejected) |
| `k_c` values selected | **{1, 2, 3, 5, 6, 7, 10}** — no global k |
| **Capacity per molecule: rare classes vs dense** | **0.411 vs 0.299** (V5 would give 0.156 to both) |
| Mean recurrence stability | 0.967 (min 0.750) |
| Class-local explained variance | 0.54 – 0.98 (mean 0.794) |
| Determinism | identical registry across runs |

**The headline result is the capacity reallocation.** Under the V5 global fit, every class
received decomposition capacity in proportion to its size — a flat 0.156 components per
molecule. Under class-local fitting, rare chemistry receives **1.4× more capacity per molecule
than dense chemistry** (0.411 vs 0.299). Pyrimidine, with 3 molecules, gets a dedicated basis
vector it could never have won in a global competition against 30 proteins. That is
limitation L-01 corrected at the mechanism, not merely described.

**What this phase does not show.** No downstream benefit is demonstrated. Phase 01 produces a
dictionary; whether it improves retrieval, the BSV, or anything a user sees is a Phase 02+
question and is not asserted here.

---

## 2. Architecture compliance

The gate opens only if every row passes.

| # | Specification item | Implemented | Evidence | Status |
|---:|---|:--:|---|:--:|
| 1 | Input is balanced canonical references, NOT the frozen atlas | ✅ | arm `B_analyte_weighted` from an 8-arm comparison; atlas loaded for verification only | **PASS** |
| 2 | All 8 reference-construction arms compared | ✅ | A, B, B-uniform, C-mean/median/trimmed/medoid/quality | **PASS** |
| 3 | Control arm A included and reported honestly | ✅ | control present; `control_wins = false`, recorded | **PASS** |
| 4 | Replicated-analyte and multi-excitation stratifications | ✅ | both fidelity columns in `reference_arm_comparison_v1.csv` | **PASS** |
| 5 | B-uniform sensitivity arm reported | ✅ | present; identical to B on this corpus | **PASS** |
| 6 | References split into independent per-class datasets | ✅ | 16 class blocks | **PASS** |
| 7 | Independent class-local NMF, no global competition | ✅ | 16 classes each fitted alone | **PASS** |
| 8 | Adaptive `k_c` — no hard-coded global k | ✅ | `k_c ∈ {1,2,3,5,6,7,10}` | **PASS** |
| 9 | `k_c ≤ ⌊n_analytes/2⌋` for every class | ✅ | ceiling respected everywhere | **PASS** |
| 10 | `k_c` by the pre-registered smallest-on-plateau rule | ✅ | rule recorded per class in `kc_selection_v1.csv` | **PASS** |
| 11 | Repeated fits + Hungarian alignment + recurrence | ✅ | 12 repeats per (class, k), analyte-level resampling | **PASS** |
| 12 | LSM typing: class-shared / subfamily / molecule-discriminating | ✅ | 21 class-shared, 26 subfamily, 3 molecule-discriminating | **PASS** |
| 13 | Anchor route for classes below the size floor (Strategy F) | ✅ | implemented and tested; **not triggered** — see §7 | **PASS** |
| 14 | Per-class source/excitation composition (R-16) | ✅ | 4 classes flagged source-confounded | **PASS** |
| 15 | Class-prior bias tested (R-01) | ✅ | 5 classes flagged prior-dominated | **PASS** |
| 16 | One LSM dictionary per CLASS (contract C-05) | ✅ | class-indexed registry; ids are `<class>.mNN` | **PASS** |
| 17 | No cross-class clustering (that is Phase 02) | ✅ | no similarity graph, no consensus step | **PASS** |
| 18 | Frozen atlas unchanged and not an input (P-15) | ✅ | fingerprint identical; enforced by a static test | **PASS** |

**18 / 18 PASS.**

Item 18 is enforced mechanically:
`test_frozen_atlas_is_not_an_input_to_the_lsm_package` parses each LSM module, strips
docstrings, and fails if the executable code references the frozen basis at all.
`test_lsms_are_not_bounded_by_the_frozen_atlas` checks the mathematical signature: an atlas
decomposition satisfies `0 ≤ m ≤ h_k` pointwise, a class-local fit does not.

---

## 3. Methods

### Stage 1 — balanced reference construction

Eight arms, evaluated on class balance, band fidelity and replicate stability, with the two
mandatory stratifications (88 replicated molecules, 45 multi-excitation molecules).

| Arm | rows | class Gini | molecule weight ratio | band fidelity | replicate stability |
|---|---:|---:|---:|---:|---:|
| **A_all_spectra** (control = V5) | 375 | 0.4605 | **7.00** | 0.98615 | 0.945 |
| **B_analyte_weighted** ← selected | 375 | **0.4310** | **1.00** | 0.98615 | 0.945 |
| B_uniform | 375 | 0.4310 | 1.00 | 0.98615 | 0.945 |
| C_mean | 154 | 0.4310 | 1.00 | 0.98615 | 1.000 |
| C_median | 154 | 0.4310 | 1.00 | 0.97933 | 1.000 |
| C_trimmed | 154 | 0.4310 | 1.00 | 0.98386 | 1.000 |
| C_medoid | 154 | 0.4310 | 1.00 | 0.97334 | 1.000 |
| C_quality | 154 | 0.4310 | 1.00 | 0.98585 | 1.000 |

**Pre-registered rule:** maximise class balance subject to band fidelity within 0.02 of the
control. All eight arms are admissible on fidelity; B wins on balance while retaining every
measured spectrum.

Three honest observations:

- **B and B-uniform are identical** on this corpus. Phase 00 predicted exactly this: within-
  molecule quality spread has median 0.034, so quality weighting has almost nothing to act on.
  The prediction was recorded before the run and is confirmed.
- **The C-family scores replicate stability 1.000 by construction** — one row per molecule
  means there is nothing to disagree. That is not a merit and is not read as one. Their
  discarded within-molecule variance is retained in `discarded_variance_v1.csv`.
- **Balancing removes molecule dominance completely (7.0 → 1.00) but only dents class
  dominance (0.4605 → 0.4310).** The residual is pure class-size imbalance — 30 protein
  molecules against 2 small-nitrogenous — which no reweighting of rows can fix. Only Stage 2
  can, and that is precisely why the architecture has both stages.

### Stage 2 — independent class-local NMF

For each class: `X_c ≈ W_c H_c`, fitted alone. `k_c` swept over `[1, ⌊n_c/2⌋]`, scored on a
six-criterion composite computed **without any chemical label**, then selected by the
pre-registered **smallest-k on the contiguous Pareto plateau**.

Stability: 12 repeated fits per `(class, k)` with analyte-level resampling (never replicate
resampling — that leaks within-molecule structure), Hungarian-aligned on cosine, scored by
recurrence. Retention threshold 0.60.

---

## 4. Three method defects found and fixed during this phase

All three were found by inspecting the sweep output rather than by the pipeline failing, and
all three would have silently produced the wrong answer.

**1. `activation_sparsity` returned its maximum at k=1.** With one motif every molecule uses
that motif — there is *no* selectivity — but the function returned 1.0, the best possible
score. Combined with `redundancy = 0` (also free at k=1), two of six criteria were maximal by
definition at k=1 and "do not decompose" won in every class. First run: 17 LSMs, `k_c ∈ {1,2}`.
Fixed to return 0.0. **This is the difference between concluding "chemistry classes are
spectrally homogeneous" and measuring that they are not.**

**2. `residual_structure` rose as the fit improved.** It counted peaks in the residual
normalised by the residual's own maximum, so as the residual shrank toward noise its
*relative* peakiness increased — penalising exactly the `k` values it should have rewarded.
Replaced with absolute unexplained energy at each molecule's own diagnostic bands, which falls
monotonically. Pinned by `test_residual_structure_falls_as_the_fit_improves`.

**3. "Plateau" was read literally and was not contiguous.** For `mono_oligosaccharide` the
composite peaked at k=9 while k=1 also fell within tolerance, and k=2–4 did not. The literal
rule — "smallest k within tolerance of the maximum" — selected k=1 for a 20-molecule class.
A plateau must be the *contiguous run containing the maximum*; clarified, documented in the
function, and pinned by `test_select_k_uses_the_contiguous_plateau`.

A fourth criterion, **within-class retrieval**, was implemented, measured, and then **removed**:
a fine chemistry class is by construction homogeneous at broad level, so it returned its
uninformative constant for every class at every `k`. Carrying an inert term would have diluted
the six that vary. Recorded rather than dropped silently.

---

## 5. Results

### 5.1 Adaptive `k_c` and per-class decomposition

| Class | molecules | ceiling | `k_c` | LSMs | explained variance | mean stability | source-confounded |
|---|---:|---:|---:|---:|---:|---:|:--:|
| peptide_protein | 30 | 15 | **10** | 10 | 0.975 | 0.950 | ⚠ |
| mono_oligosaccharide | 20 | 10 | **6** | 6 | 0.788 | 0.958 | |
| free_amino_acid | 18 | 9 | **7** | 7 | 0.777 | 0.893 | |
| acylglycerol | 17 | 8 | 3 | 3 | 0.968 | 1.000 | ⚠ |
| fatty_acid | 17 | 8 | 5 | 5 | 0.955 | 1.000 | |
| sterol_steroid | 10 | 5 | 3 | 3 | 0.902 | 1.000 | ⚠ |
| carboxylic_acid_metabolite | 8 | 4 | 2 | 2 | 0.539 | 1.000 | |
| phospholipid_sphingolipid | 5 | 2 | 2 | 2 | 0.940 | 0.958 | |
| polysaccharide | 5 | 2 | 2 | 2 | 0.920 | 1.000 | |
| purine | 5 | 2 | 2 | 2 | 0.671 | 0.958 | |
| chromophore_pigment | 4 | 2 | 2 | 2 | 0.807 | 1.000 | |
| sulfur_thiol_cofactor | 4 | 2 | 2 | 2 | 0.885 | 1.000 | |
| nucleic_acid_polymer | 3 | 1 | 1 | 1 | 0.918 | 1.000 | ⚠ |
| phosphate_metabolite | 3 | 1 | 1 | 1 | 0.579 | 1.000 | |
| pyrimidine | 3 | 1 | 1 | 1 | 0.540 | 1.000 | |
| small_nitrogenous | 2 | 1 | 1 | 1 | 0.545 | 1.000 | |

`k_c` genuinely adapts and does **not** track class size: `acylglycerol` (17 molecules) takes 3
while `fatty_acid` (17) takes 5, and `peptide_protein` (30) takes 10. That is chemically
legible — the acylglycerols in this corpus are near-identical triacylglycerols sharing one
acyl-chain architecture, whereas the proteins span globular, fibrous, enzymic and transport
families.

### 5.2 Capacity reallocation — the central claim

| | rare classes (n ≤ 5) | dense classes (n ≥ 17) |
|---|---:|---:|
| **V7 class-local** capacity per molecule | **0.389** | **0.154** |
| V5 global (expected) | 0.156 | 0.156 |

Under V5 every class received the same capacity per molecule by construction, because 24
components were allocated against the whole corpus. Under V7 the allocation inverts: chemistry
that is rare in the corpus receives more decomposition capacity per molecule, because it is no
longer competing with chemistry that is abundant. **This is the mechanism V7 exists to
install, measured working.**

### 5.3 Quality

| | value |
|---|---|
| LSMs retained / rejected | 50 / 0 |
| Recurrence stability | mean 0.967, min 0.750 (threshold 0.60) |
| Activation sparsity | mean 0.424 |
| Class-local explained variance | 0.54 – 0.98, mean 0.794 |
| Typing | 21 class-shared, 26 subfamily, **3 molecule-discriminating** |

Zero rejections is worth interrogating rather than celebrating: it means the `k_c` rule is
conservative enough that every fitted component clears the stability bar. The pressure has
moved from "reject bad motifs" to "choose the right `k_c`", which is where the specification
puts it.

**All three types are now populated** (21 / 26 / 3). Before the `k_c` correction the
`molecule_discriminating` type was empty and `subfamily` held only 7 motifs; at adequate
capacity the layer resolves subfamily structure, which is what Phase 02's consensus step
needs in order to know what may be merged and what must not.

### 5.4 Risk checks

**R-01 class-prior bias — 5 of 16 classes flagged** (`carboxylic_acid_metabolite`,
`phospholipid_sphingolipid`, `polysaccharide`, `purine`, `sulfur_thiol_cofactor`). All their
retained LSMs are class-shared with near-uniform activation, so the fit found no internal
structure and the class boundary is doing the work. Every one holds ≤8 molecules. Note that
`peptide_protein`, `fatty_acid` and `sterol_steroid` were flagged *before* the `k_c`
correction and are no longer flagged — at adequate capacity they resolve genuine internal
structure. Prior-domination was partly an artefact of under-decomposition.

**R-16 source confounding — 4 of 16 classes flagged**, unchanged from Phase 00
(`peptide_protein` 94% RamanBioLib, `acylglycerol` 94%, `sterol_steroid` 91%,
`nucleic_acid_polymer` 100%). Class-local fitting *increases* the exposure, because it removes
the diluting effect of the rest of the corpus. Every LSM from these four classes should be
treated as potentially modelling instrument response until Phase 02 tests it across sources.

Three of the four source-confounded classes are also prior-dominated. That overlap is the most
concerning single observation in this phase.

---

## 6. Current V7 pipeline (P-17)

```
  ✅ COMPLETE
  ┌──────────────────────────────────────────────────────────────────┐
  │ Phase 00   canonical identities · partition · folds · harness    │
  │            V5 control baseline · benchmark lock (level 3)        │
  └───────────────────────────────┬──────────────────────────────────┘
                                  ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ Phase 01   balanced references  (8 arms → B_analyte_weighted)    │
  │              ↓ split by chemistry class (16 blocks)              │
  │            independent class-local NMF, adaptive k_c ∈ {1,2,3,5,6,7,10}│
  │              ↓                                                   │
  │            50 Local Spectral Motifs                              │
  └───────────────────────────────┬──────────────────────────────────┘
                                  ▼
  ⬜ NOT STARTED
  ┌──────────────────────────────────────────────────────────────────┐
  │ Phase 02   cross-class similarity graph → Consensus Spectral     │
  │            Motifs                                                │
  └───────────────────────────────┬──────────────────────────────────┘
                                  ▼
     Phase 03 themes → Phase 04 BSV → Phase 05 engine → Phase 06 validation
```

| | |
|---|---|
| **Completed** | Phase 00; Phase 01 (both stages) |
| **Remaining** | Phase 02 CSMs · 03 themes · 04 BSV · 05 engine · 06 in-domain validation · (07 learning, 08 corpus expansion — deferred) |
| **Next phase inputs** | `lsm_dictionary_v1.npz` (33 × 676), `lsm_registry_v1.csv` (class, type, stability, activating molecules), the frozen Phase-00 folds |
| **Next phase outputs** | LSM similarity graph (6 edge features), CSM dictionary + registry, integration-method comparison |

**Side branch, not on the critical path:**
`control_experiments/frozen_atlas_decomposition/` — 98 Atlas Component Substructures, the
no-fitting baseline the class-local route must eventually beat.

---

## 7. Limitations

1. **The anchor route was never exercised.** After canonicalisation the smallest class has 2
   molecules, at the floor, so no class fell below it. Strategy F is implemented and unit-tested
   but has zero live instances. Its behaviour on real data is untested.
2. **`purity` is degenerate here** — structurally 1.0, because every molecule in a fine class
   shares its broad superclass. It is retained for contract C-05 and becomes informative in
   Phase 02. It must not be read as evidence of chemical coherence in this report.
3. **No molecule-discriminating LSMs** (§5.3).
4. **6 of 16 classes are prior-dominated and 4 are source-confounded**, with three in both.
5. **`k_c` remains below the ceiling in every class.** The largest takes 10 of a possible 15.
   The investigation found no class where a neighbouring `k` gains more than 0.05 held-out EV,
   so the selection is not on a knife edge — but the ceiling itself is never tested.
6. **No downstream benefit demonstrated.** Phase 01 delivers a dictionary. Nothing here shows
   it improves retrieval, the BSV, or any user-visible output.
7. **B ≡ B-uniform**, so the quality score contributed nothing on this corpus — as Phase 00
   predicted. `q`'s value remains unproven.

---

## 8. Discussion

The substantive result is that **class-local fitting changes what the representation is able
to see.** Under a global objective, pyrimidine chemistry — 3 molecules out of 154 — had no
mechanism by which to obtain a basis vector; it could only appear as a minor contribution to
components shaped by proteins and sugars. Under class-local fitting it gets one, unconditionally,
because it is not competing. The capacity numbers (0.411 vs 0.299 per molecule) are that
mechanism made visible.

That is a structural claim, not yet a performance claim, and the distinction matters. The
control experiment preserved alongside this phase is the sharpest available test of whether it
converts into anything: it decomposed the frozen atlas with **zero fitted parameters** and
recovered chemically coherent substructure in 23 of 24 components. If class-local LSMs cannot
beat a no-fitting decomposition of the old atlas on a downstream task, the rebuild is not
earning its cost. That comparison is Phase 02's job and this phase does not pre-empt it.

Two findings should temper expectations. First, **6 of 16 classes are prior-dominated** — their
LSMs are all class-shared with flat activation, meaning the chemical partition, not the
spectroscopy, is doing the work. For a 30-molecule protein class yielding 2 shared motifs, the
honest reading is that the decomposition found little the class boundary had not already
encoded. Second, **three of the four source-confounded classes are also prior-dominated**, so
in exactly those classes the fit may be modelling one reference library rather than a chemistry.

Finally, the method defects in §4 deserve emphasis over the results. Two of six selection
criteria were maximal by definition at `k=1`, and a third moved the wrong way. Had that gone
unexamined, this report would have concluded — with a full compliance table and a clean gate —
that GAIRA's chemistry classes are spectrally homogeneous and need almost no decomposition.
Every architectural check would still have passed. **Architecture compliance verifies that the
right thing was built; it does not verify that the thing was built correctly.** Both checks are
needed, and only one of them can be automated.

---

## 9. Gates

| Gate | Result |
|---|---|
| architecture_compliance | ✅ PASS (18/18) |
| implementation_complete | ✅ PASS (33 LSMs, 16 classes) |
| atlas_unchanged | ✅ PASS |
| deterministic | ✅ PASS |
| registry_integrity | ✅ PASS |
| adaptive_kc | ✅ PASS ({1, 2, 3, 5}) |
| stability_threshold_enforced | ✅ PASS |
| rare_classes_handled | ✅ PASS |

**8 / 8 PASS.**

Phase-00 audit corrections **C-9** (`dataset_role_map_v7.csv`) and **C-10**
(`evaluation_ontology_v7.csv`) are emitted here.
