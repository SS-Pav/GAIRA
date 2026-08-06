# GAIRA V7 — Phase 00 Report
## Architecture lock, benchmark freeze and canonical data foundation

**Branch** `gaira-v7-rebuild` · **Status** COMPLETE · **Gates** 12/12 PASS ·
**Validation** 84 PASS / 3 WARN / 0 FAIL · **Tests** 63 passed

**Atlas fingerprint before and after: `09ed804a40836f4a05a91ba10900cded` — unchanged.**

Reproduce:

```bash
export GAIRA_DATA_ROOT=/path/to/GAIRA_DATA/raw       # optional; degraded mode without it
python results/v7_rebuild/phase00/code/run_phase00.py
python results/v7_rebuild/phase00/code/validate_phase00.py
python results/v7_rebuild/phase00/code/make_figures.py
pytest tests/test_v7_phase00.py
```

---

## 1. What Phase 00 was for

Phase 00 builds nothing scientific. It fixes the yardstick — the corpus, the molecule
identities, the chemical partition, the folds, the metrics and the control baseline — *before*
anything is built with it. The reason is specific to this project's history: V6.3 ran a full,
careful ontology revalidation and established a genuinely useful negative result, but only
because its harness was sound. Measuring the wrong thing carefully is still measuring the
wrong thing.

Everything downstream is measured against what is frozen here.

---

## 2. Benchmark lock — three levels, all reached

| Level | What it proves | Checks | Result |
|---|---|---:|---|
| 1 **DECLARED** | the fingerprint recorded in the manifests is the expected string | 6 | 6/6 PASS |
| 2 **RECOMPUTED** | the fingerprint recomputed *from the basis array*, and every frozen file re-hashed against `MANIFEST.json` | 12 | 12/12 PASS |
| 3 **REBUILT** | the basis **refitted from raw** through canonical preprocessing and NMF(k=24, seed=0), compared element-by-element | 4 | 4/4 PASS |

**Level 3 is the result that matters:**

```
max |H_rebuilt − H_frozen|   =  0.0
mean row-wise cosine         =  1.0
recomputed fingerprint       =  09ed804a40836f4a05a91ba10900cded
```

The V7 corpus loader, the canonical preprocessing chain and the frozen V5 atlas are the same
object — not merely three things that agree on a hash string someone copied. Any future
change to preprocessing, to the loaders, or to the corpus will break this check loudly.

Level 3 requires the raw root. Without it the lock degrades to level 2 and the manifest
records `benchmark_lock_level: 2`; it does not silently claim more than it verified.

---

## 3. Corpus — reproduced exactly, with no lab path committed

The V7 loader resolves the raw root through `GAIRA_DATA_ROOT` (never a hard-coded volume) and
reproduces the frozen V5 corpus card on every field:

| Field | Frozen V5 | V7 load | |
|---|---:|---:|---|
| spectra | 375 | 375 | ✅ |
| surface analytes | 167 | 167 | ✅ |
| bins | 676 | 676 | ✅ |
| RamanBioLib / Gobbato / amino-acid | 202 / 153 / 20 | 202 / 153 / 20 | ✅ |
| excitations (9 values) | 234/55/50/29/3/1/1/1/1 | identical | ✅ |
| multi-excitation analytes | 41 | 41 | ✅ |
| analytes with replicates | 87 | 87 | ✅ |
| V5 replicate groups | 272 | 272 | ✅ |

Window 450–1800 cm⁻¹, 2.0 cm⁻¹, `asls → savgol → L2` — unchanged. Ag-SERS, Au-SERS, DART and
the serum colloid sets remain excluded by construction.

> **Preprocessing gotcha, confirmed and handled.** `pipeline.common_grid()` defaults to the
> legacy Ag-SERS-constrained **520–1750** window. The V7 loader passes 450–1800 explicitly, as
> the V5 build did. A caller relying on the default would silently produce a different grid.

---

## 4. Canonical molecule identities — 167 surface forms → 154 molecules

### The design decision

**Canonical identity is a metadata layer, not a corpus edit.** The fitting corpus stays at
375 spectra over 167 surface analytes — which is why the atlas still reproduces bit-exactly —
and a `canonical_id` column collapses surface forms onto molecules. **Cross-validation groups
by `canonical_id`.** That removes the leakage without redefining the corpus.

### What was found

| | |
|---|---:|
| surface forms | 167 |
| canonical molecules | **154** |
| merged forms | 13 |
| **cross-source merges** | **11** |
| **spectra affected by a cross-source merge** | **49** |

A cross-source merge is one molecule that appeared in two reference libraries under two
spellings. **Eleven of the thirteen duplicates are cross-source**, so under surface-name
grouping those 49 spectra would have been split across folds and the same molecule scored
against itself. This is risk R-09 measured rather than asserted.

The merges, each with a written chemical justification (`alias_table_v1.csv`):

| Kind | Pairs | Example |
|---|---:|---|
| Unicode ligature | 1 | `riboﬂavin` (U+FB02) → `riboflavin` |
| truncation | 3 | `alb` → `albumin`, `gluth` → `glutathione`, `ure` → `urea` |
| orthographic | 2 | `acetyl coenzyme a` → `acetyl-coa` |
| protonation state | 1 | `aspartic acid` → `aspartate` |
| synonym | 1 | `(+)-dextrose` → `(+)-glucose` |
| generic stereo prefix | 5 | `glucose` → `(+)-glucose`, `fructose` → `(-)-fructose`, … |

**Two of these were also class-assignment conflicts**, exactly as predicted in the V7 spec:
`acetyl coenzyme a` was filed under *protein* in one source and *cofactor* in the other;
`aspartic acid` under *organic_acid* vs *amino_acid*. Under V7's class-partitioned
decomposition an unresolved conflict would have put one molecule into two independent local
fits. All five conflicts are recorded in `class_conflicts_v1.csv` with their resolution.

### What was deliberately NOT merged

| Pair | Decision | Reason |
|---|---|---|
| `(+)-arabinose` / `(-)-arabinose` | PROTECTED | enantiomers — different molecules |
| `(+)-glucose` / `β-d-glucose` | PROTECTED | anomers — distinct reference spectra |
| `(-)-ribose` / `2-deoxy-d-ribose` | PROTECTED | the 2′-OH is the RNA/DNA distinction |
| `carotene` / `β-carotene` | **UNRESOLVED** | see §9 |

Every non-merge is a recorded decision with a reason, so a reader can disagree with it.

---

## 5. Replicate grouping — the key was changed, with evidence

| Key | Groups | Median | Max | Singletons |
|---|---:|---:|---:|---:|
| V5 `analyte │ source │ excitation` | 272 | 1.0 | 3 | 220 |
| **V7 `canonical_id │ excitation`** (ratified) | **231** | 1.0 | **6** | 179 |

The V5 key also split on *source*, so the same molecule measured at 785 nm in two reference
libraries formed two replicate groups. Under the V7 key those are one group — which is what a
replicate is: the same molecule under the same measurement condition.

Excitation stays in the key because it is a tracked nuisance factor: peak *position* is
excitation-invariant, relative *intensity* is not. Balancing then applies per canonical
molecule **across** its groups, so the 41 multi-excitation molecules buy no extra weight.

---

## 6. Chemical partition — the three V7 problems, resolved

All three problems the V7 specification flagged are resolved by adopting the **V6.3 cleaned
ontology** (16 fine / 6 broad) rather than authoring a fourth ontology:

| Problem | Resolution |
|---|---|
| `unknown` (6 analytes) is not a chemistry | **Dissolved.** No `unknown` class exists in the frozen partition. |
| `lipid` (5) overlaps `fatty_acid` (12) and `triglyceride` (15) | **Split three ways** — `fatty_acid` (17), `acylglycerol` (17), `phospholipid_sphingolipid` (5), on the ester carbonyl (~1745) and C-O-C (~1160). |
| `polysaccharide` (5) vs `saccharide` (27) | **Kept separate**, with rationale: glycosidic polymerisation is spectroscopically real and V6 derived distinct motifs for each. |

Adopting V6.3 also keeps the Phase-07 comparison like-for-like with published V5/V6 numbers.
Every one of the 16 classes carries a written chemical rationale (`partition_rationale_v1.json`).

### Class census after canonicalisation

| Fine class | Molecules | Spectra | `k_c` ceiling | Source flag |
|---|---:|---:|---:|---|
| peptide_protein | 30 | 80 | 15 | ⚠ 94% RamanBioLib |
| mono_oligosaccharide | 20 | 43 | 10 | |
| free_amino_acid | 18 | 75 | 9 | |
| acylglycerol | 17 | 23 | 8 | ⚠ 94% RamanBioLib |
| fatty_acid | 17 | 27 | 8 | |
| sterol_steroid | 10 | 13 | 5 | ⚠ 91% RamanBioLib |
| carboxylic_acid_metabolite | 8 | 23 | 4 | |
| phospholipid_sphingolipid | 5 | 8 | 2 | |
| polysaccharide | 5 | 10 | 2 | |
| purine | 5 | 17 | 2 | |
| chromophore_pigment | 4 | 10 | 2 | |
| sulfur_thiol_cofactor | 4 | 16 | 2 | |
| nucleic_acid_polymer | 3 | 3 | 1 | ⚠ 100% RamanBioLib |
| phosphate_metabolite | 3 | 11 | 1 | |
| pyrimidine | 3 | 9 | 1 | |
| small_nitrogenous | 2 | 7 | 1 | |
| **total** | **154** | **375** | | |

Two things worth noting.

**Canonicalisation moved the imbalance.** `mono_oligosaccharide` drops from 27 to **20**
molecules — seven of the "27 saccharides" were duplicate spellings of five sugars. The
imbalance ratio narrows from 32:1 to **15:1**, which is still severe but materially less so
than the V7 planning documents assumed from the surface-name census.

**Four classes are source-confounded** (⚠ above, `dominant_source_fraction ≥ 0.9`,
`n ≥ 3`). This is risk **R-16** materialising, and Strategy D makes it *worse*, not better:
partitioning removes the diluting effect of the rest of the corpus, so a class drawn ~entirely
from one library may model that library's instrument response as if it were chemistry. Phase
02 must report per-class source composition and flag every LSM from these four classes.

---

## 7. Frozen cross-validation splits

5 folds, grouped by `canonical_id`, stratified by fine class, balanced on spectrum count,
deterministic under a fixed seed.

| Fold | Molecules | Spectra | Fine classes | Broad classes |
|---:|---:|---:|---:|---:|
| 0 | 34 | 76 | 14 | 6 |
| 1 | 31 | 76 | 14 | 6 |
| 2 | 33 | 75 | 15 | 6 |
| 3 | 28 | 72 | 12 | 5 |
| 4 | 28 | 76 | 13 | 6 |

**All three leakage checks read `false`:**

| Check | Result |
|---|---|
| `canonical_id_across_folds` | false ✅ |
| `alias_collision` | false ✅ |
| `replicate_across_folds` | false ✅ |

Determinism is verified by re-cutting the folds from the written tables and comparing the
assignment element-wise (`splits.deterministic_recut` PASS).

---

## 8. Quality score `q` — frozen, after the first design was found to be degenerate

`q` is frozen **before** Phase 01 so it cannot become a hyperparameter tuned to a preferred
answer (risk R-10). Getting there took two attempts, and the first failure is worth recording
because it is a fact about the corpus rather than a coding slip.

**The first design measured nothing.** It used classical acquisition-quality terms — cosmic-ray
spike count, detector saturation, first-difference SNR. Measured on this corpus, all three were
degenerate: `spike_free` scored **0.000 for all 375** spectra, `not_saturated` **1.000 for all
375**, and the SNR term saturated at its ceiling for all 375. Median `q` collapsed to 0.062 and
every spectrum fell below the QC floor.

The reason is that this is a *curated reference library*: digitized, already baseline-corrected
and smoothed upstream. Acquisition artefacts have been removed before GAIRA ever sees the data,
and a first-difference noise estimate on such a spectrum measures the sharpness of genuine
Raman bands, not noise.

**The frozen design (`v7_q_v2`)** measures what genuinely varies here — band structure and
contrast — using a second-difference noise estimate (insensitive to smooth band shape), grid
coverage, peak-to-body contrast, and resolvable peak count, combined as a geometric mean:

| | |
|---|---:|
| range | 0.444 – 0.967 |
| median | 0.869 |
| max / min | **2.18** |
| below QC floor (0.35) | **0** |
| median within-analyte spread | 0.034 |
| max within-analyte spread | 0.483 |
| weights sum to 1.0 per molecule | ✅ |

**This is narrower than "acquisition quality" and is documented as such rather than dressed
up.** A useful consequence, stated now as a prediction rather than discovered later: because
within-analyte spread is small, **Strategy B and the mandatory B-uniform arm should come out
close in Phase 01**. Both arms are still run, so the prediction is tested rather than assumed.

---

## 9. The V5 control baseline, under the frozen V7 harness

The harness (metrics, nulls, CIs, paired tests) is adopted wholesale from the V6.3
revalidation, which is the strongest methodology this project has produced.

| Level | fine (16) | broad (6) | old (18) | random | gain beyond mechanical (fine) |
|---|---:|---:|---:|---:|---:|
| coord (24) | 0.6467 | **0.8204** | 0.6886 | 0.0883 | +0.558 |
| **mss (17)** | **0.6707** | 0.8084 | 0.7006 | 0.0863 | **+0.584** |
| theme (6) | 0.6228 | 0.7665 | 0.6287 | 0.0823 | +0.541 |
| system (4) | 0.5689 | 0.6946 | 0.5269 | 0.0963 | +0.473 |

**The fine and broad numbers reproduce the published V6.3 values exactly** (0.6467 / 0.6707 /
0.6228 / 0.5689 fine; 0.8204 / 0.8084 broad). That is the strongest available evidence that
the V7 harness is the V6.3 harness.

**One number moved, and it moved for a reason.** Under the *old 18-class* labels the V7
harness gives MSS 0.7006 where V6.3 published 0.6766 (+0.024). The cause is exactly three
label changes, all of them canonicalisation resolving a conflict:

| analyte | old label | canonical label |
|---|---|---|
| `acetyl coenzyme a` | protein | cofactor |
| `gluth` | unknown | cofactor |
| `ure` | unknown | small_nitrogenous |

Those labels were wrong before. The shift is a correction, not a discrepancy — and the fine
ontology, which is what Phase 07 will actually use, is unaffected.

The random-ontology control sits at 0.082–0.096 (V6.3: 0.096–0.113; different draws, same
conclusion). **Coarse chemistry is genuinely present** — +0.72 beyond mechanical at broad
level. The fine ceiling of 0.647–0.671 is what V7 exists to move.

### Component baseline (frozen registry, recomputed)

| | |
|---|---:|
| components with purity ≥ 0.5 | **3 of 24** |
| median purity | 0.328 |
| median bootstrap stability | 0.799 |

---

## 10. Success criteria — FROZEN

The provisional criteria in `GAIRA_v7_rebuild/plan/SUCCESS_CRITERIA.md` are hereby **frozen**
against the baseline measured in §9, under the harness `v7_harness_v1` and the splits
`v7_cv_v1`. They are not adjusted afterwards — not upward if V7 looks strong, not downward if
it looks weak (principle P-13).

| ID | Criterion | Frozen threshold | Baseline |
|---|---|---|---|
| S-01 | CSM/MSS-equivalent fine top-1 | **≥ 0.7507** (+8 pts) | 0.6707 |
| S-02 | CSM top-3 | ≥ 0.90 | — |
| S-03 | fine-family retrieval improvement | **≥ +8 pts** | 0.6707 |
| S-04 | broad-superclass retrieval | **≥ 0.8084** (maintained) | 0.8084 |
| S-05 | true projection failures | **< 31** at CSM level | 31 of 54 |
| S-06 | deterministic reproduction | byte-identical, two machines | — |
| S-07 | clean inference from a frozen package | no lab volume | — |
| S-08 | diagnostic-band fidelity | improved | — |
| S-09 | LSM/CSM stability | ≥ 0.799 median | 0.799 |
| S-10 | pathological collisions | fewer than V5 | — |
| S-11 | rare-class coverage | ≥ 1 CSM per named chemistry | — |
| S-12–S-14 | interpretability, provenance, calibration | not degraded | — |

S-01 and S-03 must be significant at α = 0.05 after correction, using McNemar + permutation
on the frozen folds.

---

## 11. Analysis

### What was found

1. **The atlas is reproducible from raw, bit-exactly.** This was not guaranteed. It means
   preprocessing, loaders and corpus are all pinned by one check.
2. **Cross-source aliasing was a real leakage path**: 11 molecules, 49 spectra.
3. **Canonicalisation narrowed the class imbalance** from 32:1 to 15:1 — seven "saccharides"
   were duplicate spellings.
4. **Two alias pairs were also class conflicts**, exactly as the V7 spec predicted.
5. **Four classes are source-confounded**, and class partitioning will amplify that.
6. **The first quality-score design was degenerate on this corpus** — three of five components
   constant across all 375 spectra.
7. **The harness reproduces V6.3 exactly**, and the one number that moved moved because three
   labels were corrected.

### What changed

Molecule identity (167 → 154 for grouping), the replicate key, and the evaluation partition
(18 old → 16 fine / 6 broad). Nothing scientific: no model, no basis, no representation.

### What remained unchanged

The frozen atlas and all ten of its files, canonical preprocessing, the corpus definition,
the Raman-only exclusion list, and every V5/V6/V6.2/V6.3 artefact. Verified by hash.

### Scientific implications

- The fine-resolution ceiling (0.647–0.671) is confirmed under an independently re-run harness,
  so it is a property of the representation and not of one evaluation script.
- Class imbalance is real but 2× less extreme than the surface-name census suggested. Strategy
  D's expected benefit should be revised down accordingly — worth stating before Phase 02
  rather than after.
- Source confounding is the most under-weighted risk in the V7 plan. Four classes, including
  the largest, are ~single-source.
- `q` measures band structure, not acquisition quality, so Strategy B's headroom on this
  corpus is limited.

### Engineering implications

- `GAIRA_DATA_ROOT` works; no lab path is committed; degraded mode is defined and recorded.
- One duplicate `spectrum_id` existed in the V5 id scheme (`amino_acid_raman::glutamate`, from
  two spreadsheet columns canonicalising to one name). Two distinct measured spectra sharing
  one id would silently collapse in any id-keyed join. Fixed by disambiguation in the V7
  loader; the underlying V5 asset is untouched.
- 153 spectra carry exactly 2 NaN bins each (306 total) — a grid-edge resampling effect in the
  Gobbato source. Harmless for NMF (which zero-fills), but it is now measured and recorded.

### Risks remaining

| Risk | Status after Phase 00 |
|---|---|
| R-09 alias/replicate leakage | **Closed for known aliases**; one unresolved near-miss remains |
| R-10 in-sample evaluation | Mitigated: splits, metrics and `q` frozen before Phase 01 |
| R-16 source/excitation confounding | **Elevated** — 4 classes measured as confounded |
| R-02 rare classes too small | Confirmed: 4 classes have `k_c ≤ 1` |
| R-01 class-prior bias | Untested until Phase 02 |
| R-17 corpus is the binding constraint | Unchanged; 154 molecules is a small corpus |

### Future dependencies

Phase 01 consumes `canonical_analytes_v1.csv`, `replicate_groups_v1.csv`,
`spectrum_quality_v1.csv` (both weight columns) and `cv_folds_v1.csv`. Phase 02 additionally
consumes `chemical_partition_v1.csv` and `class_census_v1.csv` (`k_c_ceiling`,
`source_confounded`). Phase 07 consumes `phase00_baseline_metrics.csv` and the frozen criteria.

---

## 12. Validation results

| Category | PASS | WARN | FAIL |
|---|---:|---:|---:|
| artifacts | 24 | 0 | 0 |
| baseline | 5 | 0 | 0 |
| benchmark_lock | 6 | 0 | 0 |
| corpus | 6 | 0 | 0 |
| identity | 7 | 1 | 0 |
| isolation | 2 | 0 | 0 |
| partition | 6 | 1 | 0 |
| provenance | 8 | 0 | 0 |
| quality | 7 | 1 | 0 |
| splits | 9 | 0 | 0 |
| state | 4 | 0 | 0 |
| **TOTAL** | **84** | **3** | **0** |

### The three warnings — all genuine findings, none blocking

**W-1 `carotene` vs `β-carotene` unresolved.** Loose-key collision across two sources, both
filed as `chromophore_pigment`. "Carotene" most likely denotes the β isomer (the common
commercial form), but the source spreadsheet does not say so and α-carotene is a real
alternative. **NOT MERGED.** Merging on a guess would destroy a distinct reference; not merging
risks one leaked spectrum, which would slightly inflate any within-carotenoid result. Resolve
from the source datasheet before Phase 02.

**W-2 four source-confounded classes.** `peptide_protein` (93.8% RamanBioLib), `acylglycerol`
(94.4%), `sterol_steroid` (90.9%), `nucleic_acid_polymer` (100%). Phase 02 must report per-class
source composition and flag LSMs from these classes; a motif that tracks a library rather than
a chemistry must be visible as such.

**W-3 306 NaN bins across 153 spectra.** Exactly 2 bins per affected spectrum, all Gobbato —
a grid-edge resampling effect. NMF zero-fills, so it does not affect the reproduction; recorded
so it is not rediscovered as a surprise.

---

## 13. Artefacts

**Tables** (16) — `canonical_analytes_v1.csv` · `alias_table_v1.csv` ·
`alias_near_miss_audit_v1.csv` · `chemical_partition_v1.csv` · `class_conflicts_v1.csv` ·
`class_census_v1.csv` · `replicate_groups_v1.csv` · `replicate_group_key_comparison_v1.csv` ·
`spectrum_quality_v1.csv` · `cv_folds_v1.csv` · `cv_fold_summary_v1.csv` ·
`benchmark_lock_v1.csv` · `frozen_dependency_graph_v1.csv` · `phase00_baseline_metrics.csv` ·
`phase00_baseline_gain_v1.csv` · `phase00_corpus_checks.csv`

**Manifests** (7) — `phase_00_manifest_v1.json` (inputs, config, seeds, code SHA, environment,
outputs, gates, decisions) · `dataset_card_v7.json` · `cv_splits_v1.json` ·
`alias_leakage_report_v1.json` · `partition_rationale_v1.json` · `quality_summary_v1.json` ·
`phase00_component_baseline_v1.json`

**State** — `PHASE_STATE.json`

**Figures** (8, SVG + PNG) — canonical resolution workflow · alias graph · replicate grouping ·
dataset composition · provenance flow · benchmark lock · frozen dependency graph ·
V5 control baseline

**Code** (10) — `v7_paths` · `v7_corpus` · `v7_canonical` · `v7_partition` · `v7_quality` ·
`v7_splits` · `v7_harness` · `v7_benchmark` · `run_phase00` · `validate_phase00` ·
`make_figures`

**Tests** — `tests/test_v7_phase00.py` (63 tests)

---

## 14. Gates

| Gate | Result |
|---|---|
| no_alias_leakage | ✅ PASS |
| no_replicate_leakage | ✅ PASS |
| cv_checks_all_false | ✅ PASS |
| baseline_reproduced | ✅ PASS |
| atlas_rebuilt_bit_exact | ✅ PASS |
| inputs_versioned_and_hashed | ✅ PASS |
| splits_deterministic | ✅ PASS |
| class_rationale_written | ✅ PASS |
| unknown_class_resolved | ✅ PASS |
| no_uncovered_analytes | ✅ PASS |
| quality_score_frozen | ✅ PASS |
| success_criteria_frozen | ✅ PASS |

**12 / 12 PASS.**
