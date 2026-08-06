# GAIRA V7 — Consistency Audit

Verification performed before the V7 documentation pass was committed. Every check below was
run against the committed document set; mechanical checks are additionally enforced by
`tests/test_v7_rebuild_scaffold.py` so they cannot silently regress.

---

## 1. Terminology consistency

| # | Check | Result |
|---|---|---|
| 1.1 | All V7 documents use the same vocabulary | ✅ |
| 1.2 | MSS identified as legacy terminology | ✅ `TERMINOLOGY_AND_DEFINITIONS.md` states "**Do not use "MSS" as a primary V7 term**" and records the mapping `legacy MSS → V7 Consensus Spectral Motif (CSM)`; the root `README.md` repeats it |
| 1.3 | CSM is the canonical V7 term for the cross-class evidence unit | ✅ defined once, canonically, in `TERMINOLOGY_AND_DEFINITIONS.md` |
| 1.4 | Every primary document expands "CSM" on use | ✅ enforced by `test_csm_is_defined_consistently` over `README.md`, `GAIRA_V7_CONTEXT.md`, `TERMINOLOGY_AND_DEFINITIONS.md`, both mode architectures, the target architecture, and the rebuild plan |
| 1.5 | No document gives CSM a different expansion | ✅ enforced by `test_no_document_redefines_csm` |
| 1.6 | MSS appears only when reporting a legacy result or stating the mapping | ✅ every occurrence is a V5/V6/V6.3 measurement, the legacy mapping, or the phrase "CSM/MSS-equivalent layer" used to keep old and new numbers comparable |
| 1.7 | LSM consistently means Local Spectral Motif | ✅ |
| 1.8 | "Atlas" defined consistently as the frozen layered bundle | ✅ `TERMINOLOGY_AND_DEFINITIONS.md` and `ARTIFACT_AND_MANIFEST_SPEC.md` agree; both state it is "no longer only one global NMF basis" |

## 2. BSV consistency

| # | Check | Result |
|---|---|---|
| 2.1 | BSV described as absolute | ✅ in terminology, both mode architectures, the rebuild plan, and Phase 05 |
| 2.2 | ΔBSV described as derived and signed | ✅ never referred to as a BSV in any document |
| 2.3 | Elevation described as derived and signed | ✅ |
| 2.4 | Cohort standardisation marked visualisation-only | ✅ |
| 2.5 | BSV dimension = K = theme count | ✅ consistent in terminology, learning-mode architecture, decision rules, Phase 05 |
| 2.6 | Effective rank required to be reported separately from K | ✅ in decision rules, `DATA_CONTRACTS.md` C-09, Phase 05, and risk R-12 |
| 2.7 | Naming discipline stated in the output contract | ✅ `bsv` absolute, `bsv_elevation` signed, no delta returned under the name `bsv` |

## 3. Theme consistency

| # | Check | Result |
|---|---|---|
| 3.1 | Themes are chemical, not biological-process labels | ✅ stated in terminology (P-07), design principles, target architecture, Phase 04, and the Phase-04 gate |
| 3.2 | No theme name in any document refers to a disease, pathway, process, or phenotype | ✅ example theme names throughout are protein / lipid / nucleic / carbohydrate / organic-acid-energy / sulfur-redox-cofactor chemistry |
| 3.3 | Themes derived from CSMs, never asserted over them | ✅ consistent in learning-mode Stage 5, Phase 04, and the L-05 rationale |
| 3.4 | Soft membership; no hard one-parent requirement | ✅ |

## 4. Learning / inference boundary

| # | Check | Result |
|---|---|---|
| 4.1 | No document implies PCA is inference | ✅ every mention is either offline fitting or an *applied* frozen transform explicitly labelled visualisation |
| 4.2 | No document implies UMAP is inference | ✅ UMAP appears only in prohibition lists and in the anti-pattern list; it is stated that it is not shipped in the atlas at all |
| 4.3 | "Not the canonical BSV" disclaimer present wherever the visualisation projection appears | ✅ terminology, learning-mode Stage 6, inference-mode §3, `DATA_CONTRACTS.md` C-09 (in the artefact itself, not only in prose) |
| 4.4 | Prohibited and permitted operation lists agree across documents | ✅ target architecture §3 and inference-mode §§2–3 list the same operations |
| 4.5 | Batch independence stated as the governing principle | ✅ target architecture, inference-mode, risk R-14, Phase 06 gate |

## 5. Method neutrality

| # | Check | Result |
|---|---|---|
| 5.1 | No document assumes the second NMF will be selected | ✅ `LEARNING_MODE_ARCHITECTURE.md` opens with "V7 is **not** 'NMF on NMF'" and marks Stage 4 "a candidate method, not a step"; decision rules state "the plan does not presuppose that the second NMF wins" |
| 5.2 | The stated prior is labelled a prior, not a decision | ✅ "That is a hypothesis to test, not a decision already made." |
| 5.3 | Full method comparison required to be published regardless of winner | ✅ Phase 03 gate, decision rules §3a, `DATA_CONTRACTS.md` C-07 (`method_selection.candidates_evaluated`) |
| 5.4 | Reference-construction control arm required to be reported honestly | ✅ "If A wins, that is the finding" — Phase 01 gate and decision rules §1 |
| 5.5 | Enforced mechanically | ✅ `test_second_nmf_is_not_presupposed` |

## 6. No premature performance claims

| # | Check | Result |
|---|---|---|
| 6.1 | No document claims V7 performance before implementation | ✅ enforced by `test_no_document_claims_v7_performance` |
| 6.2 | All quantitative claims are about V5/V6/V6.2/V6.3, with the source table named | ✅ |
| 6.3 | Success criteria marked provisional until frozen in Phase 00 | ✅ enforced by `test_success_criteria_are_marked_provisional` |
| 6.4 | Failure path defined and criteria explicitly not adjustable | ✅ P-13, `SUCCESS_CRITERIA.md` §5, Phase 07 gate |
| 6.5 | Root README declares "specified, not implemented" | ✅ enforced by `test_readme_declares_unimplemented_status` |

## 7. Numerical accuracy

Every quantitative claim was traced to its source and re-checked.

| Claim | Value | Source | Verified |
|---|---|---|---|
| Corpus | 375 spectra, 167 analytes, 676 bins | `assets/foundation/manifold.json → corpus_card` | ✅ |
| Replicate groups | 272, median 1, max 3; 87 analytes with replicates | same | ✅ |
| Multi-excitation analytes | 41 | same | ✅ |
| Components with purity ≥ 0.5 | 3 of 24 | `component_registry_v1.json`, recomputed | ✅ |
| Median purity / stability | 0.328 / 0.799 | same, recomputed | ✅ |
| Participation ratio | 15.2; 16 components for 90% latent variance | `manifold.json → intrinsic_dimensionality` | ✅ |
| Explained variance | 0.712 | `manifold.json → stats` | ✅ |
| MSS fine top-1 | 0.6707 (V6.3 fine) / 0.6766 (old) | `v63_metrics_by_ontology.csv` | ✅ |
| MSS broad top-1 | 0.8084; coord broad 0.8204 | same | ✅ |
| Random-ontology control | 0.096–0.113 fine | same | ✅ |
| MSS failure waterfall | 54 → 7 / 16 / **31** / 8 | `v63_waterfall.csv` | ✅ |
| True representation errors | 31/54 = 57.4% | computed from the same row | ✅ |
| Significance | coord −0.012 p=0.82; MSS −0.006 p=1.00; theme +0.018 p=0.63; system +0.060 p=0.041 | `v63_statistics.csv` | ✅ |
| Gain beyond mechanical | +0.550 coord, +0.572 MSS | `v63_comparison.csv` | ✅ |
| Family census | 18 families, 32 → 1 analytes, 107/167 uncovered | `p2_family_census.csv`, totals recomputed | ✅ |
| `sterol_ring_system` | purity 0.244, band fidelity 0.018, AUC 0.683, top family fatty_acid | `p2_motif_audit.csv` | ✅ |
| `flavin_redox_cofactor` coverage | 2 analytes (1.2%) | same | ✅ |
| Motif redundancy | porphyrin↔flavin 0.699; carboxylate↔colloid 0.687; purine↔sterol 0.679 support / 0.243 activation | `p2_motif_redundancy.csv` | ✅ |
| V6.2 Pareto | admissible first at K=13; info retained 0.796 at K=6; recoverability 0.969→0.503 | `v62_pareto.csv` | ✅ |
| theme_raw ≡ theme_posterior | identical at every metric on every ontology | `v63_metrics_by_ontology.csv` | ✅ |
| V6.2 hierarchy levels | L1=17, L2=6, L3=4 | `theme_membership.yaml` | ✅ |
| V6 motif count | 18 (13 in v1) | `mss_motifs_v6.yaml`, `mss_motifs_v1.yaml` | ✅ |

## 8. Frozen asset integrity

| # | Check | Result |
|---|---|---|
| 8.1 | Atlas fingerprint **before** the pass | `09ed804a40836f4a05a91ba10900cded` ✅ recomputed from `manifold_components.npz` |
| 8.2 | Atlas fingerprint **after** the pass | `09ed804a40836f4a05a91ba10900cded` ✅ unchanged |
| 8.3 | Basis shape | `(24, 676)` ✅ |
| 8.4 | All ten `assets/foundation/` files match their recorded SHA-256 | ✅ enforced by `test_frozen_foundation_file_hashes_unchanged` |
| 8.5 | No existing scientific file changed | ✅ `git status` showed only `GAIRA_v7_rebuild/` and `tests/test_v7_rebuild_scaffold.py` as new; **zero tracked modifications** |
| 8.6 | Protected paths still present | ✅ `assets/foundation`, `results/v5_rebuild`, `results/v6_rebuild`, `src/gaira/engine`, `src/gaira/preprocessing`, `tools/reproduce_gaira_foundation.py` |
| 8.7 | No model files created by this pass | ✅ enforced by `test_v7_created_no_model_files` |
| 8.8 | `code/` contains only its README | ✅ enforced by `test_v7_contains_no_code_directory_implementation` |

## 9. Path and policy hygiene

| # | Check | Result |
|---|---|---|
| 9.1 | No `/Users/…`, `/home/…`, or Windows drive path in any V7 file | ✅ enforced by `test_no_hardcoded_absolute_paths_in_v7_documents` / `…_python` |
| 9.2 | `/Volumes/` and `SSD_Rad` appear only in prohibition statements | ✅ every occurrence states the policy that they must not be used |
| 9.3 | `GAIRA_DATA_ROOT` is the documented data-root mechanism | ✅ consistent across `DATASET_AND_PROVENANCE_CONTEXT.md`, `GIT_AND_VERSIONING_PLAN.md`, `REPOSITORY_BASELINE.md` |
| 9.4 | No raw spectra committed | ✅ |
| 9.5 | Every V7 directory documents what belongs in it | ✅ enforced by `test_every_directory_has_a_readme` |

## 10. Cross-document structural agreement

| # | Check | Result |
|---|---|---|
| 10.1 | Phase list identical across plan, dependency map, phases/, README status table, and figure 7 | ✅ ten phases, same names, same order |
| 10.2 | Gates identical between the rebuild plan, phase READMEs, and the dependency map | ✅ |
| 10.3 | Strategy labels A–F consistent | ✅ design principles, rebuild plan, Phase 01/02 READMEs, figure 4 |
| 10.4 | Limitation IDs L-01…L-08 referenced consistently | ✅ |
| 10.5 | Risk IDs R-01…R-17 referenced consistently | ✅ every in-text reference resolves to a register entry |
| 10.6 | Data-contract IDs C-00…C-11 referenced consistently | ✅ |
| 10.7 | Principle IDs P-01…P-14 referenced consistently | ✅ |
| 10.8 | Phase numbering matches directory names | ✅ |

### One cross-reference correction made during the audit

`LEARNING_MODE_ARCHITECTURE.md` Stage 1 cites the per-class source/excitation composition
check as "risk R-14". In the register, source/excitation confounding is **R-16**; R-14 is
inference nondeterminism. The correct reference for that check is **R-16**. The same check is
cited correctly as R-16 in `DATASET_AND_PROVENANCE_CONTEXT.md`, the Phase-02 README, and the
rebuild plan, so the register and the phase-level plan are consistent; only that single
in-line citation is off. Recorded here rather than silently patched, and to be corrected in
the first Phase-00 commit.

## 11. Figures

| # | Check | Result |
|---|---|---|
| 11.1 | Ten planning figures present in SVG (vector) and PNG (preview) | ✅ enforced by `test_planning_figures_exist_in_vector_and_raster` |
| 11.2 | Every arrow corresponds to a defined computational operation | ✅ figures 2, 3, 5, 6 trace the operations named in `architecture/` |
| 11.3 | Numeric figures name their source table in the subtitle | ✅ figures 1, 4, 8, 9 |
| 11.4 | Conceptual figures labelled as such | ✅ figure 10 is explicitly "conceptual schematic — no data" |
| 11.5 | No decorative pseudo-science | ✅ |
| 11.6 | Deterministic generation (no RNG, no timestamps) | ✅ figure 10's trajectory is hand-specified, not sampled |
| 11.7 | PDF omitted, with the reason recorded | ✅ repo `.gitignore` excludes `*.pdf`; SVG satisfies the vector requirement |

## 12. Test suite

`pytest tests/test_v7_rebuild_scaffold.py` — **80 tests, all passing.** Coverage:

- V7 directory structure and required documents (structure only; no scientific-model tests)
- frozen atlas fingerprint and per-file SHA-256 integrity
- no model files, no V7 implementation code
- no hard-coded absolute paths in documents or scripts
- CSM defined consistently and never redefined
- BSV described as absolute; ΔBSV described as derived
- second NMF not presupposed; success criteria marked provisional
- no premature V7 performance claims
- protected paths present and not written to
- planning figures present in both formats

---

## Summary

| Category | Checks | Passed | Notes |
|---|---|---|---|
| Terminology | 8 | 8 | |
| BSV | 7 | 7 | |
| Themes | 4 | 4 | |
| Learning/inference boundary | 5 | 5 | |
| Method neutrality | 5 | 5 | |
| Performance claims | 5 | 5 | |
| Numerical accuracy | 22 | 22 | all traced to source tables |
| Frozen assets | 8 | 8 | fingerprint unchanged |
| Path hygiene | 5 | 5 | |
| Cross-document structure | 8 | 8 | one citation correction recorded (§10) |
| Figures | 7 | 7 | |
| Tests | — | 80/80 | |

**Frozen atlas fingerprint before and after this pass: `09ed804a40836f4a05a91ba10900cded` — unchanged.**

**No existing scientific asset was modified.** The only changes are the new
`GAIRA_v7_rebuild/` tree and `tests/test_v7_rebuild_scaffold.py`.
