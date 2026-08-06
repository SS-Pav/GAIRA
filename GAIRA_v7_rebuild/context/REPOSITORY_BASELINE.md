# GAIRA V7 — Repository Baseline

Recorded at the moment the V7 branch was created. This file is the immutable
reference point for every later V7 claim of the form "relative to the current atlas".

## Branch and commit

| Field | Value |
|---|---|
| Source branch | `gaira-v5-rebuild-plan` |
| Source commit | `ddbb3945d670eee58f5ad99f868fb3c36b2a2c06` |
| Source commit subject | `test(v6.2): add regression tests for the semantic hierarchy (15 cases)` |
| New branch | `gaira-v7-rebuild` |
| Branch base | identical to source commit (no rebase, no reset, no history rewrite) |
| Remote | `origin` → `https://github.com/SS-Pav/GAIRA.git` (fetch and push) |
| Default/integration branch | `main` |

## Working-tree state at branch creation

- **Tracked modifications: 0.** `git status --porcelain` reported no `M`, `A`, `D`, or `R`
  entries. The tracked tree was clean, so creating the branch was safe.
- **Untracked entries: 53.** These are pre-existing, unrelated to V7, and were
  **not staged, not committed, and not discarded**. They carry across the branch switch
  unchanged. They fall into four groups:

| Group | Examples | Disposition |
|---|---|---|
| Legacy exploratory report trees | `reports/gaira_bsv_*`, `reports/embedding_v*_smoke/`, `reports/small2023_ev_v*/`, `reports/gaira_phase1_registry_audit_v*/` | Left untouched. Pre-V5-rebuild experiment output. |
| Regenerable binary intermediates under existing result trees | `results/v5_rebuild/*/artifacts/*.npz`, `results/v6_rebuild/semantic_validation/`, `results/v6_rebuild/v63_ontology_revalidation/` | Left untouched. Some are canonical V6.3 outputs not yet committed on this branch. |
| Loose top-level working directories | `figures/`, `outputs/`, `pathogen/` | Left untouched. |
| Machine-local files | `.zshrc`, `streamlit_apps/*/.streamlit/` | Left untouched; not V7 concerns. |

**Consequence for V7:** the V6.3 ontology revalidation outputs
(`results/v6_rebuild/v63_ontology_revalidation/`) are present on disk but not yet
committed. Phase 00 must decide whether to commit them as a versioned input or to
re-derive them under a V7 manifest. Until that decision is made, every V7 document that
cites a V6.3 number also cites the on-disk table it came from.

## Frozen atlas fingerprint

| Field | Value |
|---|---|
| Fingerprint | `09ed804a40836f4a05a91ba10900cded` |
| Definition | first 32 hex characters of `sha256(ascontiguousarray(H).tobytes())`, where `H` is the NMF basis |
| Basis array | `assets/foundation/manifold_components.npz` → key `components`, shape `(24, 676)` |
| Declared in | `assets/foundation/manifold.json` → `fingerprint`; `assets/foundation/MANIFEST.json` → `atlas_fingerprint` |
| Pinned in code | `tools/reproduce_gaira_foundation.py` → `CANONICAL_FINGERPRINT` |
| **Verified before this documentation pass** | `09ed804a40836f4a05a91ba10900cded` ✅ recomputed from the array |
| **Verified after this documentation pass** | see `context/CONSISTENCY_AUDIT.md` |

## Frozen asset inventory (must not change during V7 planning)

`assets/foundation/` — the frozen scientific product, ten files:

| File | Role | SHA-256 (from `MANIFEST.json`) |
|---|---|---|
| `manifold_components.npz` | NMF basis **H** (24 × 676) + grid + mean | `ca385847146c7a9b72bd5c7ecfae85105ecf8740e43abe4a83ce894587444b9f` |
| `manifold.json` | frozen NMF metadata, corpus card, selection, validation | `fc0fae1e130af6725eb7f834cbd0b688da2c2636c5ad97cf8f02a1e079d2ab87` |
| `component_registry_v1.json` | per-component provenance, purity, stability, bands | `d9f205f80b787b675bdc5f72cf14d64fa065b90735ffcad683318c40cb29626d` |
| `component_theme_weights_v1.json` | component→theme weights **W** (24 × 13) | `ab75d76cca02cade290aa7e56a5614da36708fa84fa988b17b5d9c61fc1006bf` |
| `biochemical_ontology_v2.yaml` | 13 theme definitions (11 biochemical + 2 non-biochemical) | `b37cdc8ee3b2ab9ca341defc661328b08fecc0f5a7ce25908be604be91513aea` |
| `mss_motifs_v1.yaml` | 13 legacy MSS motif definitions | `3983c2346e410a543a681143775546b2c75cb5ea738668ffebada8ec4ab3c702` |
| `reference_normalization_v1.json` | per-component centre/spread reference frame | `f2a015ac795dea41e8716b17e9304fed87b67c7529f8d15dd98e24f92ff43614` |
| `reference_support.npz` | reference support vectors for the OOD score | `65b45ff59b4d3f05dbec1acb728c4ddc15722e2501a2705146297cf97a72819e` |
| `MANIFEST.json` | fingerprint, stack versions, preprocessing config, per-file hashes | — |
| `README.md` | frozen-bundle documentation | — |

## Path register — what V7 reads (never writes)

### Canonical preprocessing
| Item | Path |
|---|---|
| Preprocessing primitives | `src/gaira/preprocessing/pipeline.py` |
| Entry point | `preprocess(wn, y, config, grid=None, window=...)` |
| Baseline / smooth / norm used by the atlas | `asls` / `savgol` / `l2` (from `MANIFEST.json → preprocessing`) |
| Atlas analysis window | 450–1800 cm⁻¹, 2.0 cm⁻¹ step, 676 bins |
| Note | `common_grid()` defaults to the **legacy Ag-SERS-constrained 520–1750** window. The atlas window is passed explicitly. V7 must pass the window explicitly too and never rely on the default. |

### Frozen basis and inference engine
| Item | Path |
|---|---|
| Frozen bundle | `assets/foundation/` |
| Asset resolution with legacy fallback | `src/gaira/engine/paths.py` (`ASSETS` → `LEGACY_FOUNDATION` → `LEGACY_ENGINE` → `PKG_DATA`) |
| Engine entry points | `src/gaira/engine/pipeline.py` (`GAIRAEngine`, `GAIRAInference`) |
| NNLS projection, BSV, MSS, radar, evidence, OOD | `src/gaira/engine/{bsv,mss,radar,evidence,normalization,ontology,registry,domain,dart,versioning}.py` |
| Legacy build tree (fallback source) | `results/v5_rebuild/foundation/artifacts/`, `results/v5_rebuild/engine_v1/artifacts/` |

### V6.2 semantic hierarchy
| Item | Path |
|---|---|
| Core implementation | `results/v6_rebuild/code/v62/core.py` |
| V6 MSS + theme implementation | `results/v6_rebuild/code/v6_semantic/{mss_v6.py,themes_v6.py}` |
| 18 V6 motif definitions | `results/v6_rebuild/artifacts/mss_motifs_v6.yaml` |
| Motif registry | `results/v6_rebuild/artifacts/mss_registry_v6.json` |
| Soft motif→theme membership (L1=17 / L2=6 / L3=4) | `results/v6_rebuild/artifacts/theme_membership.yaml` |
| Soft hierarchy + membership matrices | `results/v6_rebuild/artifacts/{v62_soft_hierarchy.json,v62_membership.npz,v62_spaces.npz}` |
| Information graph | `results/v6_rebuild/artifacts/v62_information_graph.json` |
| Pareto / bottleneck / uncertainty tables | `results/v6_rebuild/tables/v62_*.csv` |
| Motif audit, family census, redundancy | `results/v6_rebuild/tables/p2_*.csv` |
| Theme sweep | `results/v6_rebuild/tables/p4_theme_sweep.csv` |
| Reports | `results/v6_rebuild/reports/V62_*.pdf`, `THEME_HIERARCHY_AUDIT.md` |

### V6.3 cleaned evaluation ontology
| Item | Path |
|---|---|
| Revalidation tree | `results/v6_rebuild/v63_ontology_revalidation/` |
| Metrics by ontology | `.../tables/v63_metrics_by_ontology.csv` |
| Old vs fine vs broad comparison | `.../tables/v63_comparison.csv` |
| Failure waterfall | `.../tables/v63_waterfall.csv` |
| Significance tests | `.../tables/v63_statistics.csv` |
| Per-family / per-analyte audit | `.../tables/{v63_per_family.csv,v63_analyte_audit.csv}` |
| Error reclassification | `.../tables/v63_error_reclassification.csv` |
| Confusion matrices | `.../tables/{v63_confusion_fine.csv,v63_confusion_old.csv}` |
| Status | **uncommitted on disk** — see working-tree note above |

### Reproduction tooling
| Item | Path |
|---|---|
| Deterministic rebuild orchestrator | `tools/reproduce_gaira_foundation.py` |
| Modes | `full` (raw corpus → NMF → compare) and `interpretation-only` (frozen basis, no raw data, no SSD) |
| Data-root precedence | `--data-root` > `$GAIRA_DATA_ROOT` > optional documented default > error |
| Committed reference coordinates | `results/v5_rebuild/reproduction/manifests/nmf_reference_coordinates.npz` |
| Dataset role map | `results/v5_rebuild/reproduction/audits/dataset_role_map.csv` |
| Reproduction test | `tests/test_reproduce_foundation.py` |

V7 inherits the same data-root policy: **no absolute lab paths, no `SSD_Rad` defaults,
`GAIRA_DATA_ROOT` only.**

## Corpus baseline (from `manifold.json → corpus_card`)

| Field | Value |
|---|---|
| Domain | Raman only (canonical observation domain) |
| Excluded domains | Ag-SERS, Au-SERS, DART, serum Ag-colloid, metabolite-63 (633 nm), adenine Ag-SERS series, European multi-instrument adenine |
| Spectra | 375 |
| Analytes | 167 |
| Bins | 676 (450–1800 cm⁻¹, 2.0 cm⁻¹) |
| Sources | RamanBioLib 202, `gobbato_raman_metabolites` 153, `amino_acid_raman_grounding` 20 |
| Excitations | 785 nm ×234, 1064 ×55, 532 ×50, 488 ×29, 514.5 ×3, others ×4 |
| Analytes with >1 excitation | 41 |
| Replicate groups | 272 (median size 1, max 3) |
| Analytes with replicates | 87 |
| External projection-only set | `covid_serum_raman` (never used for fitting) |

## Repository policy constraints inherited by V7

From the root `.gitignore`:

- `*.pdf` is ignored → **V7 planning figures ship as SVG (vector) + PNG (preview)**, not PDF.
- `checkpoints/` is ignored globally → `GAIRA_v7_rebuild/results/checkpoints/` is re-included
  by a scoped `GAIRA_v7_rebuild/.gitignore` negation so V7 checkpoint manifests can be tracked.
- `data/`, `GAIRA_DATA/`, `/Volumes/`, `*.mat` are ignored → **no raw spectra in Git, ever.**
- `.npz`, `.npy`, `.csv`, `.json`, `.png`, `.md` are deliberately tracked — they are the product.
