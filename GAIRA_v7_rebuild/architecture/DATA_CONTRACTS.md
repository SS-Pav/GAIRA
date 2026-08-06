# GAIRA V7 — Data Contracts

Schemas for every object crossing a phase boundary. Each is versioned; a breaking change
requires a version bump and a migration note.

Conventions: `D = 676` (canonical grid bins); `M` = CSM count (Phase 03); `K` = theme count
= BSV dimension (Phase 04); all spectra, activations, memberships, and BSVs are **non-negative**.

---

## C-00 Canonical analyte table — *Phase 00*

`canonical_analytes_v1.csv`

| Column | Type | Notes |
|---|---|---|
| `canonical_id` | str | stable, unique, the unit of account |
| `canonical_name` | str | NFKC-normalised preferred name |
| `aliases` | str | `;`-separated observed surface forms |
| `chemical_class` | str | exactly one; from the frozen partition |
| `class_rationale_ref` | str | pointer to the written justification |
| `n_spectra` | int | |
| `n_replicate_groups` | int | |
| `excitations` | str | `;`-separated |
| `sources` | str | `;`-separated dataset IDs |
| `inchikey` / `smiles` | str | optional; when present, authoritative for identity |
| `is_anchor_candidate` | bool | flagged for Strategy F review |
| `notes` | str | merge/no-merge decisions, stereochemistry calls |

**Invariants.** `canonical_id` unique. Every spectrum maps to exactly one `canonical_id`.
Every `canonical_id` has exactly one `chemical_class`. Enantiomers and anomers are distinct
unless a written justification says otherwise.

---

## C-01 Replicate group table — *Phase 00*

`replicate_groups_v1.csv`

| Column | Type | Notes |
|---|---|---|
| `group_id` | str | |
| `canonical_id` | str | FK → C-00 |
| `excitation_nm` | float | group key component (see below) |
| `spectrum_ids` | str | `;`-separated |
| `n_spectra` | int | |
| `quality_score` | float | frozen `q`; see C-02 |

**Group definition (to be ratified in Phase 00):** `(canonical_id, excitation_nm)`. Analyte
balancing then applies at the `canonical_id` level *across* its groups, so the 41
multi-excitation analytes do not buy extra weight while excitation remains a tracked nuisance
factor.

---

## C-02 Quality metadata — *Phase 00*

`spectrum_quality_v1.csv`

| Column | Type | Notes |
|---|---|---|
| `spectrum_id` | str | |
| `snr_estimate` | float | |
| `baseline_residual` | float | asls fit quality |
| `grid_coverage` | float | fraction of 450–1800 cm⁻¹ measured |
| `spike_flag` | bool | cosmic ray / detector spike |
| `saturation_flag` | bool | |
| `source_prior` | float | per-dataset quality prior |
| `quality_score` | float | the frozen composite `q` used by Strategy B |
| `qc_pass` | bool | |

**Frozen before Phase 01.** `q` must not be tuned against Phase-01 outcomes — it would become
a hidden hyperparameter. A uniform-`q` sensitivity arm is mandatory in Phase 01.

---

## C-03 CV split manifest — *Phase 00*

`cv_splits_v1.json`

```
{ "schema": "cv_splits_v1", "seed": <int>, "n_folds": <int>,
  "grouping": "canonical_id",
  "folds": [ { "fold": 0, "train": [canonical_id...], "test": [canonical_id...] } ],
  "leakage_checks": { "alias_collision": false, "replicate_across_folds": false,
                      "canonical_id_across_folds": false } }
```

**Invariant — a hard gate.** Splits are grouped by `canonical_id`. No canonical ID, and no
replicate of one, may appear in more than one fold. All three leakage checks must read
`false` or Phase 00 does not pass.

---

## C-04 Balanced reference matrix — *Phase 01*

`balanced_references_v1.npz` — arrays `X (N×D)`, `weights (N,)`, `grid (D,)`; plus
`balanced_references_v1.csv`:

| Column | Type | Notes |
|---|---|---|
| `row_id` | str | |
| `canonical_id` | str | FK → C-00 |
| `chemical_class` | str | |
| `strategy` | str | `all_spectra` \| `analyte_weighted` \| `prototype_{mean,median,trimmed,medoid,quality}` |
| `weight` | float | Strategy B row weight; 1.0 otherwise |
| `n_source_spectra` | int | how many measurements this row summarises |
| `excitation_nm` | float | or `mixed` for cross-excitation prototypes |
| `provenance` | str | contributing spectrum IDs |

**Invariants.** `X ≥ 0`. Under Strategy B, weights sum to 1.0 per `canonical_id`. Under
Strategy C, exactly one row per `(canonical_id, excitation)` or per `canonical_id` — the
choice recorded in `strategy`.

---

## C-05 LSM dictionary — *Phase 02*

`lsm_dictionary_v1.npz` — per class, `H_c (k_c × D)`; plus `lsm_registry_v1.json`:

```
{ "schema": "lsm_registry_v1", "atlas_build": "<manifest hash>",
  "classes": { "<class>": { "k_c": <int>, "k_c_selection": {...},
                            "n_analytes": <int>, "source_composition": {...},
                            "excitation_composition": {...} } },
  "lsms": [ { "lsm_id": "...", "class": "...", "index": <int>,
              "stability": <float>, "recurrence": <float>,
              "type": "class_shared|subfamily|molecule_discriminating",
              "dominant_bands": [{"center_cm": ..., "width_cm": ..., "weight": ...}],
              "activating_analytes": ["..."], "redundancy_max": <float>,
              "retained": true, "retention_reason": "..." } ] }
```

**Invariants.** `H_c ≥ 0`. Only `retained: true` LSMs proceed to Phase 03; discarded LSMs stay
in the registry with a reason, so "what was thrown away" is answerable. `k_c ≤ ⌊n_analytes/2⌋`.
Per-class source and excitation composition are mandatory (risk R-14).

---

## C-06 LSM similarity graph — *Phase 03*

`lsm_graph_v1.json`

```
{ "schema": "lsm_graph_v1",
  "nodes": [ {"lsm_id": "...", "class": "...", "type": "..."} ],
  "edges": [ { "source": "...", "target": "...", "weight": <float>,
               "features": { "spectral_cosine": ..., "band_overlap": ...,
                             "peak_agreement": ..., "bootstrap_cooccurrence": ...,
                             "activation_cooccurrence": ..., "provenance_overlap": ... } } ],
  "threshold_sweep": [ {"threshold": ..., "n_edges": ..., "community_stability": ...} ],
  "selected_threshold": <float>, "selection_rationale": "..." }
```

**Invariants.** All six edge features present on every edge. The threshold sweep is
mandatory — a single unswept threshold is not acceptable evidence (risk R-07).
`provenance_overlap` must be computed with within-class overlap discounted (risk R-01).

---

## C-07 CSM dictionary — *Phase 03*

`csm_dictionary_v1.npz` — `CSM (M × D)`, `grid (D,)`; plus `csm_registry_v1.json`:

```
{ "schema": "csm_registry_v1", "atlas_build": "<hash>", "M": <int>,
  "integration_method": "consensus_clustering|graph_community|spectral|meta_nmf|hybrid",
  "method_selection": { "candidates_evaluated": [...], "criteria": {...},
                        "rationale": "..." },
  "csms": [ { "csm_id": "...", "index": <int>,
              "contributing_lsms": [{"lsm_id": "...", "weight": ...}],
              "supporting_classes": ["..."], "supporting_analytes": ["..."],
              "n_lsms": <int>, "n_classes": <int>, "n_analytes": <int>,
              "dominant_bands": [...], "band_fidelity": <float>,
              "uncertainty": <float>, "stability": <float>,
              "is_singleton": <bool>, "is_anchored": <bool>,
              "anchor_justification": "...", "provenance": {...} } ] }
```

**Invariants.** `CSM ≥ 0`. Every CSM resolves to LSMs → classes → analytes → sources.
`is_singleton` ⇔ `n_lsms == 1`. `is_anchored` requires a non-empty `anchor_justification` and
`n_analytes == 1`. `method_selection.candidates_evaluated` must list every candidate from
`LEARNING_MODE_ARCHITECTURE.md` Stage 4, so the choice is auditable whichever way it goes.

---

## C-08 Theme registry and membership — *Phase 04*

`theme_membership_v1.npz` — `S (M × K)`; plus `theme_registry_v1.yaml`:

```
schema: theme_registry_v1
K: <int>
K_selection: { criteria: {...}, pareto: [...], rationale: "..." }
themes:
  - theme_id: "..."
    name: "..."                    # CHEMISTRY ONLY — no disease/pathway/process
    chemical_definition: "..."
    top_csms: [{csm_id: ..., membership: ...}]
    n_supporting_csms: <int>
    membership_entropy: <float>
    chemically_admissible: <bool>
```

**Invariants.** `S ≥ 0`; rows sum to 1.0; `S` is sparse under the pre-registered sparsity
target. No theme name refers to a disease, pathway, process, or phenotype. No CSM is required
to have exactly one parent.

---

## C-09 BSV specification — *Phase 05*

`bsv_reference_v1.json`

```
{ "schema": "bsv_reference_v1", "K": <int>, "atlas_fingerprint": "...",
  "axes": [ {"theme_id": "...", "index": <int>,
             "reference_mean": <float>, "reference_spread": <float>,
             "reference_distribution": {...}, "support_breadth": <int>,
             "uncertainty_inflation": <float>} ],
  "effective_rank": {"participation_ratio": <float>, "entropy_rank": <float>,
                     "n_axes_90pct": <int>},
  "ood_support": "bsv_ood_support_v1.npz",
  "visualisation": {"P": "bsv_pca_v1.npz", "note": "frozen transform, VISUALISATION ONLY;
                    not the canonical BSV; applied never fitted"} }
```

**Invariants.** `BSV ≥ 0`. `K` = number of themes. `effective_rank` is reported alongside `K`
(risk R-12). `uncertainty_inflation` is > 1 for axes dominated by singleton or anchored CSMs.
The visualisation block carries its disclaimer in the artefact itself, not only in prose.

---

## C-10 Inference output — *Phase 06*

`gaira_v7_inference_v1`

```
{ "schema": "gaira_v7_inference_v1",
  "atlas_version": "...", "atlas_fingerprint": "...",
  "preprocessing_config_hash": "...",
  "csm_activations": [<float> × M],
  "theme_activations": [<float> × K],
  "bsv": [<float> × K],                    // ABSOLUTE
  "bsv_elevation": [<float> × K],          // derived, signed, z-scored
  "uncertainty": {"per_axis": [...], "method": "..."},
  "qc": {"grid_coverage": ..., "reconstruction_residual": ...,
         "band_fidelity": ..., "ood_score": ..., "flags": [...]},
  "confidence_tier": "...",
  "evidence": {"top_csms": [...], "supporting_analytes": [...],
               "singleton_or_anchored_axes": [...]},
  "provenance": {...} }
```

**Invariants.** `bsv ≥ 0` and absolute. `bsv_elevation` is signed and **never** named `bsv`.
A ΔBSV is never returned by the inference path — it is produced by a separate call over two
BSVs. `atlas_fingerprint` is present on every output.

---

## C-11 Build manifest — *every phase*

Every phase writes a manifest recording: input artefact IDs and hashes, config, seeds, code
version (git SHA), environment (Python, numpy, scipy, sklearn, BLAS), timestamp, output
artefact IDs and hashes, and the gate-check results. Full spec:
`ARTIFACT_AND_MANIFEST_SPEC.md`.

---

## Cross-cutting invariants

| # | Invariant | Where checked |
|---|---|---|
| 1 | All spectra, activations, memberships, and BSVs are non-negative | every phase |
| 2 | Every artefact resolves to canonical analytes and source datasets | Phase 06 |
| 3 | No canonical ID crosses a CV fold boundary | Phase 00 gate |
| 4 | No raw spectra in Git | CI / repo policy |
| 5 | No absolute lab paths; `GAIRA_DATA_ROOT` only | scaffold test + CI |
| 6 | Every artefact carries the build manifest hash | Phase 06 |
| 7 | Schema version present on every artefact | all |
| 8 | Singleton and anchored status propagates to output uncertainty | Phase 05 / 06 |
