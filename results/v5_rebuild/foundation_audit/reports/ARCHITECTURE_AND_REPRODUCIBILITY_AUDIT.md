# ARCHITECTURE & REPRODUCIBILITY AUDIT
### Does the GAIRA implementation faithfully match the intended architecture, and can the whole foundation be regenerated independently?

*Audit only — no code was modified. Every claim below is verified against the current
source and committed artifacts (paths + quoted mechanisms given). Complements the 11-part
Foundation Model audit in this folder with two new lenses: (a) architecture faithfulness
and (b) provenance of every learned/annotated/interpreted layer.*

**Headline verdict.** The learned representation is data-driven, deterministic, and
byte-for-byte reproducible *in a pinned environment*; the interpretation layers (registry,
component→theme weights, MSS, normalization) are all **derived by committed, deterministic
scripts** from the frozen atlas + curated knowledge — not hand-tuned numbers. Two real
gaps: (1) **exact reproducibility is environment-sensitive and not version-stamped**, and
(2) the implemented layering is **`Component → Theme` (direct) with MSS as a parallel
overlay** — the *inverse* of the intended `Component → MSS → Theme` grouping. Neither is a
scientific error, but both should be documented so they are not mis-read.

---

## 1 · NMF Foundation Audit

**Training corpus** — `src/gaira/foundation/dataset.py::load_reference_corpus`.

| | |
|---|---|
| Loaded from | RamanBioLib (`data/loader.py`), Gobbato Raman (`data/gobbato.py`, Raman subset only), amino-acid sheet (`aa.xlsx`) — all under `/Volumes/SSD_Rad/GAIRA_DATA/raw` (GAIRA_Lab) |
| Included | 3 pure-Raman sources: RamanBioLib 202/141, gobbato_raman_metabolites 153/51, amino_acid_raman_grounding 20/19 |
| Excluded | **Ag-SERS, Au-SERS, DART, serum** — enforced by `assert (meta.modality == "raman").all()` (line 121). Gobbato's SERS half is skipped by an explicit `modality != "raman": continue` |
| Spectra / analytes | **375 spectra / 167 labels** (≈161 distinct molecules) |
| Duplicate handling | **None** — spectra are not de-duplicated; 34 analytes intentionally replicate across sources; **6 canonicalization duplicate labels** (`alb`/`albumin`, riboflavin ligature, …) are *not* merged |
| Inclusion/exclusion filters | modality==raman; a finite-value check drops all-NaN preprocessed vectors; no outlier/QC filter |

**Preprocessing** (exact, deterministic) — `preprocessing/pipeline.py` config `P2` in
`dataset.py::PREPROC`:
`crop [450,1800] cm⁻¹ → ASLS baseline (λ=1e5, p=0.01, 8 iter) → Savitzky-Golay (win=9,
poly=3) → resample to a fixed 2 cm⁻¹ grid (676 bins, NaN outside range) → L2 normalize →
clip ≥ 0 (NMF only)`. The non-negativity clip removes < 1 % of signal mass.

**NMF implementation** — `foundation/representation.py::fit_nmf`,
`sklearn.decomposition.NMF`:
`n_components=24, init="nndsvda", solver="cd" (default), beta_loss="frobenius" (default),
alpha_W=alpha_H=0 (default), l1_ratio=0 (default), max_iter=1500, tol=1e-4 (default),
random_state=0, shuffle=False (default)`. Fit on the clipped, L2-normalized matrix.

**Determinism.** Deterministic **given the same data, seed, and numerical stack**. The
audit reproduced the frozen basis **byte-for-byte** (max abs diff `0.0` → identical
fingerprint) — but only in the *pinned* environment. **Sources of variability are NOT the
seed** — they are the sklearn/numpy/scipy/BLAS versions (coordinate-descent NMF numerics
can change across releases). The build stamps **no** environment record (see §2).

**Outputs.** `H` (the 24×676 basis) is frozen in `manifold_components.npz["components"]`
(+ `grid`, `mean`). **`W` (the 375×24 training activations) is NOT persisted** — it is
recomputed per query by non-negative least squares against the frozen `H`
(`foundation/serialization.py::FrozenAtlas.project`, `update_H=False`). The basis is
frozen and fingerprinted; both Streamlit apps load the *same* frozen basis via
`gaira.engine.paths` (→ `assets/foundation/`, fingerprint-checked on load). ✅

---

## 2 · Exact Reproducibility of the NMF Basis

**Can a new developer regenerate the identical basis?** — **Yes, with caveats.**

Exact command chain (deterministic, seed 0):
```
python results/v5_rebuild/foundation/code/run_c1_benchmark.py   # corpus → benchmark → c1_selection.json (NMF k=24)
python results/v5_rebuild/foundation/code/run_c2_c7.py          # corpus → NMF → freeze → foundation/artifacts/  (fingerprint 09ed804a…)
```
Inputs: the 3 raw Raman sources (SSD/GAIRA_Lab), `foundation/{dataset,benchmark,
representation,latent_space,serialization}.py`, seed 0.

**Undocumented dependencies / manual interventions found:**
1. **Environment not pinned in the primary path.** `scikit-learn==1.8.0` appears only in
   `requirements_vm.txt`; the primary `requirements.txt` and `pyproject.toml` say
   `scikit-learn>=1.3`, and `manifold.json` records **no** sklearn/numpy/BLAS versions.
   A clone that resolves a different sklearn may produce a *different* basis (⚠️).
2. **Absolute repo path hard-coded** in 8 build scripts (`REPO =
   Path("/Users/surajpg/projects/GAIRA")` in `foundation/code/*` and `engine_v1/code/*`).
   A new developer must edit these (⚠️, rebuild-only; runtime is unaffected).
3. **SSD corpus path** hard-coded in `dataset.py` (`RAW = /Volumes/SSD_Rad/…`).

Protocol (all steps documented above): reference spectra → preprocessing → matrix → NMF
(seed 0) → W,H → freeze H (sha256) → store metadata (manifold.json + MANIFEST.json).

---

## 3 · Component Audit

- **24 components.** Chosen by the multi-criteria benchmark (`benchmark.py`, 5
  representations × 6 k, analyte-grouped CV) + a **pre-stated non-negativity tie-break**:
  raw winner is signed ICA k=32; NMF k=24 is selected as the best *parts-based* candidate
  in the 0.02 tie band. Justified in `NMF_REBUILD.md`.
- **Ordering is deterministic but NOT semantically anchored.** `fit_nmf` does **not** sort
  components; order is whatever NNDSVD-a + coordinate descent return for the given
  seed+data+sklearn. Component IDs (`c0…c23`) are therefore stable **only** while that
  environment is fixed.
- Per-component facts (all in `component_registry_v1.json` + `component_audit_summary.csv`):
  strongest analytes, dominant bands, reconstruction contribution (per-component variance
  0.018–0.119), purity, effective-contributor entropy, bootstrap stability (0.65–0.97,
  mean 0.81), biochemical interpretation. Max pairwise basis cosine 0.52 → **no redundant
  component**.
- **Numbering stability across rebuilds: NOT guaranteed** if the numerical stack changes.
  Because the registry and the component→theme weights key everything by **integer
  component index**, an environment-shifted rebuild could silently misalign annotations to
  components (⚠️ — the single biggest reproducibility risk).

---

## 4 · MSS Audit — how a component becomes an MSS

MSS are created in `src/gaira/engine/mss.py::MSSLayer` (and serialized by
`engine_v1/tools/build_mss_registry.py`). **Two halves:**

- **Curated definition** (`src/gaira/engine/data/mss_motifs_v1.yaml`, 13 motifs): `id`,
  `name`, characteristic `bands_cm`, `exemplars` (reference chemistries), `parent_theme`.
  Hand-authored from Raman spectroscopy.
- **Derived contributors** (deterministic, pure function of frozen artifacts): for every
  component *j*, `score(j) = 0.40·band + 0.35·exemplar + 0.25·theme`; keep `score ≥ 0.15`,
  take top 6, normalize → contributor weights. `confidence = weighted stability ×
  evidence-breadth`; perturbation evidence pulled from the registry.

**Is "Component 7 → Purine MSS" explicit?** — **No.** There is no hard-coded
component→motif table. A component *becomes part of* a motif by **scoring against the
motif's curated bands/exemplars/theme**; the mapping is *derived*, not asserted, and a
motif is a *weighted set of components*, not a single one.

**Actual implementation diagram:**
```
mss_motifs_v1.yaml (curated: bands, exemplars, parent_theme)
        +                                    → score each component (0.40 band + 0.35 exemplar + 0.25 theme)
component_registry_v1.json (derived: bands, loadings, stability)   → keep ≥0.15, top-6, normalize
        +                                    → MSS motif = {contributing components + weights + confidence}
component_theme_weights W (derived)
```

---

## 5 · Literature & Knowledge Audit

| Knowledge source | Location | Format | Version-controlled? | Provenance |
|---|---|---|---|---|
| Theme definitions + `characteristic_bands_cm` + `family_theme_affinity` + literature notes | `src/gaira/engine/data/biochemical_ontology_v2.yaml` | YAML | ✅ | curated from Raman literature |
| MSS motif bands / exemplars / parent_theme | `src/gaira/engine/data/mss_motifs_v1.yaml` | YAML | ✅ | curated |
| RamanBioLib metadata | `raw/ramanbiolib/…` (SSD) | CSV | lab | **external, cited** (Terán 2025, ODbL) |
| Peak assignments ("knowledge core") | `raw/raman_knowledge_core/` (SSD) | CSV | ❌ (on SSD) | **self-authored — `authors: GAIRA Curated Seed Pack`, `license: internal-curated`, no DOI**; used only for C3 axis interpretation, **not** by the runtime engine |

**Knowledge that currently lives only in code** (version-controlled as `.py`, but not
declarative/inspectable data):
- `PERTURBATION_THEME` dict — analyte→theme for perturbation evidence
  (`engine_v1/code/build_theme_weights.py:35`).
- Chemical-family rules + name sets (`PROTEIN_NAMES`, `PURINES`, …) in
  `foundation/families_raman.py`.
- `AA_NAME_FIX` canonicalization map (`foundation/dataset.py:40`).

These are transparent but embedded in Python; §10–11 recommend externalizing them.

---

## 6 · Theme Audit

- **Themes are hand-defined**, not inferred: 13 themes (11 biochemical + `background_matrix`
  + `unknown_mixed`) in `biochemical_ontology_v2.yaml` (id, description,
  `characteristic_bands_cm`, `family_theme_affinity`, literature, caveats).
- **Component→theme weights are DERIVED, deterministically** —
  `engine_v1/code/build_theme_weights.py`:
  `w = 0.50·loading (families→theme via family_theme_affinity) + 0.25·spectral
  (band overlap ±20 cm⁻¹) + 0.25·perturbation (driving analytes → PERTURBATION_THEME)`,
  residual mass → background/unknown, normalized so each component's weights sum to 1.
  **Every weight records its three evidence lines + a confidence label.** No weight is
  hand-typed. Reads only committed tables (no SSD).
- **Complete mapping — as intended vs as built:**
  - *Intended:* `Component → MSS → Theme` (themes group MSS).
  - *Built:* `Component →(W)→ Theme` **directly**, and `Component →(M)→ MSS` **in
    parallel**. `MSSLayer` and `BSVBuilder` are independent; **`bsv.py` never imports
    MSS** (`composition = Wᵀ·coord`). A motif's `parent_theme` is a *curated label*, and
    the MSS contributor score even *consumes* the theme weight (25 %). So the intended
    MSS→Theme dependency is **inverted**: MSS depends on themes, not the reverse.

---

## 7 · Streamlit Audit

Both apps read only committed assets (fingerprint-checked); neither recomputes analysis.

- **Foundation Explorer** cleanly separates *learned* (Page 4: the NMF benchmark, framed
  as data-driven) from *interpreted* (Page 6: MSS + ontology). No wording implies the
  literature created the components. **One inaccuracy:** the Page-6 flow diagram renders
  `Components → MSS motifs → Themes → BSV`, implying MSS feeds themes — which the code
  contradicts (themes are computed directly from components; MSS is parallel). The cards
  below the diagram describe it correctly, but the arrow order is misleading.
- **Reasoning demo (v4)** bills **"the MSS layer is the centerpiece"** (`p3_reasoning.py`).
  This is accurate as the *interpretive/human-facing* centerpiece, but it overstates MSS's
  computational role: the **BSV is computed `Component →W→ Theme` without MSS**, so the
  true load-bearing interpretive object is the component→theme weight matrix, not MSS.
- **Recommendation:** re-order the Explorer Page-6 flow to show themes and MSS as *parallel*
  projections of the components, and add one sentence to the demo clarifying MSS is a
  finer-grained explanatory overlay parallel to the BSV, not its computational input.

---

## 8 · Repository Audit — where each stage lives

| Stage | Location | Runtime SSD? |
|---|---|---|
| Raw spectra | `/Volumes/SSD_Rad/GAIRA_DATA/raw` (GAIRA_Lab) | build only |
| Preprocessing | `src/gaira/preprocessing/pipeline.py` + `foundation/dataset.py` | no |
| NMF | `foundation/{representation,benchmark,latent_space}.py`, `foundation/code/run_c1_benchmark.py` + `run_c2_c7.py` | build only |
| Frozen basis | `assets/foundation/manifold_components.npz` (+ legacy `results/v5_rebuild/foundation/artifacts/`) | no |
| MSS | `engine/mss.py` + `data/mss_motifs_v1.yaml` (+ `engine_v1/tools/build_mss_registry.py`) | no |
| Themes | `data/biochemical_ontology_v2.yaml` + `component_theme_weights_v1.json` + `engine/ontology.py` (built by `engine_v1/code/build_theme_weights.py`) | no |
| Registry / normalization | `engine_v1/code/build_registry.py`, `build_reference_norm.py` | registry: no · norm: **yes (corpus)** |
| BSV | `src/gaira/engine/bsv.py` | no |
| Demos | `gaira_foundation_explorer/`, `gaira_demo_reasoning_v4/` | no |

**Dependencies flagged:** (a) `build_reference_norm.py` calls `load_reference_corpus()` →
needs SSD to rebuild; (b) 8 build scripts hard-code the absolute repo path; (c)
`raman_knowledge_core` lives only on SSD. **No hidden notebooks, temp outputs, or manual
edits are in the runtime path.** Regeneration order is documented in
`engine_v1/MIGRATION_NOTES.md` (`build_registry → build_theme_weights → build_reference_norm
→ run_validation/emit_examples`; "the first three are deterministic").

---

## 9 · Foundation Reproducibility Audit (per stage)

| Stage | From a clone (runtime) | Full rebuild FROM RAW |
|---|---|---|
| Preprocessing | ✅ committed + deterministic | ✅ (needs SSD corpus) |
| Reference matrix | ✅ | ✅ (needs SSD corpus) |
| Latent components (H) | ✅ committed & fingerprinted | ⚠️ byte-identity requires exact `sklearn==1.8.0` (unpinned in primary reqs; not stamped) |
| Component ordering / IDs | ✅ (frozen) | ⚠️ env-dependent; annotations keyed by integer index |
| MSS | ✅ derived from committed artifacts + yaml | ✅ deterministic, no SSD |
| Biochemical themes / weights | ✅ | ✅ deterministic from committed tables + yaml, no SSD |
| BSV | ✅ | ✅ deterministic transform |
| Reference normalization | ✅ committed | ⚠️ rebuild needs SSD corpus |
| Streamlit outputs | ✅ clone-and-run proven (no SSD) | ✅ |

**Runtime = fully reproducible (✅).** **Full from-raw rebuild = reproducible but ⚠️**:
requires the SSD corpus, the exact numerical environment, and editing hard-coded absolute
paths.

---

## 10 · Missing-Assets Audit

| Asset | Exists? | Where |
|---|---|---|
| NMF basis | ✅ | `assets/foundation/manifold_components.npz` |
| Component annotations (registry) | ✅ | `assets/foundation/component_registry_v1.json` |
| MSS catalogue | ✅ defs (`mss_motifs_v1.yaml`) + derived registry (`foundation_audit/tables/mss_registry.json`) |
| Theme definitions | ✅ | `assets/foundation/biochemical_ontology_v2.yaml` |
| Ontology (component→theme W) | ✅ | `assets/foundation/component_theme_weights_v1.json` |
| Peak references | ⚠️ | `raman_knowledge_core` (SSD, self-authored, not in repo) |
| Metadata / fingerprint | ✅ | `assets/foundation/MANIFEST.json` |
| Preprocessing config | ⚠️ | **in code** (`PREPROC` dict); only summarized in MANIFEST — no standalone file |
| Environment lock | ❌ | not stamped in the foundation manifest |
| In-code knowledge (PERTURBATION_THEME, family rules) | ⚠️ | in `.py`, not declarative data |

---

## 11 · Recommended Canonical Architecture (additions only)

`assets/foundation/` already holds the basis, registry, weights, ontology, MSS, metadata
+ MANIFEST. Recommended additions to close the gaps above — all additive, none change the
frozen model:

```
assets/foundation/
    nmf_basis.npz               ✅ (manifold_components.npz)
    component_annotations.json  ✅ (component_registry_v1.json)
    mss_catalog.json            ✅ (mss_motifs_v1.yaml + derived registry)
    biochemical_themes.json     ✅ (biochemical_ontology_v2.yaml)
    component_theme_weights.json✅
    preprocessing.yaml          ➕ NEW — lift PREPROC out of code (window, grid, ASLS/SG params, norm)
    environment.lock            ➕ NEW — exact sklearn/numpy/scipy/BLAS versions used for the freeze
    knowledge_maps.json         ➕ NEW — externalize PERTURBATION_THEME + family_theme rules from .py
    references.md               ➕ NEW — external citations + explicit provenance of the self-authored knowledge-core
    fingerprint / MANIFEST.json ✅
```
Each exists so that **every learned representation, annotation, and interpretation is
explicit, version-controlled, inspectable, and reproducible** from declarative assets
rather than embedded code.

---

## 12 · Final Scientific Assessment

- **What exists:** a data-driven frozen NMF basis + a fully *derived*, evidence-carrying
  interpretation stack (registry, component→theme weights, MSS, normalization), all built
  by committed deterministic scripts, plus curated theme/motif YAML — and two portable
  demos that run from a clone with no SSD.
- **What is scientifically sound:** the NMF is honest and reproducible; the component→theme
  weights are transparently derived (3 evidence lines, fixed mixing, per-weight evidence);
  MSS derivation is deterministic; themes/motifs are curated and version-controlled; the
  representation↔interpretation boundary is respected (SERS validates, never fits).
- **What is implicit rather than explicit:** the preprocessing config, the perturbation
  and family knowledge maps, and the environment lock live in code, not in declarative
  assets; `raman_knowledge_core` is self-authored and off-repo.
- **Manual steps / hidden dependencies remaining:** editing hard-coded absolute repo paths
  in the 8 build scripts; supplying the exact numerical environment for a byte-identical
  rebuild; the SSD corpus for `build_reference_norm` and the foundation rebuild.
- **Could another developer independently regenerate the whole foundation?** **Runtime:
  yes, from the clone alone (verified).** **From raw: yes in principle, but only with the
  SSD corpus, the pinned environment, and minor path edits** — so "independently, on any
  machine, byte-for-byte" is ⚠️ until the environment lock + relative paths are added.
- **Does the implementation faithfully represent the intended architecture?** **Largely
  yes, with one structural deviation:** the pipeline is `Reference → preprocessing → NMF →
  data-driven components → curated/derived annotation`, exactly as intended — but the
  **themes are derived directly from components (a many-to-many weight matrix), and MSS is
  a parallel explanatory overlay, not the intermediate that themes are grouped from.** The
  intended `Component → MSS → Theme` chain is implemented as `Component → {Theme, MSS}` in
  parallel. This is defensible (a direct, auditable component→theme map is more robust than
  an MSS-grouping cascade) but should be stated plainly in the docs and demos.

---

## Foundation Reproducibility Checklist

*Everything required to rebuild the GAIRA foundation from scratch.*

**Inputs (GAIRA_Lab / SSD)**
- [ ] RamanBioLib (`raw/ramanbiolib/…`) · Gobbato Raman (`raw/serum_ag_colloids/…zip → Raman metabolites/`) · amino-acid sheet (`raw/amino_acid_raman_grounding/aa.xlsx`)

**Code — representation**
- [ ] `src/gaira/preprocessing/pipeline.py` (P2: crop→ASLS→SG→resample→L2→clip)
- [ ] `src/gaira/foundation/{dataset,benchmark,representation,latent_space,serialization,axes,bsv,mss}.py`
- [ ] `results/v5_rebuild/foundation/code/run_c1_benchmark.py` → `run_c2_c7.py`

**Code — interpretation (deterministic; order matters)**
- [ ] `engine_v1/code/build_registry.py` → `build_theme_weights.py` → `build_reference_norm.py` (→ `run_validation.py`, `emit_examples.py`)  *(per `engine_v1/MIGRATION_NOTES.md`)*
- [ ] `src/gaira/foundation/families_raman.py` (family rules) · `PERTURBATION_THEME` dict

**Curated knowledge (version-controlled)**
- [ ] `src/gaira/engine/data/biochemical_ontology_v2.yaml` (themes, `characteristic_bands_cm`, `family_theme_affinity`)
- [ ] `src/gaira/engine/data/mss_motifs_v1.yaml` (motif bands/exemplars/parent_theme)
- [ ] `raman_knowledge_core/peak_assignments.csv` *(self-authored; move into repo + cite)*

**Configuration & parameters**
- [ ] NMF: `n_components=24, init=nndsvda, solver=cd, beta_loss=frobenius, max_iter=1500, random_state=0`
- [ ] Benchmark: `ks=(4,8,12,16,24,32)`, GroupKFold(4), seed 0, selection weights + non-negativity tie-break
- [ ] Theme-weight mixing `{loading 0.50, spectral 0.25, perturbation 0.25}`, band tol 20 cm⁻¹
- [ ] MSS derivation `{band 0.40, exemplar 0.35, theme 0.25}`, keep ≥0.15, top-6
- [ ] **Environment lock (MISSING — add):** `scikit-learn==1.8.0`, numpy, scipy, BLAS

**Frozen output assets**
- [ ] `assets/foundation/`: manifold.json, manifold_components.npz, component_registry_v1.json, component_theme_weights_v1.json, biochemical_ontology_v2.yaml, mss_motifs_v1.yaml, reference_normalization_v1.json, reference_support.npz, MANIFEST.json (fingerprint `09ed804a40836f4a05a91ba10900cded`)

**Verify**
- [ ] Rebuilt `manifold_components.npz` SHA-256 == `09ed804a…`
- [ ] `GAIRAEngine()` loads + fingerprint assertion passes
- [ ] Both Streamlit apps render from the clone with no SSD
