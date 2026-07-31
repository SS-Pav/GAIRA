# NMF Foundation + Interpretation Build Trace

*Parts 2–6 of the reproduction audit. The exact wired route from raw Raman datasets to the
frozen basis and every derived interpretation layer, verified against current source.
Machine-readable params: `nmf_parameters.json`, `preprocessing_parameters.json`.*

---

## Part 2 · Raw Raman → `manifold_components.npz`

**Canonical route (functions actually called):**
```
data.loader.load_ramanbiolib()  ┐
data.gobbato.load_gobbato_785() ├─► dataset.load_reference_corpus()   [parse + canonicalize + preprocess + stack]
dataset._load_amino_acids()     ┘        │
                                         ▼  X_ref  (375 × 676, L2-normalized, clip≥0)
foundation.code.run_c1_benchmark.py ─► benchmark.run_benchmark ─► c1_selection.json (NMF, k=24)
foundation.code.run_c2_c7.py        ─► latent_space.build_manifold ─► representation.fit_nmf  [NMF.fit → H, transform → A_ref]
                                    ─► serialization.freeze_manifold ─► manifold_components.npz (+ manifold.json, fingerprint)
```
**Every step performs real computation — no step loads a cached basis.** `fit_nmf` calls
`NMF(...).fit(Xc)`; `freeze_manifold` writes `components_` (H) fresh and SHA-256-hashes it.
(The *benchmark* step re-derives k=24; a direct rebuild can skip it since k is known.)

**Input corpus** (`load_reference_corpus`): 375 spectra / 167 labels (~161 distinct
molecules; 6 canonicalization duplicate labels not merged). Exclusions: `assert modality ==
"raman"` drops all SERS; a finite-value check drops all-NaN preprocessed vectors. **Ordering:**
spectra are stacked in loader order (RamanBioLib → Gobbato → amino-acid). NMF with a fixed
NNDSVD-a init + seed is **not** invariant to row order in general, but the loader order is
deterministic, so ordering is fixed and reproducible; it is *not* independently shuffled
(`shuffle=False`).

**Preprocessing (verified from `preprocessing/pipeline.py` + `dataset.PREPROC`):** crop
450–1800 cm⁻¹ · ASLS λ=1e5, p=0.01, 8 iter · Savitzky–Golay window=9, poly=3 · resample to
a fixed **2 cm⁻¹** grid → **676 bins** · L2 · clip ≥ 0 (NMF only). Matches the spec exactly.

**NMF parameters (verified from the fitted estimator, this environment):**
`n_components=24, init="nndsvda", solver="cd", beta_loss="frobenius", alpha_W=0.0,
alpha_H="same"(→0), l1_ratio=0.0, max_iter=1500, tol=1e-4, random_state=0, shuffle=False`.
> **Correction to the expected list:** `alpha_H` is the sklearn default **`"same"`** (it
> inherits `alpha_W=0`), not the literal `0`. Effective regularization is zero either way.

**Environment (this run):** Python 3.12.7 · NumPy 2.4.3 · SciPy 1.17.1 · scikit-learn
**1.8.0** · macOS/arm64. **Byte-identical rebuilding depends on all of these** — coordinate-
descent NMF numerics can change across sklearn/BLAS versions. The seed alone does **not**
guarantee it. The manifest records **no** environment; `requirements_vm.txt` pins
`scikit-learn==1.8.0`, the primary reqs say `>=1.3`.

**Outputs.** `H` = `components_` (24 × 676) = the basis, frozen. `A_ref` = `transform(X)`
(375 × 24) = training activations — **not currently persisted** (recomputed by NNLS at
projection). `grid` (676) is stored alongside H. Fingerprint = `sha256(H.tobytes())[:32] =
09ed804a40836f4a05a91ba10900cded`. Explained variance 0.712; components stable
(0.65–0.97), non-redundant (max pairwise cosine 0.52).

---

## Part 3 · Component registry (`build_registry.py`) — DERIVED, no raw

Reads **frozen H + committed tables only** (verifies fingerprint, refuses on mismatch):
`reference_atlas_audit/tables/{p1_component_inventory, p2_full_analyte_composition,
p4_chemical_coherence}` + `perturbation_response/tables/{part1_component_dose_response,
part2_response_fingerprints, part8_uricase.json, part11_component_robustness}` + `AA.component_bands(H, grid)`.

| Field | Source | Derived? | Needs raw? |
|---|---|---|---|
| dominant_raman_peaks_cm | `find_peaks` on basis loading (`atlas_audit.component_bands`) | ✅ from H | frozen only |
| reference_analyte_loadings / top_families | Component Audit p2 (analyte→component composition) | ✅ | committed table |
| purity | Component Audit p4 | ✅ | committed table |
| bootstrap_stability | Component Audit p1 | ✅ | committed table |
| dose_response / serum_spike / depletion | Perturbation Response p1/p2/p8 (**SERS-derived**) | ✅ | committed table |
| current_interpretation / confidence / caveats | rubric in `_interpret()` (stability × purity × perturbation) | ✅ deterministic | — |

**Regenerable from `frozen H + committed tables` — does NOT need the raw corpus or A_ref.**
(The committed tables are keyed by **integer component index**, so they are valid only for
the canonical basis ordering.)

---

## Part 4 · MSS (`engine/mss.py`, `mss_motifs_v1.yaml`, `tools/build_mss_registry.py`) — DERIVED

Curated motif def (bands, exemplars, `parent_theme`) + frozen component evidence. Verified
coefficients from `mss.py`: for each component *j*,
`score = 0.40·band + 0.35·exemplar + 0.25·theme` → keep `score ≥ 0.15` → top **6** →
normalize. `confidence = weighted stability × evidence_breadth`; perturbation from registry.
- MSS are **not** learned by NMF; MSS do **not** create components; MSS do **not** compute
  the BSV; `bsv.py` never imports MSS.
- MSS **consumes the theme weight** (25 % of the score) → MSS is **downstream of / parallel
  to** themes, never their source.
- Inputs to reproduce identical MSS: `component_registry_v1.json` + `component_theme_weights_v1.json`
  + `mss_motifs_v1.yaml`. **No raw.** Pure function → byte-reproducible.

---

## Part 5 · Themes + BSV (`build_theme_weights.py`, `biochemical_ontology_v2.yaml`, `bsv.py`)

**Component→theme matrix T** (verified from `build_theme_weights.py`), per component:
`w_t = 0.50·loading(families→theme via family_theme_affinity) + 0.25·spectral(band overlap
±20 cm⁻¹) + 0.25·perturbation(driving analytes → PERTURBATION_THEME)`; residual mass →
`background_matrix` (generic components) or `unknown_mixed`; drop < 0.02; renormalize so each
component's weights sum to 1. Every weight stores its 3 evidence lines + confidence.
- Reads **committed tables + ontology yaml** — **no raw.** The `PERTURBATION_THEME` map is a
  hard-coded dict in the script (knowledge-in-code).

**BSV** (`bsv.py`, unambiguous names):
```
X_ref   reference spectral matrix (375×676)
A_ref   NMF training activations (375×24)          H_basis  NMF basis (24×676)
z_query query component coordinates (24, L1 share) T        component→theme matrix (24×13)
b_query biochemical state vector                   b_query = Tᵀ · z_query        (composition)
                                                   elevation = Tᵀ · z_scored(z_query)
```
`composition = T.T @ coord`, `elevation = T.T @ z`, `display = 0.5+0.5·tanh(elev/3)`,
`confidence = stability · evidence · (1−OOD)`. Themes/family-affinities/perturbation-maps/
residual handling are all in `biochemical_ontology_v2.yaml` + the build script. BSV needs
only **committed assets** (T + reference frame). Δ-BSV = composition(query) − composition(baseline).

---

## Part 6 · Reference normalization (`build_reference_norm.py`) — the one raw-dependent derived layer

```
Z = atlas.coordinates(corpus.X)            # NNLS-project the 375 Raman refs onto frozen H  → 375×24 (L1)
center = median(Z, axis=0)                 # robust center
spread = 1.4826 · MAD(Z, axis=0), floored 1e-3
support_unit = Z / ||Z||                   # OOD reference cloud
```
- **Projects the Raman reference corpus** — so as written it **requires raw data**.
- **BUT the only raw-derived input is `Z` (the 375×24 projected reference coordinates).**
  If `Z` is saved once (a ~36 KB `.npz`), `center`, `spread`, and `support_unit` are pure
  functions of it — **reference normalization can then be rebuilt with no raw data.**
  **→ Saving `A_ref`/`Z` removes the last raw dependency for the whole interpretation layer.**
  This is the single small committed asset that makes Mode B complete (Part 7).

---

## Raw-vs-cached summary (does each step recompute or load a cache?)

| Stage | Recomputes? | Needs raw corpus? |
|---|---|---|
| Preprocess + X_ref | ✅ real | **yes** |
| NMF → H, A_ref | ✅ real (`NMF.fit`) | **yes** |
| Freeze + fingerprint | ✅ real | no |
| Component registry | ✅ real (from frozen H + committed tables) | **no** |
| Component→theme weights | ✅ real | **no** |
| MSS registry | ✅ real | **no** |
| Reference normalization | ✅ real | **yes** (unless `Z` is saved) |
| Validation fixtures | ✅ real (from committed `phase3_projection_*`) | **no** |
