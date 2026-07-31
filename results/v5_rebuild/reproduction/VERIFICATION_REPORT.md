# GAIRA V5 Foundation — Verification Report

Authoritative verification that `tools/reproduce_gaira_foundation.py` regenerates the
GAIRA foundation exactly as documented. Every number below is from an actual run on the
reference machine; nothing here is modified in `assets/foundation/`.

## Environment

| | |
|---|---|
| OS | macOS-26.5.1-arm64 (Darwin, aarch64) |
| Python | 3.12.7 |
| NumPy | 2.4.3 |
| SciPy | 1.17.1 |
| scikit-learn | **1.8.0** |
| BLAS/compiler | clang 15.0.0 / aarch64 (see `tmp_full_rebuild/reproduction_run/environment.json`) |

Byte-identical NMF requires this stack (see Known caveats).

## Commands executed

```bash
# full raw rebuild (into a throwaway dir; never overwrites canonical)
python tools/reproduce_gaira_foundation.py --mode full \
    --data-root "$GAIRA_DATA_ROOT" \
    --output-dir results/v5_rebuild/reproduction/tmp_full_rebuild
# interpretation-only (no raw, no SSD)
python tools/reproduce_gaira_foundation.py --mode interpretation-only \
    --foundation-root assets/foundation \
    --output-dir results/v5_rebuild/reproduction/verification/interp_run
python -m pytest tests/test_reproduce_foundation.py -q
```

## Runtime
- full rebuild: **11.4 s** · interpretation-only: **2.8 s**

## Dataset counts (verified)
- **375** Raman spectra · **167** analyte labels (~161 distinct molecules) · **676** bins
- sources: RamanBioLib **202**, gobbato_raman_metabolites **153**, amino_acid_raman_grounding **20**
- SERS spectra used for NMF training: **0** (Raman-only representation; `assert modality=="raman"`)

## Matrix dimensions
- `X_ref`: 375 × 676 · `H_basis`: 24 × 676 · `A_ref`: 375 × 24 · reference coordinates `Z`: 375 × 24

## NMF convergence / reconstruction
- explained variance **0.7120** · reconstruction relative error **0.4231**
- `n_components=24, init=nndsvda, solver=cd, beta_loss=frobenius, alpha_W=0.0, alpha_H="same", l1_ratio=0.0, max_iter=1500, tol=1e-4, random_state=0, shuffle=False`

## Basis equality (full mode)
| metric | value |
|---|---|
| exact array equality | **true** |
| max absolute difference | **0.0** |
| grid match | **true** |
| rebuilt fingerprint | **`09ed804a40836f4a05a91ba10900cded`** |
| canonical fingerprint | `09ed804a40836f4a05a91ba10900cded` |
| Hungarian alignment | not required (basis byte-identical → identity permutation) |

## Downstream equality (both modes)
| artifact | verdict |
|---|---|
| `component_theme_weights_v1.json` | **identical** |
| `mss_registry_v1.json` | **identical** |
| `component_registry_v1.json` | numeric content **identical** (cosmetic `(up/down)` set-order text is `PYTHONHASHSEED`-dependent) |
| `reference_normalization_v1.json` | center/spread **identical** (only the `note` metadata string differs) |
| BSV regression fixtures | **identical** (deterministic across runs and across modes) |

## Frozen-asset integrity
- combined SHA-256 of all `assets/foundation/*` files, **before** verification: `4606ce98ba752d19`
- combined SHA-256 **after** verification: `4606ce98ba752d19`  → **UNCHANGED**
- per-file canonical hashes: `manifests/canonical_foundation_files.json` (10 files)

## Tests
```
tests/test_reproduce_foundation.py .............  13 passed, 0 failed, 0 skipped
```
(includes the optional full-raw integration test, which ran because the data-root was
available on the reference machine; it is skipped automatically when no raw data is present.)

## Remaining caveats
1. **Numerical-environment sensitivity.** The seed alone does not guarantee byte identity;
   coordinate-descent NMF can differ across sklearn/BLAS versions. Pin
   `scikit-learn==1.8.0`. A different stack may give equivalent but reordered components —
   transfer index-keyed annotations only after Hungarian alignment.
2. **Registry cosmetic nondeterminism.** `build_registry.py` joins a `set` of direction
   labels for `current_interpretation` (`PYTHONHASHSEED`-dependent). Numeric content is
   fully reproducible; only that text's token order can flip. (A one-line `sorted()` would
   make it byte-stable; not changed in this documentation pass.)
3. **RamanBioLib parquet** is not committed (git-ignored); full mode needs it in-repo or
   under the data-root.
