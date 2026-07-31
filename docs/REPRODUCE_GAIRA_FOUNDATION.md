# Reproducing the GAIRA Foundation Model

How to regenerate the complete GAIRA foundation — the learned NMF basis and every
derived interpretation layer — on **another computer**, with or without the raw datasets,
without assuming `/Volumes/SSD_Rad`.

One orchestration script drives everything:
**`tools/reproduce_gaira_foundation.py`**. It *orchestrates the existing canonical
functions* (it re-implements no scientific logic) and writes only to your `--output-dir`;
it never modifies `assets/foundation/` or any canonical artifact.

---

## The conceptual model (keep this fixed)

```
LEARNED    Raman reference spectra ──NMF──► 24 basis spectra H  (the only learned object)
DERIVED    H + reference evidence  ──────► component registry, component→theme weights T,
                                            MSS registry, reference normalization, BSV
CURATED    motif definitions, theme definitions, characteristic bands, family→theme
           affinity, perturbation maps                 (hand-authored, version-controlled)
RUNTIME    new spectrum → NNLS coordinates z → PARALLEL:  BSV (b = Tᵀz)  and  MSS activations
```

- **What is learned:** the 24 NMF basis spectra `H` (fingerprint `09ed804a…`). Nothing else.
- **What is derived deterministically:** component registry, component→theme weights, MSS
  contributors, reference normalization, BSV.
- **What is curated:** biochemical theme definitions, MSS motif names/bands/exemplars, the
  `family_theme_affinity` map, the `PERTURBATION_THEME` map.
- **Datasets by role:** representation is Raman-only; SERS is external validation **plus**
  perturbation evidence to the interpretation layer — it never fits the NMF or the
  normalization frame. Full table: `results/reproduction_audit/dataset_role_map.csv`.

---

## Two modes

| | Mode `full` | Mode `interpretation-only` |
|---|---|---|
| Needs raw datasets? | **yes** (Gobbato zip + amino-acid xlsx + RamanBioLib parquet) | **no** |
| Needs SSD? | no — pass any `--data-root` | **no** |
| Rebuilds the NMF basis? | yes (and verifies the fingerprint) | no (uses the committed basis) |
| Rebuilds registry / theme-weights / MSS? | yes | yes |
| Rebuilds reference normalization? | yes | only if you pass `--reference-coords` |

---

## Run it on another computer

### Full raw rebuild (verifies the fingerprint)

```bash
export GAIRA_DATA_ROOT=/path/to/GAIRA_DATA/raw     # or pass --data-root
python tools/reproduce_gaira_foundation.py \
    --mode full \
    --data-root "$GAIRA_DATA_ROOT" \
    --output-dir results/reproduction/my_run
```

Data-root precedence: `--data-root` → `$GAIRA_DATA_ROOT` → a documented optional default →
clear error. The raw layout expected under the data-root:
`serum_ag_colloids/dataset_spectral_data.zip`, `amino_acid_raman_grounding/aa.xlsx`, and the
RamanBioLib parquet (in-repo at `streamlit_apps/gaira_demo/data/grounding_molecule_spectra.parquet`,
or pass `--ramanbiolib-parquet`).

### Rebuild without any raw data

```bash
python tools/reproduce_gaira_foundation.py \
    --mode interpretation-only \
    --foundation-root assets/foundation \
    --output-dir results/reproduction/interp
```

To also rebuild reference normalization without raw data, feed it the reference
coordinates saved by a previous `full` run (a ~43 KB file):

```bash
python tools/reproduce_gaira_foundation.py --mode interpretation-only \
    --foundation-root assets/foundation \
    --reference-coords results/reproduction/my_run/reproduction_run/nmf_reference_coordinates.npz \
    --output-dir results/reproduction/interp
```

---

## How to verify success

The run writes `reproduction_run/` with `basis_comparison.json`, `downstream_comparison.json`,
`component_alignment.csv`, `bsv_regression_results.json`, `run_manifest.json`, and
`reproduction_report.md`. Check:

1. **Basis fingerprint** — `basis_comparison.json → fingerprint_match: true` and
   `exact_array_equality: true` (max abs diff 0.0). *This was verified on the reference
   machine: the basis rebuilds byte-for-byte.*
2. **Component alignment** — `component_alignment.csv`; when byte-identical it is the identity
   permutation. **If the basis is NOT byte-identical (different numerical stack), do NOT
   compare component *i* to component *i*.** The script computes a Hungarian (max-cosine)
   permutation first; transfer any integer-index annotation only through that alignment.
3. **Downstream equality** — `downstream_comparison.json`:
   - `component_theme_weights_v1.json`: **identical**
   - `mss_registry_v1.json`: **identical**
   - `component_registry_v1.json`: **numeric content identical** (one cosmetic caveat below)
   - `reference_normalization_v1.json`: identical center/spread (only the `note` differs)
4. **BSV regression** — `bsv_regression_results.json` is deterministic across runs/modes
   (same fingerprint → same biochemical state on the fixed coordinate fixtures).

Run the tests (no raw data needed):
```bash
python -m pytest tests/test_reproduce_foundation.py -q
```

---

## Known limitations

- **Exact NMF byte-identity requires the same numerical stack.** The reference build used
  Python 3.12 · NumPy 2.4 · SciPy 1.17 · **scikit-learn 1.8.0** (see `environment.json`).
  Coordinate-descent NMF numerics can change across sklearn/BLAS versions. A changed stack
  may yield an *equivalent but reordered* basis; because the registry and theme-weights are
  keyed by **integer component index**, downstream annotations must only be transferred
  after the explicit Hungarian component alignment above. Pin `scikit-learn==1.8.0`
  (`requirements_vm.txt`) to reproduce the fingerprint.
- **`component_registry_v1.json` has one cosmetic nondeterminism:** the free-text
  `current_interpretation` joins a *set* of direction labels (`(up/down)`), whose order
  depends on `PYTHONHASHSEED`. Every numeric field is fully reproducible; only the text
  order of `up`/`down` can flip. (A one-line `sorted(...)` in `build_registry.py` would make
  it byte-stable; not changed in this pass.)
- **Reference normalization needs the projected reference coordinates.** In `full` mode the
  script saves them as `nmf_reference_coordinates.npz` (~43 KB). Committing that one small
  file would make `interpretation-only` able to rebuild reference normalization with no raw
  data at all — the only downstream artifact currently missing from a raw-free rebuild.
- **The RamanBioLib parquet is not committed** (it's git-ignored). `full` mode needs it
  in-repo or under the data-root.
