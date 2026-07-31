# GAIRA V5 Foundation — Reproduction Checklist

Concise, operational. Full detail in `README.md`; authoritative results in
`VERIFICATION_REPORT.md`.

## Before running
- [ ] Clone the correct branch: `git checkout gaira-v5-rebuild-plan`.
- [ ] Create a clean environment (venv/conda).
- [ ] Install exact dependencies: `pip install -r requirements.txt` (pin
      `scikit-learn==1.8.0` from `requirements_vm.txt` for byte-identical NMF).
- [ ] Verify raw datasets exist (full mode only): `serum_ag_colloids/dataset_spectral_data.zip`,
      `amino_acid_raman_grounding/aa.xlsx`, RamanBioLib parquet.
- [ ] Set the data root: `export GAIRA_DATA_ROOT=/path/to/GAIRA_DATA/raw`.
- [ ] Confirm `--output-dir` is **not** inside `assets/foundation/` (the script refuses this).

## Full rebuild
```bash
python tools/reproduce_gaira_foundation.py --mode full \
    --data-root "$GAIRA_DATA_ROOT" --output-dir results/reproduction/my_run
```
- [ ] Corpus counts: **375 spectra / 167 analytes** (sources 202 / 153 / 20).
- [ ] Preprocessing matrix shape: **375 × 676**; grid step 2 cm⁻¹, 676 bins.
- [ ] NMF: explained variance ≈ **0.712**, reconstruction rel. error ≈ **0.423**.
- [ ] Basis fingerprint: `basis_comparison.json` → `fingerprint_match: true`,
      `exact_array_equality: true`, `max_abs_diff: 0.0`, rebuilt fingerprint
      **`09ed804a40836f4a05a91ba10900cded`**.
- [ ] Component alignment: identity permutation when byte-identical (else use Hungarian).
- [ ] Downstream equality (`downstream_comparison.json`): theme weights **identical**, MSS
      **identical**, registry **numerically identical**, reference center/spread **identical**.

## Interpretation-only
```bash
python tools/reproduce_gaira_foundation.py --mode interpretation-only \
    --foundation-root assets/foundation --output-dir results/reproduction/interp
```
- [ ] Runs with **no raw data / no SSD** (uses committed frozen assets + coordinates).
- [ ] Theme weights, MSS, BSV fixtures **identical** to canonical; registry numerically
      identical; reference normalization rebuilt from committed coordinates.

## Final sign-off
- [ ] Tests pass: `python -m pytest tests/test_reproduce_foundation.py -q` → **13 passed**.
- [ ] Canonical assets unchanged: combined `assets/foundation/*` hash `4606ce98ba752d19`
      before == after (per-file hashes in `manifests/canonical_foundation_files.json`).
- [ ] Report saved: `VERIFICATION_REPORT.md`.
- [ ] Commit hash recorded: __________________.
