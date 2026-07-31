# Reproduction report — mode: full

- runtime: 11.4 s
- canonical asset unmodified: **True**
- environment: m['environment']['scikit_learn']='1.8.0' · numpy 2.4.3 · py 3.12.7

- basis fingerprint match: **True** (exact equality True, max abs diff 0.0)

## Downstream vs canonical

- `component_registry_v1.json`: **identical except cosmetic interpretation-text ordering (PYTHONHASHSEED-dependent set join in build_registry.py; numeric content identical)**
- `component_theme_weights_v1.json`: **identical**
- `mss_registry_v1.json`: **identical**
- `reference_normalization_v1.json`: **identical (numeric center/spread); only the 'note' metadata string differs**

## Output files

- `basis_comparison.json` (705 bytes)
- `bsv_regression_results.json` (2968 bytes)
- `component_alignment.csv` (406 bytes)
- `component_registry_v1.json` (144958 bytes)
- `component_theme_weights_v1.json` (45381 bytes)
- `corpus_manifest.csv` (39934 bytes)
- `dataset_role_map.csv` (2869 bytes)
- `downstream_comparison.json` (380 bytes)
- `environment.json` (596 bytes)
- `mss_registry_v1.json` (38616 bytes)
- `nmf_basis.npz` (69359 bytes)
- `nmf_metrics.json` (830 bytes)
- `nmf_reference_coordinates.npz` (44051 bytes)
- `nmf_training_activations.npz` (23848 bytes)
- `preprocessed_reference_matrix.npz` (942045 bytes)
- `preprocessing_config.json` (348 bytes)
- `rebuilt_foundation` (128 bytes)
- `reference_normalization_v1.json` (1077 bytes)
- `reference_support.npz` (23870 bytes)
- `run_manifest.json` (2127 bytes)
