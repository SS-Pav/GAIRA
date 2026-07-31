# Interpretation-only verification

Verifies that the frozen foundation's deterministic interpretation layers can be
regenerated **without any raw Raman data and without SSD_Rad**, from the committed
`assets/foundation/` bundle + the committed reference-coordinate artifact.

## Command

```bash
python tools/reproduce_gaira_foundation.py \
    --mode interpretation-only \
    --foundation-root assets/foundation \
    --output-dir results/v5_rebuild/reproduction/verification/interp_run
```

- **No** `--data-root`, **no** SSD, **no** NMF refit.
- Reference normalization is rebuilt from the committed
  `results/v5_rebuild/reproduction/manifests/nmf_reference_coordinates.npz` (auto-detected;
  no raw corpus needed).

## Result

- **Runtime:** ~2.8 s.
- **Canonical assets unmodified:** `True`.
- **Regenerated files:** `component_registry_v1.json`, `component_theme_weights_v1.json`,
  `mss_registry_v1.json`, `reference_normalization_v1.json`, `reference_support.npz`,
  `bsv_regression_results.json`, `run_manifest.json`, `environment.json`,
  `downstream_comparison.json`, `reproduction_report.md`.

### Byte comparison vs canonical committed outputs (`results/v5_rebuild/engine_v1/artifacts/`)

| Artifact | Verdict |
|---|---|
| `component_theme_weights_v1.json` | **identical** |
| `mss_registry_v1.json` | **identical** |
| `component_registry_v1.json` | **identical** (numeric content always; the `current_interpretation` set-ordered text is `PYTHONHASHSEED`-dependent and matched byte-for-byte on this run) |
| `reference_normalization_v1.json` | **identical** numeric center/spread; only the `note` metadata string differs (rebuilt-from-coordinates provenance line) |

## Remaining caveats

- The `component_registry_v1.json` free-text `current_interpretation` joins a Python `set`
  of direction labels (`(up/down)`), so its order depends on `PYTHONHASHSEED`. Every numeric
  field is fully reproducible; only that string's token order can differ between processes.
- `reference_normalization_v1.json` rebuilt from the committed coordinates carries a
  provenance `note` distinct from the canonical script's `note`; the numeric frame
  (center/spread/support) is identical.
