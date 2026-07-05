# GAIRA demo v3 — QC sweep

**Date:** 2026-04-20
**App:** `streamlit_apps/gaira_demo_v3/gaira_demo_v3.py`
**Previous versions:** `gaira_demo` (v1) and `gaira_demo_v2` — **untouched.**
**Harness:** `streamlit.testing.v1.AppTest` + subprocess isolation + live HTTP boot.

## Required QC checks

| # | Check | Status | Evidence |
|---|---|---|---|
| 1 | Methods tab cleaned up successfully | PASS | Removed hero image, stages cards, BSV-pipeline / BSV-vs-peak-matching expanders. Replaced with: concise intro box; Plotly-native `pipeline_diagram_figure()`; 3-card layered grounding overview; family + source-kind panels; honest atlas coverage note; atlas ruler explorer. |
| 2a | Grounding overview includes overall grounding layer | PASS | `grounding_layer_summary.csv` row `layer=atlas` ·  64 bands · 8 axes · 450–1800 cm⁻¹ |
| 2b | Pure-molecule layer included | PASS | Layer 1 card · 202 spectra · 202 unique molecules · 9 families · RamanBioLib + amino-acid grounding dataset named |
| 2c | Literature-linked layer included | PASS | Layer 2 card · 23 unique sources · 18 papers · 5 core refs · source-kind bar chart |
| 2d | Atlas layer included | PASS | Layer 3 card + atlas explorer + honest coverage caveat |
| 3 | Grounding family filter no longer crashes | PASS | AppTest rotates through Lipids → Proteins → NucleicAcids → AminoAcids → Saccharides → (all families); every rendered multiselect's `value` is a subset of its `options`. Family-scoped widget key (`grd_molecules_v3__{family}`) means a new widget is instantiated per family with family-appropriate seed defaults. |
| 4a | Multiple molecules render correct spectra | PASS | One trace per unique component; `dict.fromkeys(...)` dedupes the selection list; each trace drawn from `grounding_molecule_spectra.parquet` joined to the single `id` picked per component (`first_by_component`). |
| 4b | Separate BSV bar plots per molecule | PASS | New behavior: one `bsv_bar_figure()` per selected molecule, stacked vertically. Amalgamated-average bar removed. |
| 4c | Radar: one trace per molecule, no duplicates | PASS | Iterates `selected_unique`; traces list has unique names. |
| 5 | Calibration uses human-readable labels | PASS | `load_calibration_metadata()` supplies `rich_label`; multiselect `format_func` uses rich label; bar titles, heatmap rows, radar legend all use rich label. |
| 6 | Radar clearly shows baseline vs treated | PASS | Radar title reads "\|ΔBSV\| magnitude · perturbed vs baseline"; bar sub-header shows `(baseline_label → perturbed_label)`; condition-metadata block makes it explicit. |
| 7 | Regression tab only exposes supported ordered series | PASS | `regression_registry.csv` filtered to `supported==True`; `reg_dataset_v3` has exactly one option (`ergothioneine_titration`). Unsupported reasons shown in an expander below. |
| 8 | Uricase included only if justified | PASS | Uricase is **not** in the regression tab. Registry reason: "Endpoint comparison, not an ordered series." Uricase remains in calibration. |
| 9 | Previous demo versions untouched | PASS | Subprocess-isolated boot of v1, v2, v3 all return OK. No files under `streamlit_apps/gaira_demo/` or `streamlit_apps/gaira_demo_v2/` modified. |

## Additional interaction coverage

- Tab 1: atlas axis filter + class filter + companion-only toggle.
- Tab 2: family → molecule rotation (6 families); empty selection; 4-molecule per-bar stacking.
- Tab 3: bar + heatmap + radar views, 4-contrast overlay.
- Tab 4: primary-axis switch, slider steps (0.0 → 1.0 → 2.0 µM).
- Live HTTP boot on port 8602 returns HTTP 200.

## Seed defaults observed per family (auto-generated, no crash)

| Family | Seed defaults |
|---|---|
| Lipids | 12-methyltetradecanoic acid, 13-methylmyristicacid |
| Proteins | lactalbumin, albumin |
| NucleicAcids | adenine, cytosine |
| AminoAcids | glycine, l-alanine |
| Saccharides | amylopectin, amylose |
| (all families) | First two components in corpus |

## Files

- `streamlit_apps/gaira_demo_v3/gaira_demo_v3.py` — main app (new)
- `streamlit_apps/gaira_demo_v3/helpers.py` — dark-theme helpers + pipeline diagram + molecule-aware shading helpers (new)
- `streamlit_apps/gaira_demo_v3/build_v3_assets.py` — derivation script for v3 tables (new)
- `streamlit_apps/gaira_demo_v3/.streamlit/config.toml` — dark theme defaults (new)
- `streamlit_apps/gaira_demo_v3/data/` — 4 derived tables (new)
- `docs/gaira_demo_v3_data_audit.md`, `docs/gaira_demo_v3_qc.md`, `docs/gaira_demo_v3_run_instructions.md`

## Result

**QC PASS.** Selector hygiene is fixed (family filter never crashes; residuals
never leak); evidence-layer clarity is explicit (three named layers with
honest metrics and a coverage caveat); calibration tab uses rich condition
labels with explicit baseline/perturbed framing; regression tab exposes only
the single truly-ordered series that is wired through the GAIRA pipeline.
