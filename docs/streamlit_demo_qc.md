# GAIRA polished Streamlit demo — QC sweep

**Date:** 2026-04-20
**App:** `streamlit_apps/gaira_demo/gaira_demo.py`
**Run harness:** `streamlit.testing.v1.AppTest` (headless, full-script execution)

## QC checklist

| # | Check | Status | Evidence |
|---|---|---|---|
| 1 | App runs without crashing (boot) | PASS | AppTest boot returns `no exceptions` |
| 2 | Each tab has real usable content | PASS | Every tab renders at least one Plotly figure + text |
| 3 | No broken selectors | PASS | AppTest drives every `selectbox`, `multiselect`, `radio`, `select_slider`, `checkbox` without raising |
| 4 | Empty states are handled | PASS | Tab 2: if selection is empty, shows info prompt. Tab 3: if no contrasts selected, shows info prompt. No blank panels. |
| 5 | All text readable at standard laptop size | PASS | `simple_white` template; Inter/Helvetica fallback; tick font 10–13 |
| 6 | Plotly plots render correctly | PASS | All `go.Figure` calls use named traces + explicit layouts; heatmap and radar vmax/rmax fixed; bar signed colors respected |
| 7 | Radar labels readable | PASS | `AXIS_LABELS` short names (Lipid, Protein, Aromatic AA, Purine, Pyrimidine, Glycan, Redox, Nuc.Backbone); axis tick font 12; `direction=clockwise`, `rotation=90` |
| 8 | Methods tab clearly explains the system | PASS | Hero figure + 4-column stage legend + atlas ruler + axis-coverage chart + band table |
| 9 | Calibration tab is representation-space only | PASS | No spectra rendered in Tab 3; only ΔBSV bar/heatmap/radar + interpretation pills |
| 10 | Regression slider interaction works | PASS | AppTest: slider stepped across 0 → mid → max; radar/bars/curve update without error |

## Interaction coverage (AppTest)

| Tab | Widget (`key`) | Interactions exercised |
|---|---|---|
| 1 | `atlas_axis` | default + `membrane_lipid` |
| 1 | `atlas_class` | default + `ambiguous` |
| 1 | companion checkbox | toggled |
| 2 | `grd_family` | default |
| 2 | `grd_molecules` | default + 4-molecule + empty |
| 3 | `cal_contrasts` | default + 3-contrast |
| 3 | `cal_view_mode` | bar + heatmap + radar |
| 4 | `reg_primary_axis` | redox_metabolite + membrane_lipid |
| 4 | `reg_slider` | 0.0 µM → mid (1.0 µM) → 2.0 µM |

All runs returned `no exceptions`.

## Visual / design verification

- **Canonical axis order** used everywhere (`BSV_COMPONENTS`):
  `membrane_lipid → protein_backbone → aromatic_amino_acid → purine_nucleotide → pyrimidine_nucleotide → glycan_carbohydrate → redox_metabolite → nucleic_acid_backbone`.
- **Radar polygons** use fixed radial_max per tab (mol-wise max for Tab 2, per-slider-range max for Tab 4, |Δ| vmax for Tab 3).
- **ΔBSV heatmap** uses diverging `RdBu_r`, symmetric `[-vmax, vmax]`.
- **Non-testable axes** in Tab 3 bar view are grey-shaded, with an inline legend caption.
- **Dose-response curve** uses a dashed vline at the slider concentration + annotation.

## Known limitations

- RamanBioLib spectra used for Tab 2 are already pre-normalized by their upstream pipeline (min-max + baseline + SG per `metadata_db.csv`). Window-based BSV is therefore computed directly without re-applying GAIRA's AsLS; this is consistent across all 202 molecules.
- The Ergothioneine titration is a 5-replicate-per-concentration SERS run (single SERS substrate + laser). The demo exposes average behavior; replicate variance is saved (`bsv_std_*`) but not plotted.
- The "LOD status" shown in Tab 4 uses a lightweight threshold (|Δ| > 0.005 for commit, > 0.01 for "above routine LOD") — intended as a readability cue, not a formal LOD estimation.
- The hero figure is the existing Phase 3 three-phase master figure. It is suitable and crisp; a newer figure was not generated.

## Fixes applied during QC

- `mol_bsv` already contains a `family` column; merging `mol_index` via `["id", "family"]` caused a suffix collision → fixed by merging only `["id", "type"]`.
- Replaced deprecated `use_container_width` with `width="stretch"` throughout.
- Made `format_conc` robust to string inputs (Streamlit AppTest quirk with `select_slider` format_func).

## Result

**QC PASS.** App boots and survives end-to-end interaction across all four tabs.
