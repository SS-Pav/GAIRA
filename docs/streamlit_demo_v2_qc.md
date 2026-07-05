# GAIRA polished Streamlit demo v2 — QC sweep

**Date:** 2026-04-20
**App:** `streamlit_apps/gaira_demo_v2/gaira_demo_v2.py`
**v1 status:** **UNTOUCHED.** Data, helpers, app code at `streamlit_apps/gaira_demo/` are unchanged.
**Run harness:** `streamlit.testing.v1.AppTest` (headless) + live HTTP health probe.

## Scope of this pass

Targeted visual readability pass for a dark page background:

1. Regression tab — bright axis titles, tick labels, legends, annotations, polar labels.
2. Grounding tab radar — bright angular axis labels, readable radial ticks, legible legend.
3. Physics atlas plot — higher-contrast bands, brighter labels, classification legend.
4. Global Plotly polish — centralized theme, consistent font sizes, visible grid.

No scientific logic was changed. Data is reused verbatim from v1's
`streamlit_apps/gaira_demo/data/` folder (confirmed by shared loader paths).

## QC checklist

| # | Check | Status | Evidence |
|---|---|---|---|
| 1 | Regression tab — all plot labels readable on dark bg | PASS | `apply_dark_theme()` sets `font.color=#F1F5F9`, `title.font.color=#F8FAFC`; dose-curve y2 axis styled manually (color=#F1F5F9 title, #CBD5E1 ticks); slider vline annotation uses `#60A5FA` on dark bg |
| 2 | Regression radar labels readable | PASS | `apply_polar_dark()` sets angular tickfont `color=#F1F5F9, size=13`; radial tickfont `color=#94A3B8, size=10` |
| 3 | Grounding radar labels readable | PASS | same central `apply_polar_dark()` call; molecule legend box uses `bgcolor=rgba(17,24,39,0.75)`, `bordercolor=rgba(148,163,184,0.25)` |
| 4 | Atlas plot brightness improved | PASS | New `atlas_ruler_figure()` helper with per-classification alpha (anchor 0.95, secondary 0.65, ambiguous 0.38) + visible white borders (opacity 0.75/0.45/0.22), central markers `#F8FAFC` on dark navy inset, brighter axis palette (tailwind-400), explicit legend entries for classification |
| 5 | All major plots readable at laptop size | PASS | Default heights 320–480 px, margins 28–62 px, tick fonts 10–13 px, widths stretch to container |
| 6 | No broken interactivity | PASS | AppTest drives every selectbox / multiselect / radio / select_slider / checkbox without raising |
| 7 | No regressions in functionality | PASS | All 4 tabs render the same information; only style + layout helpers changed |
| 8 | v1 app left untouched | PASS | `streamlit_apps/gaira_demo/{gaira_demo.py,helpers.py,data/}` unchanged; v2 reads v1 data via relative path |
| 9 | Live boot returns HTTP 200 | PASS | `curl http://127.0.0.1:8601/_stcore/health` → `ok`, HTTP 200 |
| 10 | Empty-state / edge cases | PASS | Tab 2 empty selection and 1-mol selection both run without error |

## Styling helpers added (central)

In `streamlit_apps/gaira_demo_v2/helpers.py`:

| Helper | Purpose |
|---|---|
| `apply_dark_theme(fig, title, height, show_legend, margin)` | Single-pass styler for every figure: dark paper/plot bg, bright font, styled cartesian axes, styled legend, styled hoverlabel. |
| `apply_polar_dark(fig, radial_max)` | Forces angular tickfont white (#F1F5F9), radial tickfont readable (#94A3B8), bright grid lines on dark polar bg. |
| `atlas_ruler_figure(view, axes_unique, height)` | Dedicated atlas ruler builder with per-classification opacity, visible borders, central markers, alternating row stripes, classification legend. |
| `radar_figure`, `bsv_bar_figure`, `spectra_overlay_figure`, `delta_heatmap_figure` | All rebuilt to call `apply_dark_theme()` (and `apply_polar_dark()` where polar). |

## Palette

- Page bg: `#0B1220` (deep navy) with radial gradient to `#101B31` at top
- Panel bg: `#111827`
- Primary text: `#F1F5F9`
- Secondary text: `#CBD5E1`
- Muted text: `#94A3B8`
- Title text: `#F8FAFC`
- Grid: `rgba(148,163,184,0.18)`
- Axis line: `#64748B`

**Axis palette (tailwind-400s, tuned for dark bg):**
- Lipid `#60A5FA` · Protein `#FBBF24` · Aromatic AA `#34D399`
- Purine `#F87171` · Pyrimidine `#22D3EE` · Glycan `#C084FC`
- Redox `#FDE68A` · Nuc.Backbone `#F472B6`

**Diverging scale for ΔBSV heatmap:** red-400 → red-900 → deep navy → emerald-900 → emerald-400 (symmetric around zero).

## Interaction coverage (AppTest)

| Tab | Widget | Interactions exercised |
|---|---|---|
| 1 | `atlas_axis` | default + `membrane_lipid` |
| 1 | `atlas_class` | default + `ambiguous` |
| 1 | companion checkbox | toggled |
| 2 | `grd_molecules` | default + 4-mol + empty + 1-mol |
| 3 | `cal_contrasts` | default + 3-contrast |
| 3 | `cal_view_mode` | bar + heatmap + radar |
| 4 | `reg_primary_axis` | redox_metabolite + membrane_lipid |
| 4 | `reg_slider` | 0.0 µM → mid (1.0 µM) → 2.0 µM |

All runs returned `no exceptions`.

## Files

- `streamlit_apps/gaira_demo_v2/gaira_demo_v2.py` — main app (new)
- `streamlit_apps/gaira_demo_v2/helpers.py` — dark-theme helpers (new)
- `streamlit_apps/gaira_demo_v2/.streamlit/config.toml` — theme config (new, scoped to this folder; also set via CLI flags)
- `docs/streamlit_demo_v2_qc.md` — this file
- `docs/streamlit_demo_v2_run_instructions.md` — run guide

## Result

**QC PASS.** Regression tab labels are fixed, grounding radar labels are fixed,
atlas plot brightness is improved, every major plot is readable at laptop size,
interactivity is intact, and the v1 app is untouched.
