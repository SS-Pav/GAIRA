# GAIRA v4 — UI Cleanup

**Scope:** an incremental UI pass over v3. No scientific pipeline changes. v3 is
preserved unchanged at `streamlit_apps/gaira_v3/`; v4 lives in parallel at
`streamlit_apps/gaira_v4/`.

## Run

```bash
cd /Users/suraj/projects/GAIRA
PYTHONPATH=src streamlit run streamlit_apps/gaira_v4/gaira_v4.py
```

## Files

- `streamlit_apps/gaira_v4/gaira_v4.py` — landing page, describes the two
  sub-pages and what changed from v3
- `streamlit_apps/gaira_v4/pages/1_📝_Text_Query.py` — text query page
- `streamlit_apps/gaira_v4/pages/2_🔬_Spectral_Query.py` — spectral query page

## What changed

### Text Query

- **Per-condition trust graphs.** Comparison queries (e.g. "HCC vs healthy vs
  CCA") now render a separate trust graph per detected condition, stacked
  vertically. Uses the existing `build_per_condition_traversals()` helper — the
  pipeline is unchanged, only the rendering is split per condition.
- **Comparison summary table.** Above the stacked graphs, a small table shows
  evidence-item count, motif count, and active BSV-axis count per condition.
  Makes it visually obvious which conditions have stronger literature support.
- **Expected-BSV visualization.** Added a condition-by-axis heatmap and — when
  `healthy_control` is one of the conditions — a delta-vs-healthy heatmap. The
  radar is still rendered but the heatmaps are the clearer read for
  multi-condition comparisons.
- **Evidence labels.** Retained from v3 (human-readable source labels with
  hover explanations); grouped into the trust graph consistently across
  single-condition and multi-condition rendering.

### Spectral Query

- **Four labeled sections.** The page is now structured explicitly as
  (1) Measured Spectral Structure,
  (2) Spectral Band Drivers,
  (3) Expected Literature Comparator,
  (4) Observed vs Expected Comparison.
- **Expected-comparator trust graphs per cohort.** For each cohort's chosen
  expected comparator, v4 runs a literature-side retrieval and renders the
  trust graph for that condition. This exposes the evidence basis for each
  comparator, not just its BSV vector.
- **Clean comparator card/table.** Before any comparison plots, cohort →
  comparator pairing is shown in a small table with the comparator source
  count and similarity heuristic.
- **Demoted overlay radars.** The observed-vs-expected radar overlays are
  moved to the bottom and wrapped in an expander explicitly labeled
  "visual comparison only". They are no longer the hero figure.
- **Primary validation visuals.** Similarity matrix, cohort-to-comparator
  alignment table, and delta-shift (disease-vs-reference) comparison are now
  the top-of-section figures for the comparison section.
- **Consistent terminology.** Uses "expected comparator", "observed spectral
  BSV", "literature-grounded", "disease-vs-reference shift" throughout.

## What is intentionally unchanged

These are preserved from v3 and earlier phases; v4 does not touch them:

- Spectral BSV computation — preprocessing (AsLS + SG + L2), 22 windows,
  projection to 8 BSV components.
- Motif / theme / BSV mapping for literature evidence
  (`map_evidence_to_motifs_themes_bsv`).
- Dataset loaders (`HCC Holdout`, `CCA/HCC/LM`, `Diabetes Plasma EV`).
- Comparator selection logic (cohort → expected literature profile).
- Text retrieval backbone (`TextQueryRetriever`, packet builder, section
  linker, confidence composer).
- Gemini fallback chain (flash → flash → flash-lite).
- Response schema (`GAIRAResponse`) and local synthesizer.

## What is explicitly not in v4

- **Calibration-dataset benchmarking** is not yet wired up. v4 remains a
  visualization / UI cleanup pass; the next phase is evaluating GAIRA's
  expected-comparator alignment against held-out calibration cohorts.
- No new scientific claims, no changes to confidence tiers, no changes to
  BSV weighting or composition.

## Verification

All three v4 files compile cleanly (`py_compile`). Imports resolve against
`src/gaira/...`. v3 folder is untouched and still launches independently at
`streamlit_apps/gaira_v3/gaira_v3.py`.
