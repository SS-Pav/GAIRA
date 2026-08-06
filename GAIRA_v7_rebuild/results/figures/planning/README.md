# V7 planning figures

Ten architecture diagrams for the V7 specification. Regenerate with:

```bash
python GAIRA_v7_rebuild/results/figures/planning/make_planning_figures.py
```

Deterministic — no RNG, no timestamps, no data loading. The script is documentation tooling,
not V7 implementation code.

| # | File | What it shows |
|---|---|---|
| 1 | `fig01_flat_vs_hierarchical` | V5/V6 flat global NMF vs the V7 hierarchical architecture |
| 2 | `fig02_learning_pipeline` | the offline build, corpus → frozen atlas, with phase attribution |
| 3 | `fig03_inference_pipeline` | the live path, spectrum → BSV → interpretation, with permitted operations |
| 4 | `fig04_coverage_imbalance` | the 32:1 class imbalance and strategies A–F |
| 5 | `fig05_representation_hierarchy` | LSM → CSM → theme → BSV, and the derived quantities that are not BSVs |
| 6 | `fig06_offline_vs_live` | the learning/inference boundary and why it is absolute |
| 7 | `fig07_phase_roadmap` | phase sequence with gate summaries |
| 8 | `fig08_failure_taxonomy` | the V6.3 failure waterfall — 57.4% true representation errors |
| 9 | `fig09_atlas_structure` | V5 vs V7 atlas assets, and why the single-hash scheme does not generalise |
| 10 | `fig10_dart_trajectory` | BSV trajectories and the requirements they place on the representation |

## Formats

**SVG (vector) + PNG (preview).** No PDF — the root `.gitignore` excludes `*.pdf` under the
repo's "track the source, not the binary export" policy. SVG is a vector format and satisfies
the vector requirement; text is kept as text (`svg.fonttype: none`), so the diagrams remain
searchable and editable.

## Editorial rule

**Every arrow corresponds to a defined computational operation** described in
`../../../architecture/`. No decorative flow, no implied mechanism that does not exist in the
specification.

Figures 1, 4, 8, and 9 carry real numbers from committed or on-disk tables. Their sources are
printed in the figure subtitles. Figure 10 is explicitly labelled a conceptual schematic — it
contains no data, and its trajectory is a hand-specified illustration, not a measurement.
