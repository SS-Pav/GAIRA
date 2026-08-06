# `GAIRA_v7_rebuild/results/`

All V7 generated artefacts. **Currently empty apart from this README and the planning
figures.**

## Layout

| Directory | Contents |
|---|---|
| `tables/` | CSV tables: sweeps, comparisons, metrics, per-class and per-analyte results |
| `figures/` | figures — `planning/` holds the architecture diagrams for this documentation pass; phase figures get their own subdirectories |
| `manifests/` | per-phase build manifests (C-11), dataset role maps, split manifests |
| `checkpoints/` | frozen V7 atlas bundles, one directory per version |
| `phase_outputs/` | per-phase intermediate artefacts that later phases consume |

## What belongs here

- every artefact a manifest points at
- every table and figure referenced by a phase report
- frozen atlas bundles, versioned, each with its own `MANIFEST.json`

## What must not be stored here

- **Raw spectra.** `data/`, `GAIRA_DATA/`, `/Volumes/`, `*.mat` are gitignored, and V7 does
  not circumvent that.
- **Large regenerable intermediates** — full sweep tensors, per-run NMF outputs. Keep the
  summary tables and the manifest that regenerates the rest.
- **Anything under `results/v5_rebuild/` or `results/v6_rebuild/`.** V7 outputs live here,
  never there.
- **PDFs.** Repo policy gitignores `*.pdf`; Markdown and SVG are tracked instead.

## Two gitignore interactions

**`checkpoints/` is ignored globally** by the root `.gitignore`. A scoped
`GAIRA_v7_rebuild/.gitignore` re-includes `results/checkpoints/` so atlas bundles can be
tracked. Bulk intermediates remain excluded by extension.

**`*.pdf` is ignored.** Planning and phase figures ship as **SVG (vector) + PNG (preview)**.
SVG satisfies the vector-format requirement.

## Rules

1. Every artefact is referenced by a manifest. An artefact with no manifest has no provenance
   and is not evidence.
2. Frozen bundles are immutable — a change means a new version directory, never an edit.
3. Nothing here contains an absolute local path.
