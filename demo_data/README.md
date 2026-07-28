# `demo_data/` — precomputed inputs for the Streamlit apps

The Streamlit apps **never require SSD_Rad, raw spectra, or recomputation**. They read
only the frozen model in `assets/foundation/` and the precomputed outputs listed here.
This file documents where each app's demo data lives (kept in place to avoid breaking the
apps' data loaders; all paths are in-repo and committed).

## Foundation Explorer (`gaira_foundation_explorer/`)

Reads the complete audit under **`results/v5_rebuild/foundation_audit/`**:

| What | Location |
|---|---|
| Reports (11 parts, rendered as Markdown) | `foundation_audit/reports/*.md` |
| Figures (basis grid, benchmark, components, validation, …) | `foundation_audit/figures/*.png` |
| Tables (corpus, benchmark, components, MSS, validation) | `foundation_audit/tables/*.json` · `*.csv` |
| Per-component pages | `foundation_audit/components/*.md` |
| Frozen atlas metadata | `assets/foundation/manifold.json` |

Loader: `gaira_foundation_explorer/explorer_core/data.py`.

## Reasoning demo (`gaira_demo_reasoning_v4/`)

Reads its own committed, sanitized artifacts plus the frozen engine:

| What | Location |
|---|---|
| Biological cohort states (sanitized engine outputs) | `gaira_demo_reasoning_v4/biological_artifacts/*.json` |
| Matched Raman↔SERS pairs | `gaira_demo_reasoning_v4/reference_artifacts/*.json` |
| Serum-spike / dose / uricase projections | `results/v5_rebuild/spike_validation/tables/*.csv` |
| Frozen inference engine | `assets/foundation/` (via `src/gaira/engine`) |

Loaders: `gaira_demo_reasoning_v4/demo_core/data.py`, `biological.py`, `serum.py`.

## Regenerating demo data (optional, GAIRA_Lab only)

The `tools/` scripts in each app regenerate these artifacts from the raw lab volume
(`/Volumes/SSD_Rad/GAIRA_DATA/`). That is lab work and is **not** needed to run the apps —
the committed outputs above are sufficient for a fresh clone.
