# GAIRA demo v3 — run instructions

v3 is the **semantic-cleanup + evidence-layer + selector-hygiene** pass over v2.

## 1. Prerequisites

- v1 must already have its derived assets built (reused unchanged):
  `streamlit_apps/gaira_demo/data/` should contain `grounding_molecule_*.csv`,
  `calibration_*.csv`, `ergothioneine_*.csv`, `atlas_explorer.csv`.
  If missing, rebuild:

  ```bash
  cd /Users/suraj/projects/GAIRA
  PYTHONPATH=src .venv/bin/python streamlit_apps/gaira_demo/build_demo_assets.py
  ```

## 2. Build v3 derived tables

One-time (or after the v1 derived tables change):

```bash
cd /Users/suraj/projects/GAIRA
PYTHONPATH=src .venv/bin/python streamlit_apps/gaira_demo_v3/build_v3_assets.py
```

Writes to `streamlit_apps/gaira_demo_v3/data/`:

- `grounding_layer_summary.csv`
- `literature_evidence_layer.csv`
- `calibration_metadata_v3.csv`
- `regression_registry.csv`

## 3. Run the v3 app (recommended)

```bash
cd /Users/suraj/projects/GAIRA
PYTHONPATH=src .venv/bin/streamlit run streamlit_apps/gaira_demo_v3/gaira_demo_v3.py \
  --theme.base dark \
  --theme.backgroundColor "#0B1220" \
  --theme.secondaryBackgroundColor "#111827" \
  --theme.primaryColor "#60A5FA" \
  --theme.textColor "#F1F5F9"
```

Streamlit will print a local URL (usually `http://localhost:8501`). Open it.

## 4. Alternative: launch from the app folder (uses `.streamlit/config.toml`)

```bash
cd /Users/suraj/projects/GAIRA/streamlit_apps/gaira_demo_v3
PYTHONPATH=../../src ../../.venv/bin/streamlit run gaira_demo_v3.py
```

The folder contains a `.streamlit/config.toml` that enforces the dark theme.

## 5. What changed from v2

### Tab 1 — Methods / Pipeline
- Removed the hero image, "Stages" cards, and the two secondary explainer images.
- Added a concise what-GAIRA-does box.
- Added a Plotly-native current-pipeline diagram.
- Added a **three-layer grounding overview** (Pure-molecule / Literature-linked / Atlas) with dataset provenance named on each card.
- Added an honest coverage note.
- Kept and polished the atlas explorer.

### Tab 2 — Grounding
- Family filter uses a family-scoped widget key (`grd_molecules_v3__{family}`) → switching family never leaks a residual selection, never crashes on invalid defaults.
- Multiple selected molecules render **one trace per molecule** in the spectra overlay (deduped).
- **Separate BSV bar plots per molecule**, stacked vertically. The old averaged bar is gone.
- One radar trace per selected molecule.
- Atlas band shading is now **molecule-aware**: highlights bands whose primary axis matches the selected molecules' dominant BSV axes.
- Context table is deduped.

### Tab 3 — Calibration
- Rich, human-readable condition labels (e.g. "Serum baseline vs Ergothioneine spike").
- Every view (bar / heatmap / radar) shows explicit baseline → perturbed framing.
- Per-condition metadata block: analyte, matrix, substrate, perturbation type, concentration info, behavior class, caveat.

### Tab 4 — Regression
- Narrowed to only **truly supported** ordered series — currently `ergothioneine_titration`.
- Uricase is **excluded** (endpoint comparison, not a ladder).
- CSPP Fig 7 spikes are **excluded** (single spike level, `conc.nunique()==1`).
- Adenine SERS raw CSVs are noted as "not yet wired through GAIRA's BSV pipeline".
- An expander under the controls documents why each excluded dataset was kept out.

## 6. Side-by-side comparison

```bash
# v1 (light theme, original baseline)
PYTHONPATH=src .venv/bin/streamlit run streamlit_apps/gaira_demo/gaira_demo.py --server.port 8600

# v2 (dark-theme readability pass)
PYTHONPATH=src .venv/bin/streamlit run streamlit_apps/gaira_demo_v2/gaira_demo_v2.py \
  --server.port 8601 --theme.base dark

# v3 (semantic + selector-hygiene pass)
PYTHONPATH=src .venv/bin/streamlit run streamlit_apps/gaira_demo_v3/gaira_demo_v3.py \
  --server.port 8602 --theme.base dark
```

## 7. Headless / CI

```bash
PYTHONPATH=src .venv/bin/streamlit run streamlit_apps/gaira_demo_v3/gaira_demo_v3.py \
  --server.headless true --server.port 8602 --theme.base dark
```

Python smoke test:

```python
import sys
sys.path.insert(0, "streamlit_apps/gaira_demo_v3")
from streamlit.testing.v1 import AppTest
app = AppTest.from_file("streamlit_apps/gaira_demo_v3/gaira_demo_v3.py", default_timeout=60)
app.run()
assert not app.exception
```

## 8. Demo narration (~5 minutes)

1. **Methods tab** — read the intro box; trace the pipeline diagram; point at the three grounding layers and at the honest coverage note.
2. **Grounding tab** — pick L-ergothioneine and Hypoxanthine; notice the atlas shading tracks *their* dominant axes. Switch family to `AminoAcids` — selection cleanly reseeds.
3. **Calibration tab** — compare "Serum baseline vs Ergothioneine spike" with "Commercial serum — Uricase untreated vs treated" in heatmap view; read the per-condition metadata boxes.
4. **Regression tab** — only one option (Ergothioneine titration); step the slider 0 → 2.0 µM; open the "Why other datasets are not in this tab" expander to show honest audit.
