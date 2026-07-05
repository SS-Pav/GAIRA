# GAIRA polished Streamlit demo v2 — run instructions

v2 is a **dark-theme readability pass** over v1. It preserves v1's scientific
logic and reuses v1's derived assets (`streamlit_apps/gaira_demo/data/`).

## 1. Prerequisites

- v1 must already have its derived assets built (they are reused verbatim).
  If the `streamlit_apps/gaira_demo/data/` folder is empty or stale, rebuild:

  ```bash
  cd /Users/suraj/projects/GAIRA
  PYTHONPATH=src .venv/bin/python streamlit_apps/gaira_demo/build_demo_assets.py
  ```

- The `.venv` already has `streamlit >= 1.55` and `plotly >= 6.x`.

## 2. Run the v2 app (recommended)

Use the CLI theme flags so the dark theme is enforced even if Streamlit's
default config is set elsewhere:

```bash
cd /Users/suraj/projects/GAIRA
PYTHONPATH=src .venv/bin/streamlit run streamlit_apps/gaira_demo_v2/gaira_demo_v2.py \
  --theme.base dark \
  --theme.backgroundColor "#0B1220" \
  --theme.secondaryBackgroundColor "#111827" \
  --theme.primaryColor "#60A5FA" \
  --theme.textColor "#F1F5F9"
```

Streamlit will print a local URL (usually `http://localhost:8501`). Open it in a browser.

## 3. Alternative: launch from the app folder (uses .streamlit/config.toml)

```bash
cd /Users/suraj/projects/GAIRA/streamlit_apps/gaira_demo_v2
PYTHONPATH=../../src ../../.venv/bin/streamlit run gaira_demo_v2.py
```

The folder contains a `.streamlit/config.toml` that Streamlit picks up when
launched from this directory. Theme flags are therefore not needed.

## 4. Headless / CI use

```bash
PYTHONPATH=src .venv/bin/streamlit run streamlit_apps/gaira_demo_v2/gaira_demo_v2.py \
  --server.headless true --server.port 8601 --theme.base dark
```

For pure-Python smoke testing (no browser):

```python
from streamlit.testing.v1 import AppTest
app = AppTest.from_file("streamlit_apps/gaira_demo_v2/gaira_demo_v2.py", default_timeout=60)
app.run()
assert not app.exception
```

## 5. What you will see

Same four tabs as v1 — same data, fresher skin:

- **Methods / Pipeline** — hero figure, grounding corpus metrics, atlas band
  ruler with classification legend and high-contrast bands.
- **Grounding** — up to 5 pure molecules. Spectra overlay uses a bright
  tailwind-400 palette; BSV radar polygons have crisp angular labels.
- **Calibration** — ΔBSV bar / heatmap / radar, with a diverging scale
  red-400 → navy → emerald-400 tuned for dark bg.
- **Regression / Dose-response** — concentration slider updates the radar,
  ΔBSV bars, and dose curve together; both y-axes and the secondary "commit
  fraction" axis have bright, labeled titles and ticks.

## 6. v1 ↔ v2 side-by-side

To compare directly, launch them on different ports:

```bash
# v1 (light theme, unchanged)
PYTHONPATH=src .venv/bin/streamlit run streamlit_apps/gaira_demo/gaira_demo.py \
  --server.port 8600

# v2 (dark theme, readability pass)
PYTHONPATH=src .venv/bin/streamlit run streamlit_apps/gaira_demo_v2/gaira_demo_v2.py \
  --server.port 8601 --theme.base dark
```

## 7. What changed under the hood

- New file: `streamlit_apps/gaira_demo_v2/helpers.py` with:
  - `apply_dark_theme(fig, ...)` — single-pass dark styling for every figure.
  - `apply_polar_dark(fig, radial_max)` — ensures radar angular/radial labels are bright.
  - `atlas_ruler_figure(view, axes_unique)` — dedicated atlas plot with per-classification alpha, visible borders, and a classification legend.
- New file: `streamlit_apps/gaira_demo_v2/gaira_demo_v2.py` — same tabs as v1, with tighter CSS (pills, boxes, tabs) tuned for the dark page.
- New file: `streamlit_apps/gaira_demo_v2/.streamlit/config.toml` — theme defaults when launched from the app folder.
- Data: **reused from v1** (no duplicate copies).
