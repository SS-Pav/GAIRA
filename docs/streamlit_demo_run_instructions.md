# GAIRA polished Streamlit demo — run instructions

## 1. First-time setup

The repo already has all dependencies installed in `.venv/`. If not:

```bash
cd /Users/suraj/projects/GAIRA
.venv/bin/pip install -r requirements_vm.txt
```

## 2. Build (or rebuild) demo assets

The app reads from a small derived `data/` folder so it loads fast and does not
depend on the external SSD volume being mounted while demoing (as long as the
assets are already built).

```bash
cd /Users/suraj/projects/GAIRA
PYTHONPATH=src .venv/bin/python streamlit_apps/gaira_demo/build_demo_assets.py
```

This reads from:
- `config/spectral_anchor_windows_v1.csv`
- `/Volumes/SSD_Rad/.../ramanbiolib/…`
- `/Volumes/SSD_Rad/.../ergothioneine_serum/ERG_calibration.csv`
- `/Volumes/SSD_Rad/.../gaira_calibration_eval_v3/tables/*.csv`

…and writes to `streamlit_apps/gaira_demo/data/` (~3 MB total).

## 3. Run the app

```bash
cd /Users/suraj/projects/GAIRA
PYTHONPATH=src .venv/bin/streamlit run streamlit_apps/gaira_demo/gaira_demo.py
```

Streamlit will print a local URL (usually `http://localhost:8501`). Open it in a browser.

## 4. What you will see

- **Methods / Pipeline** — hero figure, pipeline stages, grounding corpus + atlas band explorer.
- **Grounding** — pick up to 5 pure molecules from RamanBioLib, compare their processed spectra, BSV bars, and BSV radar overlay.
- **Calibration** — pick up to 4 calibration contrasts (from `gaira_calibration_eval_v3`), view ΔBSV as bars, heatmap, or |Δ| radar, with SAEL-backed pass/inconsistent pills.
- **Regression / Dose-response** — slider over Ergothioneine 0.0 → 2.0 µM (11 levels); the radar, ΔBSV bars, and dose curve all update together.

## 5. Typical demo script (3–5 minutes)

1. Open **Methods** — point at the hero figure, then the atlas ruler (hover a band to show range + ambiguity + source count).
2. Switch to **Grounding** — show L-ergothioneine and Hypoxanthine side-by-side; note how the spectra overlay drives the BSV radar.
3. Switch to **Calibration** — pick `cspp_fig7_ergothioneine_spike` + `uricase_sigma_depletion` in heatmap view; ergo spike lights up Lipid/Protein/Redox (expected up), uricase shows the moderate-confidence disagreement that drives the "inconsistent" label.
4. Switch to **Regression** — step the slider from 0 to 2 µM, show ΔRedox tracking dose; point at the dashed vline on the dose curve.

## 6. Headless / CI use

```bash
PYTHONPATH=src .venv/bin/streamlit run streamlit_apps/gaira_demo/gaira_demo.py --server.headless true --server.port 8599
```

For pure-Python smoke testing (no browser), use Streamlit's AppTest:

```python
from streamlit.testing.v1 import AppTest
app = AppTest.from_file("streamlit_apps/gaira_demo/gaira_demo.py", default_timeout=60)
app.run()
assert not app.exception
```
