# GAIRA V7 Phase 11 — Screenshot Gallery

Captured from the **running application** by a headless Chromium driver walking the real user
flow — not mocked, not composited. Every number visible is live engine output for the built-in
**cholesterol** reference spectrum.

Viewport 1560×1000, full-page captures, downscaled 2× for the repository.

| # | file | screen |
|---|---|---|
| 01 | `01_home.png` | Landing page — hero, stat tiles, the three claims |
| 02 | `02_upload_empty.png` | Upload — drop zone and built-in examples |
| 03 | `03_upload_loaded.png` | Upload — file stats, raw spectrum, parse diagnostics, metadata |
| 04 | `04_preprocess_idle.png` | Preprocessing — checklist before the run |
| 05 | `05_preprocess_done.png` | Preprocessing — all four stages complete, engine output plotted |
| 06 | `06_analysis_running.png` | Analysis — the six-stage sequence mid-flight |
| 07 | `07_results_hero.png` | **Results** — verdict card, spectrum, radar, analogues |
| 08 | `08_chemistry.png` | Chemical evidence — all sixteen axes |
| 09 | `09_csm.png` | CSM contributions — activation bars and heatmap |
| 10 | `10_reconstruction.png` | Reconstruction and residual |
| 11 | `11_confidence.png` | Confidence — gauge, factor decomposition, open-set limitation |
| 12 | `12_provenance.png` | Provenance — the evidence chain and atlas identity |
| 13 | `13_documentation.png` | Docs — fingerprints and validated performance, pulled live |
| 14 | `14_architecture.png` | Architecture — the seven stages |
| 15 | `15_about.png` | About — what GAIRA is and why |

Regenerate with `results/v7_rebuild/phase11/code/capture_gallery.py` while the app is running.
