# BUILD REPORT — RESTORED FIRST WORKING V1

**Date:** 2026-04-26
**Phase:** GAIRA_COMMAND_CENTER_STREAMLIT_HARD_REVERT_TO_FIRST_WORKING_V1
**Decision:** SHIPPED

---

## Source folder used for restore

`streamlit_apps/gaira_command_center_v1/` — the working v1 restored on
2026-04-26 from the post-Tab-2-Visual-Intelligence-Upgrade state, prior
to the v2 family-first redesign attempts. That folder was itself a
reconstruction (the original `streamlit_apps/gaira_command_center/` was
moved to `_v2/` during the v2 branch and never recovered into git).

Files copied verbatim into `streamlit_apps/gaira_command_center_restored_v1/`:

| file | role |
|---|---|
| `app.py` | orchestrator, sidebar, Tab 1 + Tab 2 wiring, manifest cache |
| `README.md` | original README |
| `assets/README.md` | placeholder |
| `components/__init__.py` | empty package marker |
| `components/overview_tab.py` | Tab 1 — Overview / Evidence Stack |
| `components/motif_mss_bsv_tab.py` | Tab 2 — Motif · MSS · BSV construction |
| `components/ui_blocks.py` | dark-theme cards / metrics / pipeline-flow / interpretation primitives |
| `config/app_config.yaml` | paths + roadmap + 22 phase folders |
| `config/evidence_layers.yaml` | pipeline steps + 8 evidence layers + BSV taxonomy |
| `utils/__init__.py` | package marker |
| `utils/artifact_loader.py` | manifest scanner + lookup helpers |
| `utils/embedding_loader.py` | MSS / motif embedding + signature + saliency loaders |
| `utils/figure_loader.py` | safe image load |
| `utils/markdown_loader.py` | safe markdown load |
| `utils/table_loader.py` | safe CSV load |

The cached `config/artifact_manifest.yaml` and Python `__pycache__/`
directories were NOT copied — they auto-rebuild on first launch.

The previous `BUILD_REPORT_v1_RESTORED.md` (which described the
gaira_command_center_v1 reconstruction) was not carried over; this
report replaces it.

## Artifacts successfully loaded

Verified before launch:
- `/Volumes/SSD_Rad/GAIRA_BUILD/gaira_representation_cluster_analysis_v1/tables/mss_analyte_embedding_v1.csv` (15,509 bytes)
- `/Volumes/SSD_Rad/GAIRA_BUILD/gaira_representation_cluster_analysis_v1/tables/motif_analyte_embedding_v1.csv` (15,543 bytes)

Verified at first launch (auto-built manifest, 164 KB):
- 22 phase folders detected, 0 missing
- 426 artifacts indexed (207 csv · 126 png · 93 md)
- All Tab 2 figure targets resolve:
  - `gaira_representation_cluster_analysis_v1/figures/fig_mss_dendrogram_v1.png`
  - `gaira_representation_cluster_analysis_v1/figures/fig_motif_dendrogram_v1.png`
  - `gaira_base_4_hybrid_bsv_build_v1/figures/fig_hybrid_family_confusion_heatmap_v1.png`
  - `gaira_base_4_hybrid_bsv_build_v1/figures/fig_hybrid_confidence_vs_accuracy_v1.png`

## Artifacts still missing

None blocking. Every loader degrades gracefully via a soft `gaira-warn`
card if a file is absent; the smoke test reported **zero** missing-artifact
warnings.

## Changes made (paths / imports only)

None required. The restored folder is byte-identical to v1 except for:
1. Removal of stale `config/artifact_manifest.yaml` (auto-rebuilds on first launch).
2. Removal of stale `__pycache__/` directories.
3. Removal of the inherited `BUILD_REPORT_v1_RESTORED.md` (replaced by this report).
4. Addition of `VERSION_LOCK.md` declaring the folder is locked.

No code edits were necessary because v1 already pointed at the correct
artifact paths under `/Volumes/SSD_Rad/GAIRA_BUILD/`.

## What was deliberately excluded (vs broken v2)

Restored v1 does **not** include any of:
- artificial family centroid + Fibonacci-jitter plot
- forced family-first map at the top of Tab 2
- random "Top 10 bands driving G05" auto-table
- broken axis-overlap network as primary
- 11-axis BSV taxonomy as a huge dataframe
- experimental hierarchical-representation rewrite
- inline drilldown dropdown (per-family) — that is a v2-only addition

Tab 2 in restored v1 keeps the working Plotly UMAPs + side-by-side MSS↔Motif
+ static dendrograms + BSV saliency heatmap + hybrid evidence flow.

## Run command

```bash
streamlit run streamlit_apps/gaira_command_center_restored_v1/app.py
```

Default port 8501. Use `--server.port=N` to pick another. Use the
sidebar **🔄 Rebuild manifest** button after adding new phase folders
to `/Volumes/SSD_Rad/GAIRA_BUILD/`.

## Smoke test result

| test | result |
|---|---|
| All modules import (`utils.*`, `components.*`, `app`) | ✅ |
| Stubbed render: Plotly figures issued | ✅ 5 |
| Stubbed render: image loads attempted | ✅ 4 |
| Stubbed render: dataframes rendered | 0 (Tab 2 v1 has no inline tables — by design) |
| Stubbed render: warnings + infos | ✅ **0 / 0** |
| Live `streamlit run` boots | ✅ HTTP 200 on `/` and `/_stcore/health` (port 8767) |
| Live server log errors / tracebacks | ✅ none |
| Artifact manifest auto-built on first launch | ✅ 164 KB, 22 phases / 426 artifacts |

**Smoke test PASSED.** No "Embedding not found" warnings, no missing
artifacts, no crashes.

## Layout

```
streamlit_apps/gaira_command_center_restored_v1/
├── BUILD_REPORT_RESTORED_FIRST_WORKING_V1.md   # this file
├── README.md
├── VERSION_LOCK.md                             # do-not-edit notice
├── app.py
├── assets/
│   └── README.md
├── components/
│   ├── __init__.py
│   ├── motif_mss_bsv_tab.py                    # Tab 2
│   ├── overview_tab.py                         # Tab 1
│   └── ui_blocks.py
├── config/
│   ├── app_config.yaml
│   └── evidence_layers.yaml
└── utils/
    ├── __init__.py
    ├── artifact_loader.py
    ├── embedding_loader.py
    ├── figure_loader.py
    ├── markdown_loader.py
    └── table_loader.py
```

## What remains to fix later (not in this revert)

These are deferred to a future *experimental* folder
(`streamlit_apps/gaira_command_center_experimental/`) when the user
chooses to pick the v2 work back up:

- Family-first scientific figure that is genuinely readable.
- Per-family drilldown with all 11 BSV axes selectable inline.
- Confusion-style axis-overlap matrix.
- BSV taxonomy as a compact card grid (vs huge dataframe).

None of these are required for the restored v1 to be useful — they
were experimental work that broke the demo and has been set aside.

## Strict invariants preserved

- GAIRA core unchanged.
- No GAIRA scoring rerun.
- No new visuals invented.
- `gaira_command_center_v1/` and `gaira_command_center_v2/` were not modified by this phase.
- VERSION_LOCK declares this folder must not be overwritten.
