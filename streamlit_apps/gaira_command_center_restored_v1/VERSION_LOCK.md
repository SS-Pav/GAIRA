# VERSION LOCK — gaira_command_center_restored_v1

**This is the restored first working Streamlit version.**
**Do not overwrite during future redesigns.**

Locked on: **2026-04-26**
Source: `streamlit_apps/gaira_command_center_v1/` (the working v1 from
the post-Tab-2-Visual-Intelligence-Upgrade state, prior to the v2
family-first attempts).

## What this folder contains

The first stable, readable Streamlit demo of GAIRA Command Center:
- Tab 1 — Overview / Evidence Stack
- Tab 2 — Motif · MSS · BSV construction
  (concept overview · MSS evolution · interactive UMAP · side-by-side
  MSS↔Motif · annotated dendrograms · BSV saliency heatmap · hybrid
  evidence flow · Tab 3 link card)

## Strict rules

- **Do not modify this folder.** Any code edits should go into a NEW folder.
- Future experimental changes must go into:
  `streamlit_apps/gaira_command_center_experimental/`
- The artifact manifest (`config/artifact_manifest.yaml`) auto-rebuilds
  on first launch and is safe to refresh via the sidebar **🔄 Rebuild
  manifest** button. No other state in this folder mutates at runtime.

## Run

```bash
streamlit run streamlit_apps/gaira_command_center_restored_v1/app.py
```

## Smoke-test result (2026-04-26)

- Stubbed render: 5 Plotly figures · 4 image loads · 0 warnings · 0 infos.
- Live `streamlit run`: HTTP 200 on `/` and `/_stcore/health` (port 8767),
  no errors / tracebacks in the server log, manifest auto-built (164 KB).
