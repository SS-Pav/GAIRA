# BUILD REPORT — v1 RESTORED (stable readable demo)

**Date:** 2026-04-26
**Phase:** GAIRA_COMMAND_CENTER_STREAMLIT_VERSION_REPAIR_AND_V2_REBUILD
**Decision:** SHIPPED

---

## What this is

`streamlit_apps/gaira_command_center_v1/` is the recovered stable Streamlit
demo. It pre-dates the broken family-first hierarchical rewrite and matches
the post-Tab-2-Visual-Intelligence-Upgrade state.

## What was restored

| section | restored from | notes |
|---|---|---|
| `app.py` | last shared version | identical to v2 — orchestrates Tab 1 + Tab 2 with cached manifest |
| `config/app_config.yaml` | shared | unchanged |
| `config/evidence_layers.yaml` | shared | unchanged |
| `components/ui_blocks.py` | shared | already had `cluster_card` + `interpretation` |
| `components/overview_tab.py` | shared | Tab 1 unchanged |
| `utils/*.py` (5 files) | shared | `embedding_loader.py` includes the optional `attach_bsv_family` helper but Tab 2 v1 does not call it |
| `components/motif_mss_bsv_tab.py` | **reconstructed** | the stable post-upgrade-v1 layout, no family-first hull plot, no axis-overlap network, no traffic-light bar |

Reconstruction was needed because the folder was untracked in git (so no
`git checkout` recovery path) and the prior file had been overwritten by the
v2 family-first rewrite. The file was rebuilt from the contents documented
in `BUILD_REPORT_tab2_upgrade_v1.md` (which lives in v2).

## Tab 2 sections (v1)

```
1. Concept overview                    — 3 cards (L1 primitives → L2 MSS → L3 BSV)
2. MSS evolution                       — v4.1 → v4.2 (2 cards)
3. Interactive UMAP                    — Plotly Scattergl, hulls + labels + hover
4. Side-by-side · MSS vs Motif         — Plotly subplots
5. Hierarchical dendrograms            — pre-rendered images + interpretation
6. BSV saliency map                    — heatmap + canonical labels + shared-band bar
7. Hybrid BSV evidence flow            — 6-node Plotly diagram + supporting figs
↓
Tab 3 link card
```

## What was removed vs the broken intermediate

- The molecule-explorer skeleton stays out of Tab 2 (per the original v1 upgrade decision; reverting that would need a third reconstruction and the user said *"do not force if reverting is difficult"*).
- The family-first hull plot, axis-overlap network, and traffic-light ambiguity bar are absent — they live in v2.

## Acceptance check

| criterion | result |
|---|---|
| All modules import cleanly | ✅ utils.* + components.* + app |
| Live `streamlit run` boots | ✅ HTTP 200 on `/` and `/_stcore/health` (port 8765); no errors / tracebacks |
| Tab 1 unchanged + still works | ✅ overview_tab.py is byte-identical to v2 |
| Tab 2 readable | ✅ no broken family-first hulls, no missing artifacts |
| App runs even if artifacts missing | ✅ all loaders have `show_warning` fallbacks |

## How to run

```bash
streamlit run streamlit_apps/gaira_command_center_v1/app.py
```

Default port 8501. Use `--server.port=N` to pick another.

## Strict invariants preserved

- GAIRA core unchanged (no edits under `src/gaira/`).
- No GAIRA scoring rerun.
- Manifest path-configurable + cached.
- Missing-artifact tolerance preserved.
