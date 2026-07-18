# GAIRA Demo v2 — Migration Hardening

**Created:** 2026-07-15 · **Basis:** the migration issues found in
`GAIRA_FULL_CONTEXT_AND_STATE_AUDIT_2026-07-15.md`.

v2 is a **self-contained successor to v1**. v1 is preserved unchanged at
`../gaira_demo_reasoning_v1`. v2 shares no files with v1 — you can run, move,
or delete either independently.

The scientific engine (`gaira_core/{preprocessing,motif_scoring,mss_scoring,
substrate_physics,bsv_projection,evidence_synthesis,report_builder}.py`) is
**identical to v1** — v2 changes only *how the demo finds its data and reports
that to the user*. It is a robustness release, not a science release.

---

## What was fragile in v1 (and is fixed here)

| # | v1 problem | v2 fix |
| --- | --- | --- |
| 1 | Launch path hardcoded a stale username (`/Users/suraj/...`) in README + `app.py`. | No username anywhere; launch is `streamlit run app.py` or `./run_demo.sh`, resolved from the script's own location. |
| 2 | Data volume hardcoded to `/Volumes/SSD_Rad/GAIRA_DATA` in `config.py`. | Resolved by `gaira_core/paths.py`: `$GAIRA_DATA_ROOT` → candidate mounts/home → `None`. |
| 3 | Legacy CSVs read only from `../streamlit_apps/gaira_demo/data` (breaks if the demo is moved out of the repo). | The 5 tiny CSVs are **bundled** in `data/legacy/`; `paths.py` prefers `$GAIRA_LEGACY_DEMO_DATA` → repo copy → bundled. |
| 4 | If the SSD wasn't mounted, sections **silently** fell back to placeholders (badge only). | An explicit **data-source banner** at the top of the app shows REAL / DEGRADED / PLACEHOLDER mode, the resolved root, and a per-section table. |
| 5 | EV-diabetes caveat text named the wrong loader/file. | Caveat text now branches on the loader that actually ran. |
| 6 | No way to verify data resolution without launching Streamlit. | `selfcheck.py` prints the full resolution and exits non-zero if not in REAL mode. |

---

## How data is resolved (order of precedence)

**External GAIRA_DATA volume** (holds `raw/` + `processed/`; large, not bundled):
1. `$GAIRA_DATA_ROOT` (if it exists)
2. `/Volumes/SSD_Rad/GAIRA_DATA`, `/Volumes/SSD_Rad2/GAIRA_DATA`, `/Volumes/GAIRA_DATA`
3. `~/GAIRA_DATA`, `~/projects/GAIRA_DATA`, `<repo>/GAIRA_DATA`
4. `<demo>/data/external` (make this a symlink for a local dev copy)
5. otherwise unresolved → demo runs in DEGRADED/PLACEHOLDER mode (no crash)

**Legacy demo CSVs** (202-molecule BSV, ergothioneine, uric-acid SAEL; tiny):
1. `$GAIRA_LEGACY_DEMO_DATA` (if it looks valid)
2. `<repo>/streamlit_apps/gaira_demo/data`
3. **bundled** `<demo>/data/legacy` (always present → calibration + biochemical-space
   tabs work with zero external dependencies)

---

## Running on a new machine

```bash
cd gaira_demo_reasoning_v2

# 1. verify what will resolve (no server started)
python selfcheck.py

# 2a. if you have the GAIRA_DATA drive, point at it and launch
export GAIRA_DATA_ROOT=/Volumes/YourDrive/GAIRA_DATA
./run_demo.sh

# 2b. or launch with only bundled data (calibration + space tabs are real,
#     adenine + biological pilots show honest placeholders)
./run_demo.sh
```

Dependencies: `streamlit pandas numpy plotly scipy scikit-learn` (required),
`umap-learn` (optional — PCA fallback otherwise). The repo `.venv`
(Python 3.12) already has them.

---

## What v2 does NOT change (still true, still honest)

- The BSV is a **transparent band-evidence heuristic** (11 curated motifs →
  noisy-OR), not a calibrated or learned model. Biological cohort deltas are
  exploratory and composition-relative. v2 keeps every honest caveat from v1
  and adds one to the EV tab.
- v2 does **not** wire in the production `src/gaira` engine, domain-aware
  reranking, or per-sample BSV distributions. Those remain open items from the
  audit (see the audit report's §16 action plan) — they are science changes,
  out of scope for this migration-hardening pass.
