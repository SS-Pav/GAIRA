# GAIRA Physics Atlas Audit

**Date:** 2026-07 · Full table: `data_audit/physics_atlas_registry.csv`.

## What the atlas is
- **`config.py ATLAS_REGIONS`**: 8 wavenumber zones (400–700, 700–760, 760–900, 900–1150, 1150–1350, 1350–1500, 1500–1700, 1700–1800), each mapped to candidate G-axes.
- **`app.py atlas_details`**: a Python dict of prose per region — assignments / ambiguity / substrate / confounders / treatment.

## How it was formed & what it affects
- **Formed from literature/curated knowledge** (band-assignment prose), NOT from measured GAIRA data. There is no measured "atlas" dataset behind it.
- **Runtime effect: UI text / caveats ONLY.** It renders in `st.expander` blocks. **It does NOT change any BSV number, does not mask bands, and does not alter retrieval** (confirmed: caveat count varies but BSV is unaffected — see ablation `caveats_change_bsv = no`).
- One region (700–760 purine) and one (1500–1700 carotenoid) overlap with substrate rules that *do* have numeric effect, but that effect lives in `substrate_physics.py`, not the atlas.

## Answers
- **How formed?** Manually curated from Raman/SERS literature band assignments.
- **Literature vs measured vs curated?** Entirely literature-derived, manually curated prose. **0 entries from measured GAIRA data.**
- **Changes BSV numbers?** **No.** **Masks bands?** No. **Alters retrieval?** No. **Only generates caveats / UI explanation.**
- **Testable?** **Not currently testable** — there is no measured atlas ground truth to validate against; it is an interpretation aid.

## Related (production, dormant)
The production `src/gaira/atlas/atlas_loader.py` + GAIRA_BUILD phase4 YAMLs implement a *real* band-constraint / ambiguity / companion-band engine (evidence-derived), and `config/spectral_anchor_windows_v1.csv` (64 source-backed windows) — but these are **not imported by any runtime module** and are not the demo's atlas. The demo atlas is prose-only.
