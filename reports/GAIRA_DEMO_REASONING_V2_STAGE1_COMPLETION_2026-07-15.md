# GAIRA Reasoning Demo v2 — Stage 1 Completion Report

**Date:** 2026-07-15
**Scope:** finish Stage 1 verification + provenance hardening for `gaira_demo_reasoning_v2`, without integrating `src/gaira` and without changing any scientific inference behavior. v1 frozen; v1 untouched.

---

## 1. What was completed previously (prior pass)

- Created `gaira_demo_reasoning_v2` as a self-contained successor to v1.
- Portable path resolution (`gaira_core/paths.py`): `GAIRA_DATA_ROOT` / `GAIRA_LEGACY_DEMO_DATA` env overrides → candidate mounts → bundled fallback.
- Bundled the 5 tiny legacy CSVs in `data/legacy/` (self-containment).
- Explicit REAL / DEGRADED / PLACEHOLDER data-source banner.
- Corrected EV-diabetes provenance text to name the loader that actually runs.
- `selfcheck.py`, portable `run_demo.sh`, `MIGRATION_HARDENING.md`.
- Preserved the v1 scientific engine unchanged.

## 2. What was completed in this pass

1. **v1 freeze proof** — SHA-256 manifest of all 16 v1 source/doc files; verified `shasum -c` OK; git diff vs HEAD empty. **0 v1 files modified.**
2. **Numerical regression v1↔v2** — 9 full-decomposition reports + 4 loader BSV tables across all required inputs; **all 10 output classes 0.00e+00 at atol=1e-9 → PASS**.
3. **Serum-liver provenance reconciliation** — canonical **212 unique patients**; 213 BSV rows = 212 + duplicate measurement of `SER-CCA-58`; 212 mean spectra; **no 214 exists**. v2 serum caveat text corrected.
4. **Per-axis grounding placeholder replaced** — real, NA-aware evidence table generated from the 202-molecule reference table + grounding registry; wired into v2 only; caption explains every count.
5. **DuckDB accidental files** — inspected and deleted (conclusively accidental, empty, untracked, gitignored, unreferenced).
6. **Final validation** — REAL + DEGRADED selfcheck, v2 boot, v1 boot, grounding load, no `src/gaira` import.

## 3. Files added / changed

### v2 — scientific engine (UNCHANGED, byte-identical to v1)
`preprocessing.py`, `primitive_extraction.py`, `motif_scoring.py`, `mss_scoring.py`, `substrate_physics.py`, `bsv_projection.py`, `evidence_synthesis.py`, `report_builder.py`, `plotting.py`, `__init__.py` — **all identical to v1** (verified by `diff` and by the numerical regression).

### v2 — changed (non-inference only)
| File | Change |
| --- | --- |
| `gaira_core/config.py` | path constants now resolved via `paths.py`; added `GENERATED_DIR`. No axis/threshold/scoring change. |
| `gaira_core/data_loader.py` | `load_family_counts` reads the generated evidence table (real-first, `keep_default_na=False` to preserve literal `NA`); added `load_grounding_corpus_summary`. **No change to any spectrum/cohort/inference loader.** |
| `app.py` | title/docstring/footer → v2; data-source banner; EV caveat names actual loader; serum caveat states canonical 212; per-axis grounding evidence table (NA-aware) + captions; adenine captions use resolved path. |
| `README.md` | v2 launch, env vars, self-containment note. |

### v2 — new files
`gaira_core/paths.py`, `selfcheck.py`, `run_demo.sh`, `MIGRATION_HARDENING.md`, `tools/build_grounding_evidence.py`, `data/legacy/*.csv` (5 bundled), `data/generated/per_axis_grounding_evidence.csv`, `data/generated/grounding_corpus_summary.json`.

### Repository
| File | Change |
| --- | --- |
| `.gitignore` | narrow negations so v2's small bundled/generated data is tracked (the broad `data/` rule still ignores everything else). |
| root `<_duckdb.DuckDBPyConnection object at 0x…>` ×2 | **deleted** (accidental, empty, untracked). |
| `reports/GAIRA_DEMO_REASONING_V1_FREEZE_MANIFEST_2026-07-15.md` | new |
| `reports/gaira_demo_reasoning_v1_sha256_2026-07-15.txt` | new |
| `reports/GAIRA_DEMO_V1_V2_NUMERICAL_REGRESSION_2026-07-15.md` | new |
| `reports/SERUM_LIVER_PROVENANCE_RECONCILIATION_2026-07-15.md` | new |
| `reports/GAIRA_DEMO_REASONING_V2_STAGE1_COMPLETION_2026-07-15.md` | new (this file) |

## 4. Regression results

All output classes, atol = 1e-9, PASS (max abs diff `0.00e+00`): preprocessed_spectra, primitive_counts, motif_scores, mss_scores, substrate_corrections, bsv_values, confidence_output, evidence_axis_ordering, caveat_generation, loader_bsv_tables. Cases: 6 adenine concentrations, 4 serum cohorts, 2 EV cohorts, 8 SHINE Day0/Day2 cells, 1 synthetic end-to-end. The only nonzero diffs (synthetic cases, pre-fix) were `hash()`-seeded synthetic-noise, resolved with `PYTHONHASHSEED=0` — a property of shared unchanged code, not a v1/v2 divergence. Detail: `GAIRA_DEMO_V1_V2_NUMERICAL_REGRESSION_2026-07-15.md`.

## 5. Serum-liver reconciliation

- **212 unique patients** (CCA 66 · HA 48 · HCC 49 · LM 49).
- `patient_level_bsv.csv` = 213 rows (212 + duplicate measurement of CCA `SER-CCA-58`).
- `patient_level_mean_spectra.csv` = 212 (one mean/patient; the demo's source).
- **214 does not exist** in any pilot4 table (earlier miscount).
- Secondary cosmetic issue: LM patient stored as `SER-LM-11-1_01` (BSV) vs `SER-LM-11` (spectra) — same patient, inconsistent ID; no count effect.
- v2 displays 212; v1 unchanged; raw/processed data unchanged. Detail: `SERUM_LIVER_PROVENANCE_RECONCILIATION_2026-07-15.md`.

## 6. Grounding-count methodology

Generator `tools/build_grounding_evidence.py`; output `data/generated/per_axis_grounding_evidence.csv` + `grounding_corpus_summary.json`. Columns kept strictly separate: `axis, unique_reference_analytes, measured_reference_spectra, direct_spectral_sources, supporting_literature_sources, unmapped_records, mapping_status, mapping_notes`.

- **Unique reference analytes** (from the 202-molecule RamanBioLib table, by dominant 8-axis, unique compounds — not files or augmented spectra):
  - Resolved (1:1 legacy→v11): Pyrimidine **13**, Nuc-phosphate **31**, Glycan **25**, Protein **81**, Aromatic **12**.
  - **NA** (legacy 8-axis splits into two v11 children — cannot defensibly assign): Purine-nuc/Purine-met (shared pool 12), Lipid/Sterol (shared pool 25), Redox/Metabolite (shared pool 3). Displayed as **NA, not 0**.
- **Measured reference spectra**, **direct spectral sources**, **supporting literature sources** are recorded per-source in the registry, not per-axis → **NA per axis**, reported corpus-wide:
  - Measured reference spectra = **160** (adenine 12, amino-acid 20, metabolite-63 64, serum-Ag 64) — never combined with literature.
  - Direct-spectral sources = **12 unique** (13 registry rows; `src_raman_ir_handbook_manuscript` duplicated).
  - Supporting-literature sources = **16 unique** (30 registry rows; duplicates collapsed).
- Legacy 8→11 mapping documented in each row's `mapping_notes`. No molecular assignment inferred from a nearby peak.

## 7. DuckDB findings & disposition

- Two repo-root files `<_duckdb.DuckDBPyConnection object at 0x100b578f0>` and `…0x1019473f0>`.
- **SHA-256 (both identical):** `de92b45109daa1605e4ce962ec8411aea68ed8f2d51095dc4226d180bb7a7985`; size 2,109,440 bytes each; `file` → "data".
- **Valid DuckDB databases but EMPTY (0 tables).**
- **Not tracked** by git; already matched by `.gitignore` line `<_duckdb*` under "🚨 ACCIDENTAL FILES".
- **Unreferenced** — filenames are process memory-address `repr()`s; no code references them.
- **Likely creation path:** `duckdb.connect(str(db_path), …)` where `db_path` was accidentally a `DuckDBPyConnection` object, so `str(connection)` produced the repr filename (the `duckdb.connect(str(...))` idiom appears in `src/gaira/inference.py` and several scripts).
- **Disposition: DELETED** — conclusively accidental, empty, untracked, unreferenced. No legitimate database was touched.

## 8. Remaining known issues (out of Stage-1 scope)

- Scientific limitations from the audit remain by design: the BSV is a transparent band-evidence heuristic (not a calibrated/learned model); production `src/gaira` engine, domain-aware reranking, and per-sample BSV distributions are **not** wired (explicitly out of scope).
- Upstream data provenance items not owned by the demo: the `SER-CCA-58` duplicate and `SER-LM-11` ID inconsistency live in the autoresearch export (not modified here).
- SHINE remains a 3-axis autoresearch projection (honestly flagged), unchanged.

## 9. Exact launch commands

```bash
cd /Users/surajpg/projects/GAIRA/gaira_demo_reasoning_v2
python selfcheck.py            # verify data resolution (exit 0 = REAL)
./run_demo.sh                  # portable launch (REAL if SSD/GAIRA_DATA_ROOT present)
# or:  ../.venv/bin/streamlit run app.py
# DEGRADED test:  GAIRA_DATA_ROOT=/tmp/empty ./selfcheck.py
```

v1 (frozen reference) still launches: `cd ../gaira_demo_reasoning_v1 && ../.venv/bin/streamlit run app.py`.

## 10. Confirmation — scientific inference behavior unchanged

- All 10 scientific engine modules in v2 are **byte-identical** to v1 (`diff` clean).
- Numerical regression: **0.00e+00** across every output class (preprocessing, primitives, motif, MSS, substrate, BSV, confidence, evidence ordering, caveats, loader BSV tables).
- Changes were confined to path resolution, provenance/display text, and the grounding-count table — none touch preprocessing, motif scoring, MSS scoring, BSV projection, thresholds, cohort selection, or inference.
- `src/gaira` is **not** imported anywhere in v2.

> **Scientific inference behavior is unchanged. v1 remains the frozen audited reference build and was not modified.**
