# GAIRA V6 Demo — Data Provenance Audit

Every dataset displayed in `gaira_demo_reasoning_v4`, its canonical source, count
verification, status, the generated artifact, the engine version, known limitations,
and any discrepancy found and how it was resolved. Engine/atlas fingerprint for all
V6 outputs: **`09ed804a40836f4a05a91ba10900cded`** (verified on load).

## 1. Frozen Raman Reference Atlas (all pages)

| field | value | source |
|---|---|---|
| spectra | **375** | `foundation/artifacts/manifold.json` corpus_card (live) |
| analytes | **167** | corpus_card |
| sources | RamanBioLib 202 · gobbato_raman_metabolites 153 · amino_acid_raman_grounding 20 | corpus_card |
| representation | NMF k=24 (chosen by benchmark, not PCA by default) | manifold.json selection |
| explained variance | 0.712 | manifold.json stats |
| excitation transfer | 0.918 | foundation validation |
| status | **REAL** (committed: `manifold.json` + `manifold_components.npz`) | — |

Reference family map (Page 2): 167 analytes × 24 components from
`foundation/tables/c3_analyte_activation_matrix.csv` (committed), L1-normalised;
family per analyte derived from the frozen registry loadings (129 analytes carry a
family; the rest are `unassigned`). PCA is explanatory only.

## 2. Calibration datasets (Page 4)

| dataset | source (committed) | units | status | notes |
|---|---|---|---|---|
| adenine | `spike_validation/tables/phase3_projection_ils_adenine.csv` | 3381 rows; **cAg@785 subset used** (recoverable regime) | REAL | inter-lab; other substrates noisier |
| ergothioneine | `…/phase3_projection_ergothioneine.csv` | 55 (cAg@785) | REAL | cleanest calibrant |
| uricase | `…/phase3_projection_uricase.csv` | 20 (serum; spiked / spiked+uricase / serum_reference) | REAL | depletion (before/after) |

All are Ag/Au-SERS → out-of-domain for a Raman atlas by construction (a property to
characterise, not a defect). Langmuir fits are visual overlays, not part of the
engine.

## 3. Serum spike stress test (Page 5)

| field | value | source |
|---|---|---|
| analytes | **53** | `spike_validation/tables/phase7_serum_vs_pure.csv` (committed) |
| serum baseline | 15 unspiked spectra | `phase3_projection_serum_baseline.csv` |
| spiked spectra | 265 (53 analytes × ~5 replicates) | `phase3_projection_spiked_serum.csv` |
| recoverability tiers | strong 6 · partial 8 · poor 39 | derived (cos ≥0.35 / ≥0.10 / else) |
| above permutation null | 7 / 53 | `phase7_summary.json` |
| status | **REAL** | — |

**Discrepancy resolved.** Historical notes sometimes cite a different analyte count.
Per the brief, the **file-derived count (53)** from the canonical validation table is
used, and the discrepancy is surfaced on the page.

## 4. Biological cohorts (Page 6) — genuine V6, generated + committed

Built by `tools/build_biological_v6.py` from the raw data volume, projected live
through `GAIRAEngine.infer` → committed **sanitised** artifacts in
`biological_artifacts/`. Fresh checkout reads the artifact (no volume needed).

| dataset | source (raw volume) | units | groups | aggregation | status |
|---|---|---|---|---|---|
| COVID serum Raman | `raw/covid_serum_raman` | 465 spectra | COVID 159 · Healthy 150 · Suspected 156 | spectrum | REAL |
| HCC serum SERS | `raw/hcc_serum/data.csv` | 144 spectra | HCC 72 · control 72 | spectrum | REAL |
| Diabetes plasma-EV SERS | `raw/diabetes_plasma_ev_sers` | 63 patients | Impact 39 · Strong-D 24 | **patient** | REAL |

**Sanitisation (verified by test + grep).** Records contain only: anonymised `id`,
`group`, 24 `coord`, 13 `themes`, 13 `motifs`, `ood`, `confidence`, `background`. **No
demographics** (hba1c/bmi/race/gender/age/height/weight/waist) and no patient
identifiers are read or stored.

**Discrepancies resolved.**
- Diabetes metadata lists **40** Impact rows but the `.mat` has **39** Impact objects
  with a positional patient↔spectrum mapping. Resolution: cohort labels are taken from
  the `.mat` FILE (Impact vs Strong-D), the demographic `patient_data.csv` is **never
  read**, identifiers are anonymised indices, and analysis is **patient-level** (1
  patient = 1 n; mean over scans) to avoid pseudoreplication. Reported n = 39 + 24.
- COVID source file is misspelled `raw_Helthy.txt` (handled in the loader).
- HCC CSV has negative-shift instrument-padding columns; cropped to ≥ 200 cm⁻¹.

**Not displayed as V6.** The prior demo's per-cohort radars came from a **pre-V6**
8-axis autoresearch engine (`processed/gaira_autoresearch/`) and are **not** reused or
relabelled. The committed `results/diabetes_gaira_audit_*` tables are that legacy
audit (with demographics) and are not surfaced by this demo.

## 5. UNAVAILABLE (registry shows them honestly; no output fabricated)

| dataset | reason |
|---|---|
| Liver serum SERS (CCA/HCC/LM/NC) | 4-class per-sample txt; ingestion not yet wired for V6 |
| Ovarian plasma Raman+SERS | paired Raman/SERS zips; not yet wired for V6 |

*(SHINE EV-SERS and small2023 EV were UNAVAILABLE in the first build; the correction
pass wired both as REAL — see the "Correction pass — added datasets" section below.)*

## 6. Frozen-artifact completeness fixes (prerequisites for a working fresh checkout)

Two frozen files required by the engine had been silently dropped by broad
`.gitignore` rules and were never committed by earlier work. Both were force-added
with content unchanged (fingerprint verified):
- `src/gaira/engine/data/biochemical_ontology_v2.yaml` (rule `data/`)
- `results/v5_rebuild/foundation/artifacts/manifold_components.npz` (rule `*.npz`) —
  **the frozen NMF basis itself**; without it `gaira.engine` cannot load.

A `git archive HEAD` fresh-checkout simulation (only committed files, no volume) then
loads the engine, imports all 8 pages, and renders every figure successfully.

---

## Correction pass — added datasets

| dataset | source | units | groups | status | notes |
|---|---|---|---|---|---|
| shine_ev_sers | raw/shine_ev_sers (RawDataSet91 `clustered`) | 720 (subsampled) | D0 360 / D2 360 × dose | REAL | cell-culture EV; wavenumber from Fig4D cubic; Day1 omitted |
| small2023_ev | raw/small2023_ev (NormedProbe1 + SI xlsx axis) | 600 (subsampled) | c00…c100 | REAL | probe titration; characterization-only |

See BIOLOGICAL_DATA_AUDIT.md for full per-dataset detail, discrepancies (SHINE 15,027
vs measured counts; Day1 empty; cohort-level pairing; small2023 axis in the SI), and
sanitization. Both added as genuine V6 outputs; liver/ovarian remain UNAVAILABLE.
