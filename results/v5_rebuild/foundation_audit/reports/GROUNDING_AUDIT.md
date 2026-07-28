# GROUNDING_AUDIT
### Every source that touches the frozen coordinate system — and everything that does not

*Part 1 of the GAIRA Foundation Model audit. A whole-repository census of spectral data,
classified strictly by its relationship to the **frozen V5 Raman atlas** (the NMF
manifold `results/v5_rebuild/foundation/artifacts/manifold.json`). Compiled from a
full sweep of `/Volumes/SSD_Rad/GAIRA_DATA/raw`, `src/gaira/`, `scripts/`, and
`results/v5_rebuild/*/code`.*

---

## 0. The one distinction that matters

"TRAINING / VALIDATION / UNUSED" is meaningful **only relative to the frozen Raman
atlas**. The repo contains two spectral systems:

- **The frozen V5 Raman atlas** (this audit's subject): fit by `gaira.foundation` on a
  **Raman-only** corpus of **exactly three sources**, frozen to fingerprint
  `09ed804a…`. `dataset.py` hard-asserts `(meta.modality == "raman").all()` and
  explicitly excludes Ag-SERS / Au-SERS / DART.
- **A separate legacy "global_v2 / base_4" embedding engine** (`scripts/build_global_v2_*`,
  `run_gaira_base_4_*`, `processed/embedding_v5_full/`) that consumed ~20 biofluid SERS
  datasets. **None of those feed the atlas.** They are UNUSED here.

**There is no fourth Raman reference dataset feeding the atlas, and none hidden.** The
only Raman sources in the NMF are RamanBioLib, Gobbato Raman, and the amino-acid sheet.

---

## 1. TRAINING — the three Raman sources in the frozen NMF

| Dataset | Modality / excitation | Spectra | Analytes | Raw path → loader | Citation / origin | Metadata |
|---|---|---:|---:|---|---|---|
| **RamanBioLib** | spontaneous Raman; 9 excitations | 202 | 141 | `raw/ramanbiolib/…/db/*.csv` → `data/loader.py::load_ramanbiolib` | **Terán, Ruiz, Loza-Alvarez, Masip, Merino, *Chemom. Intell. Lab. Syst.* 264 (2025) 105476**, doi:10.1016/j.chemolab.2025.105476; licence **ODbL** | **Rich** (full metadata_db + schema) |
| **gobbato_raman_metabolites** | Raman powder; 785 nm B&WTek | 153 | 51 | `raw/serum_ag_colloids/dataset_spectral_data.zip → Raman metabolites/` → `data/gobbato.py::load_gobbato_785` (Raman subset) | **Gobbato et al. 2025**, *Anal. Bioanal. Chem.*, doi:10.1007/s00216-025-06192-5 (PMC12680727); Bonifacio group, Trieste | Sparse (filename-encoded; abbrev map in `synonyms.py`) |
| **amino_acid_raman_grounding** | Raman; 785 nm (hard-coded) | 20 | 19* | `raw/amino_acid_raman_grounding/aa.xlsx` → `dataset.py::_load_amino_acids` | **Origin undocumented** in repo (column-per-analyte sheet) | **Sparse / none** (no README, DOI, or instrument record) |
| **TOTAL (deduped union)** | Raman only | **375** | **167** | `dataset.py::load_reference_corpus` | — | — |

\*19 unique after `glutamate` appears in two columns ("Glutamic Acid" + "L-Glu"). See
FOUNDATION_CORPUS_REPORT §6 for the ≈161 distinct-molecule count.

**Provenance grade of the training corpus:** RamanBioLib is fully citable and licensed;
Gobbato is citable (its powder Raman is the pure-compound arm of PMC12680727); the
**amino-acid sheet is the weakest link — no citation, README, DOI, or instrument
metadata anywhere in the repo.** For a *foundation* model this is the one provenance gap
worth closing (it contributes 20 spectra / 19 analytes including several duplicates).

---

## 2. INTERPRETATION AID — not spectra, not training

| Resource | What it is | Usage | Provenance |
|---|---|---|---|
| **raman_knowledge_core** | 7 CSVs of curated Raman **peak assignments / region cautions** — **zero spectra** | `dataset.py::load_peak_assignments` — used ONLY to *corroborate* post-hoc axis themes (Part 6), never to fit the NMF and never to *define* a theme | **Internal "GAIRA Curated Seed Pack" 2026, licence `internal-curated`, blank DOI/URL** — self-authored |

**Caveat surfaced by the audit:** the literature that "corroborates" component themes is, in
this table, internally authored rather than externally peer-reviewed. The axis-building
code already treats it as corroboration-only (and deliberately down-weights it because it
is biofluid/protein-biased), but the label "literature support" in the ontology should be
read as *"consistent with GAIRA's curated seed pack,"* not *"confirmed by an external
reference."* This does not affect the representation (Parts 4–5) — only the *naming* of
axes (Parts 6–8).

---

## 3. VALIDATION — Raman/SERS projected through the frozen atlas, never fitted

| Dataset | Modality / excitation | ~Spectra | Role | Origin |
|---|---|---:|---|---|
| **covid_serum_raman** | biological serum **Raman** | ~477 (4 groups) | `dataset.py::load_serum_raman` — C7 out-of-domain projection / blank control | COVID serum Raman study (readme; excitation unrecorded — metadata gap) |
| **serum_ag_colloids (Gobbato SERS)** | Ag-SERS 785 nm | ~271 spiked + 15 baseline + 265 pure + 20 uricase + 73 isotopic | serum-spike recoverability + pure-SERS transfer + uricase depletion (`spike_validation/spike_lib.py`) | Gobbato 2025 (same archive as the training Raman) |
| **european_multi_instrument_adenine (ILS)** | Ag/Au-SERS, 532/785 nm | thousands (dose × 15 labs × 4 substrates) | adenine dose-response / inter-lab robustness | European inter-laboratory adenine study |
| **ergothioneine_serum** | SERS calibration; cAg 785 nm | ~55 (11 conc.) | ergothioneine dose-response | **Fornasaro 2024, Zenodo 10.5281/zenodo.13785349** (same group) |
| **Gobbato donor sera** | Ag-SERS 785 nm | 81 | serum-SERS characterization (demo Page 5) | Gobbato 2025 archive |

These are the datasets Parts 9–10 exercise. All are **SERS or biological Raman** — by
design they test the atlas, they never define it.

---

## 4. UNUSED (by the atlas) — legacy engine + unwired data

~20 further datasets exist on the volume but feed only the **legacy embedding engine** or
nothing. None touches the frozen NMF. Representative (full list in the audit sweep):

- **Legacy Stage-A/B / base_4 SERS** (all UNUSED by atlas): `sers_metabolite_63` (633 nm,
  PMC6989628) + its derivative `metabolite_sers63_support` (**overlap/duplicate**),
  `adenine_sers_control` (bAgNPs 785), `ag_flakes_metabolites_23` (peak-only),
  `hcc_serum`, `cca_hcc_lm_serum_sers`, `diabetes_plasma_ev_sers`, `shine_ev_sers`,
  `small2023_ev`, `cspp_serum`, `serum_protocol_comparison`, `nature_serum_sers`,
  `otc_drugs`.
- **Never wired / unextracted:** `ovarian_plasma_raman_sers` (has a Raman half!),
  `single_vesicle_ev_raman` (`.rar`), `stroke_urine_sers`, `coeliac_faecal_sers`,
  `mycoplasma_na_sers`, `ucla_saliva_sev_gc`.
- **Provenance ghosts:** `serum_ag_colloids_grounding/` and
  `serum_ag_colloids_literature_grounding/` are **0 files on disk** yet a processed
  derivative (368 spectra / 64 classes) persists in `embedding_v5_full/` — cannot be
  re-derived from raw. **Dangling code references** to absent datasets also exist
  (`fornasaro_raman4clinics_3572359`, `tumor_purine_secretome_sers`,
  `tear_dopamine_sers_support`, …) — these are stale, not missing training data.

**Notable unused Raman:** `ovarian_plasma_raman_sers` contains a genuine Raman half and
`single_vesicle_ev_raman` is single-vesicle Raman — both are *candidate* future atlas
inputs (pure/near-pure Raman) that are currently not ingested. Everything else unused is
SERS (correctly excluded).

---

## 5. Grounding verdict

- **Training is exactly what it should be:** three pure-Raman sources, SERS rigorously
  excluded by code assertion. Confirmed reproducible (Part 4).
- **One real provenance gap:** the amino-acid grounding sheet has no citation or
  instrument record. Recommend documenting or re-sourcing it (does not require an atlas
  rebuild unless the spectra change).
- **One naming caveat:** "literature corroboration" of themes derives from a
  self-authored seed pack, not external literature — the axis *labels* (not the
  representation) inherit that softness.
- **The 20-dataset SERS zoo is correctly firewalled** from the representation; it is
  available for validation/observation-model work, not for fitting.

The frozen biochemical coordinate system is grounded in a clean, Raman-only, mostly
well-cited pure-compound corpus, with the caveats above fully surfaced rather than hidden.
