# GAIRA Scientific Reasoning Demo v2 (migration-hardened)

A Streamlit-based scientific reasoning demo for Raman/SERS spectra.

> **v2 vs v1.** v2 is a self-contained, portable successor to
> `../gaira_demo_reasoning_v1` (which is preserved unchanged). Same scientific
> engine; hardened data resolution. All paths resolve via
> `gaira_core/paths.py` (env overrides → candidate mounts → bundled data), the
> app shows an explicit **data-source banner** (REAL / DEGRADED / PLACEHOLDER)
> instead of degrading silently, and the tiny grounding/calibration CSVs are
> **bundled** in `data/legacy/` so the calibration and biochemical-space tabs
> run with no external drive. See [MIGRATION_HARDENING.md](MIGRATION_HARDENING.md).

**GAIRA** = GenAI Raman Analysis. This demo shows how GAIRA was constructed
and why its outputs are scientifically grounded. It is *not* a black-box
classifier; it is an evidence-grounded biochemical reasoning engine that
converts Raman/SERS spectra into interpretable **biochemical state vectors
(BSVs)**, with evidence, confidence, caveats, and substrate-aware
interpretation.

The core message:

> GAIRA transforms ambiguous Raman/SERS spectra into evidence-grounded
> biochemical state representations.

---

## Install & run

```bash
cd gaira_demo_reasoning_v2

# check what data will resolve on this machine (no server started)
python selfcheck.py

# portable launcher (finds repo .venv or python3, prints resolution, launches)
./run_demo.sh

# ...or directly, using the repo's existing .venv
streamlit run app.py
```

Point the demo at your data (optional — bundled data works without it):

```bash
export GAIRA_DATA_ROOT=/Volumes/YourDrive/GAIRA_DATA   # must contain raw/ and processed/
./run_demo.sh
```

If you don't have a virtual environment yet:

```bash
python -m venv .venv && source .venv/bin/activate
pip install streamlit pandas numpy plotly scikit-learn scipy
pip install umap-learn   # optional: UMAP for the 11-axis space tab (PCA fallback otherwise)
streamlit run app.py
```

Required: `streamlit`, `pandas`, `numpy`, `plotly`, `scipy`, `scikit-learn`.
Optional: `umap-learn` (falls back to PCA if missing).

---

## What the demo shows

Three top-level modes:

### 1. How GAIRA Works
Answers *"Why should I trust the BSV radar?"*

- **Construction Overview** — the GAIRA pipeline diagram, end-to-end.
- **Grounding Corpus Map** — sources feeding the grounding layer, with
  evidence tiers (Tier 1 direct spectral, Tier 2 supporting).
- **11-Axis Biochemical Space** — PCA/UMAP of reference molecules colored
  by dominant 11-axis BSV family.
- **MSS / Motif Explorer** — pick a molecule, see anchors / supports /
  anti-evidence regions and its 11-axis contribution profile.
- **Collision Viewer** — paired molecule comparisons with shared/unique
  bands and collision score; the place GAIRA admits ambiguity.
- **Physics-Aware Atlas** — interactive wavenumber atlas with
  assignments, ambiguity notes, substrate sensitivity, and GAIRA's
  treatment for each region.
- **End-to-End Workflow** — full pipeline run on a chosen scenario:
  spectrum → preprocessing → primitives → MSS → motifs → substrate
  adjustment → BSV → interpretation report.

### 2. Calibration Evidence
Answers *"Do GAIRA axes respond correctly to known chemistry?"*

- **Ergothioneine Dose Slider** — slider over concentration; G10
  sulfur/thiol/redox axis rises monotonically.
- **Adenine Detection** — concentration-conditioned BSV, with collision
  caveat for purine-metabolite overlap.
- **Uric Acid / Isotope Validation** — spike, ¹⁵N isotope, uricase
  depletion, with substrate-sensitive caveats for carotenoid overlap in
  serum SERS.

### 3. Biological Pilot Interpretation
Answers *"Can GAIRA organize real biological spectra into interpretable
state changes?"*

- **Serum Liver Disease** — Healthy / HCC / CCA / Liver-Metastases
  cohort comparison.
- **EV Diabetes** — Normal-Weight Control / Overweight Diabetic /
  Normal-Weight Diabetic.
- **SHINE Liver Injury / Hepatotoxicity** — Day-0 / low / mid / high
  dose progression.

All biological interpretations use *consistent-with* language. GAIRA does
not classify disease.

---

## Architecture

```
gaira_demo_reasoning_v1/
├── app.py                       # main Streamlit app — three modes
├── gaira_core/
│   ├── __init__.py
│   ├── config.py                # 11-axis BSV definition, labels, palette, atlas regions
│   ├── data_loader.py           # real-data-first loaders with demo-flagged fallback
│   ├── plotting.py              # radar, spectrum, ΔBSV bar, biochemical space, atlas, pipeline
│   ├── preprocessing.py         # ASLS baseline + Savitzky–Golay smoothing + L2 normalize
│   ├── primitive_extraction.py  # peak detection, prominence, width
│   ├── mss_scoring.py           # curated MSS for ~11 reference molecules
│   ├── motif_scoring.py         # 11 class-level motifs (one anchor per axis chemistry)
│   ├── substrate_physics.py     # substrate-aware multiplicative adjustments + caveats
│   ├── bsv_projection.py        # motif/MSS → 11-axis noisy-OR projection
│   ├── evidence_synthesis.py    # per-axis evidence cards + caveats + confidence
│   └── report_builder.py        # orchestrates everything → GAIRAReport dict
├── data/
│   ├── calibration/
│   ├── pilots/
│   ├── grounding/
│   └── cached/                  # drop real cached GAIRAReport dicts here later
├── assets/
└── README.md
```

The **11 BSV axes** are the canonical interpretation surface:

| ID  | Axis                          | Short label    |
| --- | ----------------------------- | -------------- |
| G01 | purine_nucleotide             | Purine-nuc     |
| G02 | purine_metabolite             | Purine-met     |
| G03 | pyrimidine_nucleotide         | Pyrimidine     |
| G04 | nucleic_acid_phosphate        | Nuc-phosphate  |
| G05 | glycan_carbohydrate           | Glycan         |
| G06 | protein_peptide_backbone      | Protein        |
| G07 | aromatic_residue              | Aromatic       |
| G08 | lipid_acyl_membrane           | Lipid          |
| G09 | sterol_neutral_lipid          | Sterol         |
| G10 | sulfur_thiol_redox            | Redox          |
| G11 | metabolic_small_molecule      | Metabolite     |

---

## Data: real vs placeholder

The demo runs out-of-the-box without any external data files. Every loader
in `gaira_core/data_loader.py` tries real cached data first, then falls
back to a clearly-labelled demo placeholder. Whenever placeholders are
used, the UI renders a purple **`Demo placeholder — replace with cached
GAIRA output`** badge in that section.

### What is wired to real data (post-audit 2026-06-18, SSD_Rad pass)

| Section                          | Loader                                          | Source file (SSD_Rad is source of truth)                                                                                                                      | Status                                                       | Badge |
| -------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------ | ----- |
| 11-Axis Biochemical Space        | `load_reference_points`                         | `streamlit_apps/gaira_demo/data/grounding_molecule_bsv.csv` + `grounding_molecule_index.csv`                                                                 | **REAL** (legacy 8-axis remapped to 11-axis)                 | none  |
| Ergothioneine dose slider         | `load_ergothioneine_dose`                       | `streamlit_apps/gaira_demo/data/ergothioneine_dose_response.csv`                                                                                             | **REAL** (legacy 8-axis remapped to 11-axis)                 | none  |
| **Adenine detection**            | `load_adenine_calibration` → `_load_adenine_real` | `/Volumes/SSD_Rad/GAIRA_DATA/raw/adenine_sers_control/Adenine_bAgNPs_*.CSV` (6 concentrations)                                                              | **REAL** (raw cp1252 CSVs parsed; live `build_report(substrate='Ag colloid SERS')` per concentration) | none  |
| Uric acid validation             | `load_uric_acid_validation`                     | `streamlit_apps/gaira_demo/data/calibration_conditions.csv` + `calibration_delta_bsv.csv`                                                                    | **REAL** (3 SAEL contrasts; legacy 8-axis remapped)          | none  |
| **Grounding corpus map**         | `load_grounding_corpus` → `_load_grounding_corpus_real` | `/Volumes/SSD_Rad/GAIRA_DATA/processed/.../gaira_evidence_warehouse_grounding_backbone_v1/tables/warehouse_source_registry.csv` + `grounding_peak_support_summary.csv` | **REAL** (43 sources; 12 reference_molecule + 30 disease_or_stress_paper + 1 serum_grounding) | none  |
| SHINE liver injury                | `load_pilot_cohorts('shine_liver_injury')` → `_load_shine_real` | `/Volumes/SSD_Rad/GAIRA_DATA/.../pilot3_shine_single_set_day0_day2/tables/class_mean_bsv_day0_day2.csv` + `pilot3_shine_ev_sers/tables/per_sample_bsv.csv` | **REAL** (Day 0 + Day 2 × C0/C10/C20/C40; Day-0 n=0 bug fixed; autoresearch 8-axis remapped) | none  |
| **EV diabetes pilot**            | `load_pilot_cohorts('ev_diabetes')` → `_load_ev_diabetes_real` | `/Volumes/SSD_Rad/GAIRA_DATA/.../pilot2_target_validation_v1/tables/class_mean_bsv.csv`                                                                | **REAL** (2 cohorts: Impact n=39, Strong-D n=24; project-specific labels) | none  |
| **Serum liver disease pilot**    | `load_pilot_cohorts('serum_liver')` → `_load_serum_liver_real` | `/Volumes/SSD_Rad/GAIRA_DATA/.../pilot4_1_cca_hcc_lm_serum_patient_level/tables/patient_level_bsv.csv`                                                | **REAL** (213 patients aggregated to 4 cohorts: HA n=48, CCA n=67, HCC n=49, LM n=49) | none  |
| Per-axis family analyte counts    | `load_family_counts`                            | curated demo counts                                                                                                                                          | **PLACEHOLDER** (no `per_axis_grounding_counts.csv` in corpus) | purple|
| End-to-end workflow spectra       | `synth_reference_spectrum`                      | Gaussian-bump synthesis                                                                                                                                      | **SYNTHETIC by design** (labelled in every plot title)       | n/a   |

Bold = newly real in this round (round 3 / SSD_Rad pass).

Notes:
- **8→11 axis remap** is a demo-grade projection (legacy or autoresearch 8-axis values duplicated equally across their v11 children). Two remap tables live in [gaira_core/config.py](gaira_core/config.py): `LEGACY8_TO_V11` (for old ergothioneine / uric-acid / molecule files) and `AUTORESEARCH8_TO_V11` (for SHINE / EV diabetes / serum liver). Both are flagged in UI captions wherever applied.
- **Adenine** real BSVs are computed live on each cache-cold demo load — first load takes a few seconds (six `build_report` calls on parsed raw SERS spectra), subsequent loads are instant via `@st.cache_data`.
- **EV diabetes labels** "Impact" and "Strong-D" are project-specific (per autoresearch pilot2 documentation). They are NOT a generic Normal vs Diabetic split and the demo says so verbatim.
- **No isotope (¹⁵N / ¹³C) uric-acid data exists in the GAIRA corpus.** A previous fabricated isotope condition has been removed.
- **No SHINE Day 3 or Day 7 data exists.** A previous fabrication of those days has been removed. Day 1 (n=4) exists in the per-sample file but not in the per-class means file; it is flagged as a caveat, not silently merged.
- **Uric acid uricase-depletion** contrast records a SAEL *disagree* verdict on several axes; this is surfaced honestly rather than hidden.

### How to wire a placeholder to real data

After the SSD_Rad pass only one section is still placeholder:

| Placeholder              | File to create                              | Required columns                                          |
| ------------------------ | ------------------------------------------- | --------------------------------------------------------- |
| Family analyte counts    | `data/cached/family_counts.csv`             | `axis`, `n_analytes` (one row per of the 11 BSV axes)     |

For everything else, the demo reads directly from `/Volumes/SSD_Rad/GAIRA_DATA/`.
If the SSD_Rad volume is not mounted at demo time, every loader gracefully
falls back to its honest placeholder and flips the purple badge back on.

---

## GAIRAReport schema

Every scenario in the End-to-End Workflow produces a dict with this shape:

```python
{
    "sample_id": str,
    "title": str,
    "domain": str,                  # serum / calibration / EV / etc.
    "substrate": str,               # "Raman" | "Ag colloid SERS" | etc.
    "preprocessing": str,           # human-readable summary
    "spectrum": {"wavenumber": [...], "raw_intensity": [...], "processed_intensity": [...]},
    "features": {"anchors": [...], "support": [...], "anti_evidence": [...], "n_peaks": int},
    "motif_scores_raw": {motif_id: float},
    "motif_scores_adjusted": {motif_id: float},
    "substrate_events": [{"rule_id", "motif_id", "before", "after", "multiplier", "note"}],
    "mss_fires": {mol_id: {"fire", "anchor", "support", "anti"}},
    "bsv": {axis_id: float},        # 11-axis biochemical state vector
    "delta_bsv": {axis_id: float},  # vs reference, if provided
    "top_axes": [{"axis", "value", "direction", "evidence_strength"}],
    "evidence": [{"axis", "evidence_type", "bands", "summary", "confidence", "motif_id"}],
    "caveats": [{"type", "summary", ...}],
    "confidence": {"overall", "substrate_sensitivity", "molecular_specificity"},
}
```

`gaira_core/report_builder.py:build_report` is the single entry point.

---

## Scientific guardrails (encoded in the code)

- **Class-level by default.** Molecule-level claims require co-band
  corroboration in the evidence layer.
- **"Consistent with" language.** No "this proves molecule X" anywhere in
  the templates.
- **Substrate-aware corrections.** Ag-SERS purine amplification is
  explicitly dampened; thione/thiol is mildly boosted; carotenoid overlap
  in serum SERS soft-dampens G02 specificity.
- **Ambiguity routing.** Adenine/uric-acid collision, lipid/sterol
  collision, and protein/aromatic overlap all produce automatic caveats
  whenever co-firing is detected.
- **Anti-evidence aware.** MSS scoring subtracts anti-evidence terms
  (e.g. uric-acid 725 cm⁻¹ anti-evidence dampens adenine call).

---

## Assumptions / caveats

- The motif and MSS sets in this demo are intentionally small (curated 11
  motifs, ~11 reference molecules) so the demo is auditable end-to-end.
  Production GAIRA's `src/gaira/base3/mss_engine.py` is far richer.
- The synthesised reference spectra are Gaussian-bump approximations of
  curated anchor/support/anti positions — they are *plausible* but not
  empirically measured. Replace with real cached spectra under
  `data/cached/` to ground every visualization in lab data.
- The 8→11 BSV remap for legacy data is demo-only.
- UMAP is preferred for the biochemical-space tab; PCA fallback runs if
  `umap-learn` is not installed.

---

## Next steps to make this production-real

1. Drop real cached GAIRAReport dicts into `data/cached/` keyed by
   scenario name and update `data_loader.build_report`-equivalent loaders
   to read them first.
2. Replace synthesised reference spectra with the cached spectra used in
   the actual MSS build (`gaira_base_3` MSS spectra registry).
3. Connect the substrate-rule list to GAIRA's full substrate physics
   library (47 effects per memory) rather than the curated 5 used here.
4. Wire the real grounding corpus registry CSV (RamanBioLib, Gobbato,
   NIHMS1547448, etc.) into the Grounding Corpus Map tab.
5. Add `DART-Met Dynamic Mode` as Mode 4 once spec is ready.
