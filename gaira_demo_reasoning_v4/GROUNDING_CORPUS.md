# GAIRA Grounding Corpus — the whole thing, in its entirety

*What actually grounds GAIRA, where every spectrum comes from, and which parts train
the frozen atlas vs. which parts are only projected through it. Verified July 2026
against the raw volume, the atlas dataset card (`results/v5_rebuild/foundation/tables/
raman_dataset_card.json`) and the phase-2 input manifest.*

There are **two tiers** and the distinction is the whole point of the architecture:

1. **The atlas grounding corpus** — *pure Raman only.* These 375 spectra are the ONLY
   data the NMF basis was fitted on. This is "the grounding corpus" in the strict sense.
2. **The reference / calibration / application corpora** — SERS, serum, dose series and
   biological cohorts. **None of these touched the fit.** They are *projected through*
   the frozen atlas as validation, calibration and demonstration.

The frozen atlas fingerprint is `09ed804a40836f4a05a91ba10900cded`. Nothing below
changes it; adding SERS/serum/biological data is deliberately kept OUT of the fit.

---

## TIER 1 — The atlas grounding corpus (pure Raman; trains the NMF basis)

- **375 spectra · 167 unique analytes**, all pure-compound Raman
- Window **450–1800 cm⁻¹**, 2 cm⁻¹ grid → **676 bins**; ASLS baseline + Savitzky–Golay + L2
- Representation **NMF, k = 24**, seed 0, explained variance **0.712**
- SERS, serum, DART are **excluded by design** (they are observation modalities, not the
  biochemical reference frame)

| Source | Spectra | Analytes | What it is |
|---|---:|---:|---|
| **RamanBioLib** | 202 | 141 | Digitized public reference library, 9 excitations (785/1064/532/488/514.5/632.8/457.9/850/633 nm). Broad chemistry: lipids, fatty acids, triglycerides, sugars, nucleobases, amino acids, cofactors. |
| **Gobbato Raman metabolites** | 153 | 51 | B&WTek i-Raman Plus 785 nm pure-metabolite powders (Gobbato 2025). The **clinical serum metabolite panel**. |
| **Amino-acid grounding** | 20 | 20 | Pure amino-acid + key-metabolite Raman references (see list below). |
| **UNION (deduped by canonical analyte)** | **375** | **167** | 141 + 51 + 20 = 212 source-analyte entries → 167 unique (45 cross-source overlaps merged). |

### The 20 amino-acid grounding analytes (`raw/amino_acid_raman_grounding/aa.xlsx`)
Valine · Glutamic acid · L-glutamate · Leucine · Phenylalanine · Proline · Alanine ·
Arginine · Aspartate · Glycine · Histidine · Methionine · Serine · Glutathione ·
Glucose · Urea · Malic acid · Carotene · Albumin · Tryptophan
*(the canonical amino-acid panel plus glutathione, glucose, urea, malic acid, carotene
and albumin — the small-molecule backbone of serum biochemistry).*

### Gobbato Raman ↔ RamanBioLib cross-match (why Gobbato was added)
Of Gobbato's 51 Raman metabolites, **30 already exist in RamanBioLib** (deduped) and
**21 are unique to Gobbato** — and those 21 are precisely the serum-critical metabolites
RamanBioLib lacks:

> **acetyl-CoA, creatinine, cysteine, ergothioneine, fructose, galactose, glucose,
> hydroxyproline, hypoxanthine, isoleucine, lactate, leucine, mannose,
> N-acetylglucosamine, phosphate, phosphatidylinositol, riboflavin, urate, urea,
> xanthine, methionine.**

RamanBioLib is broad but lipid/fatty-acid-heavy; Gobbato contributes the clinical
metabolite core (glucose, urate, the oxopurines hypoxanthine/xanthine, creatinine,
lactate, sugars). Together they cover both the general chemistry and the serum panel.

**No pure-Raman analyte is missing.** All 51 Gobbato Raman analytes are ingested; the
grounding corpus is complete for its Raman-only design.

---

## TIER 2 — Reference / calibration / application corpora (projected, never fitted)

Every item here is run *through* the frozen atlas (component coords → BSV → MSS). None
of it changed the basis.

### 2a. Matched pure Ag-SERS reference (the observation twin)
| Source | Spectra | Analytes | Use |
|---|---:|---:|---|
| **Gobbato SERS metabolites** | 265 | 53 | Pure-analyte Ag-SERS, 785 nm, same instrument/group as the Raman powders. **51 analytes have BOTH a pure Raman AND a pure Ag-SERS spectrum.** |

→ `reference_artifacts/matched_raman_sers_pairs.json` (`tools/build_matched_pairs.py`):
all 51 pairs projected through the frozen engine; **median coordinate cosine 0.42**;
the oxopurines (hypoxanthine 0.84, xanthine 0.81) preserve their Raman signature on
silver, glucose (0.20) / amino acids / uracil (0.06) are scrambled. Surfaced on **Page 7
· Section E** as the empirical seed of an observation model (Ag, descriptive, not applied
inside the engine). DNA and RNA are SERS-only (no Raman twin).

### 2b. Calibration dose/depletion series (Page 4)
| Experiment | Source | Provenance |
|---|---|---|
| **Adenine dose–response** (0–9 µM, cAg/sAg/cAu/sAu, 532/785 nm) | `raw/european_multi_instrument_adenine/ILSdata.csv` | **European inter-laboratory (ILS) round-robin, 15 labs** — a metrology study, **NOT** Gobbato. |
| **Ergothioneine dose–response** (0–2 µM, cAg @ 785 nm) | `raw/ergothioneine_serum/ERG_calibration.csv` | **Fornasaro et al. 2024, "Detection and quantification of ergothioneine in human serum using SERS", Zenodo 10.5281/zenodo.13785349** (same Trieste/Bonifacio group as Gobbato; byte-identical to the user-supplied download). |
| **Uricase depletion** (serum ± uricase) | Gobbato archive → `dataset uricase/` | **Gobbato 2025**, DOI 10.1007/s00216-025-06192-5 (PMC12680727). |

### 2c. Serum stress-test datasets (Page 5) — all Gobbato 2025
| Folder in the archive | Spectra | Use |
|---|---:|---|
| SERS spiked serum Merck | 271 | 53-analyte serum spike-in (recoverability tiers). |
| SERS serum Merck | 15 | Unspiked serum baseline. |
| **donors serum SERS** | **81** | **Real healthy-donor sera → `biological_artifacts/gobbato_donor_sera.json`** (Page 5 · B++). Mean BSV purine-dominated — reproduces the paper's PC1≈70% urate+hypoxanthine. |
| isotopic | 73 | Pure UA / ¹³C-UA ± HSA (out-of-domain projection). |
| SERS metabolites for fitting | 31 | Hypoxanthine + free/HSA uric acid (paper Fig 9). |

All from one archive: `raw/serum_ag_colloids/dataset_spectral_data.zip` (914 .txt, 785 nm
B&WTek). The paper is **Gobbato, Fornasaro, Sergo, Bonifacio (2025), *Anal. Bioanal.
Chem.* — PMC12680727**.

### 2d. Biological cohorts (Page 6) — sanitized V6 artifacts
| Cohort | Domain · modality | Units | Source |
|---|---|---:|---|
| COVID serum Raman | serum · Raman | 465 | `raw/covid_serum_raman` |
| HCC serum SERS | serum · SERS | 144 | `raw/hcc_serum` |
| Diabetes plasma-EV SERS | ev · SERS | 63 (patient) | `raw/diabetes_plasma_ev_sers` |
| SHINE EV-SERS (hepatotox) | ev · SERS | 720 | `raw/shine_ev_sers` |
| EV single-vesicle Raman (small2023) | ev · Raman | 600 | `raw/small2023_ev` |
| **Gobbato donor sera** | serum · SERS | **81** | Gobbato archive (see 2c) |

Built by `tools/build_biological_v6.py`; every value is a genuine `GAIRAEngine.infer`
output; NO demographics; anonymised IDs; committed so the demo runs on a fresh checkout.

### 2e. On disk but deliberately NOT incorporated
- `raw/sers_metabolite_63/` (63-metabolite Ag-SERS, 633 nm) — SERS-only, no matched Raman.
- `raw/ag_flakes_metabolites_23/`, `raw/sers24_metabolite_support/` — metadata/supplement only.
- `cca_hcc_lm_serum_sers`, `ovarian_plasma_raman_sers` — present, not yet wired for V6
  (shown as UNAVAILABLE on Page 6, no fabricated output).

---

## The pipeline in one line

Reference **pure Raman** (Tier 1, 375 spectra) → ASLS+SG+L2 on the 450–1800 cm⁻¹ grid →
**NMF k=24** → 24 basis motifs (components) + activations → *frozen*. Any query (a serum
SERS spectrum, a dose point, a donor) → same preprocessing → **NNLS onto the frozen
basis** → 24 coords → **ontology W** → 11 biochemical themes (**BSV**) and **MSS** motif
layer → domain context (OOD/confidence). Tier-2 data only ever enters at "any query" — it
is measured against the frame, never allowed to redefine it.

## Guardrail (honoured)
The atlas is Raman-only and frozen. Adding SERS/serum would break fingerprint
`09ed804a…` and every downstream validation, and would conflate the *observation*
modality with the *biochemical reference frame*. So SERS/serum/biological data is wired as
Tier-2 (projected/validated/demonstrated) — never re-fitted into Tier-1. All 51 Gobbato
pure-Raman analytes were already in Tier-1, so no re-fit was needed or done.
