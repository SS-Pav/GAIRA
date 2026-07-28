# Calibration & Atlas Data-Source Re-evaluation (Gobbato + inter-lab)

A forensic re-check of where GAIRA's atlas-input, calibration, and serum datasets
actually come from, triggered by the Gobbato 2025 paper. Read-only audit; no science
changed. Verified against the raw volume, the atlas corpus card, and
`spike_validation/tables/phase1_dataset_audit.csv`.

## The Gobbato paper

Gobbato, Fornasaro, Sergo, Bonifacio (2025), *"Adsorption of serum components on Ag
colloids: on the biochemical interpretation of SERS of human serum"*, Anal. Bioanal.
Chem., **DOI 10.1007/s00216-025-06192-5** — **the same paper as PMC12680727**
(pubmed 41249629). It reports pure-analyte Raman AND Ag-SERS, serum spike-ins, a uricase
depletion, and isotopically-labelled uric acid.

## One archive holds the entire Gobbato corpus

`/Volumes/SSD_Rad/GAIRA_DATA/raw/serum_ag_colloids/dataset_spectral_data.zip` (914 .txt
spectra, 785 nm, B&WTek i-Raman Plus), confirmed by its bundled `Instructions.docx`.
Contents and how GAIRA uses each:

| folder in zip | files | modality | pure/serum | analytes | in atlas? | used as |
|---|---|---|---|---|---|---|
| Raman metabolites/ | 153 | **Raman** 785 nm powders | pure | **51** | **YES** → atlas source `gobbato_raman_metabolites` | NMF atlas input |
| **SERS metabolites/** | 265 | **Ag-SERS** 785 nm | pure (buffer) | **53** | **NO** | out-of-domain `pure_sers` only |
| SERS spiked serum Merck/ | 271 | Ag-SERS 785 nm | serum spike | ~53 | NO | `spiked_serum` (Page 5) |
| SERS serum Merck/ | 15 | Ag-SERS 785 nm | serum baseline | — | NO | `serum_baseline` |
| **donors serum SERS/** | **81** | Ag-SERS 785 nm | 81 healthy-donor sera | — | NO | **UNUSED anywhere** |
| dataset uricase/ | 20 | Ag-SERS 785 nm | serum ± uricase | urate/hypoxanthine | NO | `uricase` (Page 4) |
| isotopic/ | 73 | Ag-SERS 785 nm | pure UA / ¹³C-UA ± HSA | uric acid | NO | `isotopic` |
| SERS metabolites for fitting/ | 31 | Ag-SERS 785 nm | hypoxanthine + UA | 2 | NO | partial (paper Fig 9) |

So **Gobbato's contribution to the frozen atlas is only the 153 pure-Raman powders**;
everything SERS is (correctly, by the Raman-only design) excluded — but sits on disk.

## Matched pure Raman ↔ Ag-SERS pairs — CONFIRMED (the key asset)

Inside the same zip, two sibling folders pair 1:1 by analyte, same 785 nm, same
instrument, same group:
- `Raman metabolites/Raman_pwd_<Ab>_s_0N.txt` → 51 analytes (Raman powder)
- `SERS metabolites/SERS_met_<Ab>_<conc>uM_0N.txt` → 53 analytes (Ag-SERS)

**51 analytes have BOTH a pure 785 nm Raman AND a pure 785 nm Ag-SERS spectrum** (only
DNA/RNA are SERS-only). Independently cross-checked at the analyte level: 51 of the 53
`pure_sers` analytes also appear in the atlas's Raman analyte list. **This is the
matched Raman→SERS transfer set GAIRA lacks a model for — present, but not leveraged.**

## Calibration-experiment provenance (corrected)

| experiment | raw source | Gobbato? |
|---|---|---|
| **adenine dose series** (cAg/sAg/cAu/sAu, 532/785, 0–9 µM) | `raw/european_multi_instrument_adenine/ILSdata.csv` — a **15-lab European inter-laboratory (ILS)** round-robin | **NO** |
| (separate) adenine bAgNPs LOD series | `raw/adenine_sers_control/` (Anal. Chim. Acta 2025) | **NO** |
| **ergothioneine dose series** (cAg@785, 0–2 µM) | `raw/ergothioneine_serum/ERG_calibration.csv` — **Fornasaro et al. 2024, "Detection and quantification of ergothioneine in human serum using SERS", Zenodo 10.5281/zenodo.13785349** (byte-identical to the user-provided download; same Trieste/Bonifacio group as Gobbato) | **NO** (sibling group) |
| **uricase depletion** | Gobbato zip → `dataset uricase/` | **YES** (Instructions.docx) |
| **serum spike-in (53) + baseline (15)** | Gobbato zip → `SERS spiked serum Merck/` + `SERS serum Merck/` | **YES** |
| pure-SERS reference (53) | Gobbato zip → `SERS metabolites/` | **YES** |
| isotopic uric acid | Gobbato zip → `isotopic/` | **YES** |

**Correction to earlier docs:** the adenine gradient is NOT Gobbato — it is the European
ILS adenine reference-standard study (15 labs). Ergothioneine is **Fornasaro 2024 (Zenodo
13785349)**, a sibling paper from the same group (now confirmed against the user-provided
download — byte-identical). Only the serum/uricase/pure-SERS/isotopic experiments are Gobbato.

## Non-Gobbato metabolite SERS libraries on disk (not in atlas, no matched Raman)

- `raw/sers_metabolite_63/` (+ `metabolite_sers63_support/`) — 63-metabolite Ag-SERS
  library (633 nm, NIHMS1547448). SERS-only.
- `raw/ag_flakes_metabolites_23/` — 23 metabolites on Ag flakes (SAA 2023); supplement
  docx only, no spectra extracted.
- `raw/sers24_metabolite_support/` — metadata only, no spectra.

## What is present but NOT incorporated (ranked by value)

1. **Gobbato `SERS metabolites/` — 265 spectra, 53 analytes.** The matched pure Ag-SERS
   twin of the 51 Raman analytes already in the atlas (same 785 nm, same instrument).
   Enables a **learned pure-Raman → Ag-SERS transfer** across ~51 analytes — precisely
   the cross-domain "observation model" the demo currently only gestures at (DART/Au-SERS
   page). Today used only as a throwaway out-of-domain projection.
2. **Gobbato `donors serum SERS/` — 81 healthy-donor serum SERS spectra — entirely
   unused.** A real biological serum-SERS cohort (the paper's inter-individual dataset).
3. `SERS metabolites for fitting/` (31; hypoxanthine + free/HSA uric acid) — partial.

## Follow-ups — STATUS (done July 2026)

- **Do NOT re-fit the atlas on SERS** — the Raman-only design is deliberate and correct.
  Confirmed: all 51 Gobbato pure-Raman analytes are already in the frozen atlas; nothing
  pure-Raman is missing, so no re-fit is warranted. ✅ (guardrail honoured)
- ✅ **Matched Raman↔Ag-SERS observation reference built** — `tools/build_matched_pairs.py`
  projects both sides of all 51 pairs through the frozen engine and writes
  `reference_artifacts/matched_raman_sers_pairs.json` (median coord-cosine 0.42; signature
  preserved for the oxopurines, scrambled for glucose/amino-acids). Surfaced on **Page 7 ·
  Section E** as the empirical seed of the observation layer. NOT applied inside the engine.
- ✅ **81 donor sera wired** as a genuine V6 serum-SERS characterization cohort
  (`biological_artifacts/gobbato_donor_sera.json`, single group, sanitized). Surfaced on
  **Page 5 · Section B++** — mean BSV is purine-dominated (nucleic_purine 0.27, top MSS motifs
  oxopurine + purine-ring), independently reproducing the paper's PC1≈70% urate+hypoxanthine.
- ✅ **Demo provenance corrected** — adenine = "European ILS (15-lab)"; ergothioneine =
  "Fornasaro 2024, Zenodo 13785349"; uricase/serum/pure-SERS = Gobbato 2025. Applied on
  Page 4 (per-calibrant captions) and Page 8 (data-provenance table).
- Still deferred: a *learned/validated* Raman→SERS transfer model over the 51 pairs (the
  artifact here is descriptive, not trained); the 265-spectrum SERS reference and 31 fitting
  spectra remain out-of-domain projections only.
