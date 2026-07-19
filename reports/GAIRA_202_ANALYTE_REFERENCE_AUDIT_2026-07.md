# GAIRA 202-Analyte Reference Audit

**Date:** 2026-07 · Read-only. Sources: `streamlit_apps/gaira_demo/data/grounding_molecule_{bsv,index}.csv`, `grounding_molecule_spectra.parquet`, `/Volumes/SSD_Rad/GAIRA_DATA/raw/ramanbiolib/`.

## What the "202" is
The demo's "grounding molecule" table is **digitized RamanBioLib** (a published Raman spectral search library for biological molecule identification; DOI 10.1002/jrs.1734; RamanBioLib README states **"database of 140 components"**).

| Quantity | Value |
| --- | --- |
| Rows in `grounding_molecule_bsv.csv` / `_index.csv` | **202** |
| Unique `id` | 202 (unique) |
| **Unique `component` (chemical) names** | **141** |
| Duplicate rows (same chemical, different measurement) | **61** (39 names repeat) |
| — of which same-substrate true duplicates | 6 |
| — of which same analyte on a different substrate/laser | 33 |
| Raw spectra available? | **Yes** — `grounding_molecule_spectra.parquet` = 272,902 points across all 202 ids (byte-identical to DB `reference_spectrum_points`) |
| BSV generated from raw or imported? | **Generated from raw** (`build_demo_assets.py`); **0 precomputed-only** |

**So "202 analytes" is not 202 chemicals.** It is **~141 unique reference compounds** represented by 202 (compound × substrate × laser) measurement rows. Examples of duplication: `melanin` ×4 (785/632.8/514.5/457.9 nm); ~20 proteins (trypsin, albumin, ferritin, ubiquitin…) ×2–3 (glass-slide Raman + metal-ring SERS-ish); fatty acids ×2 (785 + 532 nm on CaF₂).

## Entity types (index `type`, 21 categories)
Proteins 69, Lipids/FattyAcids 21, Lipids/Triglycerides 20, AminoAcids 13, Saccharides/Monosaccharides 12, PrimaryMetabolites 10, NucleicAcids 8, Proteins/Hemeproteins 6, Lipids/Sterols 5, Pigments 5, Lipids/Hormones 5, Saccharides/Disaccharides 5, … (family rollup: Proteins 76, Lipids 57, Saccharides 30, AminoAcids 13, PrimaryMetabolites 10, NucleicAcids 8, Pigments 5, Vitamins 2).
- **"Proteins (69)" are individual named proteins/enzymes** (trypsin, pepsin, papain, ferritin, ubiquitin…), NOT biochemical classes — but with duplication, unique protein chemicals ≪ 69.
- Triglycerides/fatty acids are unique named chemicals (triolein, trilinolein, 12-methyltetradecanoic acid…), several duplicated.

## Modality & substrate
- **No explicit modality column.** Inferred from `sample_substrate`: CaF₂ slide 60(+18), Glass slide 51 ⇒ **spontaneous Raman**; Metal rings 21 / Metal ring 17 / Metal discs 14 / gold-coated ⇒ SERS-ish; NaN 17.
- RamanBioLib source metadata: `raman_technique` = Spontaneous (142) / Fourier Transform (54). **It is a Raman reference library — NOT SERS.**
- Lasers: 785(61), 1064(55), 532(50), 488(29), 514.5(3), 632.8/457.9/850/633(1 each) — **multi-instrument digitized literature**, not one acquisition setup.

## 8-axis ontology & 8→11 mapping
The 8 legacy axis columns: `membrane_lipid, protein_backbone, aromatic_amino_acid, purine_nucleotide, pyrimidine_nucleotide, glycan_carbohydrate, redox_metabolite, nucleic_acid_backbone`.
`LEGACY8_TO_V11` (config.py) **replicates** a legacy value across children where it splits: membrane_lipid→(G08,G09), purine_nucleotide→(G01,G02), redox_metabolite→(G10,G11); the other 5 map 1→1. Code comment: *"not a scientifically rigorous projection."* So **6 of 11 axes are inherited splits**, not independent.

## Defensible answer to "how many analytes / how many map to each axis"
- **141 unique reference compounds** (202 measurement rows).
- Per-axis defensible unique-analyte counts (dominant legacy axis → resolved v11): Protein 81-rows, Nuc-phosphate 31, Glycan 25, Pyrimidine 13, Aromatic 12 are resolved; the purine/lipid/redox pools (12/25/3) cannot be split to a single child (NA at v11 resolution).
- Modality of the 202 rows: **133 Raman / 52 SERS-likely / 17 unknown** (by substrate heuristic); the authoritative source (RamanBioLib) is **spontaneous Raman**.

## Caveats
Do NOT cite "202 analytes" without stating: it is 202 digitized RamanBioLib rows = ~141 unique Raman reference compounds (multi-laser, spontaneous Raman), with 6 of 11 GAIRA axes inherited by proportional split.
