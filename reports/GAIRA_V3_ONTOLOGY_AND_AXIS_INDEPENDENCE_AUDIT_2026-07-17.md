# GAIRA V3 — Ontology & Axis-Independence Audit

**Date:** 2026-07-17 · **Ontology:** GAIRA Biochemical Ontology v1 (`config/biochemical_ontology_v1.yaml`)
**Purpose:** before building global coordinates, audit whether the 11 v11 axes are independently grounded or inherited from a lower-dimensional legacy 8-axis grounding.

---

## Legacy 8-axis → v11 mapping (from `config.LEGACY8_TO_V11`, mirrored in the ontology)

| Legacy axis (8) | v11 children | Kind | Reference analytes (RamanBioLib dominant-axis) |
| --- | --- | --- | --- |
| protein_backbone | G06 | 1:1 | 81 |
| nucleic_acid_backbone | G04 | 1:1 | 31 |
| glycan_carbohydrate | G05 | 1:1 | 25 |
| pyrimidine_nucleotide | G03 | 1:1 | 13 |
| aromatic_amino_acid | G07 | 1:1 | 12 |
| **purine_nucleotide** | **G01 + G02** | **split** | 12 (shared pool) |
| **membrane_lipid** | **G08 + G09** | **split** | 25 (shared pool) |
| **redox_metabolite** | **G10 + G11** | **split** | 3 (shared pool) |

5 of 8 legacy axes map 1:1; **3 legacy axes split into 6 v11 children**. Apparent 11-dimensionality is inherited from **8 independent source dimensions** — the split children are proportional divisions of one shared legacy signal and are **not independently resolvable** at v11 resolution.

---

## Per-axis grounding status

| Axis | Status | Basis |
| --- | --- | --- |
| G04 Nuc-phosphate | **independently_grounded** | 1:1 legacy, 31 analytes, PO2 motif |
| G05 Glycan | **independently_grounded** | 1:1, 25 analytes, glucose MSS anchor |
| G06 Protein | **independently_grounded** | 1:1, 81 analytes (largest family) |
| G03 Pyrimidine | **partially_grounded** | 1:1, 13 analytes, but no curated MSS analyte |
| G07 Aromatic | **partially_grounded** | 1:1, 12 analytes, rests on one narrow 1003 cm⁻¹ motif |
| G01 Purine-nuc | **derived_split** | shares 12-analyte purine pool with G02 |
| G02 Purine-met | **derived_split** | shares 12-analyte purine pool with G01; substrate-confounded |
| G08 Lipid | **derived_split** | shares 25-analyte lipid pool with G09 |
| G09 Sterol | **derived_split** | shares 25-analyte lipid pool with G08 |
| G10 Redox | **insufficiently_grounded** | shares 3-analyte redox pool with G11; high dynamic range |
| G11 Metabolite | **insufficiently_grounded** | shares 3-analyte redox pool with G10 |

**Independently or partially grounded: 5 axes (G03–G07). Derived split / insufficiently grounded: 6 axes** (the three split families). No fabricated independence.

### The three audited unresolved families
- **Purine split (G01/G02):** one 720–740 cm⁻¹ ring-breathing signal; nucleotide vs metabolite cannot be separated by the 8-axis grounding. Adenine calibration exercises G01; hypoxanthine/uricase exercise G02 — but the split itself is inherited.
- **Lipid split (G08/G09):** 1440 cm⁻¹ CH₂ is shared; sterol-specific 548 cm⁻¹ is the only separator when it co-fires.
- **Redox split (G10/G11):** the smallest pool (3 analytes). Ergothioneine calibration exercises G10 strongly, but G10/G11 remain a proportional split.

---

## Consequences for global coordinates
- The global calibration standardizes all 11 axes, but **6 axes are not independent measurements** — their coordinates co-move with their split siblings by construction.
- The UI exposes `grounding_status` per axis (Ontology tab + Axis Coverage). NA is preserved (never shown as 0) for analyte counts on split axes.
- **Recommendation for a future ontology v2:** consider merging each split pair into a single grounded axis (8 grounded axes) unless independent reference spectra are acquired to resolve them; or add hierarchical structure (purine → {nucleotide, metabolite}).

The 11-axis ontology is the first operational interpretable system, **not** assumed final.
