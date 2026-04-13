# GAIRA LFM v1 — Phase 7A: Literature-BSV + Radar Plots + GAIRA-Native Graph

## What Phase 7A Adds

This phase transforms the text-query app from "document retrieval + LLM" into a GAIRA-native biochemical reasoning interface:

1. **Literature-grounded BSV profiles** loaded from GAIRA landscape v4 matrices
2. **Radar plots** for single-condition and multi-condition queries
3. **Motif → Theme → BSV mapping** layer that organizes evidence through GAIRA's reasoning stack
4. **6-column trust graph** reflecting the full GAIRA reasoning flow
5. **Cleaner caveats** appropriate for text-query mode (no spectral-transfer language)

## How Literature-Grounded BSV Is Computed

BSV profiles come from pre-computed landscape v4 matrices (`outputs/landscape_v4/bsv_compositional_matrix.csv` and `bsv_delta_matrix.csv`), which aggregate literature evidence across 37 conditions and 8 BSV components.

The builder:
1. Detects condition names in the query (via alias table: "HCC" → `HCC`, "fatty liver" → `NAFLD_NASH`, etc.)
2. Looks up pre-computed compositional profiles and deltas vs healthy
3. Returns profiles for all detected conditions

This is **literature-grounded composition**, not measured spectral composition. Scores represent relative evidence support from the GAIRA corpus, not spectral intensities.

## How Motifs/Themes/BSV Are Linked

22 motifs are defined with keyword patterns, grouped into 8 themes that map 1:1 to BSV components:

| Example Motifs | Theme | BSV Component |
|---|---|---|
| CH2/CH3 deformation, cholesterol | Membrane / Lipid | membrane_lipid |
| amide I/II/III, collagen | Protein Backbone | protein_backbone |
| phenylalanine, tryptophan, tyrosine | Aromatic Amino Acids | aromatic_amino_acid |
| adenine, guanine | Purine Nucleotides | purine_nucleotide |
| PO2 stretch, C-O-C | Nucleic Acid Backbone | nucleic_acid_backbone |

For each query, the mapper scans retrieved evidence for motif keywords and builds a chain: evidence → motifs → themes → BSV components.

## How Radar Plots Are Generated

Plotly scatterpolar radar with:
- 8 BSV axes arranged radially
- One trace per detected condition (HCC red, NAFLD orange, healthy green, etc.)
- Filled polygons with 12% opacity fill
- Dark background consistent with trust graph
- Overlay for comparisons, single trace for single-condition queries

Supports:
- Single condition: "What biochemical changes are associated with HCC?"
- Comparison: "How does HCC differ from healthy?" → overlay radar
- Multi-condition: "Compare HCC, CCA, and liver metastases" → 3-trace overlay

## How the Graph Flow Was Redesigned

6-column GAIRA-native flow replacing the old 4-column structure:

```
Query → Evidence → Motifs → Themes → BSV → Output
         (lanes)    (22)     (8)     (8)
```

| Column | Content | Node shape |
|---|---|---|
| 0 - Query | User's question | Square |
| 1 - Evidence | Retrieved items, color-coded by tier | Circle |
| 2 - Motifs | Detected molecular features | Hexagon |
| 3 - Themes | Broader biochemical categories | Pentagon |
| 4 - BSV | Active BSV components (with scores) | Diamond |
| 5 - Output | BSV Profile / Answer | Star |

Edges are batched by weight into strong/normal/weak opacity tiers. 10 plotly traces total.

## Caveats Cleanup

Old (Phase 3-6):
- "Evidence is drawn from curated GAIRA documents, not the full literature..."
- "Substrate-dependent enhancement (Au vs AgNP) can change peak intensities and even reverse apparent disease-associated directions."
- "Literature assignments are not ground truth. Many papers overclaim molecule specificity from single peaks."

New (Phase 7A):
- "Raman/SERS peak assignments are many-to-many..."
- "Literature support varies in specificity..."
- "Substrate and sample preparation differences between studies may alter the expression of certain biochemical themes."

No spectral-transfer, HCC-holdout, or dataset-specific caveats unless the evidence packet specifically triggers them.

## How to Run

```bash
cd /Users/suraj/projects/GAIRA
PYTHONPATH=src streamlit run streamlit_apps/gaira_lfm_v1_text_query.py
```

## What Remains

- Embedding-based retrieval upgrade
- Spectral query integration (separate mode)
- Full corpus graph
- Evidence quality scoring / calibration
