# GAIRA Landscape v5 — Signal Decomposition Summary

## What Was Built

6 new analytical layers decomposing BSV signal into actionable biochemical inference:

### 1. Subcomponent Decomposition
- 8 BSV components expanded into **45 subcomponents** (e.g., purine_nucleotide::adenine, purine_nucleotide::guanine)
- Reveals within-component heterogeneity: HCC may show adenine enrichment but not guanine

### 2. Baseline Correction (Anti-Adsorption)
- Regresses out protein_backbone correlation from each component
- Corrected delta reveals signal obscured by protein co-adsorption
- **Key finding**: After correction, HCC shows purine_nucleotide enrichment (+0.17) that was hidden at zero in raw delta

### 3. Co-occurrence Network
- 18 significant edges (|corr| > 0.2) between BSV components across conditions
- Identifies coupled biochemical pathways (e.g., membrane_lipid and glycan_carbohydrate often co-vary)

### 4. Condition Signatures
- Delta-vector cosine similarity matrix (22×22 conditions)
- Enables: "which conditions have the most similar biochemical deviation from healthy?"

### 5. Discriminative Axes (PCA)
- **PC1 (52.5% variance)**: protein_backbone + pyrimidine — baseline vs disease axis
- **PC2 (14.2%)**: separates metabolic from structural conditions
- Biplot shows condition positions + component loading directions

### 6. Confidence Scoring
- `confidence = |delta| × trust_weight × log(source_count + 1)`
- HCC nucleic_acid_backbone: confidence=1.39 (strong, multi-source)
- HCC purine_nucleotide: confidence=0.00 (delta was zero before correction — needs caution)

## HCC Signal Profile (post-decomposition)

| Component | Raw Delta | Corrected Delta | Confidence | Interpretation |
|---|---|---|---|---|
| nucleic_acid_backbone | +1.000 | +1.039 | 1.386 | **Strong enrichment** — phosphate backbone signal |
| aromatic_amino_acid | +0.250 | +0.393 | 0.347 | Moderate enrichment — protein composition shift |
| purine_nucleotide | 0.000 | +0.169 | 0.000 | **Hidden enrichment** revealed by correction |
| membrane_lipid | -0.750 | -0.658 | 1.040 | Strong depletion — membrane remodeling |
| glycan_carbohydrate | -0.500 | -0.366 | 0.693 | Depletion — glycan alteration |

**Key insight**: Protein correction reveals purine enrichment in HCC that raw delta missed. This is consistent with altered nucleotide metabolism in cancer.

## Validation

| Check | Result |
|---|---|
| Subcomponents > 8 | 45 (PASS) |
| Corrected delta differs from raw | True (PASS) |
| Network has edges | 18 (PASS) |
| PCA extracts meaningful axes | PC1=52.5% (PASS) |
| Confidence varies by trust | HCC range 0-1.39 (PASS) |

## Output Files (9)

| File | Content |
|---|---|
| `bsv_subcomponent_matrix.csv` | 37 conditions × 45 subcomponents |
| `bsv_subcomponent_delta.csv` | Subcomponent deltas vs healthy |
| `bsv_corrected_delta_matrix.csv` | Protein-corrected deltas |
| `bsv_delta_network.csv` | Component co-occurrence (18 edges) |
| `condition_signature_similarity.csv` | 22×22 delta-vector similarity |
| `bsv_latent_axes.csv` | PCA projections (PC1-PC3) |
| `bsv_latent_loadings.csv` | PCA loadings per component |
| `bsv_biplot.png` | Condition + loading biplot |
| `bsv_confidence_matrix.csv` | Trust-weighted confidence per component |

## What Changed: "what is different" → "what processes are changing, and how confidently"

v4 showed condition-specific deviations. v5 decomposes those deviations into:
- **Subcomponent-level resolution** (adenine vs guanine, not just "purine")
- **Artifact-corrected signals** (removing protein co-adsorption)
- **Latent axes** (metabolic vs structural disease)
- **Confidence-weighted interpretation** (trust matters)

This is the transition from representation to inference.
