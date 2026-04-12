# Phase C1.1 — Example Outputs

## Example: "What does GAIRA know about HCC?"

### Ranked Themes (with specificity)

| Theme | Final Score | Support | Specificity | Confidence | Direct | Sources | Motifs |
|---|---|---|---|---|---|---|---|
| protein [broad] | 12.3 | 150.0 | 0.40 | medium | 40 | 8 | 5 |
| lipid [broad] | 10.8 | 120.0 | 0.38 | medium | 30 | 6 | 4 |
| nucleic acid | 8.5 | 80.0 | 0.72 | high | 25 | 5 | 3 |
| carbohydrate [broad] | 5.2 | 40.0 | 0.55 | medium | 12 | 3 | 2 |

**Note**: Protein and lipid are flagged as [broad] because they are ubiquitous biochemical classes. Nucleic acid scores higher on specificity because it's more concentrated in this condition's evidence.

### Top Motifs

| Subfamily | Family | Members | Enrichment |
|---|---|---|---|
| tryptophan | protein_support | 15 | condition-enriched |
| lipid | lipid_support | 42 | broadly-shared |
| adenine | nucleic_acid_support | 10 | condition-enriched |
| collagen | protein_support | 8 | condition-enriched |

### Caveats
- Top themes (protein, lipid) are broad biochemical classes present in most biofluids. They may reflect shared background biology rather than condition-specific signal.
- Most supporting motifs are broadly shared across many conditions. Condition-specific interpretation requires deeper per-peak analysis.

### Graph Preview
Interactive PyVis subgraph showing:
- Yellow HCC condition diamond at center
- Green motif nodes radiating outward
- Blue evidence dots connecting motifs to themes
- Purple/teal theme and biomolecule stars at periphery
- Solid edges = direct evidence, dashed = inferred

---

## Example: "What does peak 1005 mean?"

### Ranked Themes

| Theme | Final Score | Specificity | Confidence |
|---|---|---|---|
| protein | 8.5 | 1.2 | high |
| amino acid | 4.2 | 0.9 | medium |

### Top Biomolecules
- phenylalanine (high count, multi-source)

### Caveats
- None — strong multi-source agreement at this peak

**Interpretation**: Peak 1005 cm-1 is one of the most reproducible Raman/SERS peaks. It corresponds to the phenylalanine ring breathing mode. The protein theme is dominant with high confidence.
