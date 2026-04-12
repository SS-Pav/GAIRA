# BSV v1 — Component Definitions

## 8 Biochemical State Vector Components

### 1. membrane_lipid
**What it captures**: Lipid membrane integrity and composition
**Key motifs**: lipid, cholesterol, phosphatidylcholine, phosphatidylserine, sphingomyelin
**Biological meaning**: Changes in membrane lipid content/composition are common in cancer (membrane remodeling) and metabolic disease (lipid accumulation)

### 2. protein_backbone
**What it captures**: Overall protein/peptide content as seen via backbone vibrations
**Key motifs**: collagen, proline, glycine + protein/peptide themes
**Biological meaning**: Amide I/II/III bands, collagen changes in fibrosis, broad protein abundance

### 3. aromatic_amino_acid
**What it captures**: Aromatic amino acid signatures (phenylalanine, tryptophan, tyrosine)
**Key motifs**: tryptophan, phenylalanine, tyrosine, histidine
**Biological meaning**: These are the strongest individual amino acid SERS markers. Changes often reflect altered protein composition or metabolic state

### 4. purine_nucleotide
**What it captures**: Purine bases and nucleotides (adenine, guanine, AMP)
**Key motifs**: adenine, guanine, AMP + nucleic acid themes
**Biological meaning**: DNA/RNA metabolism, cell turnover, purine pathway changes in cancer

### 5. pyrimidine_nucleotide
**What it captures**: Pyrimidine bases (cytosine, thymine, uracil)
**Key motifs**: cytosine, thymine, uracil
**Biological meaning**: Complementary to purine — captures the other half of nucleic acid signature

### 6. glycan_carbohydrate
**What it captures**: Saccharide and glycan signatures
**Key motifs**: glucose, galactose, mannose, glycogen + carbohydrate theme
**Biological meaning**: Glycosylation changes, metabolic glucose, fibrosis-associated glycan alterations

### 7. redox_metabolite
**What it captures**: Antioxidant/metabolite signatures
**Key motifs**: glutathione, ergothioneine, uric acid, carotenoids, dopamine
**Biological meaning**: Oxidative stress markers, antioxidant capacity, metabolite-level changes

### 8. nucleic_acid_backbone
**What it captures**: Phosphodiester backbone chemistry (PO2- stretching)
**Key inputs**: phosphodiester and phosphate functional groups
**Biological meaning**: DNA/RNA backbone integrity, complementary to base-specific components
**Note**: Lower weight (0.5x) because sourced from chemistry-only functional groups

## Design Principles
- Components are biologically interpretable, not just statistical clusters
- Each maps to specific motif subfamilies with documented rationale
- Overlap between components is minimal but not zero (e.g., nucleic acid appears in both purine + pyrimidine)
- The "unresolved_mixed" category was deliberately excluded from BSV to prevent noise domination
