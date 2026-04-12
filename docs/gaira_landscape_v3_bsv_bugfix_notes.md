# Landscape v3 — BSV Bugfix Notes

## Bug Diagnosis

### Root Cause
The v2 BSV landscape builder iterated over motif matrix columns (e.g., `motif_L3_0026`) and checked whether `mid_lower in comp_def['subfamilies']`. But subfamilies contains values like `tryptophan`, `lipid`, `adenine` — not motif IDs.

**The motif ID was never resolved to its subfamily before the BSV mapping check.**

### Why It Produced All Zeros
Every motif column name is `motif_L3_NNNN`. None of these strings match `tryptophan`, `lipid`, etc. So every motif was classified as unmapped, and every BSV component accumulated zero signal.

### Fix
Added a motif metadata resolution step:
1. Load `phaseO3_batch1_refreshed_motifs.csv` to get `motif_id → subfamily` and `motif_id → family` mappings
2. For each motif column, resolve its subfamily/family
3. Match the resolved subfamily/family against BSV component definitions
4. Accumulate into the correct component

### Validation
After fix:
- **32/37 conditions** have nonzero BSV (was 0/37)
- **5 conditions** remain zero (legitimate: their motif profiles contain only unmapped chemistry/unresolved motifs)
- **44/95 motif columns** map to BSV components (47%)
- **51 motif columns** are unmapped (mostly chemistry_support and unresolved_support families)

### HCC Profile (post-fix)
```
protein_backbone:    1.000
aromatic_amino_acid: 0.750
purine_nucleotide:   0.750
pyrimidine_nucleotide: 0.500
glycan_carbohydrate: 0.500
membrane_lipid:      0.500
redox_metabolite:    0.250
nucleic_acid_backbone: 0.250
```
This is biologically plausible: protein and aromatic amino acids dominate serum SERS, with purine/pyrimidine and glycan contributions reflecting liver disease metabolism.
