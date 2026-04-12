# D0 — Figure Digitization Pilot Assessment

## Blunt Assessment: NOT RECOMMENDED AT THIS STAGE

### Evidence
Across all GAIRA extraction campaigns (critical_A, critical_B, liver_SERS_DB), figure digitization has yielded **exactly 0 approved evidence rows**. This is not because figures don't contain useful information — it's because:

1. **Most spectral figures in Raman/SERS papers show overlaid spectra without peak labels.** They require manual inspection or algorithmic peak-picking to extract assignments.
2. **Peak-labeled figures exist** but are rare and typically redundant with the paper's text/table assignments.
3. **Difference spectra and class-comparison plots** would be the most valuable figure type for directionality, but these require:
   - Spectral axis reading
   - Class label parsing
   - Sign/direction interpretation
   
   None of which our current extraction pipeline can do.

### How Many Sources Look Digitizable?
- **Cannot assess from extracted text alone.** The audit classified 0 sources as Category C (figure-derivable) because figure content is not represented in the evidence rows.
- A manual inspection of 10-20 PDFs would be needed to estimate how many contain usable comparison figures.
- Based on liver SERS literature conventions, approximately 30-50% of disease papers include overlaid mean spectra — but only ~10% label specific peaks on those figures.

### What Figure Types Would Be Most Promising?
1. **Peak intensity bar plots** with disease vs control groups — these directly encode directionality
2. **Overlaid mean spectra** with labeled bands — moderate value if peak positions are annotated
3. **Difference spectra** (disease minus control) — highest value but least common

### What Makes a Figure Too Noisy/Ambiguous?
- Overlaid raw spectra without averaging
- ML-derived visualizations (PCA scores, t-SNE) — no spectral assignment content
- Heavily processed derivative spectra without peak labels
- Composite figures where spectral region is a small inset

### Recommendation
**Defer figure digitization.** If pursued later:
- Start with 5 liver-disease papers that have known peak-labeled comparison figures
- Use a manual annotation workflow, not automated extraction
- Expected yield: 2-5 directional assignments per figure, 10-25 total from a 5-paper pilot
- This is lower ROI than re-extracting differential text from discussion sections
