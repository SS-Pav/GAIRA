# D1.1 — Liver Extraction Failure Modes

## Why 17/20 Sources Yielded Zero Directional Evidence

### Failure Mode 1: Classification-First Papers (most common)
Many liver SERS papers follow this structure:
1. Collect serum SERS spectra from disease + control groups
2. Apply PCA/LDA/SVM/DNN to classify
3. Report classification accuracy
4. Show a few "important features" or "discriminative peaks"

**The problem**: Steps 1-3 produce no per-peak directionality. Step 4 often lists peak positions without saying whether they go up or down. "Peak at 1445 cm-1 was selected as discriminative" does not tell us the direction.

### Failure Mode 2: Vague Comparison Language
Papers mention "differences between groups" without specifying direction:
- "spectral differences were observed between HCC and healthy"
- "the SERS spectra showed distinguishable features"
- "significant discrimination was achieved"

These are comparisons without directionality.

### Failure Mode 3: Direction in Figures Only
Several papers show overlaid mean spectra where differences are visible but:
- The text doesn't describe the direction explicitly
- The figure caption says "spectra of disease (red) and control (blue)" without peak-level annotation
- Recovering direction would require visual spectral comparison, not text extraction

### Failure Mode 4: Peak Tables Without Direction
Tables list peaks and their molecular assignments but not their behavior in disease:
```
1003 cm-1    phenylalanine ring breathing
1445 cm-1    CH2 deformation (lipids/proteins)
1655 cm-1    Amide I
```
These are descriptive, not differential.

### Failure Mode 5: Discussion-Only Direction
Some papers do discuss directionality in narrative form but in ways the regex doesn't capture:
- Complex multi-sentence reasoning: "The ratio of amide I to lipid bands shifted, suggesting altered protein/lipid balance in disease"
- Indirect language: "The spectral profile of HCC patients showed a trend toward higher protein content"

These require more sophisticated NLP or manual curation.

## Implication for GAIRA
The liver SERS literature is **classification-heavy and assignment-rich but direction-poor** in its explicit per-peak textual descriptions. Directionality often exists implicitly (in figures, in broad discussion themes) but rarely in the extractable "peak X increased in disease Y vs control" format that our pipeline captures.

## What Would Help
1. **Manual curation of 5-10 key papers** by a domain expert who can interpret figures + discussion
2. **Direct spectral data integration** — computing intensity differences from raw spectra rather than extracting them from text
3. **Feature importance tables** from ML papers that include direction (SHAP values, not just VIP scores)
