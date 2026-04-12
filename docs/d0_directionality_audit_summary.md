# D0 — Directionality Viability Audit Summary

## Corpus Audited
- **137 unique sources**, 1,887 evidence rows
- Sources span: critical_A (23), critical_B (35), liver_SERS_DB (40), literature OA/curated (39)

## Category Distribution

| Category | Count | % | Description |
|---|---|---|---|
| **A — Explicit text directionality** | 1 | 0.7% | True "increased/decreased in disease vs control" language |
| **B — Tabular/weak directionality** | 9 | 6.6% | Directional keywords present, weak or no explicit comparator |
| **C — Figure-derivable only** | 1 | 0.7% | Comparison mentioned, no textual direction |
| **D — No usable directionality** | 126 | 92.0% | Assignment-only evidence, no differential content |

## Key Findings

### Blunt Overall Judgment: WEAK VIABILITY

The current GAIRA evidence corpus was built from peak assignment tables and spectral interpretation text. It captures "peak X corresponds to molecule Y" — NOT "peak X is elevated in disease Z vs healthy."

**Only 10/137 sources (7.3%)** contain any true directional language (increased/decreased/elevated/reduced). Of these, only **1 liver-disease source** has both clear directionality AND explicit comparison context.

### Why So Low?
1. **Extraction pipeline was designed for assignments, not differentials.** The regex patterns match "X cm-1 assigned to Y" — not "X cm-1 increased in disease."
2. **Many disease papers DO contain directionality** — but it lives in discussion paragraphs, figure legends, and statistical tables that the current extractor does not capture.
3. **The evidence that IS extracted is the assignment portion** of differential papers. The differential context was stripped during structured decomposition.

### What Would Help?
A **new extraction pass** targeting differential language patterns would likely recover directionality from many of the same source PDFs. The information is probably in the documents — it just wasn't extracted.

## Viability for Key Queries

### HCC vs healthy serum within liver sources
- **1 liver-disease source** with weak directionality (Category B)
- **Insufficient** for meaningful differential analysis from current evidence alone
- The underlying PDFs likely contain much more differential content than was extracted

### NAFLD vs healthy serum within liver sources
- **0 sources** with usable directionality
- Cannot perform differential analysis from current evidence

## Answers to Key Questions

### 1. Is within-source directionality extraction viable from the current corpus?
**No — from the current extracted evidence.** The evidence rows contain assignment-grade content, not differential content. However, the underlying source PDFs likely contain substantial directionality that could be recovered with a targeted re-extraction pass.

### 2. What fraction of sources appear usable?
**~8% (11/137)** have any detectable directionality in the currently extracted text. Only **~1.5% (2/137)** are viable pilot candidates.

### 3. Best next step?
**Option A (recommended): Targeted re-extraction pilot** — go back to the top 10-20 disease-comparison source PDFs and extract differential language specifically. This means new regex patterns targeting "increased/decreased/elevated/reduced in [condition]" in discussion sections.

**Option B: Defer to dataset integration.** If spectral data (not just literature assignments) is available, within-dataset contrasts are far more reliable than text-derived directionality.

**Option C (not recommended): Figure digitization.** Cannot assess viability from text alone — would require inspecting the actual PDFs, and historical GAIRA figure extraction has yielded 0 rows across all campaigns.

## Recommendation
**Defer contrast evidence schema implementation.** The current corpus does not contain enough directionality to justify a new schema layer now. Instead:
1. Build BSV from the existing motif differential + stability layers (which work at cross-source level)
2. If a directionality extraction pilot is pursued later, target the liver-disease PDFs with new differential regex patterns
3. Do not invest in figure digitization at this stage
