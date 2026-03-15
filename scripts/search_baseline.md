# GAIRA Search Engine – Current Baseline

This version establishes a stable baseline for Raman spectrum search and candidate identification.

## Core Workflow

The GAIRA search pipeline currently performs the following steps:

1. **Load query spectrum**
   - Accepts CSV with two columns (wavenumber, intensity).

2. **Crop spectrum**
   - Restricts range to **450–1800 cm⁻¹**, matching RamanBioLib coverage.

3. **Normalize spectrum**
   - Min–max scaling of intensity values to the range 0–1.

4. **Peak detection**
   - Uses `scipy.signal.find_peaks` with configurable prominence and height thresholds.

5. **Reference peak retrieval**
   - Query peaks are matched against reference peaks stored in DuckDB (`reference_peaks` table).
   - Matching tolerance: **±5 cm⁻¹**.

6. **Peak-based candidate scoring**
   - For each reference spectrum:
     - matched peak fraction
     - mean peak delta
     - intensity-weighted match score
     - strong peak coverage
     - missing strong peak penalty
   - These metrics are combined into an **overall peak evidence score**.

7. **Full-spectrum reranking**
   - Top peak candidates are reranked using:
     - cosine similarity
     - Pearson correlation
   - Reference spectra are interpolated onto the query grid before comparison.

8. **Component-level aggregation**
   - Multiple references representing the same biochemical component are combined.
   - Component score = weighted combination of best and mean reference scores.

9. **Biochemical class summary**
   - Component results are grouped by biochemical class.
   - Summary statistics include:
     - candidate count
     - mean score
     - max score.

10. **Interpretation**
   - Produces a simple textual interpretation of the likely sample composition.

---

## Current Performance Example

For the collagen example query:

**Ref-level ranking (top results)**

1. superoxide dismutases  
2. collagen  
3. pepsin  
4. superoxide dismutases  
5. elastase  

**Component-level ranking**

1. superoxide dismutases  
2. collagen  
3. pepsin  
4. elastase  

Collagen consistently appears at **rank 2** at both reference and component levels.

This behavior is considered acceptable for the current baseline, given overlapping protein Raman features.

---

## Experimental Layers Removed

Several experimental ranking layers were explored and removed from the active pipeline because they degraded robustness:

- hard biochemical class filtering
- soft class prior reranking
- diagnostic peak reranking
- derivative-based similarity
- region-specific similarity weighting
- confuser-aware reranking
- contrastive pairwise reranking

The current baseline intentionally keeps the ranking logic simple and interpretable.

---

## Next Development Phase

Development will now focus on:

1. **Evaluation harness**
   - benchmark queries
   - ranking statistics
   - reproducibility.

2. **Interactive visualization**
   - Streamlit interface for inspecting candidate spectra and peak matches.

3. **Mixture analysis experiments**
   - synthetic mixture generation
   - evaluation of multi-component recovery.

The baseline ranking logic will remain frozen while these tools are developed.