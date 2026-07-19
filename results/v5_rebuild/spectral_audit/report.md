# Matched-Analyte Spectral Audit — Raman vs Ag-SERS

**Read-only spectroscopic audit** of the 51 matched analytes in the frozen GAIRA V5 Stage B grounding corpus. No models were fitted, no preprocessing was altered, and no GAIRA implementation, plan, architecture document, or hypothesis register was modified.

**Question:** do Raman and Ag-SERS actually observe fundamentally different spectra for the same analyte, or largely the same vibrational modes with different relative enhancement?

**Answer:** neither, as it turns out. The Ag-SERS arm is **background-dominated** — its spectra are ~95% identical to one another regardless of analyte. Stage A/B's weak-similarity conclusion is spectroscopically corroborated, but the cause is a measurement-contrast problem in the Ag-colloid acquisition, not different chemistry and not a representation-learning failure.

---

## Reproduce

```bash
cd results/v5_rebuild/spectral_audit/code
python run_audit.py            # peaks, correspondence, 13 metrics, alignment, bands  (~6 s)
python run_nulls.py            # mismatched-analyte + random null controls
python run_sensitivity_l2.py   # SNV (primary) vs L2 (sensitivity)
python run_degeneracy.py       # Ag-SERS analyte discriminability
python run_background_test.py  # cross-modal retrieval after removing shared component
python make_pdf.py             # GAIRA_V5_MATCHED_ANALYTE_SPECTRAL_AUDIT.pdf (61 pages)
```

**Dataset:** exactly the frozen Stage B corpus — 51 matched analytes, 435 spectra (180 Raman / 255 Ag-SERS), 785 nm both arms, grid 520–1750 cm⁻¹ @ 2 cm⁻¹.
**Preprocessing (primary):** `A2_asls_savgol_snv` — the exact Stage B SNV pipeline, unmodified.
**Peak-match tolerance:** ±12 cm⁻¹. **Sensitivity pipeline:** `A1_asls_savgol_l2` (clearly labelled secondary).

---

## Headline numbers

| Quantity (median over 51 analytes) | Value |
| --- | --- |
| Full-profile cosine (Raman vs Ag-SERS) | **+0.101** |
| Matched-minus-mismatched cosine separation | **+0.037** (SNV) / +0.027 (L2) |
| % Raman bands with an Ag-SERS peak within ±12 cm⁻¹ | 80% |
| — same, for a **different** analyte's Ag-SERS (null) | **71%** |
| — same, for uniform-random peaks (null) | 60% |
| Analytes with band correspondence specific at p<0.05 | **0 / 51** |
| Mean \|Δν\| of matched peaks | 5.8 cm⁻¹ |
| Optimal rigid shift (±10 cm⁻¹ scan) | **0 cm⁻¹** (median cosine gain +0.036) |
| Peak-rank correlation (intensity ordering) | +0.12 |
| Between-analyte cosine, within Ag-SERS (L2) | **0.945** |
| Between-analyte cosine, within Raman (L2) | 0.349 |
| Ag-SERS variance explained by corpus-mean spectrum | **95%** |
| Ag-SERS leave-one-out 1-NN analyte ID (chance 0.020) | **0.729** |
| Within-replicate cosine, Ag-SERS: raw / L2 / SNV | 0.999 / 0.948 / **0.491** |
| Within-replicate cosine, Raman: raw / L2 / SNV | 0.999 / 0.990 / 0.986 |

---

## The five decisive findings

**1. The Ag-SERS arm is background-dominated.** Ag-SERS spectra of 51 chemically unrelated metabolites are ~95% identical to each other (between-analyte cosine 0.945 vs 0.349 for Raman), and 95% of each spectrum's variance is captured by the single corpus-mean spectrum. This is the signature of a dominant citrate/colloid surface contribution, not of analyte normal modes.

**2. Analyte identity is present, but faint.** The Ag-SERS data are *not* junk: leave-one-out 1-NN identification across 51 analytes reaches **0.729** (chance 0.020), and raw replicates agree at 0.999. The analyte contributes a real but ~5% residual on a large common component that has no Raman counterpart.

**3. Apparent band correspondence is a peak-density artefact.** Ag-SERS mean spectra yield ~46 features spaced ~24 cm⁻¹; Raman yields ~12 bands spaced ~75 cm⁻¹. A ±12 cm⁻¹ window therefore almost always finds a hit. Observed recall 0.80 vs **0.71 for a mismatched analyte** and 0.60 for random peaks — excess +0.08, and **no analyte** reaches p<0.05. The same holds under L2 (0/51).

**4. A real preprocessing defect exists, but it does not drive the result.** SNV collapses Ag-SERS replicate reproducibility from 0.999 (raw) / 0.948 (L2) to **0.491**, because SNV mean-centres each spectrum so cosine becomes Pearson correlation dominated by the low-amplitude residual. Native resolutions (1.0–1.7 cm⁻¹) are all finer than the 2 cm⁻¹ grid and Gobbato Raman and Ag-SERS share one instrument, so resampling and instrument mismatch are excluded. Crucially the negative result reproduces under L2.

**5. Neither alignment nor background removal rescues it.** The optimal rigid shift is 0 cm⁻¹ at the median (gain +0.036). Projecting out the shared component makes the residual cross-modal similarity genuinely *specific* (matched 0.089 vs mismatched −0.002) but leaves top-1 retrieval at only 0.12–0.18 against 0.02 chance.

---

## Scientific conclusions (Part 12)

1. **Does Stage A look spectroscopically believable?** Yes — corroborated directly in the spectra. Matched/mismatched separation is only +0.03 and band correspondence never exceeds a mismatched-analyte null.
2. **Same dominant vibrational modes?** Not demonstrably. The dominant Ag-SERS features are shared across all analytes — surface/colloid behaviour, not analyte normal modes.
3. **Peak positions preserved?** Not informatively. The 80% recovery is chance-level (71% for a mismatched analyte).
4. **Primarily intensity redistribution?** Redistribution is severe (rank ρ 0.12) but secondary; the band inventories themselves differ in size and identity, so this is not a re-weighting of one shared mode set.
5. **Systematic peak shifts?** No. Median optimal shift 0 cm⁻¹; mean |Δν| 5.8 cm⁻¹ is consistent with random matching inside the tolerance window.
6. **Which families transfer best?** Differences are small against the noise: polysaccharide/amino-acid/protein rank highest (correspondence score 0.20–0.30), purine/lipid/small-nitrogenous lowest (0.08–0.14); purines preserve intensity ordering best (rank ρ 0.47). No family reaches a usable level.
7. **Preprocessing artefacts?** Yes — the SNV reproducibility collapse (finding 4). Significant and worth fixing, but not the cause of the negative result.
8. **Problematic analytes?** Under SNV, 29/51 have within-Ag-SERS replicate cosine < 0.50 (6 below 0.30: hydroxyproline, oleate, fructose-6-phosphate, methionine, serine, triolein) — these are *pipeline casualties*, since the same replicates agree at 0.95 under L2. The structural caveat is that 27/51 analytes draw Raman replicates from two sources, mixing inter-instrument variation into within-Raman spread.
9. **Should Stage A be rerun?** Yes, for methodological correctness — a normalisation that halves one modality's reproducibility should not have been selected. Use L2 or explicit background removal and report similarity against the per-modality reproducibility ceiling. Expect the architectural verdict to survive.
10. **Biggest scientific insight?** *The Ag-SERS arm is background-dominated, not analyte-dominated.* The cross-modal failure is a measurement-contrast problem in the Ag-colloid acquisition — not evidence that the two techniques observe unrelated chemistry, and not a shortcoming of the representation or the encoders. It also explains the Stage B encoder collapse: embeddings with cross-analyte duplicate fraction 0.96–1.00 were faithfully representing inputs that are themselves near-identical.

---

## What this implies (analysis only — no changes made)

Progress requires Ag-SERS measurements in which the **analyte**, not the colloid, dominates the spectrum: higher effective surface coverage/concentration, blank-colloid difference spectra, or explicit background modelling at acquisition time. Until the Ag-SERS arm carries analyte-dominated signal, no representation — interpretable or learned — can recover cross-modal correspondence, because the information is not present in the measurement at usable contrast.

---

## Outputs

| Path | Contents |
| --- | --- |
| `GAIRA_V5_MATCHED_ANALYTE_SPECTRAL_AUDIT.pdf` | 61-page report: cover, TOC, executive summary, decisive-evidence figures, global statistics, band/family analysis, **one page per analyte**, conclusions |
| `tables/per_analyte_summary.csv` | All metrics + generated spectroscopic interpretation, per analyte |
| `tables/peak_correspondence_matrix.csv` | **Primary output** — every Raman↔Ag-SERS peak pair with shift, intensity ratio, confidence, note |
| `tables/peak_table_raman.csv`, `peak_table_sers.csv` | Position, prominence, width, relative intensity, local SNR |
| `tables/peak_correspondence_null_controls.csv` | Mismatched-analyte and random nulls per analyte |
| `tables/band_level_comparison.csv` | Six spectral regions × 51 analytes |
| `tables/global_statistics.json` | Distributions + top-10 rankings |
| `tables/family_analysis.csv` | Per-family aggregates |
| `tables/preprocessing_sensitivity_summary.json` | SNV vs L2 |
| `tables/sers_degeneracy_test.json` | Analyte discriminability |
| `tables/background_removal_test.json` | Retrieval after removing shared components |
| `figures/analyte_*.png` | Per-analyte composite figures (51) |
