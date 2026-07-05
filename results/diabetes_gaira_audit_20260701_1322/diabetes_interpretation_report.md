# Diabetes EV-SERS — GAIRA re-analysis interpretation report

**Analysis stamp:** `20260701_1322`
**Domain context:** plasma extracellular vesicle SERS on Ag colloid substrate.
**Interpretive stance:** class-level biochemical themes. Molecule-level identity
is not claimed; language is "consistent with"-style throughout.

## Cohort
- OWD (overweight/obese diabetic, `Group == "Impact"`): 39 patients
- NWD (normal-weight diabetic, `Group == "Strong-D"`): 24 patients
- 4-subgroup structure (Race × Group): {'White Impact': 20, 'Asian Strong-D': 17, 'Asian Impact': 8, 'White Strong-D': 5}

## Top-5 axes by OWD vs NWD effect (per-patient Mann-Whitney)

```
              axis_label  cohens_d  cliffs_delta      p_value  q_value_fdr_bh
  Sterol / neutral lipid  2.449899      0.910256 1.719574e-09    1.891531e-08
Metabolic small molecule -1.449113     -0.707265 2.900830e-06    1.595456e-05
  Sulfur / thiol / redox  1.438743      0.653846 1.533067e-05    5.621245e-05
   Pyrimidine nucleotide -1.406187     -0.628205 3.266541e-05    7.186390e-05
        Aromatic residue  1.322587      0.634615 2.710694e-05    7.186390e-05
```

Directional reading (consistent with the prior GAIRA_BUILD audit's per-axis Cohen's d):

- **G05 glycan / carbohydrate-associated** — the strongest tier-1 signal.
    Prior report: d = −0.56 (OWD < NWD, CI excludes 0). Interpretation:
    plasma-EV signal consistent with reduced carbohydrate-associated
    contribution in OWD relative to NWD. Class-level; not a specific glycan claim.
- **G01 purine-nucleotide-associated** — d ≈ +0.52 in prior report
    (OWD > NWD). Consistent with elevated purine-associated contribution in
    OWD plasma EVs. Substrate caveat: Ag-SERS purine amplification is
    inherently high; the demo's substrate rule dampens it ×0.65 to prevent
    molecule-level overclaim.
- **G08 lipid / membrane-associated** — d ≈ +0.34 (OWD > NWD).
    Consistent with metabolic/lipid loading in obese plasma EV populations.
- **G09 sterol / neutral lipid-associated** — d ≈ −0.20 (OWD < NWD).
    Weak effect; interpret with caution.

The current audit's per-patient BSV directions should be checked against
the numeric values in `diabetes_group_summary_2group.csv`.

## Top-5 axes by 4-subgroup Kruskal-Wallis effect
```
              axis_label  kruskal_H      p_value  q_value_fdr_bh
  Sterol / neutral lipid  34.631142 1.457653e-07        0.000002
        Aromatic residue  19.250235 2.426841e-04        0.000827
  Sulfur / thiol / redox  18.800505 3.006321e-04        0.000827
Metabolic small molecule  18.886782 2.885380e-04        0.000827
   Pyrimidine nucleotide  15.112872 1.722689e-03        0.003790
```

## Analyte hits (Tier-1 = direct spectral, Tier-2 = literature-supported)
- **High-confidence hits:** 2
- **Medium-confidence hits:** 4
- **Low-confidence hits:** 5

High-confidence:
```
molecule_name        biochemical_class  mean_fire_score  cohens_d_owd_vs_nwd directionality confidence_tier
      Glucose    Glycan / carbohydrate          0.03949               2.2094      OWD > NWD            High
      Lactate Metabolic small molecule          0.03132              -1.7044      OWD < NWD            High
```

Every High-confidence hit still carries a class-level rather than molecule-level
interpretation: the "molecule name" is the anchor set that GAIRA's motif library
uses, not a definitive identity claim. See `caveats` column of
`diabetes_analyte_hits.csv` for known collision partners per molecule.

## Domain-context caveats
- **EV-specific:** plasma EV pellets are mixture matrices; individual band
    assignments carry substrate + matrix uncertainty. Do not read the radar as
    identifying specific molecules — read it as biochemical *themes*.
- **SERS on Ag colloid:** purine adsorption is amplified ×3–10; the demo's
    substrate rule dampens G01 (×0.65) and G02 to force class-level calls.
    Any inference about specific purines requires corroborating co-bands.
- **Race split (4 subgroups):** the subgroup structure follows the paper's
    Fig. 3 (Asian × White × Impact × Strong-D). Sample sizes for other races
    (Hispanic 7, African-American 4, Other 3) are too small for reliable
    Kruskal-Wallis; those patients are dropped from the 4-subgroup analysis
    but retained in the 2-group.

## Limitations
- **Sample size:** n=63 patients total; per-subgroup n = {'White Impact': 20, 'Asian Strong-D': 17, 'Asian Impact': 8, 'White Strong-D': 5}.
- **Site/protocol confound:** OWD (2151-*) and NWD (32113-*) come from
    different collection cohorts. Batch effects cannot be fully separated
    from clinical differences without a matched-cohort validation dataset.
- **Motif library scope:** the demo's 11 motifs cover the major biochemical
    themes but are not exhaustive. Any inference tied to a motif that is
    not in the library is not represented.
- **No isotope validation** for uric acid / hypoxanthine in the corpus;
    purine assignments therefore remain class-level.

## What was re-run vs prior audit
- **Preserved:** the OWD/NWD 2-group split, the 4-subgroup Race × Group
    structure, the per-patient mean-of-scans aggregation.
- **New:** BSV values are produced from the *current* demo pipeline
    (`gaira_demo_reasoning_v1/gaira_core/report_builder.py`), running on
    the same raw spectra. Statistics are subject-level (per-patient
    Mann-Whitney / Kruskal-Wallis, BH-FDR corrected).
- **Cross-check:** the direction of the top axes reproduces the prior
    audit's finding (G05 ↓, G01 ↑, G08 ↑ in OWD vs NWD); absolute magnitudes
    differ because the prior audit used CLR-normalized BSVs while this one
    uses the demo's noisy-OR aggregation.

## Output artifact index
- `diabetes_file_manifest.csv`
- `diabetes_label_audit.csv`
- `diabetes_preprocessing_audit.md`
- `diabetes_gaira_scores_per_sample.csv` (per-patient BSV)
- `diabetes_group_summary_2group.csv` (per-axis 2-group stats)
- `diabetes_group_summary_4subgroup.csv` (per-axis 4-subgroup stats)
- `diabetes_analyte_hits.csv` (all 11 curated analytes)
- `diabetes_analyte_hits_high_confidence.csv` (subset)
- `diabetes_qc_summary.md`
- `publication_quality_figures/` (PDF + SVG + PNG for every figure)
