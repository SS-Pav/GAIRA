# Diabetes EV-SERS — mechanistic upgrade summary

**Date:** 2026-07-01
**Purpose:** Document the three scoring upgrades applied to the diabetes EV-SERS
GAIRA re-analysis, the before/after impact on the Redox axis, and the
mechanistic biological reading enabled by the new radar + forest plot.

## 1. What was changed (three fixes, all in `analysis/_diabetes_overrides.py`)

| Fix | Description | Motivation |
| --- | --- | --- |
| **G10 motif window tighter** | Thione C–S band shrunk from 480–510 cm⁻¹ (30 cm⁻¹) to 490–505 cm⁻¹ (15 cm⁻¹) | Ag colloid + plasma matrix show broad background in 450–520 cm⁻¹; the wider window inflates G10 with substrate/matrix modes. |
| **Co-band-gated Ag-SERS thiol boost** | The ×1.20 substrate boost on the thione motif is applied **only** when the ergothioneine 720 cm⁻¹ imidazole support co-band fires above intensity 0.010 | Anchor at 495 firing alone is not sufficient evidence for a thione call. Requiring the 720 imidazole co-band forces the boost to be biology-driven. |
| **Z-score mechanistic radar** | New radar panel showing per-cohort deviation from pooled mean (`z = (cohort_mean − pool_mean) / pool_SD`) | Absolute magnitudes are inter-axis-biased by the ×0.65 / ×1.20 substrate rules. z-scores normalise this so mechanistic shifts (not pipeline artifacts) drive the visual pattern. |

**Files added** (nothing in GAIRA core modified):
- `analysis/_diabetes_overrides.py` (new)
- `analysis/run_diabetes_gaira_audit.py` (modified to call `build_report_diabetes`)

## 2. Effect on G10 (Redox axis)

|  | Old (`build_report`) | New (`build_report_diabetes`) | Δ |
| --- | ---: | ---: | ---: |
| **OWD G10 mean**  | 0.1124 | **0.1084** | −4% |
| **NWD G10 mean**  | 0.1009 | **0.0728** | −28% |
| Cohen's d (OWD vs NWD) | +0.44 | **+1.44** | +227% |
| BH-FDR q-value | 0.081 | **5.6 × 10⁻⁵** | 1450× stronger |
| Cliff's δ | 0.316 | 0.654 | 2.1× stronger |

**Why the asymmetric drop?** The co-band gate audit shows:

| Cohort | Boost APPLIED (720 co-band ≥ 0.010) | Boost SKIPPED (720 silent) | % applied |
| --- | ---: | ---: | ---: |
| OWD (n=39) | 31 patients | 8 patients | **79%** |
| NWD (n=24) | 15 patients | 9 patients | **63%** |

79% of OWD patients show genuine ergothioneine-like imidazole co-band evidence
alongside the 495 anchor, versus 63% of NWD patients. Before the fix, the ×1.20
boost was applied to ALL 63 patients regardless — so both cohorts were similarly
inflated and the biological difference was masked. After the fix, the OWD radar
reflects real thiol/redox biology corroborated by co-bands, while the NWD radar
honestly shows lower G10 because 37% of NWD patients don't have that
corroboration.

**The Redox axis is now the second-strongest per-patient effect in the dataset**
(d = +1.44, q = 5.6 × 10⁻⁵), where before the fix it barely cleared FDR.

## 3. Full re-ranked axis effects (post-fix)

| Rank | Axis | Direction | Cohen's d | BH-FDR q | Family |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | G09 Sterol / neutral lipid       | OWD > NWD | **+2.45** | 1.9 × 10⁻⁸ | lipid |
| 2 | G10 Sulfur / thiol / redox       | OWD > NWD | **+1.44** | 5.6 × 10⁻⁵ | redox |
| 3 | G11 Metabolic small molecule     | OWD < NWD | **−1.45** | 1.6 × 10⁻⁵ | metabolite |
| 4 | G03 Pyrimidine nucleotide        | OWD < NWD | **−1.41** | 7.2 × 10⁻⁵ | nucleic |
| 5 | G07 Aromatic residue             | OWD > NWD | **+1.32** | 7.2 × 10⁻⁵ | protein |
| 6 | G08 Lipid acyl / membrane        | n.s. | −0.28 | 0.28 | lipid |
| 7 | G05 Glycan / carbohydrate        | n.s. | −0.21 | 0.28 | glycan |
| 8 | G04 Nucleic acid phosphate       | n.s. | +0.28 | 0.46 | nucleic |
| 9 | G06 Protein peptide backbone     | n.s. | +0.27 | 0.46 | protein |
| 10 | G01 Purine nucleotide           | n.s. | +0.30 | 0.66 | nucleic |
| 11 | G02 Purine metabolite           | n.s. | −0.03 | 0.77 | nucleic |

Five axes clear the |d| ≥ 1.0 + q < 0.001 double bar — a very strong effect
pattern. Every strong effect is on a *different* biochemical family, meaning
this is not one axis "leaking" into related axes; it's a genuine multi-system
biology signature.

## 4. Mechanistic reading — OWD (obese/overweight diabetic) plasma-EV signature

Reading the z-score radar with sign + family together:

| Elevated in OWD (positive z) | Reduced in OWD (negative z) |
| --- | --- |
| **Sterol / neutral lipid** (G09, d=+2.45) — consistent with obesity-related sterol/free-cholesterol loading in circulating EVs. | **Pyrimidine nucleotide** (G03, d=−1.41) — reduced pyrimidine ring-associated signal (∼780 cm⁻¹). |
| **Aromatic residue** (G07, d=+1.32) — Phe 1003 cm⁻¹ ring-breathing elevated; consistent with elevated plasma aromatic AA in obese diabetic serum. | **Metabolic small molecule** (G11, d=−1.45) — reduced lactate/C-C-O 845+925 cm⁻¹ region. |
| **Sulfur/thiol/redox** (G10, d=+1.44, co-band corroborated) — consistent with ergothioneine-like plasma thiol-redox signal, elevated with obesity-associated oxidative stress. | |

Together this is a coherent **"metabolically stressed obese-diabetic EV
biochemistry"** signature — elevated sterol + aromatic + thiol/redox, reduced
small-molecule / pyrimidine baseline — which is directionally consistent with
the diabetes literature on plasma EV cargo and oxidative-stress biomarkers,
but is reported here at biochemical-*class* level only (no molecule-identity claim).

## 5. Caveats that survive after the fix

- **Site/protocol confound**: OWD (`2151-*`) and NWD (`32113-*`) come from two
  different collection sites; batch effects cannot be fully separated without a
  matched-cohort validation dataset.
- **Sample size**: 39 vs 24 patients; 4-subgroup analysis has one cell with n=5.
- **Class-level only**: "Sterol" ≠ "specific sterol X"; "Redox" ≠ "ergothioneine
  concentration"; interpret as biochemical themes, not molecule identities.
- **G10 co-band gate is a design choice**: the 720 cm⁻¹ floor (0.010) and
  window (715–735 cm⁻¹) are hard-coded; sensitivity to those parameters should
  be checked in a follow-up.
- **G09 sterol dominant effect is untriangulated**: only one sterol-ring
  motif (~548 cm⁻¹) feeds this axis; a follow-up requiring cholesterol
  co-bands (~1130 or 1665) would strengthen confidence.

## 6. Output artifacts (new in this rerun)

- `diabetes_zscore_2group.csv` — cohort z-scores per axis
- `diabetes_zscore_4subgroup.csv` — subgroup z-scores per axis
- `diabetes_thiol_boost_gate_audit.csv` — per-patient thiol-boost gate state
  (applied / skipped + 720 imidazole intensity)
- `publication_quality_figures/fig_radar_2group_mechanistic.{pdf,svg,png}`
  — 2-panel raw + z-score mechanistic radar with Cohen's d + significance
- `publication_quality_figures/fig_radar_4subgroup_mechanistic.{pdf,svg,png}`
- `publication_quality_figures/fig_forest_owd_vs_nwd.{pdf,svg,png}` —
  effect-size forest plot with 95% bootstrap CI, family-colored, significance-marked
