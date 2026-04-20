# GAIRA Expected-BSV Axis Audit (v2)

Per-axis summary of literature evidence that feeds the expected comparator.
Numbers come from `peak_assignments` (local, peak-level), `knowledge_chunks`
(diffuse prose), and `condition_differential_profile.csv` (landscape v4
contrast-explicit directionality). Calibration status is READ-ONLY — this
audit does not adjust anything based on calibration results.

## Support strength at a glance

- **Strong:** `membrane_lipid`, `protein_backbone`
- **Moderate:** `aromatic_amino_acid`, `purine_nucleotide`, `nucleic_acid_backbone`
- **Sparse:** `pyrimidine_nucleotide`, `glycan_carbohydrate`, `redox_metabolite`

## Full audit

| axis | n_peak_rows | n_sources | n_molecules | share_high_conf | share_medium_conf | anchor_hint_hit_rate | n_conditions_explicit | n_conditions_up | n_conditions_down | locality_score | support_strength | calibration_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| membrane_lipid | 54 | 10 | 36 | 0.056 | 0.741 | 0.241 | 4 | 0 | 4 | 0.844 | strong | not_tested |
| protein_backbone | 88 | 15 | 60 | 0.034 | 0.864 | 0.216 | 3 | 0 | 3 | 0.898 | strong | not_tested |
| aromatic_amino_acid | 39 | 12 | 25 | 0.051 | 0.897 | 0.538 | 3 | 2 | 1 | 0.951 | moderate | not_tested |
| purine_nucleotide | 21 | 3 | 13 | 0 | 0.952 | 0.476 | 2 | 2 | 0 | 0.875 | moderate | weak_recovery |
| pyrimidine_nucleotide | 1 | 1 | 1 | 0 | 1 | 0 | 4 | 0 | 4 | 1 | sparse | not_tested |
| glycan_carbohydrate | 11 | 3 | 7 | 0 | 0.727 | 0.273 | 4 | 0 | 4 | 0.647 | sparse | not_tested |
| redox_metabolite | 9 | 4 | 5 | 0 | 1 | 0.556 | 1 | 0 | 1 | 0.6 | sparse | not_tested |
| nucleic_acid_backbone | 32 | 12 | 24 | 0.156 | 0.688 | 0.125 | 1 | 1 | 0 | 0.727 | moderate | not_tested |

## Ambiguous / unmapped pool

Peaks that could not be confidently attributed to a single BSV axis. Kept for
visibility so they are counted in the ambiguity tally rather than silently
dropped or force-assigned.

| n_peak_rows | n_sources | n_molecules | distinct_molecules_top5 | peak_cm_min | peak_cm_max | peak_cm_median | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 33 | 15 | 29 | glycine/proline-like low-wavenumber contribution; C-S stretching; C–H bending mode; the CC stretching vibration arising from the conjugation of the two benzene rin; vibrations | 256 | 3339 | 1448 | Mixed-Vibrational + unclassifiable Metabolite/AminoAcid rows |

## Field meanings

- `n_peak_rows` — rows in `peak_assignments` mapped to this axis via `axis_mapping.py`.
- `n_sources` / `n_molecules` — distinct `source_id` / `assigned_molecule` values.
- `share_*_conf` — distribution of `confidence_text` (high/medium/low) among the peaks.
- `anchor_hint_hit_rate` — fraction of peaks whose `peak_cm` lies inside a canonical
  anchor range for this axis (`AXIS_ANCHOR_HINTS`).
- `n_conditions_explicit` — conditions in `condition_differential_profile.csv` with
  `direction ∈ {up, down}` on this axis.
- `locality_score` — `n_peak_rows / (n_peak_rows + n_prose_chunks)`. Higher = more local, peak-level support.
- `support_strength` — `strong` if ≥40 peak rows AND ≥75% medium-or-better confidence;
  `moderate` if ≥15 peak rows; else `sparse`.
- `calibration_status` — result from `gaira_calibration_eval_v1` (downstream); not used by the audit.
