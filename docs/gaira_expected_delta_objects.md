# Expected-Delta Objects (v2)

Contrast-specific, literature-grounded expected shifts. Each object carries per-axis direction + confidence + anchor windows + ambiguity notes, rather than a single averaged profile.

## Schema

```
ExpectedDelta {
  contrast_id: str
  condition_a: str                     // perturbed / disease side
  condition_b: str                     // reference
  matrix: str                          // serum, EV, biofluid, ...
  substrate_context: str               // Ag colloid, Au, plasmonic paper, ...
  status: 'direct' | 'approximate' | 'unavailable'
  overall_confidence: 'high' | 'moderate' | 'low' | 'none'
  expected_axes: [
    {
      axis: str
      direction: 'up' | 'down' | 'flat' | 'mixed' | 'unknown'
      confidence: 'high' | 'moderate' | 'low'
      anchor_windows: [[start_cm, end_cm], ...]
      ambiguity_notes: [str, ...]
      source_ids: [str, ...]
      rationale: str
    }, ...
  ]
  ambiguity_summary: str
  rationale: str
  provenance: [str, ...]
}
```

## Status semantics

- `direct` — contrast-specific landscape-v4 row exists; numeric delta vector is attached.
- `approximate` — contrast inferred from analyte fingerprint (e.g. hypoxanthine peak assignments) rather than a direct contrast statement in the source literature.
- `unavailable` — no literature support; downstream should not treat this as a real comparator.

## Registered contrasts

| contrast_id | condition_a | condition_b | matrix | status | overall_confidence | #up | #down | #mixed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hcc_vs_healthy_serum | HCC | healthy_control | serum | direct | moderate | 2 | 4 | 0 |
| nafld_vs_healthy_serum | NAFLD_NASH | healthy_control | serum | direct | moderate | 0 | 5 | 0 |
| hepatitis_vs_healthy_serum | hepatitis | healthy_control | serum | direct | moderate | 1 | 5 | 0 |
| liver_cancer_unspecified_vs_healthy_serum | liver_cancer_unspecified | healthy_control | serum | direct | moderate | 2 | 3 | 0 |
| hypoxanthine_spike_literature | serum_plus_hypoxanthine | serum_baseline | serum | approximate | moderate | 1 | 0 | 0 |
| uricase_depletion_literature | serum_plus_uricase | serum_baseline | serum | approximate | moderate | 0 | 1 | 2 |
| ergothioneine_spike_literature | serum_plus_ergothioneine | serum_baseline | serum | approximate | moderate | 1 | 0 | 1 |

## Per-contrast detail

### `hcc_vs_healthy_serum`

- **Label**: HCC vs healthy_control  ·  matrix: serum  ·  substrate: mixed (Au / Ag)
- **Status**: `direct`  ·  **Overall confidence**: `moderate`
- **Rationale**: Hepatocellular carcinoma serum vs healthy control; landscape v4 derives axis-level deltas from aggregated Raman/SERS evidence.
- **Ambiguity**: Landscape v4 has 6 non-flat axes for HCC (coarse aggregation; use with caution).

| axis | direction | confidence | anchor windows | ambiguity | rationale |
| --- | --- | --- | --- | --- | --- |
| membrane_lipid | down | high | 1109–1151; 1435–1455; 853–887; 1059–1070; 1355–1405 | window 1260–1330: ambiguity_score=0.67; window 635–672: ambiguity_score=0.78; window 535–553: ambiguity_score=0.00; w... | landscape-v4 delta=-0.75; anchor present: True |
| protein_backbone | down | low | 1540–1680; 550–565; 662–689; 1500–1515 | window 1175–1343: ambiguity_score=0.43; window 1392–1470: ambiguity_score=0.46; window 727–755: ambiguity_score=0.63;... | landscape-v4 delta=-0.20; anchor present: True |
| aromatic_amino_acid | up | low | 991–1025; 1590–1605; 1148–1165; 1548–1565; 1512–1522 | window 819–856: ambiguity_score=0.55; window 628–642: ambiguity_score=0.57; window 1192–1208: ambiguity_score=0.50; w... | landscape-v4 delta=+0.25; anchor present: True |
| purine_nucleotide | flat | low | 715–734 | window 491–501: ambiguity_score=0.00; window 635–645: ambiguity_score=0.39; window 884–894: ambiguity_score=0.75; win... | landscape-v4 delta=+0.00; anchor present: False |
| pyrimidine_nucleotide | down | moderate | — | window 1605–1615: ambiguity_score=0.00 | landscape-v4 delta=-0.33; anchor present: False |
| glycan_carbohydrate | down | moderate | 1100–1110 | window 935–985: ambiguity_score=0.43; window 580–595: ambiguity_score=0.00; window 880–902: ambiguity_score=0.67; win... | landscape-v4 delta=-0.50; anchor present: False |
| redox_metabolite | flat | low | 479–489 | window 1502–1536: ambiguity_score=0.67; window 1153–1163: ambiguity_score=0.75; window 1210–1220: ambiguity_score=0.0... | landscape-v4 delta=+0.00; anchor present: False |
| nucleic_acid_backbone | up | high | 1075–1090; 1566–1588; 1045–1055 | window 745–813: ambiguity_score=0.44; window 611–621: ambiguity_score=0.50; window 1335–1367: ambiguity_score=0.50; w... | landscape-v4 delta=+1.00; anchor present: True |

### `nafld_vs_healthy_serum`

- **Label**: NAFLD_NASH vs healthy_control  ·  matrix: serum  ·  substrate: mixed (Raman / SERS)
- **Status**: `direct`  ·  **Overall confidence**: `moderate`
- **Rationale**: NAFLD/NASH vs healthy serum; landscape v4 encodes the disease-minus-healthy shift from evidence aggregation.
- **Ambiguity**: Landscape v4 has 5 non-flat axes for NAFLD_NASH (coarse aggregation; use with caution).

| axis | direction | confidence | anchor windows | ambiguity | rationale |
| --- | --- | --- | --- | --- | --- |
| membrane_lipid | down | high | 1109–1151; 1435–1455; 853–887; 1059–1070; 1355–1405 | window 1260–1330: ambiguity_score=0.67; window 635–672: ambiguity_score=0.78; window 535–553: ambiguity_score=0.00; w... | landscape-v4 delta=-1.00; anchor present: True |
| protein_backbone | down | moderate | 1540–1680; 550–565; 662–689; 1500–1515 | window 1175–1343: ambiguity_score=0.43; window 1392–1470: ambiguity_score=0.46; window 727–755: ambiguity_score=0.63;... | landscape-v4 delta=-0.40; anchor present: True |
| aromatic_amino_acid | flat | low | 991–1025; 1590–1605; 1148–1165; 1548–1565; 1512–1522 | window 819–856: ambiguity_score=0.55; window 628–642: ambiguity_score=0.57; window 1192–1208: ambiguity_score=0.50; w... | landscape-v4 delta=+0.00; anchor present: True |
| purine_nucleotide | flat | low | 715–734 | window 491–501: ambiguity_score=0.00; window 635–645: ambiguity_score=0.39; window 884–894: ambiguity_score=0.75; win... | landscape-v4 delta=+0.00; anchor present: False |
| pyrimidine_nucleotide | down | moderate | — | window 1605–1615: ambiguity_score=0.00 | landscape-v4 delta=-1.00; anchor present: False |
| glycan_carbohydrate | down | moderate | 1100–1110 | window 935–985: ambiguity_score=0.43; window 580–595: ambiguity_score=0.00; window 880–902: ambiguity_score=0.67; win... | landscape-v4 delta=-0.50; anchor present: False |
| redox_metabolite | down | moderate | 479–489 | window 1502–1536: ambiguity_score=0.67; window 1153–1163: ambiguity_score=0.75; window 1210–1220: ambiguity_score=0.0... | landscape-v4 delta=-0.50; anchor present: False |
| nucleic_acid_backbone | flat | low | 1075–1090; 1566–1588; 1045–1055 | window 745–813: ambiguity_score=0.44; window 611–621: ambiguity_score=0.50; window 1335–1367: ambiguity_score=0.50; w... | landscape-v4 delta=+0.00; anchor present: True |

### `hepatitis_vs_healthy_serum`

- **Label**: hepatitis vs healthy_control  ·  matrix: serum  ·  substrate: mixed
- **Status**: `direct`  ·  **Overall confidence**: `moderate`
- **Rationale**: Hepatitis vs healthy serum, landscape v4 aggregated delta.
- **Ambiguity**: Landscape v4 has 6 non-flat axes for hepatitis (coarse aggregation; use with caution).

| axis | direction | confidence | anchor windows | ambiguity | rationale |
| --- | --- | --- | --- | --- | --- |
| membrane_lipid | down | moderate | 1109–1151; 1435–1455; 853–887; 1059–1070; 1355–1405 | window 1260–1330: ambiguity_score=0.67; window 635–672: ambiguity_score=0.78; window 535–553: ambiguity_score=0.00; w... | landscape-v4 delta=-0.50; anchor present: True |
| protein_backbone | down | low | 1540–1680; 550–565; 662–689; 1500–1515 | window 1175–1343: ambiguity_score=0.43; window 1392–1470: ambiguity_score=0.46; window 727–755: ambiguity_score=0.63;... | landscape-v4 delta=-0.20; anchor present: True |
| aromatic_amino_acid | down | low | 991–1025; 1590–1605; 1148–1165; 1548–1565; 1512–1522 | window 819–856: ambiguity_score=0.55; window 628–642: ambiguity_score=0.57; window 1192–1208: ambiguity_score=0.50; w... | landscape-v4 delta=-0.25; anchor present: True |
| purine_nucleotide | up | low | 715–734 | window 491–501: ambiguity_score=0.00; window 635–645: ambiguity_score=0.39; window 884–894: ambiguity_score=0.75; win... | landscape-v4 delta=+0.17; anchor present: False |
| pyrimidine_nucleotide | down | moderate | — | window 1605–1615: ambiguity_score=0.00 | landscape-v4 delta=-0.33; anchor present: False |
| glycan_carbohydrate | down | moderate | 1100–1110 | window 935–985: ambiguity_score=0.43; window 580–595: ambiguity_score=0.00; window 880–902: ambiguity_score=0.67; win... | landscape-v4 delta=-0.50; anchor present: False |
| redox_metabolite | flat | low | 479–489 | window 1502–1536: ambiguity_score=0.67; window 1153–1163: ambiguity_score=0.75; window 1210–1220: ambiguity_score=0.0... | landscape-v4 delta=+0.00; anchor present: False |
| nucleic_acid_backbone | flat | low | 1075–1090; 1566–1588; 1045–1055 | window 745–813: ambiguity_score=0.44; window 611–621: ambiguity_score=0.50; window 1335–1367: ambiguity_score=0.50; w... | landscape-v4 delta=+0.00; anchor present: True |

### `liver_cancer_unspecified_vs_healthy_serum`

- **Label**: liver_cancer_unspecified vs healthy_control  ·  matrix: serum  ·  substrate: mixed
- **Status**: `direct`  ·  **Overall confidence**: `moderate`
- **Rationale**: Unspecified liver cancer vs healthy; coarser than the HCC-specific object and typically of lower confidence.
- **Ambiguity**: Landscape v4 has 5 non-flat axes for liver_cancer_unspecified (coarse aggregation; use with caution).

| axis | direction | confidence | anchor windows | ambiguity | rationale |
| --- | --- | --- | --- | --- | --- |
| membrane_lipid | down | high | 1109–1151; 1435–1455; 853–887; 1059–1070; 1355–1405 | window 1260–1330: ambiguity_score=0.67; window 635–672: ambiguity_score=0.78; window 535–553: ambiguity_score=0.00; w... | landscape-v4 delta=-0.75; anchor present: True |
| protein_backbone | flat | low | 1540–1680; 550–565; 662–689; 1500–1515 | window 1175–1343: ambiguity_score=0.43; window 1392–1470: ambiguity_score=0.46; window 727–755: ambiguity_score=0.63;... | landscape-v4 delta=+0.00; anchor present: True |
| aromatic_amino_acid | up | moderate | 991–1025; 1590–1605; 1148–1165; 1548–1565; 1512–1522 | window 819–856: ambiguity_score=0.55; window 628–642: ambiguity_score=0.57; window 1192–1208: ambiguity_score=0.50; w... | landscape-v4 delta=+0.50; anchor present: True |
| purine_nucleotide | up | moderate | 715–734 | window 491–501: ambiguity_score=0.00; window 635–645: ambiguity_score=0.39; window 884–894: ambiguity_score=0.75; win... | landscape-v4 delta=+0.50; anchor present: False |
| pyrimidine_nucleotide | down | moderate | — | window 1605–1615: ambiguity_score=0.00 | landscape-v4 delta=-1.00; anchor present: False |
| glycan_carbohydrate | down | moderate | 1100–1110 | window 935–985: ambiguity_score=0.43; window 580–595: ambiguity_score=0.00; window 880–902: ambiguity_score=0.67; win... | landscape-v4 delta=-0.75; anchor present: False |
| redox_metabolite | flat | low | 479–489 | window 1502–1536: ambiguity_score=0.67; window 1153–1163: ambiguity_score=0.75; window 1210–1220: ambiguity_score=0.0... | landscape-v4 delta=+0.00; anchor present: False |
| nucleic_acid_backbone | flat | low | 1075–1090; 1566–1588; 1045–1055 | window 745–813: ambiguity_score=0.44; window 611–621: ambiguity_score=0.50; window 1335–1367: ambiguity_score=0.50; w... | landscape-v4 delta=+0.00; anchor present: True |

### `hypoxanthine_spike_literature`

- **Label**: serum_plus_hypoxanthine vs serum_baseline  ·  matrix: serum  ·  substrate: plasmonic paper Ag / Ag colloid
- **Status**: `approximate`  ·  **Overall confidence**: `moderate`
- **Rationale**: Hypoxanthine is a purine. Literature peak assignments place its ring-breathing mode in the 700–740 cm⁻¹ window (axis: purine_nucleotide). Spiking a serum matrix with hypoxanthine is expected to raise signal on that axis.
- **Ambiguity**: Calibration-literature contrast. Expected direction encoded from 11 matching peak_assignments rows across 3 sources; substrate/matrix match not verified.

| axis | direction | confidence | anchor windows | ambiguity | rationale |
| --- | --- | --- | --- | --- | --- |
| membrane_lipid | flat | low | 1109–1151; 1435–1455; 853–887; 1059–1070; 1355–1405 | window 1260–1330: ambiguity_score=0.67; window 635–672: ambiguity_score=0.78; window 535–553: ambiguity_score=0.00; w... | — |
| protein_backbone | flat | low | 1540–1680; 550–565; 662–689; 1500–1515 | window 1175–1343: ambiguity_score=0.43; window 1392–1470: ambiguity_score=0.46; window 727–755: ambiguity_score=0.63;... | — |
| aromatic_amino_acid | flat | low | 991–1025; 1590–1605; 1148–1165; 1548–1565; 1512–1522 | window 819–856: ambiguity_score=0.55; window 628–642: ambiguity_score=0.57; window 1192–1208: ambiguity_score=0.50; w... | — |
| purine_nucleotide | up | low | 715–734 | window 491–501: ambiguity_score=0.00; window 635–645: ambiguity_score=0.39; window 884–894: ambiguity_score=0.75; win... | analyte peak evidence: adenine/choline-adjacent region used cautiously @ 720 cm⁻¹; adenine/choline-adjacent region us... |
| pyrimidine_nucleotide | flat | low | — | window 1605–1615: ambiguity_score=0.00 | — |
| glycan_carbohydrate | flat | low | 1100–1110 | window 935–985: ambiguity_score=0.43; window 580–595: ambiguity_score=0.00; window 880–902: ambiguity_score=0.67; win... | — |
| redox_metabolite | flat | low | 479–489 | window 1502–1536: ambiguity_score=0.67; window 1153–1163: ambiguity_score=0.75; window 1210–1220: ambiguity_score=0.0... | — |
| nucleic_acid_backbone | flat | low | 1075–1090; 1566–1588; 1045–1055 | window 745–813: ambiguity_score=0.44; window 611–621: ambiguity_score=0.50; window 1335–1367: ambiguity_score=0.50; w... | — |

### `uricase_depletion_literature`

- **Label**: serum_plus_uricase vs serum_baseline  ·  matrix: serum  ·  substrate: Ag colloid
- **Status**: `approximate`  ·  **Overall confidence**: `moderate`
- **Rationale**: Uricase converts uric acid to allantoin. Uric acid's peak assignments straddle the 635, 890, and 1130 cm⁻¹ regions — so depletion should reduce the purine axis primarily but with substrate-dependent leakage into aromatic_amino_acid and glycan_carbohydrate windows.
- **Ambiguity**: Calibration-literature contrast. Expected direction encoded from 11 matching peak_assignments rows across 2 sources; substrate/matrix match not verified.

| axis | direction | confidence | anchor windows | ambiguity | rationale |
| --- | --- | --- | --- | --- | --- |
| membrane_lipid | flat | low | 1109–1151; 1435–1455; 853–887; 1059–1070; 1355–1405 | window 1260–1330: ambiguity_score=0.67; window 635–672: ambiguity_score=0.78; window 535–553: ambiguity_score=0.00; w... | — |
| protein_backbone | flat | low | 1540–1680; 550–565; 662–689; 1500–1515 | window 1175–1343: ambiguity_score=0.43; window 1392–1470: ambiguity_score=0.46; window 727–755: ambiguity_score=0.63;... | — |
| aromatic_amino_acid | mixed | low | 991–1025; 1590–1605; 1148–1165; 1548–1565; 1512–1522 | window 819–856: ambiguity_score=0.55; window 628–642: ambiguity_score=0.57; window 1192–1208: ambiguity_score=0.50; w... | analyte peak evidence: uric acid [ 20 @ 640 cm⁻¹; uric acid according to the literature @ 889 cm⁻¹; uric acid skeleta... |
| purine_nucleotide | down | low | 715–734 | window 491–501: ambiguity_score=0.00; window 635–645: ambiguity_score=0.39; window 884–894: ambiguity_score=0.75; win... | analyte peak evidence: uric acid [ 20 @ 640 cm⁻¹; uric acid according to the literature @ 889 cm⁻¹; uric acid skeleta... |
| pyrimidine_nucleotide | flat | low | — | window 1605–1615: ambiguity_score=0.00 | — |
| glycan_carbohydrate | mixed | low | 1100–1110 | window 935–985: ambiguity_score=0.43; window 580–595: ambiguity_score=0.00; window 880–902: ambiguity_score=0.67; win... | analyte peak evidence: uric acid [ 20 @ 640 cm⁻¹; uric acid according to the literature @ 889 cm⁻¹; uric acid skeleta... |
| redox_metabolite | flat | low | 479–489 | window 1502–1536: ambiguity_score=0.67; window 1153–1163: ambiguity_score=0.75; window 1210–1220: ambiguity_score=0.0... | — |
| nucleic_acid_backbone | flat | low | 1075–1090; 1566–1588; 1045–1055 | window 745–813: ambiguity_score=0.44; window 611–621: ambiguity_score=0.50; window 1335–1367: ambiguity_score=0.50; w... | — |

### `ergothioneine_spike_literature`

- **Label**: serum_plus_ergothioneine vs serum_baseline  ·  matrix: serum  ·  substrate: Ag colloid (cAg)
- **Status**: `approximate`  ·  **Overall confidence**: `moderate`
- **Rationale**: Ergothioneine is a sulfur-containing imidazole metabolite. Literature assignments cluster in the metabolites group and overlap the 720 cm⁻¹ region. Expected direction on redox_metabolite is up, but the imidazole ring mode ~720 cm⁻¹ co-occupies the purine window, so direction on purine_nucleotide is deliberately left as 'mixed'.
- **Ambiguity**: Calibration-literature contrast. Expected direction encoded from 6 matching peak_assignments rows across 1 sources; substrate/matrix match not verified.

| axis | direction | confidence | anchor windows | ambiguity | rationale |
| --- | --- | --- | --- | --- | --- |
| membrane_lipid | flat | low | 1109–1151; 1435–1455; 853–887; 1059–1070; 1355–1405 | window 1260–1330: ambiguity_score=0.67; window 635–672: ambiguity_score=0.78; window 535–553: ambiguity_score=0.00; w... | — |
| protein_backbone | flat | low | 1540–1680; 550–565; 662–689; 1500–1515 | window 1175–1343: ambiguity_score=0.43; window 1392–1470: ambiguity_score=0.46; window 727–755: ambiguity_score=0.63;... | — |
| aromatic_amino_acid | flat | low | 991–1025; 1590–1605; 1148–1165; 1548–1565; 1512–1522 | window 819–856: ambiguity_score=0.55; window 628–642: ambiguity_score=0.57; window 1192–1208: ambiguity_score=0.50; w... | — |
| purine_nucleotide | mixed | low | 715–734 | window 491–501: ambiguity_score=0.00; window 635–645: ambiguity_score=0.39; window 884–894: ambiguity_score=0.75; win... | analyte peak evidence: ergothioneine @ 484 cm⁻¹; ergothioneine @ 484 cm⁻¹; ergothioneine [ 39 @ 1445 cm⁻¹; ergothione... |
| pyrimidine_nucleotide | flat | low | — | window 1605–1615: ambiguity_score=0.00 | — |
| glycan_carbohydrate | flat | low | 1100–1110 | window 935–985: ambiguity_score=0.43; window 580–595: ambiguity_score=0.00; window 880–902: ambiguity_score=0.67; win... | — |
| redox_metabolite | up | low | 479–489 | window 1502–1536: ambiguity_score=0.67; window 1153–1163: ambiguity_score=0.75; window 1210–1220: ambiguity_score=0.0... | analyte peak evidence: ergothioneine @ 484 cm⁻¹; ergothioneine @ 484 cm⁻¹; ergothioneine [ 39 @ 1445 cm⁻¹; ergothione... |
| nucleic_acid_backbone | flat | low | 1075–1090; 1566–1588; 1045–1055 | window 745–813: ambiguity_score=0.44; window 611–621: ambiguity_score=0.50; window 1335–1367: ambiguity_score=0.50; w... | — |
