# SAEL v1 — Anchor-Based Expected-Delta Objects

Expected biochemical shifts derived **from anchor windows**, not from
broad text averages. Each contrast spec resolves to a SAEL expected-delta
object with per-axis direction, per-axis confidence, and anchor windows
used as support.

## Modes

- `analyte_based` — perturbation is a known molecule (spike / depletion).
  SAEL uses assignment-level anchors to LOCATE the analyte's peaks; the
  direction comes from the spec (spike → up, depletion → down). This is
  literature-grounded location + explicit spec direction, never a text-
  inferred contrast prose claim.
- `condition_based` — disease vs reference (e.g. HCC vs healthy serum).
  Requires SAEL contrast-type rows that carry both a matching condition
  and a direction verb. If the corpus has none, SAEL reports
  `status = unavailable` rather than inventing direction.

## Registered contrasts

| contrast_id | mode | condition_a | matrix | status | overall_confidence | # per_axis entries | provenance_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hypoxanthine_spike_literature | analyte_based | serum_plus_hypoxanthine | serum | approximate | moderate | 4 | 15 |
| uricase_depletion_literature | analyte_based | serum_plus_uricase | serum | approximate | moderate | 5 | 17 |
| ergothioneine_spike_literature | analyte_based | serum_plus_ergothioneine | serum | direct | high | 3 | 12 |
| hcc_vs_healthy_serum | condition_based | HCC | serum | unavailable | none | 0 | 0 |
| nafld_vs_healthy_serum | condition_based | NAFLD_NASH | serum | unavailable | none | 0 | 0 |
| cca_vs_healthy_serum | condition_based | cholangiocarcinoma | serum | unavailable | none | 0 | 0 |

## Per-contrast detail

### `hypoxanthine_spike_literature`

- **Condition A**: `serum_plus_hypoxanthine` · **Condition B**: `serum_baseline`
- **Matrix**: serum · **Substrate**: —
- **Status**: `approximate` · **Overall confidence**: `moderate`
- **Rationale**: Hypoxanthine spike into serum. SAEL locates hypoxanthine/adenine literature anchors (ring-breathing near ~725 cm⁻¹); direction is derived from the spike spec, not from disease-contrast prose.
- **Ambiguity summary**: window 628–642 ambiguity_score=0.571; window 1192–1208 ambiguity_score=0.5; window 635–672 ambiguity_score=0.778; window 715–725 ambiguity_score=0.5; window 1261–1330 ambiguity_score=0.429; window 1200–1328 ambiguity_score=0.516; window 1200–1210 ambiguity_score=0.667
- **Anchor windows used**: purine_nucleotide:0715-0734, purine_nucleotide:1323-1333, purine_nucleotide:0635-0645, aromatic_amino_acid:0628-0642, aromatic_amino_acid:1192-1208, membrane_lipid:0635-0672, membrane_lipid:0715-0725, membrane_lipid:1261-1330, protein_backbone:1200-1328, purine_nucleotide:1200-1210

| axis | direction | confidence | supporting windows | ambiguity notes |
| --- | --- | --- | --- | --- |
| aromatic_amino_acid | up | low | aromatic_amino_acid:0628-0642, aromatic_amino_acid:1192-1208 | window 628–642 ambiguity_score=0.571; window 1192–1208 ambiguity_score=0.5 |
| membrane_lipid | up | low | membrane_lipid:0635-0672, membrane_lipid:0715-0725, membrane_lipid:1261-1330 | window 635–672 ambiguity_score=0.778; window 715–725 ambiguity_score=0.5; window 1261–1330 ambigu... |
| protein_backbone | up | low | protein_backbone:1200-1328 | window 1200–1328 ambiguity_score=0.516 |
| purine_nucleotide | up | moderate | purine_nucleotide:0715-0734, purine_nucleotide:1323-1333, purine_nucleotide:0635-0645, purine_nuc... | window 1200–1210 ambiguity_score=0.667 |

### `uricase_depletion_literature`

- **Condition A**: `serum_plus_uricase` · **Condition B**: `serum_baseline`
- **Matrix**: serum · **Substrate**: —
- **Status**: `approximate` · **Overall confidence**: `moderate`
- **Rationale**: Uricase depletes serum uric acid. SAEL locates uric-acid literature anchors; direction derives from the enzymatic depletion spec. Expect confound axes where uric-acid peaks straddle multiple anchor regions.
- **Ambiguity summary**: window 628–642 ambiguity_score=0.571; window 1192–1208 ambiguity_score=0.5; window 635–672 ambiguity_score=0.778; window 1645–1673 ambiguity_score=0.462; window 884–894 ambiguity_score=0.5; window 1200–1328 ambiguity_score=0.516; window 884–894 ambiguity_score=0.5; window 1200–1210 ambiguity_score=0.667; window 1652–1662 ambiguity_score=0.571; window 3010–3020 ambiguity_score=0.0
- **Anchor windows used**: glycan_carbohydrate:0880-0902, protein_backbone:1590-1705, purine_nucleotide:0491-0501, purine_nucleotide:0635-0645, aromatic_amino_acid:0628-0642, aromatic_amino_acid:1192-1208, membrane_lipid:0635-0672, membrane_lipid:1645-1673, protein_backbone:0884-0894, protein_backbone:1200-1328, purine_nucleotide:0884-0894, purine_nucleotide:1200-1210, purine_nucleotide:1652-1662, purine_nucleotide:3010-3020

| axis | direction | confidence | supporting windows | ambiguity notes |
| --- | --- | --- | --- | --- |
| aromatic_amino_acid | down | low | aromatic_amino_acid:0628-0642, aromatic_amino_acid:1192-1208 | window 628–642 ambiguity_score=0.571; window 1192–1208 ambiguity_score=0.5 |
| glycan_carbohydrate | down | moderate | glycan_carbohydrate:0880-0902 | — |
| membrane_lipid | down | low | membrane_lipid:0635-0672, membrane_lipid:1645-1673 | window 635–672 ambiguity_score=0.778; window 1645–1673 ambiguity_score=0.462 |
| protein_backbone | down | low | protein_backbone:1590-1705, protein_backbone:0884-0894, protein_backbone:1200-1328 | window 884–894 ambiguity_score=0.5; window 1200–1328 ambiguity_score=0.516 |
| purine_nucleotide | down | low | purine_nucleotide:0491-0501, purine_nucleotide:0635-0645, purine_nucleotide:0884-0894, purine_nuc... | window 884–894 ambiguity_score=0.5; window 1200–1210 ambiguity_score=0.667; window 1652–1662 ambi... |

### `ergothioneine_spike_literature`

- **Condition A**: `serum_plus_ergothioneine` · **Condition B**: `serum_baseline`
- **Matrix**: serum · **Substrate**: —
- **Status**: `direct` · **Overall confidence**: `high`
- **Rationale**: Ergothioneine spike. SAEL locates ergothioneine literature anchors. Direction derives from the spike spec. Expect axis overlap with the 720 cm⁻¹ imidazole/purine region.
- **Ambiguity summary**: window 474–490 ambiguity_score=0.571; window 1200–1328 ambiguity_score=0.516; window 1440–1450 ambiguity_score=0.615
- **Anchor windows used**: membrane_lipid:1435-1455, redox_metabolite:0479-0489, redox_metabolite:1210-1220, protein_backbone:0474-0490, protein_backbone:1200-1328, redox_metabolite:1440-1450

| axis | direction | confidence | supporting windows | ambiguity notes |
| --- | --- | --- | --- | --- |
| membrane_lipid | up | high | membrane_lipid:1435-1455 | — |
| protein_backbone | up | low | protein_backbone:0474-0490, protein_backbone:1200-1328 | window 474–490 ambiguity_score=0.571; window 1200–1328 ambiguity_score=0.516 |
| redox_metabolite | up | moderate | redox_metabolite:0479-0489, redox_metabolite:1210-1220, redox_metabolite:1440-1450 | window 1440–1450 ambiguity_score=0.615 |

### `hcc_vs_healthy_serum`

- **Condition A**: `HCC` · **Condition B**: `healthy_control`
- **Matrix**: serum · **Substrate**: —
- **Status**: `unavailable` · **Overall confidence**: `none`
- **Rationale**: HCC vs healthy serum. Requires SAEL contrast-type rows with condition_a='HCC' and a direction verb. Currently expected to be unavailable until targeted literature extraction lands.
- **Ambiguity summary**: No SAEL contrast rows with condition_a='HCC' and a direction verb. Status 'unavailable' is the honest answer until targeted extraction lands.
- **Anchor windows used**: _none_

_No per-axis entries (no anchor windows matched or condition evidence absent)._

### `nafld_vs_healthy_serum`

- **Condition A**: `NAFLD_NASH` · **Condition B**: `healthy_control`
- **Matrix**: serum · **Substrate**: —
- **Status**: `unavailable` · **Overall confidence**: `none`
- **Rationale**: NAFLD / NASH vs healthy serum — condition-specific SAEL rows required.
- **Ambiguity summary**: No SAEL contrast rows with condition_a='NAFLD_NASH' and a direction verb. Status 'unavailable' is the honest answer until targeted extraction lands.
- **Anchor windows used**: _none_

_No per-axis entries (no anchor windows matched or condition evidence absent)._

### `cca_vs_healthy_serum`

- **Condition A**: `cholangiocarcinoma` · **Condition B**: `healthy_control`
- **Matrix**: serum · **Substrate**: —
- **Status**: `unavailable` · **Overall confidence**: `none`
- **Rationale**: Cholangiocarcinoma vs healthy serum.
- **Ambiguity summary**: No SAEL contrast rows with condition_a='cholangiocarcinoma' and a direction verb. Status 'unavailable' is the honest answer until targeted extraction lands.
- **Anchor windows used**: _none_

_No per-axis entries (no anchor windows matched or condition evidence absent)._

## How SAEL deltas differ from expected-BSV v2 deltas

- **Source of direction**: SAEL direction comes from the contrast spec
  (spike / depletion) combined with the direction verbs detected in the
  SAEL evidence rows. Expected-BSV v2 pulled direction from the
  `condition_differential_profile.csv` landscape aggregate, which was
  itself a coarse average over a larger evidence pool.
- **Anchor support is explicit**: each per-axis direction lists the
  window_ids that supported it. Expected-BSV v2 attached windows at the
  contrast level, not per axis.
- **Ambiguity is not averaged away**: SAEL reports status =
  'unavailable' when no direction-bearing rows exist. Expected-BSV v2
  would still emit a delta based on landscape averages.
- **Context conditioning**: SAEL filters anchor windows by declared
  matrix and substrate. Expected-BSV v2 did not.
