# SAEL v1 — Anchor Window Registry

Built from the raw SAEL anchor evidence extraction (see
`outputs/gaira_spectral_anchor_evidence_raw.csv`).

## Classification rules

- **anchor** — ≥ 2 distinct sources, ambiguity ≤ 0.4, AND either a direction
  claim or a multi-source assignment supporting the window
- **secondary** — ≥ 1 source and matches a canonical axis-hint range, OR
  multi-source assignment without direction
- **ambiguous** — ambiguity > 0.4, conflicting up/down from multiple sources,
  or insufficient support

Ambiguity score = fraction of peaks inside the window that are attributed to a
different BSV axis (cross-axis overlap).

## Totals

- **anchor**: **13**
- **secondary**: **25**
- **ambiguous**: **26**

## How this differs from the previous expected-BSV v2 registry

SAEL v1 windows carry **per-window contrast metadata** — direction_distribution,
matrix_distribution, substrate_distribution, condition_count — that the prior
expected-BSV v2 registry did not. In practice, only 3 of the current
extraction rows carry a direction verb, so most direction_distribution cells
are empty. This is honest: the underlying corpus has essentially no
explicit contrast-direction sentences; SAEL v1 surfaces the gap rather than
imputing direction.

## Per-axis detail

### `aromatic_amino_acid`

- **anchors**: 4
- **secondary**: 5
- **ambiguous**: 4

**Anchors**

| window_id | start_cm1 | end_cm1 | source_count | condition_count | direction_distribution | ambiguity_score | priority_tags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aromatic_amino_acid:0814-0866 | 814 | 866 | 10 | 1 | — | 0.2 | aromatic_aa |
| aromatic_amino_acid:0991-1025 | 991 | 1025 | 8 | 1 | up=1 | 0 | aromatic_aa; glycan |
| aromatic_amino_acid:1480-1522 | 1480 | 1522 | 3 | 1 | — | 0.4 | aromatic_aa; lipid; protein |
| aromatic_amino_acid:1548-1605 | 1548 | 1605 | 8 | 1 | down=1 | 0.207 | aromatic_aa; protein; lipid |

**Secondary** (top 5)

| window_id | start_cm1 | end_cm1 | source_count | direction_distribution | ambiguity_score |
| --- | --- | --- | --- | --- | --- |
| aromatic_amino_acid:0691-0710 | 691 | 710 | 2 | — | 0 |
| aromatic_amino_acid:0745-0790 | 745 | 790 | 5 | — | 0.2 |
| aromatic_amino_acid:0898-0908 | 898 | 908 | 5 | — | 0.333 |
| aromatic_amino_acid:1070-1080 | 1070 | 1080 | 4 | — | 0 |
| aromatic_amino_acid:1148-1165 | 1148 | 1165 | 3 | — | 0.25 |

### `glycan_carbohydrate`

- **anchors**: 2
- **secondary**: 2
- **ambiguous**: 0

**Anchors**

| window_id | start_cm1 | end_cm1 | source_count | condition_count | direction_distribution | ambiguity_score | priority_tags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| glycan_carbohydrate:0880-0902 | 880 | 902 | 4 | 1 | — | 0.333 | glycan; aromatic_aa |
| glycan_carbohydrate:1100-1110 | 1100 | 1110 | 3 | 1 | — | 0 | glycan; aromatic_aa |

**Secondary** (top 5)

| window_id | start_cm1 | end_cm1 | source_count | direction_distribution | ambiguity_score |
| --- | --- | --- | --- | --- | --- |
| glycan_carbohydrate:0580-0595 | 580 | 595 | 2 | — | 0 |
| glycan_carbohydrate:0935-0985 | 935 | 985 | 3 | — | 0.143 |

### `membrane_lipid`

- **anchors**: 3
- **secondary**: 9
- **ambiguous**: 7

**Anchors**

| window_id | start_cm1 | end_cm1 | source_count | condition_count | direction_distribution | ambiguity_score | priority_tags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| membrane_lipid:1109-1151 | 1109 | 1151 | 7 | 1 | — | 0 | lipid; glycan; aromatic_aa |
| membrane_lipid:1355-1405 | 1355 | 1405 | 5 | 2 | — | 0 | lipid; aromatic_aa |
| membrane_lipid:1435-1455 | 1435 | 1455 | 6 | 1 | — | 0.062 | lipid; aromatic_aa |

**Secondary** (top 5)

| window_id | start_cm1 | end_cm1 | source_count | direction_distribution | ambiguity_score |
| --- | --- | --- | --- | --- | --- |
| membrane_lipid:0453-0463 | 453 | 463 | 2 | — | 0 |
| membrane_lipid:0535-0553 | 535 | 553 | 2 | — | 0 |
| membrane_lipid:0745-0755 | 745 | 755 | 4 | — | 0.3 |
| membrane_lipid:0810-0822 | 810 | 822 | 3 | — | 0.333 |
| membrane_lipid:0853-0875 | 853 | 875 | 3 | — | 0.333 |

### `nucleic_acid_backbone`

- **anchors**: 1
- **secondary**: 3
- **ambiguous**: 1

**Anchors**

| window_id | start_cm1 | end_cm1 | source_count | condition_count | direction_distribution | ambiguity_score | priority_tags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nucleic_acid_backbone:1045-1055 | 1045 | 1055 | 3 | 1 | — | 0 | aromatic_aa; nucleic_bb; glycan |

**Secondary** (top 5)

| window_id | start_cm1 | end_cm1 | source_count | direction_distribution | ambiguity_score |
| --- | --- | --- | --- | --- | --- |
| nucleic_acid_backbone:0611-0621 | 611 | 621 | 3 | — | 0 |
| nucleic_acid_backbone:0651-0665 | 651 | 665 | 2 | — | 0 |
| nucleic_acid_backbone:0775-0813 | 775 | 813 | 4 | — | 0.143 |

### `protein_backbone`

- **anchors**: 0
- **secondary**: 2
- **ambiguous**: 5

**Anchors**

_none_

**Secondary** (top 5)

| window_id | start_cm1 | end_cm1 | source_count | direction_distribution | ambiguity_score |
| --- | --- | --- | --- | --- | --- |
| protein_backbone:0550-0565 | 550 | 565 | 2 | — | 0.25 |
| protein_backbone:1590-1705 | 1590 | 1705 | 9 | — | 0.375 |

### `purine_nucleotide`

- **anchors**: 2
- **secondary**: 2
- **ambiguous**: 5

**Anchors**

| window_id | start_cm1 | end_cm1 | source_count | condition_count | direction_distribution | ambiguity_score | priority_tags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| purine_nucleotide:0715-0734 | 715 | 734 | 4 | 1 | — | 0.1 | purine; aromatic_aa |
| purine_nucleotide:1323-1333 | 1323 | 1333 | 3 | 1 | — | 0.286 | purine; lipid; aromatic_aa |

**Secondary** (top 5)

| window_id | start_cm1 | end_cm1 | source_count | direction_distribution | ambiguity_score |
| --- | --- | --- | --- | --- | --- |
| purine_nucleotide:0491-0501 | 491 | 501 | 2 | — | 0 |
| purine_nucleotide:0635-0645 | 635 | 645 | 2 | — | 0.385 |

### `pyrimidine_nucleotide`

- **anchors**: 0
- **secondary**: 1
- **ambiguous**: 0

**Anchors**

_none_

**Secondary** (top 5)

| window_id | start_cm1 | end_cm1 | source_count | direction_distribution | ambiguity_score |
| --- | --- | --- | --- | --- | --- |
| pyrimidine_nucleotide:1605-1615 | 1605 | 1615 | 3 | — | 0 |

### `redox_metabolite`

- **anchors**: 1
- **secondary**: 1
- **ambiguous**: 4

**Anchors**

| window_id | start_cm1 | end_cm1 | source_count | condition_count | direction_distribution | ambiguity_score | priority_tags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| redox_metabolite:0479-0489 | 479 | 489 | 2 | 0 | — | 0.333 | redox_sulfur |

**Secondary** (top 5)

| window_id | start_cm1 | end_cm1 | source_count | direction_distribution | ambiguity_score |
| --- | --- | --- | --- | --- | --- |
| redox_metabolite:1210-1220 | 1210 | 1220 | 3 | — | 0 |

