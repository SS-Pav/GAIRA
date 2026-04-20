# Expected-BSV Anchor Windows (v2)

Per-axis anchor / secondary / ambiguous windows clustered from
`peak_assignments.peak_cm`. Windows are honest — most raw peak clusters end
up ambiguous because the underlying literature evidence is sparse or
cross-axis.

## Totals

- **anchor** windows: **7**
- **secondary** windows: **13**
- **ambiguous** windows: **61**

## Clustering rules

- Peaks within an axis are sorted; a new cluster begins whenever two
  consecutive `peak_cm` values differ by more than **25 cm⁻¹**.
- Each cluster is padded by **±5 cm⁻¹** to avoid claiming false precision.
- Classification:
  - `anchor` — ≥ 3 distinct sources, ≥ 2 distinct molecules, ambiguity ≤ 0.4
  - `secondary` — ≥ 2 sources, OR anchor-hint match with thinner support
  - `ambiguous` — everything else, or ambiguity > 0.4

Ambiguity score = fraction of peaks inside the window range that are attributed
to a DIFFERENT BSV axis (i.e. cross-axis cm overlap).

## Per-axis detail

### `aromatic_amino_acid`

**Anchors** (2) — ≥3 sources, ≥2 molecules, ambiguity ≤ 0.4.

| start_cm | end_cm | n_peak_rows | n_sources | n_molecules | top_molecules | anchor_hint_match | ambiguity_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 991 | 1025 | 8 | 6 | 7 | aromatic ring breathing; phenylalanine-like aromatic marker used cautiously; phenylalanine | matched 1000–1010: phenylalanine ring breathing | 0.111 |
| 1590 | 1605 | 4 | 3 | 3 | aromatic ring vibration; C=C stretching vibrations in aromatic amino acids like tryptophan and tyrosine 1; phenyl/tyrosine-like aromatic region | matched 1520–1600: tryptophan / ring C=C | 0.214 |

**Secondary** (3) — ≥2 sources OR anchor-hint match with thinner evidence.

| start_cm | end_cm | n_peak_rows | n_sources | n_molecules | top_molecules | anchor_hint_match | ambiguity_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1148 | 1165 | 3 | 2 | 3 | tyrosine; tyrosine and phenylalanine; tyrosine/phenyl-like contribution used cautiously |  | 0.25 |
| 1548 | 1565 | 3 | 2 | 3 | tryptophan; tryptophan has exhibited a significantly decreased intensity ( p *<0; tryptophan/tyrosine-like aromatic region | matched 1520–1600: tryptophan / ring C=C | 0.4 |
| 1512 | 1522 | 1 | 1 | 1 | tyrosine | matched 1520–1600: tryptophan / ring C=C | 0 |

**Ambiguous** (9) — ambiguity score > 0.4 or single-source single-molecule.

| start_cm | end_cm | n_peak_rows | n_sources | top_molecules | ambiguity_score |
| --- | --- | --- | --- | --- | --- |
| 819 | 856 | 5 | 4 | tyrosine Fermi resonance; the ring breathing mode of tyrosine; aromatic amino acid signal | 0.545 |
| 628 | 642 | 5 | 2 | tyrosine; phenylalanine | 0.571 |
| 1192 | 1208 | 2 | 2 | Tryptophan; tryptophan/phenylalanine | 0.5 |
| 745 | 758 | 2 | 1 | tryptophan-like aromatic contribution | 0.8 |

### `glycan_carbohydrate`

**Anchors** (0) — ≥3 sources, ≥2 molecules, ambiguity ≤ 0.4.

_none_

**Secondary** (1) — ≥2 sources OR anchor-hint match with thinner evidence.

| start_cm | end_cm | n_peak_rows | n_sources | n_molecules | top_molecules | anchor_hint_match | ambiguity_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1100 | 1110 | 1 | 1 | 1 | polysaccharide-associated C-O region | matched 1080–1140: sugar C-O-C (overlaps PO2⁻) | 0 |

**Ambiguous** (5) — ambiguity score > 0.4 or single-source single-molecule.

| start_cm | end_cm | n_peak_rows | n_sources | top_molecules | ambiguity_score |
| --- | --- | --- | --- | --- | --- |
| 935 | 985 | 4 | 2 | polysaccharide structure; polysaccharide-associated region; monosaccharide-associated region | 0.429 |
| 580 | 595 | 2 | 1 | monosaccharide-associated low-wavenumber feature | 0 |
| 880 | 902 | 2 | 1 | monosaccharide-associated region | 0.667 |
| 1025 | 1035 | 1 | 1 | C-O related carbohydrate region | 0 |

### `membrane_lipid`

**Anchors** (2) — ≥3 sources, ≥2 molecules, ambiguity ≤ 0.4.

| start_cm | end_cm | n_peak_rows | n_sources | n_molecules | top_molecules | anchor_hint_match | ambiguity_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1109 | 1151 | 10 | 5 | 6 | lipid-associated signal; the carbon‒carbon bonding mode of lipids; stretch of breast lipid | matched 1140–1200: lipid CH2 twist | 0.214 |
| 1435 | 1455 | 8 | 4 | 6 | CH 2 /CH 3 bending vibration of lipids [ 20; bending vibration of lipids [ 20; lipid and protein | matched 1380–1450: lipid δCH2/CH3 | 0.25 |

**Secondary** (3) — ≥2 sources OR anchor-hint match with thinner evidence.

| start_cm | end_cm | n_peak_rows | n_sources | n_molecules | top_molecules | anchor_hint_match | ambiguity_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 853 | 887 | 3 | 2 | 2 | lipid-associated mid region; phosphatidylcholine |  | 0.4 |
| 1059 | 1070 | 2 | 2 | 2 | Lipids; membrane-associated region |  | 0 |
| 1355 | 1405 | 4 | 2 | 4 | lipid-associated CH region; Lipids; sterol/hormone-associated CH region | matched 1380–1450: lipid δCH2/CH3 | 0.375 |

**Ambiguous** (13) — ambiguity score > 0.4 or single-source single-molecule.

| start_cm | end_cm | n_peak_rows | n_sources | top_molecules | ambiguity_score |
| --- | --- | --- | --- | --- | --- |
| 1260 | 1330 | 7 | 3 | fatty acids; lipid unsaturation-associated region; lipid CH deformation region | 0.667 |
| 635 | 672 | 3 | 2 | fatty-acid-associated low-mid region; assigned to proteins Phe, Tyr, polysaccharides, Tyr, and lipids [carbonyl ν(C=O | 0.778 |
| 535 | 553 | 2 | 1 | membrane-associated low-wavenumber band | 0 |
| 691 | 710 | 2 | 1 | sterol or hormone-like low aromatic region | 0 |

### `nucleic_acid_backbone`

**Anchors** (2) — ≥3 sources, ≥2 molecules, ambiguity ≤ 0.4.

| start_cm | end_cm | n_peak_rows | n_sources | n_molecules | top_molecules | anchor_hint_match | ambiguity_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1075 | 1090 | 3 | 3 | 3 | dATP; the symmetric stretching of PO- 2 in nucleic acids; phosphate-backbone-associated region | matched 1020–1080: C-N / ribose | 0.286 |
| 1566 | 1588 | 4 | 3 | 4 | differences in nucleic acid base vibrations; base-associated aromatic region; proteins and DNA |  | 0.4 |

**Secondary** (1) — ≥2 sources OR anchor-hint match with thinner evidence.

| start_cm | end_cm | n_peak_rows | n_sources | n_molecules | top_molecules | anchor_hint_match | ambiguity_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1045 | 1055 | 1 | 1 | 1 | DNA/RNA-associated mid region | matched 1020–1080: C-N / ribose | 0 |

**Ambiguous** (10) — ambiguity score > 0.4 or single-source single-molecule.

| start_cm | end_cm | n_peak_rows | n_sources | top_molecules | ambiguity_score |
| --- | --- | --- | --- | --- | --- |
| 745 | 813 | 10 | 5 | nucleic acids; base-associated aromatic region; DNA/RNA-associated backbone region | 0.444 |
| 611 | 621 | 2 | 2 | intjnano International Journal of Nanomedicine Int J Nanomedicine Dove Press PMC6; intjnano International Journal of Nanomedicine Int J Nanomedicine Dove Press PMC1 | 0.5 |
| 1335 | 1367 | 2 | 2 | nucleic-acid-associated SERS region; nucleic acids and proteins | 0.5 |
| 453 | 463 | 2 | 1 | lipid and nucleic acid mixed signal | 0 |

### `protein_backbone`

**Anchors** (1) — ≥3 sources, ≥2 molecules, ambiguity ≤ 0.4.

| start_cm | end_cm | n_peak_rows | n_sources | n_molecules | top_molecules | anchor_hint_match | ambiguity_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1540 | 1680 | 22 | 10 | 17 | protein-associated signal; amide (general); amide II adjacent protein region |  | 0.34 |

**Secondary** (3) — ≥2 sources OR anchor-hint match with thinner evidence.

| start_cm | end_cm | n_peak_rows | n_sources | n_molecules | top_molecules | anchor_hint_match | ambiguity_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 550 | 565 | 4 | 2 | 3 | protein backbone-associated region; protein conformational change; disulfide S-S stretching |  | 0 |
| 662 | 689 | 3 | 2 | 2 | protein-related low-mid region; phenylalanine |  | 0.25 |
| 1500 | 1515 | 2 | 2 | 2 | protein aromatic/amide-adjacent region; amide I | matched 1450–1520: Amide II adjacent / δCH2 | 0.333 |

**Ambiguous** (13) — ambiguity score > 0.4 or single-source single-molecule.

| start_cm | end_cm | n_peak_rows | n_sources | top_molecules | ambiguity_score |
| --- | --- | --- | --- | --- | --- |
| 1175 | 1343 | 20 | 10 | protein-associated signal; amide III; protein-associated mid region | 0.432 |
| 1392 | 1470 | 7 | 4 | protein-associated signal; lipid-protein interaction; protein-associated CH region | 0.455 |
| 727 | 755 | 7 | 3 | hemoglobin; protein-associated ring or backbone contribution; protein-associated signal | 0.632 |
| 814 | 866 | 7 | 3 | tyrosine-like protein-associated region; tyrosine/protein-associated region; tyrosine ring breathing | 0.533 |

### `purine_nucleotide`

**Anchors** (0) — ≥3 sources, ≥2 molecules, ambiguity ≤ 0.4.

_none_

**Secondary** (1) — ≥2 sources OR anchor-hint match with thinner evidence.

| start_cm | end_cm | n_peak_rows | n_sources | n_molecules | top_molecules | anchor_hint_match | ambiguity_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 715 | 734 | 8 | 2 | 4 | hypoxanthine; adenine/choline-adjacent region used cautiously; assigned to hypoxanthine | matched 720–740: adenine / hypoxanthine ring breathing | 0.2 |

**Ambiguous** (6) — ambiguity score > 0.4 or single-source single-molecule.

| start_cm | end_cm | n_peak_rows | n_sources | top_molecules | ambiguity_score |
| --- | --- | --- | --- | --- | --- |
| 491 | 501 | 1 | 1 | uric acid | 0 |
| 635 | 645 | 7 | 1 | uric acid; uric acid [ 20; uric acid skeletal ring deformation | 0.385 |
| 884 | 894 | 1 | 1 | uric acid according to the literature | 0.75 |
| 1200 | 1210 | 1 | 1 | uric acid or hypoxanthine | 0.667 |

### `pyrimidine_nucleotide`

**Anchors** (0) — ≥3 sources, ≥2 molecules, ambiguity ≤ 0.4.

_none_

**Secondary** (0) — ≥2 sources OR anchor-hint match with thinner evidence.

_none_

**Ambiguous** (1) — ambiguity score > 0.4 or single-source single-molecule.

| start_cm | end_cm | n_peak_rows | n_sources | top_molecules | ambiguity_score |
| --- | --- | --- | --- | --- | --- |
| 1605 | 1615 | 1 | 1 | Phe, Tyr, and cytosine | 0 |

### `redox_metabolite`

**Anchors** (0) — ≥3 sources, ≥2 molecules, ambiguity ≤ 0.4.

_none_

**Secondary** (1) — ≥2 sources OR anchor-hint match with thinner evidence.

| start_cm | end_cm | n_peak_rows | n_sources | n_molecules | top_molecules | anchor_hint_match | ambiguity_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 479 | 489 | 4 | 1 | 1 | ergothioneine | matched 450–540: S-S disulfide | 0.333 |

**Ambiguous** (4) — ambiguity score > 0.4 or single-source single-molecule.

| start_cm | end_cm | n_peak_rows | n_sources | top_molecules | ambiguity_score |
| --- | --- | --- | --- | --- | --- |
| 1502 | 1536 | 2 | 2 | carotenoids with various clinical implications such as cancer treatment; carotenoids | 0.667 |
| 1153 | 1163 | 1 | 1 | β-carotene | 0.75 |
| 1210 | 1220 | 1 | 1 | ergothioneine | 0 |
| 1440 | 1450 | 1 | 1 | ergothioneine [ 39 | 0.769 |

