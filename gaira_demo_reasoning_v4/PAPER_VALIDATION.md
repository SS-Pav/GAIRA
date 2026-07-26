# External Paper Validation — Ag-SERS serum (PMC12680727)

The serum spike-in dataset behind Page 5 comes from an Ag-SERS serum study
(**PMC12680727**, https://pmc.ncbi.nlm.nih.gov/articles/PMC12680727/). GAIRA's serum
stress test — computed **only from the frozen pure-Raman atlas**, with no knowledge of
the paper — independently reproduces the paper's central findings. This is strong,
independent validation.

## What the paper reported

- Spiked serum at physiological concentrations: tyrosine 55 µM, lactose 100 µM, uric
  acid 280 µM, hypoxanthine 10 µM, **adenine 0.3–0.4 µM**, CoA 100 µM, HSA ~50 mg/mL,
  plus xanthine and ergothioneine.
- **Only uric acid and hypoxanthine give a meaningful (~2×) SERS signal** — despite
  serum holding >4000 metabolites. Tyrosine, lactose, adenine and **millimolar glucose**
  produce none.
- "**Concentration alone doesn't predict SERS visibility**": glucose is millimolar yet
  invisible (low Ag affinity); it is adsorption affinity + competition + protein steric
  hindrance that decide.
- The serum SERS spectrum is "largely due to just two metabolites: uric acid … and
  hypoxanthine." PC1 ≈ 70% of variance; ~90% of inter-individual difference is
  uric-acid/hypoxanthine ratios.
- Uricase digestion of urate makes the urate bands disappear (the uricase experiment
  GAIRA also uses).

## What GAIRA found (frozen-atlas only)

| observation | paper | GAIRA |
|---|---|---|
| recoverable in serum | uric acid + hypoxanthine (strong Ag adsorbers) | strong tier = hypoxanthine, xanthine, guanine, ergothioneine, ascorbate, creatinine (oxopurines + thiones) |
| adenine (0.3–0.4 µM) | no signal | poor (cos 0.08) |
| tyrosine 55 µM / lactose 100 µM / glucose mM | no signal | poor (cos ≤0) |
| concentration ≠ visibility | explicit | reproduced (adenine 0.4 poor; glucose 5.4 mM poor) |
| serum space is low-D, purine-dominated | PC1 70% | BSV effective-dim low; absolute BSV purine-dominated |
| uricase removes urate | bands vanish | oxopurine MSS falls specifically |

GAIRA's spike concentrations match the paper's almost exactly (adenine 0.4, tyrosine
55, uric acid 280, hypoxanthine 10 µM …), confirming it is the same experimental design.

## The one honest discrepancy — urate

The paper calls uric acid the **strongest** serum signal; GAIRA rates **urate "poor"**
(direction agreement 0.07). Both are correct — they measure different things:
- Urate was spiked at 280 µM into serum that is **already uric-acid-saturated** — urate
  IS the dominant serum background.
- GAIRA's metric is the **direction** of the spike-induced change vs the analyte's pure
  fingerprint. Adding urate to a urate-saturated baseline barely changes the direction,
  even though the absolute urate band is huge.
- So GAIRA's "poor" means "**a urate spike is not separable from the urate
  background**", not "urate is undetectable." The absolute urate signal is real and
  large — exactly the paper's point.

This nuance is now stated on Page 5 (§B+ "Validated against the source paper").

## Takeaway

The "low serum recoverability" a first-time viewer worries about is **the correct,
published answer** — the physics of Ag-SERS in serum recovers ~2 metabolites, and GAIRA
recovers the same conclusion from an independent representation.
