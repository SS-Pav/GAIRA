# glycerol  ·  cross-modal transfer card
*Family: polyol · expected theme: saccharide_glycan · 4 Raman / 5 Ag-SERS spectra · quadrant: Q4 poor transfer (both)*

## Level 1 — Latent fingerprint preservation
- component cosine **0.47** (Moderate) — similarity of the 24 NMF coordinates (adsorption-limited)

## Level 2 — Biochemical theme preservation
- dominant theme: Raman **saccharide_glycan** → Ag-SERS **nucleic_purine**  (not preserved)
- theme cosine: raw 0.8044 · **distinctive -0.7601** (null -0.7348, separation -0.0253, self-nearest False)
- expected theme rank: Raman #1 → Ag-SERS #2 (top-3 retained: True)
- MSS motif cosine 0.5963 (dominant motif preserved: False)
- redistribution: JSD 0.0715, L1 0.4481; gained **nucleic_purine**, lost **saccharide_glycan** (motif +carboxylate_organic_acid / −glycan_co_network)

## Level 3 — Perturbation sensitivity
- **Not tested** — no controlled perturbation series exists for this analyte

## Level 4 — Matrix recoverability (serum)
- serum recovery tier **moderate** (displacement 0.0206, direction cos 0.7878, vs pure-SERS 0.2557)

*OOD(SERS) 0.096 · confidence(SERS) 0.2084. Computed on the frozen atlas 09ed804a…; Levels 3-4 only where measured.*