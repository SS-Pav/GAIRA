# Serum Spike-in Recoverability Audit (Part 4)

Audited from the committed Spike Validation outputs, not assumed from the UI. Root
question: is adenine really "poorly recoverable" and ergothioneine "strongly
recoverable", and are pure-analyte and serum-spike regimes being conflated?

## Data mapping — verified

`phase7_serum_vs_pure.csv` (53 analytes) and `phase3_projection_spiked_serum.csv` were
cross-checked. Adenine maps to a single serum arm at **0.4 µM**, correctly labelled
`adenine` (not adenosine / hypoxanthine / urate / blank). Ergothioneine maps to its
own 5 µM serum arm (not the pure calibration dataset). No mislabelling found.

## Recoverability definition — separated, not one opaque number

The **validated primary** metric (from the stress test) is
`direction_agreement = cos(serum-spike displacement, that analyte's pure-SERS
fingerprint)`; it defines the tiers (strong ≥0.35, partial ≥0.10, poor). The other
evidence terms are reported **separately** (Part 4B), never merged with invented
weights:

| term | source | meaning |
|---|---|---|
| direction agreement* | `cos_spike_vs_pureSERS` | does the spike move toward the analyte's own signature |
| detectability | `spike_displacement_norm` | magnitude of the spike-induced move |
| reproducibility | `replicate_direction_cos` | cross-replicate direction consistency |
| matrix dominance | engine `background_matrix` share | how much the serum/Ag background dominates |

**Ablation.** Ranking by each single term:
- by direction agreement: hypoxanthine, xanthine, creatinine, ergothioneine, ascorbate
- by detectability: xanthine, hypoxanthine, oleate, ergothioneine, guanine
- by reproducibility: xanthine, ergothioneine, guanine, hypoxanthine, **phenylalanine**

Reproducibility alone ranks **phenylalanine** highly — it moves *consistently* but in
the *wrong* direction (toward the matrix). This is why direction agreement, not
detectability or reproducibility, is the meaningful recoverability criterion.

## Adenine audit (Part 4D) — conclusion

- **Pure adenine**: strong — the purine theme rises 0.183 → 0.320 over 0–1.8 µM
  (cAg@785).
- **Serum adenine**: spiked at only **0.4 µM**, direction agreement 0.08, tier poor.
- **Concentration confound**: 0.4 µM is **12–25× lower** than the recoverable analytes
  (ergothioneine 5, hypoxanthine 10, xanthine 50 µM). The UI was not "wrong" — the low
  score is real for that arm — but the *interpretation* must include the concentration.

**Verdict:** adenine is **strong in pure but weak in serum, consistent with the low
spike concentration plus surface competition / matrix masking** (Part 4C category 2).
It is NOT evidence of poor adenine adsorption in general. Contrast **phenylalanine**,
spiked at **78 µM** yet still poor — a genuine adsorption/matrix failure, a different
mode. The page now shows the concentration column prominently and states this.

## Ergothioneine audit (Part 4E) — conclusion

- **Pure ergothioneine**: strong sulfur response (near-textbook Langmuir on Page 4).
- **Serum ergothioneine**: 5 µM, direction agreement 0.53, replicate consistency 0.99,
  tier strong — the serum spike moves toward ergothioneine's own pure-SERS fingerprint.

**Verdict:** strong in pure **and** strong in serum. The label is supported by the
matched serum evidence (not merely by the pure-dose Langmuir), so the "strong serum
recoverability" claim is justified.

## Confidence vs recoverability (Part 5)

The engine's per-theme `confidence` bundles atlas fit, evidence concentration and OOD.
It is NOT redefined; instead the demo now shows the underlying quantities separately:
**atlas support (1−OOD)**, **OOD**, **theme specificity** (evidence concentration),
**replicate reliability**, and **matrix recoverability** (the analyte-specific serum
metric). For unknown biological spectra, matrix recoverability is **unavailable** and
is never scored as positive — the composite excludes it and flags it.

## Limitations

- Recoverability is defined only where matched serum-spike evidence exists; it is not
  extrapolated to biological cohorts.
- The concentration confound means the adenine serum arm cannot be used to rank adenine
  adsorption against higher-concentration analytes — only same-concentration
  comparisons are fair.
