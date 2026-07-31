# Interpretation Guide — how to read GAIRA's cross-modal metrics

*A practical guide to reading the representation-hierarchy numbers without over- or
under-claiming. For the equations see `HIERARCHY_METRICS_SPECIFICATION.md`.*

## The golden rules

1. **Never read one number.** Ask how far up the hierarchy agreement survives (latent → motif →
   theme → perturbation → matrix).
2. **Never read a raw metric alone.** Raw theme cosine (0.92) and raw rank ρ (0.87) are
   baseline-inflated. Always pair them with the null / separation.
3. **Separate surface physics from biochemistry.** A low Level-1 or an argmax collapse to purine
   is *adsorption physics*, not a representation failure.
4. **Perturbation and matrix are reported only where measured** (3 and 51 analytes respectively).

## Language — use this, not that

| Don't say | Say instead |
|---|---|
| "theme preserved / failed" | "identity-specific preservation is high / selective / weak" |
| "the model recovers X" | "X's biochemical abstraction transfers" / "is pulled into the purine attractor" |
| "SERS agrees with Raman" | "agreement survives to Level *n* of the hierarchy" |
| "adenine transfers poorly" | "adenine shows latent redistribution with retained, dose-responsive abstraction" |
| "purine is wrong here" | "adsorption-driven observation bias toward the purine attractor" |

## How to read each metric

- **L1 latent fingerprint (0–1).** High = coordinates line up. Low = adsorption reshaped the
  spectrum. Not a biochemistry verdict.
- **L2 MSS motif (0–1).** Mid-level structure. Between L1 and L3 by construction.
- **L3a theme raw (0–1).** *Ignore in isolation.* ≥0.9 for nearly everything — baseline.
- **L3b theme identity (−1–1) + separation.** The honest theme number. `separation > 0` ⇒ resembles
  its own Raman more than a random analyte's. Median only +0.014 → selective.
- **L4 rank ρ (−1–1) + rank_separation.** Ordering of all 11 themes. Raw ρ high but ≈ its null;
  `rank_separation` (median +0.010, positive 34/51) is the identity part.
- **L5 top-3 overlap (0/⅓/⅔/1).** The interpretable middle-ground. 0.67 ≈ 2 of 3 leading themes kept.
- **L6 argmax (bool).** Strict, unstable; on Ag almost always `nucleic_purine`. A "True" mostly
  means the analyte was already purine-dominant.
- **ΔPurine.** How much silver pulls the analyte toward purine. Positive for weak adsorbers.
- **Perturbation.** If present, the strongest evidence (functional). If absent: "not measured."
- **Matrix.** Serum recoverability; a *weak* per-analyte predictor from pure transfer (r=0.17, ns).

## Worked examples

- **adenine** — L1 0.36 (weak, latent redistribution), L3b +0.73, argmax purine→purine, ΔPurine
  −0.12, dose ρ=0.996. *Read:* latent redistribution with a retained, dynamically-validated
  purine abstraction. One of the best-validated analytes despite a low L1.
- **guanine** — L1 0.63, L3b +0.92 (identity high), ΔPurine −0.25 (sheds excess purine).
  *Read:* identity-specific preservation across the hierarchy.
- **glucose** — L1 0.26, L3b −0.46, argmax saccharide→purine, ΔPurine +0.02. *Read:* adsorption-
  driven observation bias; the saccharide abstraction is not recovered on silver.
- **uracil** — L1 0.06 (lowest), L3b +0.16, argmax pyrimidine→purine. *Read:* strong latent
  redistribution and attractor capture; only the coarsest structure survives.

## One-line summary

Report the **hierarchy**, pair every raw metric with its **null**, describe failures as
**surface physics**, and treat **dynamic perturbation** as the gold standard — available for only
three analytes.
