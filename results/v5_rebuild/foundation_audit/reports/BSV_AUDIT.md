# BSV_AUDIT
### The Biochemical State Vector — from 24 components to an 11-axis reading

*Part 8 of the GAIRA Foundation Model audit. The BSV is the top of the hierarchy: the
deterministic transform from frozen component coordinates to a biochemical theme vector
and its radar. Source: `gaira.engine.bsv` (equations in the module docstring) +
`gaira.engine.ontology`. Nothing is learned; the only inputs are the frozen projection and
the versioned ontology weights.*

---

## 1. The construction, from first principles

A query spectrum is projected into the frozen atlas as 24 non-negative activations
`a ∈ ℝ²⁴` (NNLS onto the fixed components — Part 5). From there:

```
coord_j        = a_j / Σ_k a_k                      # L1 evidence share per component (Σ=1)
z_j            = (coord_j − center_j) / spread_j    # robust z vs the reference frame (median/MAD)
W_{j,t}                                             # component→theme weight, rows sum to 1  (24×13)

composition_t  = Σ_j W_{j,t} · coord_j              # theme's SHARE of evidence   (≥0, Σ_t ≈ 1)
elevation_t    = Σ_j W_{j,t} · z_j                  # how ELEVATED the theme is vs pure references (signed)
display_t      = 0.5 + 0.5·tanh(elevation_t / 3)    # bounded 0..1 radar value
```

Two matrices define everything downstream and both are frozen/versioned:

- **W (24 × 13)** — the component→theme map. Each **row** (a component) is a probability
  distribution over themes (sums to 1): a component spreads its evidence across the themes
  it supports. Verified: all 24 rows sum to 1.0. This is the *many-to-many* bridge — a
  component feeds several themes; a theme draws from several components (e.g.
  `nucleic_purine ← c15·0.67, c3·0.47, c23·0.34, c13·0.31`).
- **The reference frame** — per-component median/MAD `center_j`, `spread_j` computed once
  over the pure reference corpus. It is what makes `z` and `elevation` mean "vs pure
  references."

There are **13 themes, of which 11 are biochemical**; `background_matrix` and
`unknown_mixed` are computed and REPORTED (a high value lowers overall confidence) but are
**excluded from the radar axes**. So the radar is **11-dimensional**.

**Confidence** per theme = `stability_t · evidence_concentration_t · (1 − OOD)`:
weighted component stability, times how concentrated (low-entropy) the theme's evidence is,
times a penalty for out-of-distribution inputs. Overall confidence = mean biochemical-theme
confidence × (1 − matrix/unknown share).

---

## 2. What each radar axis actually means

The 11 biochemical axes and their top component evidence (from W):

| Axis (theme) | Meaning | Top components | Grounding |
|---|---|---|---|
| nucleic_purine | purine ring / adenine-guanine-oxopurine | c15·.67, c3·.47, c13·.31 | **strong** (best-validated) |
| nucleic_pyrimidine | pyrimidine ring (U/C/T) | c17·.47, c13·.30 | strong |
| protein_peptide | amide backbone + Phe | c2·.48, c6·.22, c8·.19 | strong |
| aromatic_amino_acid | Phe/Tyr/Trp ring modes | c5·.19, c11·.17, c6·.15 | strong (shares Phe with protein) |
| lipid_acyl | acyl-chain CH₂ / ester | c1·.60, c16·.57, c7·.49 | strong (largest variance) |
| sterol_membrane | sterol ring system | c3·.12, c8·.10 | **weak** (coupled to acyl + adenine c3) |
| saccharide_glycan | sugar C–O–C ring modes | c10·.55, c4·.55, c12·.47 | strong |
| organic_acid_metabolism | carboxylate / small acids | c14·.42, c23·.34 | medium |
| sulfur_antioxidant | thiol/thione (ergothioneine, GSH) | c9·.32, c6·.25 | medium (perturbation-supported) |
| heme_porphyrin | porphyrin macrocycle | c8·.10, c2·.08 | **weak/provisional** (no pure porphyrin refs) |
| redox_broad | cross-cutting redox/flavin | c0·.09, c6·.04 | **weak** (catch-all) |

An axis value is **"the share of this spectrum's evidence that falls on this biochemical
system"** (composition), or **"how elevated this system is vs pure references"** (elevation/
display). The two readings answer different questions and must not be conflated.

---

## 3. Three ways to read a BSV (and when each is honest)

1. **Absolute composition radar** — `composition_t`. The theme shares, summing to ≈1. Good
   for "what is this spectrum made of," but see the closure limitation below.
2. **Δ-BSV (delta) radar** — `composition_t(query) − composition_t(baseline)`, on a shared
   centred scale. The honest headline for a *perturbation* (a spike, a depletion, a cohort
   contrast): it shows what moved, signed, independent of the dominant background.
3. **Reference-normalized radar** — `elevation_t` / `display_t`, the z-scored view vs pure
   references. Answers "is this theme unusually high/low vs the pure-compound frame."

---

## 4. The central limitation: compositional closure

`composition` sums to ≈1 by construction (it is a share). This has a hard consequence:

- **When one theme dominates the background, the absolute radar looks static.** In serum
  SERS the purine background (uric acid + hypoxanthine) can occupy a large, roughly
  constant share; adding a weak-adsorbing analyte barely changes the *proportions*, so the
  absolute radar of "before" and "after" nearly overlap even when real chemistry changed.
  This is not a bug — it is what a proportion *is* — but it makes the absolute radar the
  **wrong** view for perturbations.
- **The fix is structural, not cosmetic:** for any before/after or group contrast, GAIRA
  reads the **Δ-BSV** (mode 2) or the **elevation** (mode 3), never two overlapping
  absolute polygons. The demo enforces this; the audit endorses it.
- **A radar is one projection, not the state.** The BSV is 11 numbers with confidence and
  OOD attached; the radar is a convenience visualization of the composition. Low-confidence
  or high-OOD axes must be read as such — the polygon does not encode its own uncertainty,
  so it is always paired with the OOD/confidence read.

Other limitations, surfaced honestly:
- **Some axes are weakly grounded** (sterol_membrane, heme_porphyrin, redox_broad — §2).
  Their values are real numbers but rest on thin corpus evidence; they should carry the
  least interpretive weight. This mirrors the Part 2 coverage gaps and the Part 7 weak
  motifs — the three layers agree on where the model is soft.
- **Absolute BSV of an out-of-domain (SERS/serum) spectrum reflects the adsorbed subset,
  not whole-sample composition.** High OOD is the correct flag, and it is reported.

---

## 5. Verdict

The BSV is a **fully deterministic, documented, non-learned** transform: two frozen
matrices (component→theme W, reference frame) turn 24 reproducible coordinates into 11
biochemical axes with attached confidence and OOD. Its equations are explicit and its axes
are traceable to components → motifs → reference chemistries (Parts 6–7). Its one genuine
hazard — compositional closure making absolute radars look static under a dominant
background — is understood and mitigated by defaulting to Δ / elevation views for any
comparison. The layer honestly flags its own weak axes and out-of-domain inputs rather than
hiding them. No change to the construction is recommended; the standing guidance is only
that absolute composition radars must never be used to read a perturbation.
