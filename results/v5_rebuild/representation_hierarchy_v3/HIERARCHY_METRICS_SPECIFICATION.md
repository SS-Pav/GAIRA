# Representation-Hierarchy Metrics Specification (V3)

*Every metric GAIRA uses to measure Raman → Ag-SERS transfer, with its equation, purpose,
advantages, and limitations. Additive to V2 (`pure_ag_sers_theme_preservation/`); every V2
metric is retained and reproduced bit-for-bit (see `hierarchy_summary.json →
reproducibility_vs_v2`, max abs diff 0.0). Frozen atlas `09ed804a…` unchanged.*

For analyte *a*: `z` = mean 24-dim NMF coordinates, `t` = 11-dim biochemical theme composition,
`m` = 12-dim MSS motif activation, subscripts `R` (Raman) / `S` (Ag-SERS). `b̄` = mean Raman
theme composition over all analytes (the shared compositional baseline).

---

## The hierarchy (abstraction increases downward)

```
Level 1  Latent fingerprint     z   (24 NMF coordinates)      surface physics
Level 2  MSS motif              m   (12 biochemical motifs)         │
Level 3  Biochemical theme      t   (11 themes)                     │  abstraction
Level 4  Perturbation           dynamic response (3 analytes)       ▼
Level 5  Matrix robustness      serum competition            biochemical meaning
```

Level 3 is not one number — it is a **ladder of strictness** (raw → identity → rank → top-k →
argmax). The point of V3 is to show all of them, because each answers a different question and
the raw ones are baseline-inflated.

---

## Level 1 — Latent fingerprint preservation

```
L1 = cos(z_R, z_S)
```
- **Purpose.** Do the low-level latent coordinates line up? Median **0.42**.
- **Advantage.** Direct, interpretable, the finest-grained view.
- **Limitation.** Dominated by adsorption physics, not biochemistry — a low value is a surface
  effect, not a representation failure. (Renamed from the generic "coordinate cosine.")

## Level 2 — MSS motif preservation

```
L2 = cos(m_R, m_S)
```
- **Purpose.** Do the mid-level biochemical motifs survive? Median **0.74**.
- **Advantage.** More robust than raw coordinates; localises perturbations themes smear.
- **Limitation.** Still surface-sensitive; not identity-proof.

## Level 3a — Raw theme similarity

```
L3a = cos(t_R, t_S)
```
- **Purpose.** Gross similarity of the biochemical composition. Median **0.92**.
- **Advantage.** Simple; the historical number.
- **Limitation — critical.** **Contains compositional baseline inflation.** Every analyte's
  composition shares the same high-share background, so two *unrelated* analytes already sit at
  ≈0.9. **Never a stand-alone preservation measure.** Reported only alongside L3b.

## Level 3b — Identity-specific theme preservation *(renamed from "distinctive")*

```
d_R = t_R − b̄ ,  d_S = t_S − b̄
L3b        = cos(d_R, d_S)                          # identity-specific preservation
null(a)    = mean_{b≠a} cos(d_R^a, d_S^b)           # background floor
separation = L3b − null                             # >0 ⇒ resembles its OWN Raman more than random
```
- **Purpose.** After removing the shared baseline, does the analyte's *distinctive* biochemical
  abstraction transfer? Median **0.11**; separation median **+0.014** (positive 28/51).
- **Advantage.** Honest — corrects the baseline illusion of L3a.
- **Limitation.** Baseline-subtracted cosine is noisy per-analyte; read with the null.

## Level 3 (Layer 4) — Theme RANK preservation *(NEW)*

```
L4  = Spearman_ρ( rank(t_R), rank(t_S) )            # ordering of all 11 themes
rank_null    = mean_{b≠a} Spearman_ρ(t_R^a, t_S^b)
rank_separation = L4 − rank_null
```
- **Purpose.** Is the *ordering* of biochemical themes preserved (not just the top one)? Raw ρ
  median **0.87**.
- **Advantage.** Uses the full ordering including minor themes; less dominated by the single
  largest share than cosine; robust to argmax instability.
- **Limitation — tested, not assumed.** Raw ρ is **also baseline-inflated**: its null is ≈0.85,
  so rank_separation is only **+0.010** (positive for 34/51 — a slim majority, marginally more
  identity signal than L3b's 28/51). Report raw ρ AND its separation; the raw number alone is
  not identity evidence.

## Level 3 (Layer 5) — Top-k theme overlap *(NEW, first-class)*

```
L5_top2 = |top2(t_R) ∩ top2(t_S)| / 2
L5_top3 = |top3(t_R) ∩ top3(t_S)| / 3
```
- **Purpose.** Do the leading themes (e.g. purine/protein/organic) stay in the top set? Median
  top-2 **0.50**, top-3 **0.67** (typically 2 of 3 retained).
- **Advantage.** The most interpretable middle-ground: discrete, avoids argmax's knife-edge, and
  less baseline-inflated than cosine/rank because it looks only at the *leading* (more variable)
  themes.
- **Limitation.** Coarser than a continuous score; ignores ordering within the top set.

## Level 3 (Layer 6) — Dominant-theme agreement (argmax)

```
L6 = [ argmax(t_R) == argmax(t_S) ]
```
- **Purpose.** Does the single most-active theme survive? Agreement **35%**.
- **Advantage.** Maximally strict and interpretable.
- **Limitation — intentional.** Argmax is **strict and unstable**: on Ag-SERS 50/51 analytes
  become `nucleic_purine`-dominant, so all 18 agreements are analytes *already* purine-dominant
  in Raman. Use as a strict corner case, never as the headline.

---

## Level 4 — Perturbation validation

Dynamic response of the correct theme to a controlled change. **Measured for EXACTLY three
analytes** — never imputed:
- **adenine** (dose): purine theme ρ=0.996, Langmuir K=0.89 µM.
- **ergothioneine** (dose): sulfur theme ρ=0.927, K=1.52 µM.
- **uricase/urate** (directional depletion, NOT a dose): oxopurine motif Δ=−0.060.
- **Advantage.** Dynamic response is *stronger* evidence than any static similarity — it shows
  the abstraction is functional, not coincidental.
- **Limitation.** Rare (3/51). Absence elsewhere is stated as "Not measured," never as failure.

## Level 5 — Matrix robustness

```
serum recovery ~ pure transfer   (linear regression, n=51)
```
- **Purpose.** Does pure-Ag transfer strength *predict* serum recoverability?
- **Finding — honest.** Only weakly: L1 → serum displacement **r=0.167, R²=0.028, p=0.24
  (n.s.)**; rank → serum r=0.098 (n.s.). The top oxopurines survive both, but there is **no tight
  per-analyte law** — serum competition adds substantial matrix-specific effects.
- **Advantage.** Quantifies (with CI) a claim V2 stated only categorically.
- **Limitation.** Displacement is one recovery proxy; a non-significant slope means pure transfer
  is a poor quantitative predictor even if the categorical top-set agrees.

---

## Derived: ΔPurine (the attractor, quantified)

```
ΔPurine(a) = t_S[purine] − t_R[purine]
```
- Increases for **36/51** (median +0.058). Non-purines gain purine share on silver; already-
  purine-rich analytes lose it. **ΔPurine vs L1: r=−0.38, p=0.006** — weaker adsorption fidelity ⇒
  stronger pull into the attractor. This is the mechanistic core of the low argmax agreement.

---

## Reading rule (adopted across GAIRA)

1. Never quote a raw metric (L3a cosine, L4 rank ρ) without its null/separation.
2. Report the **hierarchy**, never a single "preservation score."
3. Use physics-aware language — *latent redistribution, adsorption-driven observation bias,
   identity-specific preservation, functional perturbation validation* — never "theme preserved
   / failed." Always separate **surface physics** from **biochemical interpretation**.
4. Levels 4–5 are reported only where measured.

Nothing here refits or modifies any frozen asset.
