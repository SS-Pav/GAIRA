# Cross-Modal Transfer — Report

### What survives when a pure compound moves Raman → Ag-SERS, measured at four levels

*Additive analysis on the frozen GAIRA atlas (`09ed804a40836f4a05a91ba10900cded`). No
retraining, no change to the NMF, ontology, MSS, weights, or preprocessing. Companion to the
existing `foundation_audit/reports/PURE_AG_SERS_VALIDATION.md`, which it extends — it does not
overwrite or reinterpret the original coordinate-cosine results. Metric definitions:
`METRICS_SPECIFICATION.md`. Verdict: `THEME_PRESERVATION_ASSESSMENT.md`. Figures: `figures/`.
Per-analyte cards: `analytes/`. Tables: `tables/`.*

---

## The question, sharpened

The original pure-Ag-SERS stage reports one number per analyte — the cosine between the
24-dimensional NMF coordinate vectors of the Raman and Ag-SERS spectra (median 0.42) — and
calls it *recoverability*. That number is correct and is kept verbatim. But "recoverability"
is too broad a word for it. A coordinate cosine measures only whether two spectra land on the
same point of the latent manifold. It does **not** tell us whether the **biochemical
interpretation** survives, whether a **perturbation** would still register, or whether the
analyte is recoverable **in serum**. Those are four different questions:

| Level | Metric | Median (51 analytes) |
|---|---|---|
| 1 · Latent fingerprint preservation | component cosine | **0.42** |
| 2 · Biochemical theme preservation | distinctive theme cosine / dominant-match | **0.11** / **35%** |
| 2 · (motif) MSS preservation | MSS cosine | **0.74** |
| 3 · Perturbation sensitivity | dose ρ / directional Δ | measured for 3 analytes |
| 4 · Matrix recoverability | serum spike displacement | 9 strong / 24 mod / 18 weak |

These are **not tiers of one quantity.** An analyte can be weak at level 1 and meaningful at
levels 2–3 (adenine), or superficially fine at level 1 and scrambled at level 2 (urate).

---

## The central result — and the trap we avoided

**Raw theme cosine exceeds component cosine for all 51/51 analytes** (median 0.92 vs 0.42).
Read naively, this "proves" the biochemical theme almost always survives even when the latent
fingerprint does not — exactly the working hypothesis.

**That reading is a compositional-baseline artifact.** Every analyte's 11-theme composition is
dominated by the same few high-share background themes, so *any two* analytes already sit at
cosine ≈ 0.9 before preservation is considered. Raw theme cosine cannot distinguish
"preserved" from "shared background." Reporting it as evidence would be precisely the
overclaim GAIRA is built to reject (peak ≠ molecule; nearby ≠ assigned).

So we measured theme preservation **against a null**:

- **Distinctive theme cosine** `C_theme*` = cosine of baseline-subtracted theme vectors —
  median **0.11**.
- **Null floor** = the same analyte's distinctive SERS profile vs *every other* analyte's —
  median **−0.06**.
- **Separation** = `C_theme* − null` — median **+0.014**, positive for **28/51**.
- **Self-nearest** (the distinctive SERS profile's closest Raman profile is the *same*
  analyte): only **4/51**; in the top-5 for **8/51**; **median self-rank 25 ≈ chance (26)**.

**Conclusion:** identity-specific theme preservation is **real but selective and weak on
average** — strong only for the oxopurines and a handful of chemisorbers, and near-chance for
most analytes. Figure 1 shows both readings side by side: raw (everything above the diagonal)
and honest (selective).

---

## The purine attractor (why dominant-match is 35%, and what that number really means)

The dominant Ag-SERS theme is **`nucleic_purine` for 50 of 51 analytes** (the 51st is
saccharide_glycan) — confirmed identically by the pre-existing committed artifact. The
Raman dominant themes are diverse (saccharide 20, purine 18, lipid 6, organic-acid 3, sulfur
3, pyrimidine 1); Ag-SERS collapses nearly all of them onto purine. Silver colloid has strong
affinity for N-heterocycles, so oxopurine-like signal dominates the SERS of weak adsorbers.

This reframes the 35% dominant-theme "match": **all 18 matches are exactly the analytes that
were already purine-dominant in Raman.** It is not "a third of analytes keep their theme" — it
is "purine-dominant analytes stay purine, and Ag makes almost everything else look purine
too." Figure 6 (confusion matrix) and Figure 3 (Raman rich → Ag-SERS homogenised) show this
directly. The true theme often survives at rank 2–3 (expected-theme top-3 retained for 22/51),
just not at rank 1.

This is the same physics as the serum result — serum SERS ≈ uric acid + hypoxanthine — one
rung earlier, with no matrix competition yet. The attractor is a property of the **silver
surface**, not of the frozen representation.

---

## The quadrant map (Figure 1B)

Placing each analyte on latent preservation × **distinctive** theme preservation:

| | theme survives | theme changes |
|---|---|---|
| **fingerprint preserved** | **Q1** (13): oxopurines, cofactors, PEP, citrate, creatinine | **Q3** (7): albumin, urate, glycogen, sugars with superficial coord overlap |
| **fingerprint redistributes** | **Q2** (4): **adenine, riboflavin, phosphate, thymine** | **Q4** (27): most amino acids, sugars, lipids |

**Q2 is the hypothesis quadrant** — redistribution of the fingerprint with survival of the
distinctive theme. It is real but a **minority**. Its canonical member is **adenine**, and
adenine is exactly where an independent, controlled **dose-response** confirms the theme is not
only present but *functional* (Level 3, below). Q1 + Q2 together (17) are the analytes whose
interpretation genuinely transfers; the honest headline is that latent and theme preservation
are **correlated through adsorption physics**, with Q2 the important, well-characterised
exception.

---

## Level 3 — Perturbation sensitivity (measured for 3 analytes only)

Perturbation evidence exists **only** for adenine, ergothioneine, and uricase. Nothing else is
imputed; every other analyte's card reads *Not tested*.

- **Adenine** (concentration, 14 levels): purine theme rises monotonically, Spearman
  **ρ = 0.996**, best-fit **saturating (Langmuir) K = 0.89 µM, R² = 0.993**.
- **Ergothioneine** (concentration, 11 levels): sulfur theme rises monotonically,
  **ρ = 0.927**, saturating **K = 1.52 µM, R² = 0.957**.
- **Uricase** (directional urate depletion — **not** a dose series): the **oxopurine-carbonyl
  motif drops sharply (Δ = −0.060)** while the purine-ring motif is unchanged (Δ = −0.001); the
  theme layer is diffuse (purine theme Δ = −0.011). This validates perturbation **direction and
  localisation at the motif layer**, not a dose magnitude. (Figure 8.)

The perturbation layer answers a question the static cosines cannot: even where the latent
fingerprint redistributes, a controlled change in the analyte still moves the correct theme in
the correct direction. That is the operationally useful form of "theme preservation."

---

## Level 4 — Matrix recoverability (serum)

Joining the serum spike-in stage (`phase7_serum_vs_pure.csv`): of the 51 pure analytes,
**9 strong / 24 moderate / 18 weak** in serum. The strong serum recoverers are the same strong
Ag chemisorbers that preserve the fingerprint and (where tested) dose-respond: hypoxanthine,
xanthine, guanine, adenine, ergothioneine, creatinine (Figure 9). Serum adds competition on
top of the modality gap, so the strong set is a subset with more scatter — but the dividing
line is again adsorption, end to end.

---

## Verdict on the hypothesis

> *"Ag-SERS may substantially redistribute the latent NMF component profile while still
> preserving the higher-level biochemical theme and yielding a reproducible perturbation
> response."*

**Partially supported, with one essential correction** (full argument in
`THEME_PRESERVATION_ASSESSMENT.md`):

1. **Correct and important:** theme-level and fingerprint-level preservation are genuinely
   **distinct** metrics — theme cosine exceeds component cosine for all 51 analytes, MSS sits
   between them, and reducing transfer to a single Raman→SERS cosine hides real structure.
2. **Correction:** the *strong* form — "theme survives even when the fingerprint redistributes,
   in general" — is largely a **baseline artifact**. Baseline-corrected, identity-specific
   theme preservation is **selective**, tracks the same adsorption physics as the fingerprint,
   and is strong only for oxopurines and a few chemisorbers. The genuine
   redistribute-but-survive case (Q2) is a minority — with **adenine** its canonical member,
   and adenine is where a reproducible dose-response independently confirms a functional theme.

That nuanced model — *most transfer is adsorption-limited at every level, but a real,
well-characterised minority preserve a dose-responsive theme through latent redistribution* — is
both more accurate and more useful than either "one cosine" or "theme always survives."

---

## Reproduce

```bash
python results/v5_rebuild/pure_ag_sers_theme_preservation/code/theme_preservation.py     # tables + summary
python results/v5_rebuild/pure_ag_sers_theme_preservation/code/make_cards_and_layers.py  # cards + L3/L4
python results/v5_rebuild/pure_ag_sers_theme_preservation/code/make_figures.py           # 9 figures
```

All three read only committed assets + the frozen engine; none modifies a frozen file. The
atlas fingerprint is asserted equal to `09ed804a40836f4a05a91ba10900cded` at load.
