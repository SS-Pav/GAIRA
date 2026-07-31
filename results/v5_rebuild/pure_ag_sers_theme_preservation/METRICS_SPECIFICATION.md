# Cross-Modal Transfer — Metrics Specification

*How GAIRA measures what survives when a pure-compound spectrum moves from Raman to
Ag-SERS. Additive to the existing pure-Ag-SERS validation; the frozen atlas
(`09ed804a40836f4a05a91ba10900cded`) is unchanged. Computed by
`code/theme_preservation.py`; outputs in `tables/` and `artifacts/`.*

---

## Why this document exists

The original pure-Ag-SERS stage reports **one** number per analyte: the cosine between
the 24-dimensional NMF coordinate vectors of the Raman and the Ag-SERS spectrum, and calls
it *recoverability*. That number is real and worth keeping — but the word "recoverability"
is too broad for what it measures. A single coordinate cosine answers only:

> *Do the two spectra land on the same point of the frozen latent manifold?*

It does **not** answer whether the **biochemical interpretation** survives, whether a
**perturbation** would still be detectable, or whether the analyte is **recoverable in a
serum matrix**. Those are four different questions with four different answers. Collapsing
them into one cosine is exactly the kind of over-compression GAIRA is built to avoid.

This spec defines **four distinct, non-interchangeable transfer metrics**, and — critically
— the **null controls** that keep the theme-level metric honest.

---

## The four levels of transfer

| # | Metric | Question it answers | Symbol | Scope here |
|--:|---|---|---|---|
| 1 | **Latent fingerprint preservation** | Do the 24 NMF coordinates line up? | `C_component` | measured, all 51 |
| 2 | **Biochemical theme preservation** | Does the broad interpretation survive? | `C_theme`, `C_theme*`, dominant-match | measured, all 51 |
| 3 | **Perturbation sensitivity** | Would a dose/depletion still register? | — | measured only where data exist |
| 4 | **Matrix recoverability** | Does it survive serum competition? | — | linked to serum stage |

**These are not tiers of the same quantity.** An analyte can score high on one and low on
another. Adenine (see below) redistributes its latent fingerprint yet keeps a dose-responsive
purine theme: **low on 1, meaningful on 2 and 3.** Uracil scrambles on all of them.

> **Non-negotiable:** levels 3 and 4 are only computed where real data exist. Perturbation
> evidence exists **only** for adenine (concentration), ergothioneine (concentration), and
> uricase (directional urate depletion). Everything else is reported as **"Not tested"** —
> never imputed, never a fabricated score.

---

## Level 1 — Latent fingerprint preservation

For analyte *a*, let `z_R` and `z_S` be the mean 24-dim NMF coordinate vectors of its Raman
and Ag-SERS spectra (projected through the frozen atlas by NNLS).

```
C_component(a) = cos(z_R, z_S)
```

This is **identical** to the existing pure-Ag-SERS `coord_cosine` (median ≈ 0.42 over 51
analytes). It is preserved verbatim; this work does not overwrite or reinterpret it. It is
renamed *latent fingerprint preservation* only to make explicit what it measures: the
similarity of the low-level latent composition, which is dominated by adsorption physics.

**Descriptive tiers** (Excellent ≥ 0.80, Good ≥ 0.65, Moderate ≥ 0.45, Weak ≥ 0.25, Poor
otherwise) are **manually chosen thresholds for reading a table**, not learned biological
classes. They carry no probabilistic meaning.

---

## Level 2 — Biochemical theme preservation

The interpretation layer. GAIRA maps each coordinate vector to an **11-theme biochemical
composition** `b ∈ Δ¹⁰` (a share vector on the biochemical themes) via the frozen engine
(`eng.infer(...).bsv.composition`). Theme preservation is measured **three ways, because no
single number is honest on its own.**

### 2a. Dominant-theme match (argmax)
```
dominant_match(a) = [ argmax(b_R) == argmax(b_S) ]
```
Does the single most-active theme survive? **The strictest, most interpretable test.**
Preserved for **18/51** analytes (35%). *(The existing stage reports 19/51 using a slightly
different dominant-theme helper; the ~1-analyte difference is definitional, not a conflict —
both say "about a third." This work does not overwrite that number.)*

### 2b. Theme composition cosine (RAW) — **and why it must not be read alone**
```
C_theme(a) = cos(b_R, b_S)
```
Raw theme cosine has **median 0.92** — far above `C_component`'s 0.42, and it exceeds
`C_component` for **all 51/51** analytes. Read naively, this "proves" the theme always
survives.

**It does not.** The 11-theme composition of *every* analyte is dominated by the same handful
of high-share background themes (compositional closure; cf. `BSV_AUDIT`). Two unrelated
analytes therefore already sit at cosine ≈ 0.9 **before** any preservation is considered. Raw
`C_theme` cannot, by construction, distinguish "theme preserved" from "shared baseline."
Reporting it as evidence of preservation would be precisely the over-claim GAIRA rejects.
It is retained only as a descriptive quantity, **always** accompanied by 2c.

### 2c. Distinctive theme preservation (baseline-subtracted) + null — **the honest signal**
Subtract the shared background `b̄` = mean Raman composition over all analytes, leaving each
analyte's **distinctive** deviation, and measure preservation against a **null**:

```
d_R = b_R − b̄ ,   d_S = b_S − b̄
C_theme*(a)      = cos(d_R, d_S)                          # distinctive preservation
null(a)          = mean_{b≠a} cos(d_R^a, d_S^b)           # background floor (other analytes)
separation(a)    = C_theme*(a) − null(a)                  # identity-specific signal, >0 = real
self_rank(a)     = rank of a among all 51 by cos(d_R^a, d_S^·)   # 1 = self is nearest; chance ≈ 26
```

`separation > 0` means an analyte's Ag-SERS theme profile resembles **its own** Raman profile
more than a random analyte's — i.e. identity-specific theme information genuinely survived.

**Observed:** median `C_theme*` = 0.11; median `separation` = +0.014 (positive for 28/51);
**median `self_rank` = 25 (chance ≈ 26)**; self is nearest for only **4/51**; in the top-5 for
**8/51**. Conclusion: distinctive theme preservation is **real but selective and weak on
average**, concentrated in the strong Ag chemisorbers.

### 2d. Expected-theme retention
Using a family→expected-theme map (`purine→nucleic_purine`, `saccharide→saccharide_glycan`,
thiol amino acids→`sulfur_antioxidant`, …; mixed families carry >1 theme, never forced to one),
we record the **rank** of the expected theme in `b_R` and `b_S`, and whether it stays in the
Ag-SERS **top-3** (retained for **22/51**).

### 2e. The purine attractor (why dominant-match is low)
The dominant-theme confusion matrix shows Ag-SERS collapses the top theme onto
`nucleic_purine` for many non-purines (saccharides 20, lipids 6, organic acids 3). Ag colloid
has strong affinity for N-heterocycles, so oxopurine-like signal dominates weak adsorbers'
SERS. The true theme often survives at rank 2–3 (hence expected-top3 > dominant-match), but
**not** at rank 1. This attractor is the mechanism behind both the low dominant-match rate and
the serum result (serum SERS ≈ uric acid + hypoxanthine).

---

## Level 2 (motif) — MSS preservation

Between coordinates and themes sits the **MSS motif layer** (12 biochemical motifs). We report
`C_MSS = cos(m_R, m_S)` over the motif activation vectors (median **0.74**), top-3 motif
overlap, and dominant-motif match. Motif preservation is **intermediate** between latent
(0.42) and raw theme (0.92) — consistent with motifs capturing mid-level structure that is
more robust than exact coordinates but still adsorption-sensitive.

---

## Redistribution structure (how the fingerprint moves)

For each analyte we record **where** the latent profile went, so redistribution is described,
not just scored:
- `theme_redistribution = 1 − C_theme`, `l1_theme_shift = Σ|b_S − b_R|`, `theme_jsd` (Jensen–Shannon).
- Largest **gained**/**lost** theme, component (`c0…c23`), and MSS motif.

A high redistribution with a preserved *distinctive* theme is the hypothesis case; a high
redistribution with a collapsed distinctive theme is scrambling.

---

## The central quadrant (component × distinctive-theme)

Each analyte is placed on **latent preservation (`C_component`)** × **distinctive theme
preservation (`C_theme*`)** — the theme axis uses the baseline-subtracted cosine, never raw:

| | `C_theme* ≥ 0.50` | `C_theme* < 0.50` |
|---|---|---|
| **`C_component ≥ 0.55`** | **Q1** identity preserved (13) | **Q3** superficial coord match, theme changes (7) |
| **`C_component < 0.55`** | **Q2** latent redistribution, theme survives (4) | **Q4** poor transfer (both) (27) |

**Q2 is the hypothesis quadrant** — redistribution of the fingerprint with survival of the
distinctive theme. It contains **4** analytes (adenine, riboflavin, phosphate, thymine). It is
real but is **not** the dominant pattern; the honest verdict is that latent preservation and
theme preservation are **correlated through adsorption physics**, not independent axes — with a
small, important set of exceptions that Q2 names.

---

## Verdict on the working hypothesis

> *"Ag-SERS may substantially redistribute the latent NMF component profile while still
> preserving the higher-level biochemical theme and yielding a reproducible perturbation
> response."*

**Partially supported, with one essential correction.**
1. **Correct:** theme-level and fingerprint-level preservation are genuinely **distinct**
   metrics; theme cosine exceeds component cosine for all 51 analytes. Interpretation should
   not be reduced to one Raman→SERS cosine.
2. **Correction:** the *strong* reading — "theme survives even when the fingerprint
   redistributes, in general" — is largely a **compositional-baseline artifact**. Once the
   baseline is removed, identity-specific theme preservation is **selective**, tracking the
   same adsorption physics as the latent fingerprint, and strong only for oxopurines and a
   handful of chemisorbers. The genuine redistribute-but-survive case (Q2) is a **minority**,
   with **adenine** as its canonical member — and adenine is exactly where a reproducible
   perturbation (dose-response) is independently confirmed (Level 3).

That nuanced model — *most transfer is adsorption-limited at every level, but a real minority
preserve a dose-responsive theme through latent redistribution* — is both more accurate and
more useful than either "one cosine" or "theme always survives."

---

## Data provenance

- Raman reference coords: frozen corpus, per-analyte mean (`gaira.foundation.dataset`).
- Ag-SERS coords: `spike_lib.load_pure_sers()`, per-analyte mean, projected through the frozen atlas.
- Themes / MSS / OOD / confidence: frozen `GAIRAEngine` + `MSSLayer`, `domain="buffer"`.
- Perturbation (Level 3): `foundation_audit/tables/validation_results.json` (adenine, ergothioneine, uricase).
- Matrix (Level 4): `spike_validation/tables/phase7_serum_vs_pure.csv`.
- Nothing here refits, retrains, or modifies any frozen asset.
