# Theme-Preservation Assessment — the honest verdict

*Does the biochemical theme survive the Raman → Ag-SERS jump even when the latent fingerprint
does not? This document states the verdict plainly, with the evidence for and against, and is
deliberately written to resist over-claiming. Frozen atlas `09ed804a…` unchanged. Numbers from
`tables/` + `artifacts/theme_preservation_summary.json`.*

---

## The claim under test

> Ag-SERS may substantially **redistribute the latent NMF component profile** while still
> **preserving the higher-level biochemical theme** and yielding a **reproducible perturbation
> response** — a more accurate transfer model than one Raman→SERS cosine.

## Verdict: **Partially supported — the framing is right, the strong universal reading is not.**

The claim decomposes into three sub-claims. They do not all hold equally.

| Sub-claim | Verdict | Key evidence |
|---|---|---|
| (a) theme ≠ fingerprint; one cosine is too coarse | **Supported** | theme cosine > component cosine for **51/51**; MSS (0.74) sits between latent (0.42) and theme |
| (b) theme survives *generally* when fingerprint redistributes | **Not supported (baseline artifact)** | raw 0.92 is compositional baseline; distinctive median **0.11**, self-rank median **25 ≈ chance** |
| (c) a real subset redistributes yet keeps a *functional* theme | **Supported (minority)** | Q2 = 4 analytes; **adenine** dose-responds (ρ=0.996) |

---

## Why (a) is supported

Theme and fingerprint measure different things, and the data show it cleanly:
- Component cosine (latent) median **0.42**; MSS motif cosine median **0.74**; raw theme cosine
  median **0.92** — a monotone rise across every biochemical family (Figure 4). Preservation
  genuinely increases as you move up the abstraction ladder from coordinates → motifs → themes.
- Collapsing all of this to the single 0.42 coordinate cosine (as "recoverability") discards
  the motif and theme structure that the higher layers retain. The four-level framework is the
  correct unit of analysis.

**This is the part of the user's insight that is correct and now formalised.** The original
result is preserved unchanged; it is *reframed* as Level 1 of four, not replaced.

## Why (b) is **not** supported — the baseline trap

The naive evidence for (b) is overwhelming-looking: raw theme cosine ≥ component cosine for
every analyte, median 0.92. But:

- The 11-theme composition of every analyte is dominated by the same high-share background
  themes (compositional closure; cf. `BSV_AUDIT`). Two *unrelated* analytes already sit at
  cosine ≈ 0.9. Raw theme cosine therefore **cannot discriminate preservation from baseline** —
  it is high by construction.
- Subtract the shared baseline and the signal collapses: **distinctive theme cosine median
  0.11**, **null floor −0.06**, **separation +0.014** (positive for only 28/51), **self-rank
  median 25** against a chance median of 26. The distinctive SERS theme profile identifies the
  correct analyte for just **4/51** (top-5 for 8/51).
- Mechanism: the **purine attractor**. Ag-SERS makes **50/51** analytes purine-dominant
  (Figure 6); the rich, analyte-specific Raman theme structure is **homogenised** on silver
  (Figure 3). The 35% dominant-theme "match" is entirely the analytes that were *already*
  purine-dominant in Raman — not evidence of per-analyte theme survival.

So the *general* form of (b) — theme survives across the board when the fingerprint moves — is
an artifact. Reporting raw theme cosine as preservation would repeat the exact error GAIRA
exists to prevent: mistaking a shared, non-specific background for a real assignment.

## Why (c) is supported — the real, useful minority

Once the baseline is removed, a genuine subset remains where the fingerprint redistributes yet
the distinctive theme survives (**Q2**: adenine, riboflavin, phosphate, thymine) or where both
are preserved (**Q1**: oxopurines, cofactors, PEP, citrate, creatinine, methionine, cysteine).
For these the interpretation genuinely transfers. And for the one analyte with independent
perturbation data in Q2 — **adenine** — the theme is not merely present but **functional**: a
14-point concentration series drives the purine theme monotonically (ρ = 0.996) along a
saturating Langmuir law (K = 0.89 µM). Ergothioneine (Q4 by the strict distinctive cut, but a
clean single-motif sulfur analyte) likewise dose-responds (ρ = 0.927). Uricase confirms
perturbation *direction* at the oxopurine motif (Δ = −0.060).

A functional, dose-responsive theme is the operationally meaningful form of preservation —
stronger evidence than any static cosine — and it exists, for the analytes where it was
actually measured.

---

## What this means for GAIRA

1. **Report transfer at four levels, never one cosine.** Latent fingerprint (0.42), MSS motif
   (0.74), biochemical theme (distinctive 0.11 / dominant 35%), and — where measured —
   perturbation and matrix recoverability. Each answers a different question.
2. **Never quote raw theme cosine alone.** Always with its null/separation. Raw theme cosine is
   a baseline, not a preservation score.
3. **The dividing line is adsorption, at every level.** Oxopurines and strong chemisorbers
   transfer, dose-respond, deplete on cue, and survive serum; weak physisorbers fail
   everywhere. Theme preservation is *correlated with* fingerprint preservation, not
   independent of it.
4. **The attractor is an observation-model target.** The purine collapse is silver physics, not
   a representation defect — the model flags the modality gap (OOD 0.16 vs 0.05) rather than
   hiding it. Correcting it belongs in a future observation model, not in the frozen atlas.
5. **The frozen atlas is untouched.** Every number here is a projection through
   `09ed804a…`; nothing was refit.

## One-paragraph honest summary

Theme-level and fingerprint-level preservation are genuinely different things, and GAIRA should
measure both — that much of the hypothesis is right and is now built in. But the eye-catching
"theme cosine 0.92, so the theme always survives" is a compositional-baseline illusion: once
the shared background is removed, identity-specific theme preservation is weak and selective
(self-rank ≈ chance), concentrated in the same strong silver adsorbers that preserve the
fingerprint, because Ag-SERS homogenises nearly everything toward a purine attractor. A real,
minority set — adenine foremost — does redistribute its latent profile while keeping a
**dose-responsive** theme, and that is the genuinely useful transfer story: not "theme always
survives," but "for the analytes silver actually reports, the theme survives and stays
functional, and GAIRA can tell which those are."
