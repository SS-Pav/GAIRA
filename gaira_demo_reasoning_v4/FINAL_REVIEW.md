# GAIRA V6 Demo — External-Reviewer Self-Critique

Written adversarially, as a skeptical spectroscopy / comp-bio reviewer would, not as
the author. Scores are 1–10 (10 = Nature-Methods-ready).

## The single biggest scientific risk

**A Raman-derived atlas is applied to SERS data.** Calibration, serum and biology are
all Ag/Au-SERS — out-of-distribution for a pure-Raman NMF basis by construction. The
demo is honest about this (OOD everywhere, an explicit "working hypothesis" banner, the
serum stress test that maps exactly where transfer fails), and that honesty is its
strongest scientific feature. But a reviewer like Dionne or Utkan will still push: the
Au-SERS observation model does not exist yet, so every biological result is a zero-shot
cross-domain application with no in-domain biological ground truth. **This is disclosed,
not solved.**

## Remaining weaknesses (honest)

1. **Biological effect sizes are mostly modest.** Only diabetes (Impact vs Strong-D,
   δ=−0.88, ratio 1.79) is robust; HCC/SHINE are moderate/exploratory; COVID is
   near-null. Reported honestly, but there is limited "wow" biology, and Impact/
   Strong-D is not a clean disease/control contrast.
2. **Compositional closure** makes theme-level deltas partly an artefact of the simplex
   (a rise in one theme mechanically lowers others). Mitigated by the delta radar +
   MSS-level views + the target-theme dose-response, but it is inherent to composition
   shares and a careful reviewer should keep it in mind.
3. **Curated MSS bands/exemplars.** Motif *definitions* (bands, exemplar chemistries)
   are expert-curated; only the contributors/weights/confidence are derived. Defensible
   (textbook Raman) but not itself learned.
4. **Spectrum panels are atlas reconstructions**, not raw spectra — the atlas's *view*
   of a spectrum. Legitimate and always-available, but not the raw measurement.
5. **SHINE/small2023 subsampled** (90/100 per group) for artifact size; documented,
   but it caps statistical power.
6. **Plotly figures (Sankey, coefficient-PCA) are not pixel-audited** (no kaleido);
   verified only to construct.
7. **DART page is conceptual.** Necessary honesty, but it is a roadmap, not a result.

## Per-page scores

| Page | Scientific clarity | Visual clarity | Spectroscopy rigor | Educational value |
|---|---|---|---|---|
| 1 Overview | 8 | 8 | 7 | 8 |
| 2 Reference space (NMF) | 9 | 8 | 9 | 9 |
| 3 How GAIRA reasons | 9 | 8 | 8 | 9 |
| 4 Calibration | 9 | 9 | 9 | 9 |
| 5 Serum stress test | 9 | 8 | 9 | 8 |
| 6 Biological studies | 7 | 8 | 7 | 7 |
| 7 Future DART | 6 | 7 | 6 | 7 |
| 8 Methods & provenance | 8 | 7 | 8 | 7 |

**Strongest:** Page 4 (Calibration) — the reasoning cascade, the delta radar with the
dose grid, and the quantified redistribution-vs-scaling-vs-depletion mechanisms make it
genuinely publication-grade. Page 2 is now a real NMF explainer, not a scatter plot.

**Weakest:** Page 7 (conceptual by necessity) and Page 6 (honest but limited biology;
SERS-OOD ceiling). Page 8 is dense and could be more visual.

## What would move the weak pages up

- Page 6: an in-domain biological validation (a cohort with orthogonal ground truth),
  a matrix-recoverability prior on confidence, and per-analyte adsorption modelling.
- Page 7: a first real DART time-series, even a pilot, to replace the conceptual
  trajectories.
- Page 8: a visual provenance graph rather than tables.

## Does it meet the bar?

For a first-time spectroscopy expert: **yes** — by the end they understand the pipeline
(reference → NMF → MSS → themes → BSV → domain interpretation), see it validated on
calibration, see exactly where Raman→SERS transfer breaks (serum), and see it applied to
real cohorts with transparent uncertainty. They should **trust the reasoning and the
disclosed limits**, which is the correct outcome — not uncritical admiration. The demo
now reads as a guided scientific explanation, not a gallery of outputs. The remaining
gap to a Nature paper is **data** (in-domain SERS/DART, biological ground truth), not
presentation.
