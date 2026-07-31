# The Representation Hierarchy — Raman → Ag-SERS transfer, told correctly (V3)

*The central scientific narrative. Additive to V2; every V2 number is retained and reproduced
exactly. Frozen atlas `09ed804a…` unchanged. Metrics: `HIERARCHY_METRICS_SPECIFICATION.md`.
Figures: `figures/`. Cards: `analytes/`. Interactive: `gaira_foundation_explorer_v3/`.*

---

## The thesis

There is no single number for "how well does Raman transfer to Ag-SERS." Transfer is a
**hierarchy of representations**, and the honest question is *how far up the hierarchy the
agreement survives* — because each level is progressively more abstract and progressively less
governed by silver's surface physics.

```
Level 1  Latent fingerprint    median 0.42   ← adsorption physics dominates
Level 2  MSS motif             median 0.74
Level 3  Biochemical theme     raw 0.92 / rank 0.87 / top-3 0.67 / identity 0.11 / argmax 35%
Level 4  Perturbation          3 analytes, dynamically validated
Level 5  Matrix robustness     serum: weak per-analyte predictor (r=0.17, n.s.)
                               ← biochemical meaning
```

Read top to bottom, two things happen at once: the **raw** agreement rises (0.42 → 0.74 → 0.92),
which looks like "more of the biochemistry survives the higher you go." But the **identity-
specific** agreement *falls* at the top (identity cosine 0.11, argmax 35%). Both are true, and
holding them together is the whole point.

## Why the raw rise is mostly an illusion

At Level 3 the raw theme cosine is 0.92 and the raw rank ρ is 0.87 — both spectacular. But both
are **baseline-inflated**: every analyte's theme composition shares the same dominant background,
so any two analytes — even unrelated ones — already agree at ≈0.9 in cosine and ≈0.85 in rank
*before* preservation is considered. The controls prove it:

- **Identity cosine** (baseline-subtracted): median **0.11**; self-identifies the correct analyte
  for only 4/51.
- **Rank separation** (raw ρ − its null): median **+0.010**; the raw ρ (0.87) barely exceeds its
  own null (0.85). Rank ordering carries *slightly* more identity signal than magnitude (positive
  for 34/51 vs 28/51 for cosine) — a real but small edge, exactly what one hopes from a
  rank metric, and no more.
- **Top-3 overlap** = 0.67 (2 of 3 leading themes retained) is the honest middle-ground: high
  enough to be meaningful, low enough not to be baseline.

So the faithful reading of Level 3 is: **the broad biochemical neighbourhood is genuinely more
robust than the latent fingerprint, but identity-specific theme information is selective**, and
raw cosine/rank overstate it.

## The mechanism — the purine attractor, quantified

The reason identity fades at the top is a specific piece of silver physics. Ag colloid binds
N-heterocycles strongly, so oxopurine-like signal dominates the SERS of weak adsorbers. The
consequence, measured:

- **50/51** analytes become `nucleic_purine`-dominant on Ag-SERS (Sankey / confusion).
- **ΔPurine > 0 for 36/51** (median +0.058): non-purines *gain* purine share; already-purine-rich
  analytes *lose* it (guanine −0.25, adenine −0.12) — regression toward the attractor.
- **ΔPurine vs latent fingerprint: r = −0.38, p = 0.006** — the weaker the adsorption fidelity,
  the harder the analyte is pulled into the attractor.

This is why argmax agreement is only 35% and why all 18 agreements are analytes that were
*already* purine-dominant. It is **adsorption-driven observation bias**, a property of the silver
surface — an observation-model target, not a defect of the frozen representation, which honestly
flags the modality gap (OOD 0.05 → 0.16).

## Where the hierarchy genuinely holds — and where it becomes functional

Identity-specific preservation concentrates in the strong chemisorbers: oxopurines, cofactors,
creatinine/urea, PEP/citrate (family heatmap). For most weak physisorbers (amino acids, sugars,
lipids, pyrimidines) the distinctive abstraction is not recovered — a surface limit, stated as
such.

The strongest evidence sits at **Level 4**, and it is rare. For the three analytes with a
controlled perturbation, the correct theme is not merely present but *functional*:

| analyte | validation | target | metric |
|---|---|---|---|
| adenine | concentration dose-response | nucleic_purine | ρ=0.996, Langmuir K=0.89 µM |
| ergothioneine | concentration dose-response | sulfur_antioxidant | ρ=0.927, K=1.52 µM |
| uricase (urate) | directional depletion (not a dose) | oxopurine motif | Δ=−0.060 |

**Adenine** is the emblem of the whole story: weak at Level 1 (0.36), yet it keeps a
purine-dominant abstraction *and* responds to dose along a textbook adsorption isotherm. A single
Raman→SERS cosine would discard it; the hierarchy shows it is one of the best-validated analytes
GAIRA has. Dynamic response is stronger evidence than any static similarity — but only three
analytes provide it, and we never imply otherwise.

## Level 5 — an honest downgrade

V2 showed categorically that the top oxopurines survive serum. V3 asks the quantitative question:
does pure-Ag transfer *predict* serum recoverability across all 51? **It does not, tightly** —
regression gives r=0.17, R²=0.028, p=0.24 (n.s.). The categorical top-set agreement is real, but
there is no per-analyte law: serum adds matrix-specific competition beyond pure adsorption
strength. We report the weaker, more honest result.

## The one-paragraph conclusion

Raman → Ag-SERS transfer is best described not by a preservation *score* but by a preservation
*hierarchy*. Latent coordinates transfer partially (0.42, adsorption-limited); motifs better
(0.74); the broad biochemical neighbourhood best of all in raw terms (theme 0.92, rank 0.87) —
but that top-level agreement is largely a compositional baseline, and once corrected, identity-
specific theme preservation is selective (identity 0.11, rank separation +0.01, top-3 0.67,
argmax 35%), because silver homogenises most analytes toward a purine attractor (ΔPurine ∝
−adsorption fidelity, p=0.006). A minority — adenine foremost — redistribute their latent
fingerprint yet retain a *dose-responsive* theme, and that functional validation, though rare, is
the strongest evidence in the ladder. GAIRA's job is not to maximise agreement but to place each
of these — latent redistribution, motif preservation, biochemical abstraction, functional
perturbation, matrix robustness — on a coherent, defensible hierarchy that separates surface
physics from biochemical meaning.
