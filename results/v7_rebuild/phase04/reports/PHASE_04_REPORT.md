# GAIRA V7 — Phase 04: Frozen Projection Engine and Hierarchical Inference

**Frozen inputs verified before any computation:** atlas `09ed804a40836f4a05a91ba10900cded`,
LSM registry `208482d6f7178b5b8f16cace91be55b0`, CSM dictionary
`0b4aa550ccefed3edabdbde5bae11c8d`, theme registry `f54d4835ffdf8aa2d50a4a203da0e8f4`.
The engine aborts if any differs. **Nothing was refitted.**

**Status: 10 of 11 gates PASS.** The failing gate is out-of-domain detection on real Ag-SERS
spectra, and it is reported rather than compensated.

---

## 1. Executive summary

The question this phase answers is not whether GAIRA reconstructs spectra. It is whether the
frozen hierarchy supports **inference about a spectrum it has never seen**, and whether
climbing the hierarchy helps or hurts.

**It does both, at different levels, and the pattern is unambiguous:**

| level | dim | split A: molecule top-1 | split B: class top-1 (unseen molecule) | replicate consistency |
|---|---:|---:|---:|---:|
| raw spectrum | 676 | 0.790 | 0.608 | 0.904 |
| **LSM** | 50 | **0.806** | 0.850 | 0.891 |
| **CSM** | 49 | 0.799 | **0.855** | 0.893 |
| theme | 4 | 0.553 | 0.405 | **0.979** |
| BSV | 4 | 0.553 | 0.405 | **0.979** |
| geometry | 5 | 0.495 | 0.541 | 0.946 |

**The representation is genuinely reusable.** Projecting an unseen molecule onto the frozen
LSM/CSM dictionaries raises chemistry-class retrieval from 0.608 to 0.855 — a 41% relative
gain over the raw spectrum, on molecules the retrieval set has never contained. This is the
central positive result: the motif layer carries transferable chemistry, not memorised spectra.

**Abstraction buys stability and costs discrimination, and the trade is quantified.** The theme
and BSV layers reach the highest replicate consistency in the stack (0.979) while falling below
the raw spectrum on both retrieval axes. A four-dimensional BSV cannot identify a molecule; it
was never supposed to. What it can do is place two measurements of the same substance in the
same place (within-molecule cosine 0.979, between/within separation ratio 7.26) and survive
noise (cosine 0.995 at σ = 0.05).

**Dictionary-level leakage is measured, not assumed away.** Refitting the class-local NMF
per fold gives held-out top-1 of 0.855 against the frozen dictionary's 0.910: **+0.055
inflation** in every in-sample number this project has produced. Small, real, and now on record.

**One failure, reported at full strength.** On real Ag-SERS spectra — deliberately excluded from
the pure-Raman atlas by the corpus audit — out-of-domain AUROC is **0.548**, chance. The SERS
spectra are *better* explained than the references (residual 1.39 vs 1.65). A non-negative
dictionary of Raman motifs reconstructs SERS of the same metabolites comfortably, so **the atlas
cannot tell modality**. On a synthetic band-shift probe the same score reaches 0.946, which is
precisely why the real probe was necessary.

---

## 2. Engine design and the benchmarks behind it

Every stage was chosen by benchmark. Full tables in `tables/`.

| stage | selected | rule | why not the incumbent |
|---|---|---|---|
| **A** projection | `elastic_net` | zero negative mass (hard), then max replicate × noise stability | NNLS reconstructs marginally better (0.822 vs 0.819) but is less replicate-consistent (0.881 vs 0.891). Ridge and ARD reconstruct at 0.24 and 0.35 — an unconstrained fit on a coherent dictionary is not usable. |
| **B** LSM → CSM | `direct_csm_projection` | max EV × replicate consistency | Aggregating LSM activations through membership gives EV 0.712; projecting directly onto the CSM basis gives 0.821. **The LSM layer is therefore not on the CSM path** — it is computed and reported for explanation only. This is a real architectural finding, not a shortcut. |
| **C** theme mode | `confidence_weighted` | zero-evidence leakage (hard veto), then replicate × class retrieval | The softmax mode scored *best* on replicate consistency (0.993) and was **rejected**: it assigns non-zero activation to themes for which the spectrum activates no member CSM at all. A constant flat vector is perfectly reproducible and completely uninformative. |
| **D** BSV | `theme_only` | max within-molecule cosine × separation ratio | Adding residual, rejected-theme mass or bridge mass *lowers* separation (7.26 → 6.22 / 7.17 / 5.10). Uncertainty belongs beside the vector, not inside it: a coordinate in the BSV participates in every downstream distance, and two spectra should not be "far apart" because one was noisier. |
| **E** geometry | `landmark_barycentric` | max leave-one-out neighbour preservation | Nyström is the principled extension for a diffusion map and the **worst** here (0.388 vs 0.690) — 49 reference coordinates are too few and too spread for a kernel average to localise. |

**Dictionary conditioning**, which is why the estimator choice matters at all: max coherence
0.97, condition number 1.4 × 10³, effective rank 21.9 of 50 components.

---

## 3. The Biochemical State Vector

**`BSV = Sᵀc` over the four accepted themes. Absolute, non-negative, four-dimensional.**
Everything else — residual, rejected-theme mass, bridge proximity, OOD, confidence — is
**metadata beside the vector**, because the benchmark says a wider vector separates molecules
worse, and because a distance in BSV space must mean a biochemical difference and nothing else.

**Effective rank 2.40 of nominal K = 4** (risk R-12). Two of four axes are effectively
independent. The V5 precedent was a 38% gap between nominal and effective dimension; here it is
40%, and it is reported alongside K rather than instead of it.

`bsv_elevation` (signed z-score against the frozen frame) is a separate field and is never
named `bsv` (contract C-10). No ΔBSV is returned by the inference path.

---

## 4. Validation

### Two splits, because one cannot answer both questions

The brief asks for molecule top-k **and** molecule-grouped CV. Those are incompatible: grouping
withholds every spectrum of the held-out molecule, so its identity can never be retrieved and
top-1 is exactly zero however good the engine is. The first run reported 0.000 at every level —
that was the split, not the engine.

- **Split A** — leave-one-spectrum-out over 309 spectra of 88 replicated molecules. *Can a
  known molecule be identified from a new measurement?* Molecule top-k defined.
- **Split B** — the frozen Phase 00 molecule-grouped folds. *Can an unseen molecule be placed
  in the right chemistry?* Molecule top-k **undefined** and not reported as a result.

### Results by level

Table in §1. Activation recovery against each molecule's own reference profile (split A):
LSM cosine 0.920 / top-3 overlap 0.727 · CSM 0.923 / 0.741 · theme 0.984 / 0.916.

**Geometry**: neighbourhood purity 0.482 at **4.06× chance**. Held-out spectra land among their
own chemistry far more often than chance, which is the property the frozen manifold was meant to
provide.

**Per class** (split B, CSM level): full table in `validation/per_class_retrieval_v1.csv`.

**Calibration**: ECE **0.486** — poor. Confidence is monotone in accuracy but badly
overconfident. Reported as a failure; it is not usable as a probability today.

---

## 5. Does abstraction improve inference?

Tested, not assumed. Three separate answers:

1. **Molecule identity peaks at the LSM layer** (0.806), marginally above the raw spectrum
   (0.790) and far above the themes (0.553).
2. **Chemistry generalisation to unseen molecules peaks at the CSM layer** (0.855 vs 0.608 raw)
   — the largest single effect in the phase.
3. **Replicate consistency peaks at the theme/BSV layer** (0.979).

So the hypothesis "exact molecule less robust → theme more robust → geometry most stable" is
**partly true and partly false**. Stability does rise with abstraction, up to the themes. But
the geometry layer is *not* the most stable (0.946 < 0.979), and abstraction past the CSM layer
costs more identity than it buys stability for any task that needs to know what a spectrum is.

**The practical reading: the CSM layer is the engine's working representation; the BSV is a
summary for comparison and trajectory, not for identification.**

---

## 6. What failed

1. **Out-of-domain detection on real SERS: AUROC 0.548.** The atlas cannot tell modality.
   The first OOD formulation was worse than chance (0.409) because it scored the *reconstruction*,
   which lies inside the dictionary cone by construction; rebuilding it on the residual fixed the
   synthetic probe (0.670 → 0.946) and did not fix the real one (0.409 → 0.548). This is a
   property of the representation, not of the score.
2. **Confidence calibration, ECE 0.486.**
3. **The LSM layer is not on the CSM path** — direct projection onto the CSM basis reconstructs
   better than any aggregation of LSM activations. The LSM layer survives as an explanation
   layer, which is what LIVE_DART_COMPATIBILITY requires of it, but it is not load-bearing for
   inference.

## 7. Limitations

1. **375 spectra, 154 molecules, 88 of them replicated.** Split A rests on 309 queries.
2. **Dictionary-level leakage of +0.055 top-1 remains** in every split-A number: only the
   leakage control is fold-honest.
3. **Every number is in-domain pure Raman.** The single out-of-domain probe failed.
4. **Confidence is not a probability.**
5. **Effective rank 2.40 of 4** — the BSV is closer to a two-dimensional object than a
   four-dimensional one.
6. **Geometry extension is benchmarked only by leave-one-reference-out** over 49 CSM
   coordinates, which is a small and self-referential test set.

## 8. What Phase 05 consumes

- `artifacts/engine_config_v1.json` — the frozen engine configuration and its selection rules
- `artifacts/bsv_reference_v1.json` — reference frame, per-axis spread, effective rank (C-09)
- `artifacts/inference_v1.npz` — activations, BSV and coordinates for all 375 reference spectra
- `SpectrumState` — the canonical internal representation, with the explanation chain

Phase 05 (in-domain Raman validation) should treat the +0.055 leakage figure as the correction
to apply to any in-sample benchmark, and should not report an OOD capability until the SERS
failure is addressed.

## 9. Decision gate

See the gate returned with this report.
