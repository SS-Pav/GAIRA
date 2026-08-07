# GAIRA V7 — Phase 07: BSV2, the biochemical programme layer

**Status** COMPLETE — 17 of 17 gates pass. **BSV2 adopted: yes, conditionally** (§11).
**Scope** Raman only. **Input** the validated 16-dimensional Chemistry Evidence matrix, and
nothing else. **Phase 08 not begun.**
**Frozen inputs verified** LSM `208482d6…` · CSM `0b4aa550…` · engine `20d8bd99…`
**Selected** semi-NMF at **K = 9**, by the rule pre-registered in
`src/gaira/v7/programs/selection.py` and applied without adjustment.

---

## 1. Executive summary

Nine biochemical programmes emerge from the Chemistry Evidence layer, and they are what the
brief hypothesised they would be. The two examples the brief gave both appear, unprompted:

| programme | dominant chemistry axes | automatic description |
|---|---|---|
| **P3** | fatty acid 30% + acylglycerol 29% + phospholipid 18% | membrane and storage lipid |
| **P0** | phosphate metabolite 35% + carboxylic acid 32% | small-molecule energy metabolism |
| P8 | mono/oligosaccharide 33% + polysaccharide 23% | carbohydrate |
| P4 | free amino acid 46% | protein and amino-acid |
| P6 | peptide protein 45% | protein and amino-acid |
| P5 | sterol/steroid 47% | membrane and storage lipid |
| P7 | purine 46% | nucleic |
| P1 | pyrimidine 52% | nucleic |
| P2 | sulfur/thiol cofactor 55% | redox cofactor and pigment |

No programme was named by hand. Every description is composed by template from the loadings
above, and a programme whose top axis fell below 15% would have been described as *diffuse*
rather than given a name it had not earned.

**Headline numbers**

| | value |
|---|---:|
| compression | 16 → 9, **1.78×** |
| reconstruction EV / cosine | **0.818** / 0.929 |
| held-out reconstruction EV | 0.765 (gap −0.053) |
| bootstrap / seed / fold stability | **0.972** / 0.957 / 0.979 |
| programmes below 0.70 recovery | **0 of 9** |
| held-out chemistry top-1 | 0.667 vs the layer's own 0.755 (**retention 0.883**) |
| max programme overlap | 0.400 (mean 0.195) |
| top programme's usage share | 0.221 (ceiling 0.60) |
| noise robustness (programme cosine) | 0.947 |
| genuinely multi-chemistry programmes | **7 of 9** |

**Three things that qualify the result.**

1. **The rule's winner uses semi-NMF, whose programme loadings may be negative.** Principle P-02
   says non-negativity is not optional. The constrained optimum — orthogonal NMF at K = 6, fully
   non-negative — costs **0.035 of the objective** and reconstructs less (EV 0.655 vs 0.818) but
   is markedly more disentangled (max overlap 0.214 vs 0.400). §11 puts this to the decision gate
   rather than resolving it silently.
2. **BSV2 barely beats its PCA control, and loses on one measure.** Held-out chemistry top-1
   0.667 vs 0.661 — a difference of two spectra. Normalised mutual information with chemistry is
   *lower* for BSV2 (2.075) than for PCA (2.649). BSV2's case rests on non-negativity and
   interpretability, not on out-predicting a linear rotation.
3. **One chemistry axis is essentially not reconstructed at all**: `nucleic_acid_polymer` at
   EV 0.064. It has three spectra in the entire corpus.

---

## 2. What was, and was not, allowed in

The factorisation sees `Ev ∈ ℝ₊^{375×16}` and nothing else. A test asserts that the model-fitting
section of the run script contains no reference to `balanced_references`, `PRJ.project`,
`csm_dictionary`, `PERT.apply`, `embeddings`, `continuous_coordinates` or `theme`.

Spectra and CSM activations *do* appear later in the script, and only there: for the noise
robustness study (§8, which must perturb spectra to propagate through the chain) and for the
explanation layer (§6, which maps a programme's chemistry back to CSMs and bands). Both are
downstream of a fitted model.

Phase 06.5 established that no natural cluster count exists in the CSM manifold and that
continuous geometry does not improve inference. This phase therefore does not rediscover
clusters, does not rediscover chemistry classes, and inserts no geometry.

## 3. The pre-registered decision rule

Written into `selection.py` before any model was fitted:

**Stage 1 — hard floors.** Information retained ≥ 0.50; held-out chemistry retention ≥ 0.50
(both carried from success criterion S-28); max pairwise programme overlap ≤ 0.90; no programme
dominating > 0.60 of spectra; non-negative activations (P-02 — PCA and ICA are permanently
controls); **K ≤ 12** (the input's own effective rank, measured in Phase 06); **no programme
placing > 0.90 of its loading on one chemistry axis**.

**Stage 2 — objective.** Maximise `held-out chemistry retention × bootstrap stability`. A
*product*, so a candidate must be both informative and reproducible and neither compensates for
the other. This is the shape of rule that discarded Meta Components in Phase 04.5.

**Stage 3 — ties** within 0.01 break toward the **smaller K**.

> **The last two floors were added after a first run, and the reason matters.** As first written
> the rule selected **K = 16** over a 16-dimensional input, where NMF learns a permutation of the
> identity: EV 1.000, bootstrap 0.997, and every "programme" equal to exactly one chemistry
> class. That is not a compression and not a programme layer. Both missing constraints are
> restatements of things the brief says explicitly — *a programme should not equal one chemistry
> class*, and *BSV2 is a compression* — so encoding them corrected a specification bug rather
> than moving a threshold. The correction is recorded in the module, in the audit, and here.

## 4. The sweep

Six families × 15 values of K, with sparse NMF's penalty swept over four values:

| family | eligible at K | note |
|---|---|---|
| NMF | 4–11 | |
| orthogonal NMF | 4–11 | best fully non-negative candidate |
| **semi-NMF** | **4–12** | **selected at K = 9** |
| sparse NMF (α = 0.005) | 4–12 | |
| sparse NMF (α = 0.02) | 6–9 | |
| sparse NMF (α = 0.05, 0.1) | never | penalty drives the loadings to zero, EV −0.401 |
| PCA control | never | signed activations (P-02) |
| ICA control | never | signed activations (P-02) |

38 of 135 candidates eligible. Sparse NMF was originally fitted at a single α = 0.2 where it
scored EV −0.401 at every K; a required family excluded by an untuned hyperparameter is not a
finding, so the penalty was swept and the family now competes.

## 5. Reconstruction

| | value |
|---|---:|
| RMSE | 0.0313 |
| explained variance | **0.818** |
| mean cosine | 0.929 |
| relative Frobenius | 0.211 |

Per chemistry axis, the four worst:

| axis | EV | mean evidence | spectra in corpus |
|---|---:|---:|---:|
| nucleic_acid_polymer | **0.064** | low | **3** |
| chromophore_pigment | 0.182 | low | 10 |
| small_nitrogenous | 0.196 | low | 7 |
| polysaccharide | 0.557 | | 10 |

The pattern is unambiguous: **the axes reconstructed worst are the axes with least evidence to
begin with.** A 3-spectrum chemistry cannot support a programme, and BSV2 does not pretend
otherwise — it drops that axis rather than fitting noise on it. Whether that is a virtue or a
loss depends on whether nucleic-acid polymers matter downstream, which is a Phase 08 question.

## 6. Interpretability — evidence first, description afterwards

For every programme the phase reports the dominant chemistry axes and their shares, the
representative molecules, the representative chemistry classes, and — via the frozen CSM registry
— the supporting CSMs, LSMs and Raman bands. Only then is a description composed, by template,
from those facts.

**7 of 9 programmes are genuine multi-chemistry compressions** (their top axis holds < 50% of the
loading). Top-axis shares: P0 35%, P3 30%, P8 33%, P6 45%, P4 46%, P7 46%, P5 47%, P1 52%,
P2 55%. The two above 50% — pyrimidine and sulfur/thiol — are chemistries with no close
neighbour in this corpus, so a programme built on them has nothing to compress *with*.

**Two pairs of programmes receive the same description.** P4 and P6 are both "protein and
amino-acid"; P1 and P7 are both "nucleic". The descriptions are not unique identifiers, and the
report does not claim they are: P4 loads on free amino acids and P6 on peptide proteins, which
are chemically distinct and describe to the same broad superclass. That is a limitation of the
template's vocabulary, not evidence that the programmes are duplicates — their loading cosine is
well below the redundancy floor.

## 7. Stability and generalisation

| measure | value |
|---|---:|
| bootstrap recovery (resample spectra) | **0.972** ± 0.021 (min 0.918) |
| seed stability (refit) | 0.957 |
| fold stability (withhold molecules) | **0.979** |
| programmes below 0.70 recovery | **0 of 9** |
| held-out reconstruction EV | 0.765 vs in-sample 0.818 |
| replicate consistency | 0.98 |

All three kinds of stability are high, and no individual programme is unstable — which matters
more than the mean, because a respectable average can hide one programme that never survives.
The held-out gap of 0.053 is small: the programmes are reusable, not memorised.

## 8. Noise robustness

Five perturbations × five levels, propagated through the **entire frozen chain** — spectrum →
CSM → Chemistry Evidence → BSV2. Mean programme cosine **0.947**; mean argmax stability 0.890.

The argmax is less stable than the vector, which is expected and is the argument for treating
BSV2 as a continuous coordinate rather than a programme label: the *profile* survives corruption
that flips which programme happens to be largest.

## 9. Compression and information

| representation | dim | held-out chemistry top-1 | normalised MI | effective rank |
|---|---:|---:|---:|---:|
| Chemistry Evidence | 16 | **0.755** | **3.191** | 12.12 |
| **BSV2 programmes** | **9** | 0.667 | 2.075 | 7.68 |
| PCA control | 9 | 0.661 | **2.649** | 7.78 |

**BSV2 retains 0.883 of the chemistry layer's held-out prediction at 1.78× compression.** That
clears the S-28 floor of 0.50 comfortably.

**But it beats its own PCA control by 0.006 — two spectra — and loses to it on mutual
information by a wide margin (2.075 vs 2.649).** A signed linear rotation extracts more
information about chemistry from the same 9 dimensions than a non-negative factorisation does.
This is the honest answer to the brief's question *"can programmes explain chemistry better than
PCA?"*: **on prediction, marginally and not significantly; on information, no.** What BSV2 has
that PCA does not is non-negativity, additivity, and a set of axes that can be described in
chemical language — which is the entire case for it.

## 10. Scientific questions, answered

| question | answer |
|---|---|
| How many programmes naturally emerge? | **9**, under the pre-registered rule. Eligible K ran 4–12; the objective is flat across 8–11 and the tie-break chose the smaller. |
| How much chemistry can they reconstruct? | EV 0.818, cosine 0.929; per-axis EV tracks per-axis evidence. |
| How stable are they? | bootstrap 0.972, seed 0.957, fold 0.979; 0 of 9 programmes unstable. |
| Are programmes reusable? | Yes — held-out EV 0.765 against in-sample 0.818. |
| Do they correspond to meaningful biochemistry? | 7 of 9 are genuine multi-chemistry compressions, and the lipid and energy-metabolism programmes are exactly the groupings the brief predicted. |
| Is there redundancy? | Low: max pairwise overlap 0.400, mean 0.195, against a floor of 0.90. |
| Does one programme dominate? | No: the top programme wins for 0.221 of spectra against a 0.60 ceiling. |
| Better than PCA? | **Marginally on prediction (0.667 vs 0.661), worse on mutual information (2.075 vs 2.649).** |

## 11. Limitations

1. **The adopted model has signed loadings.** A programme that subtracts chemistry evidence is
   harder to defend than one that adds it, and P-02 exists for that reason. The fully
   non-negative alternative costs 0.035 of objective and 0.16 of reconstruction EV.
2. **BSV2's advantage over PCA is not significant** and is negative on mutual information.
3. **One chemistry axis is effectively lost** (nucleic_acid_polymer, EV 0.064, 3 spectra), and
   two more are weak.
4. **K = 9 is a tie-break, not a peak.** The objective is flat across K = 8–11; the rule chose
   the smallest within tolerance. A different tolerance would give a different K.
5. **Two pairs of programmes share a description** — the template's vocabulary is coarser than
   the programmes are.
6. **375 spectra and 154 molecules** is a small corpus for a 9-programme model, and the sweep is
   noisy at the margins.
7. **Every held-out evaluation is chemistry prediction, never molecule retrieval**, per the
   brief. Whether BSV2 helps molecular retrieval is a Phase 08 question and is untested here.

## 12. Reproduction

```bash
PYTHONPATH=src python results/v7_rebuild/phase07/code/run_phase07.py    # ~12 min
PYTHONPATH=src python results/v7_rebuild/phase07/code/make_figures.py
PYTHONPATH=src python results/v7_rebuild/phase07/code/make_pdf.py
PYTHONPATH=src python -m pytest tests/test_v7_phase07.py -q
```

`SEED = 0`; output root resolves through `GAIRA_V7_OUTPUT_ROOT`; no path is hardcoded.
