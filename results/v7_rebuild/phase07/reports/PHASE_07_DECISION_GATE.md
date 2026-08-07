# GAIRA V7 — Phase 07 Decision Gate

**Status** COMPLETE — 17 of 17 gates pass. **Phase 08 not begun.**

---

## The five questions

### 1. Does BSV2 preserve the validated Chemistry Evidence layer?

**Yes, at 1.78× compression — with one named exception.**

- reconstruction EV **0.818**, mean cosine 0.929, held-out EV 0.765
- held-out chemistry prediction 0.667 against the layer's own 0.755 — **retention 0.883**,
  against a pre-registered floor of 0.50
- **exception:** `nucleic_acid_polymer` reconstructs at EV **0.064**. It has three spectra in the
  corpus. `chromophore_pigment` (0.182) and `small_nitrogenous` (0.196) are also weak.

**Anything downstream that depends on nucleic-acid-polymer chemistry must read the Chemistry
Evidence layer directly, not BSV2.**

### 2. How many biochemical programmes should GAIRA use?

**Nine — but as a declared tie-break, not a discovered optimum.**

Eligible K ran 4–12. The objective is flat across K = 8–11 (0.858, 0.859, 0.839, 0.859), well
inside the bootstrap spread, and the pre-registered rule breaks ties toward the smaller K. The
honest statement is *"between 8 and 11, and 9 by rule"*.

This matches the guidance given with the brief: the data were allowed to determine the
dimensionality, and they determined a **range**. Nine is the parsimonious point in it.

### 3. Should BSV2 replace the current radar?

**No — not yet, and not as currently fitted.** Three reasons.

1. **The adopted model has signed loadings.** A radar spoke that can subtract chemistry evidence
   is not a defensible visual. The fully non-negative alternative — orthogonal NMF at K = 6 —
   costs 0.035 of objective and 0.163 of reconstruction EV, and is markedly more disentangled
   (max overlap 0.214 vs 0.400). **This choice must be made before any radar is built.**
2. **The 16-axis Chemistry Evidence radar validated in Phase 06 is grounded, calibrated and
   provenanced.** BSV2 has no calibration layer and no per-axis confidence. Replacing a
   calibrated display with an uncalibrated one is a regression.
3. **Two programmes share a description with two others.** A radar with two spokes labelled
   "protein and amino-acid" and two labelled "nucleic" is worse than one with sixteen distinct
   chemistry names.

**Recommendation: BSV2 becomes a *second* view, not a replacement.** The Chemistry Evidence radar
answers *what chemistry is present*; a BSV2 display would answer *what programmes explain it*.
The brief itself says these are different questions.

### 4. Should BSV2 become the canonical biochemical representation?

**No. It becomes a validated derived layer.**

The canonical representation remains the 49-dimensional CSM activation vector (A-08), with
Chemistry Evidence as the validated interpretable layer (A-19). BSV2 is adopted as A-20: a
compact, stable, interpretable *derived* view — not a replacement for either.

The decisive evidence is §9 of the report: BSV2 beats its own PCA control by **0.006** on
held-out chemistry (two spectra) and **loses** to it on mutual information (2.075 vs 2.649). A
representation that cannot clearly out-explain a linear rotation of its own input has not earned
canonical status. What it has earned is a place as an interpretable summary, on the strength of
non-negative activations, additive structure, and axes that describe in chemical language.

### 5. Does Phase 08 have a stable biochemical foundation?

**Yes, with two conditions.**

Stable: bootstrap 0.972, seed 0.957, molecule-grouped fold 0.979, with **0 of 9 programmes below
0.70 recovery**; held-out reconstruction gap 0.053; noise robustness 0.947 propagated through the
whole frozen chain.

**Condition 1 — resolve the P-02 loading-sign question first.** Phase 08 building on a signed
programme layer, and then having that layer ruled inadmissible, would waste the phase.

**Condition 2 — Phase 08 must read the CSM layer and the Chemistry Evidence layer directly**, not
only BSV2. Hierarchical retrieval needs the fine chemistry that BSV2 compresses away, and the
Phase 06 result already established that the chemistry prior should be *soft* rather than a
filter.

---

## Decision

| | |
|---|---|
| **BSV2 adopted** | **Yes**, as decision A-20, a validated derived layer |
| **K** | 9 (range 8–11; 9 by pre-registered tie-break) |
| **Family** | semi-NMF by the rule; **orthogonal NMF at K = 6 is the P-02-compliant alternative and the open question** |
| **Replaces the radar** | **No** |
| **Canonical representation** | **No** — CSM remains canonical, Chemistry Evidence remains the interpretable layer |
| **Phase 08 may proceed** | **Yes**, subject to the two conditions above |
| **Confidence** | **7 / 10** |

## Required before Phase 08

1. **Decide whether P-02 (non-negativity) binds programme loadings.** If it does, adopt
   orthogonal NMF at K = 6 and re-run the validation; the cost is quantified.
2. **Record in the architecture documents** that nucleic-acid-polymer chemistry does not survive
   the BSV2 compression.
3. **Report BSV2 vs PCA with a confidence interval**, so the +0.006 is visibly not significant.

## Not done, deliberately

Phase 08 — hierarchical molecular retrieval — has **not** been begun.
