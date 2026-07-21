# GAIRA V6 Demo — Scientific Claims Audit

Line-by-line audit of the visible scientific statements. For each: **source**,
**evidence level** (Observation = measured; Interpretation = reasoned from evidence;
Hypothesis = future/testable), whether it is **supported by the displayed data**, and
whether **caveating is adequate**. Wording was corrected where a claim outran its
evidence; no claim was strengthened for presentation.

Legend — Level: O=observation, I=interpretation, H=hypothesis.

## Page 1 — Overview

| Claim | Source | Level | Supported | Caveat |
|---|---|---|---|---|
| 375 spectra / 167 analytes / 24 components / 13 MSS / 11 themes | live registries | O | yes | — |
| atlas variance explained 71% | manifold stats | O | yes | — |
| "when adsorption is good, Raman biochemical fingerprints remain informative" | working hypothesis | **H** | flagged | explicitly labelled a working hypothesis; Page 5 shows where it fails |
| radar axes are not independent concentrations | BSV validation | I | yes | stated inline |

## Page 2 — Reference Atlas

| Claim | Source | Level | Supported | Caveat |
|---|---|---|---|---|
| atlas learns molecular *class*, not species | Component Audit | I | yes | stated |
| families overlap (Raman ≠ unique barcode) | reference PCA | I | yes | stated by the figure |
| c3 is purine-associated, not "sterol" | frozen loadings (adenine top) + adenine perturbation + nucleic_purine weight 0.47 | I | yes | framed as evidence over label |
| components → MSS → themes is many-to-many | MSS registry / ontology W | O | yes | Sankey labelled "no one-to-one implied" |
| sterol/heme under-represented | corpus | O | yes | stated |

## Page 3 — How GAIRA Reasons

| Claim | Source | Level | Supported | Caveat |
|---|---|---|---|---|
| high adenine → purine motif on top | live engine + MSS | O | yes | — |
| spectral collisions resolved in coordinate space | collision map | I | yes | stated |
| effective dimensionality ≈ 4 of 11 | BSV validation | O | yes | stated in limits |
| large SERS elevations are OOD, read ordering not magnitude | live OOD | I | yes | stated in caveats |

## Page 4 — Calibration

| Claim | Source | Level | Supported | Caveat |
|---|---|---|---|---|
| adenine = component redistribution | live component evolution (c3/c13 rise, others fall) | O/I | yes | — |
| ergothioneine = single-motif scaling; ρ≈0.96, Langmuir R²≈0.95 | live dose-response | O | yes | strong-adsorber, buffer, best case (stated) |
| uricase drops the oxopurine motif specifically | live before/after | O | yes | serum OOD; direction/specificity only (stated) |
| "validates the reasoning layer where the analyte is effectively recovered … does not imply universal SERS modality invariance" | — | I | yes | explicit per-tab caveat |

## Page 5 — Serum Spike Stress Test

| Claim | Source | Level | Supported | Caveat |
|---|---|---|---|---|
| 6 strong / 8 partial / 39 poor; 7/53 above null; median angle ≈90° | phase7 (committed) | O | yes | criteria documented |
| strong tier = strong Ag adsorbers (oxopurines, ergothioneine) | phase7 | O | yes | — |
| phenylalanine fails yet is present ("failure ≠ absence") | phase7 + live before/after | I | yes | stated repeatedly |
| confidence does NOT track recoverability | live (strong≈poor mean conf) | O | yes | flagged as the key limitation |

## Page 6 — Biological Studies

| Claim | Source | Level | Supported | Caveat |
|---|---|---|---|---|
| diabetes Impact vs Strong-D: purine Δ−0.052 (δ=−0.88, q<0.001), sulfur δ=+0.95 | patient-level MWU+FDR+bootstrap | O | yes | patient-level; "consistent with", labels verbatim |
| HCC vs control: moderate difference | spectrum-level | O | yes | flagged spectrum-level, exploratory |
| COVID vs Healthy: near-null | spectrum-level | O | yes | reported as negative result, not inflated |
| cross-study separation is domain/matrix, not biology | centroid PC1=95% | I | yes | stated in limits |
| interpretation uses "consistent with / associated with", never "contains/proves/diagnoses" | — | I | yes | enforced in template |

**Correction applied.** For the large-n spectrum-level cohorts (COVID, HCC), the audit
emphasises **effect size (Cliff's delta)** over the p-value, and the verdict text calls
COVID a near-null result despite small p — so statistical significance is never
presented as biological importance.

## Page 7 — Future DART

| Claim | Level | Caveat |
|---|---|---|
| static point → dose trajectory → DART trajectory | I (concept) | page-level "Not implemented" banner |
| eight trajectory classes | **H** (conceptual) | figure titled "CONCEPTUAL … not measurements" |
| four falsifiable predictions | **H** | explicitly "testable hypotheses, not results" |
| Au-SERS observation layer | **H** | "No Au correction model exists today" |
| calibration trajectories (the only real data) | O | labelled REAL precedent |

## Page 8 — Methods & Provenance

| Claim | Level | Supported |
|---|---|---|
| version manifest + fingerprints; atlas verified on load | O | yes (live) |
| implemented equations (BSV/MSS/OOD/ΔBSV) | O | match implementation; conceptual future equations kept separate |
| validation-library conclusions + limitations | O | each links a committed study |

## Summary

No unsupported molecule/pathway/diagnosis claim appears anywhere. Every observation
traces to the live engine, a committed artifact, or a validated study; every
interpretation is hedged; every hypothesis is labelled. The two places most at risk of
overclaiming — large-n biological significance and the SERS working hypothesis — are
explicitly caveated (effect-size framing; "does not imply universal SERS invariance").

---

## Correction pass — additional / revised claims

| Claim | Source | Level | Supported | Caveat |
|---|---|---|---|---|
| calibration radars differ across doses (delta radar) | live engine (traced) | O | yes | absolute radar's flatness is compositional closure, stated |
| adenine redistributes (R=0.46), ergothioneine scales (R=0.12) | live mechanism metrics | O | yes | — |
| adenine poor serum recovery is a 0.4 µM concentration case, not proven poor adsorption | phase7 + spike conc | I | yes | "consistent with low conc + matrix masking"; contrasts phenylalanine (78 µM, genuine failure) |
| direction agreement (not reproducibility) is the meaningful recoverability term | ablation | I | yes | ablation shows reproducibility misranks phenylalanine |
| confidence ≠ recoverability; metrics shown separately | metrics.py | I | yes | recoverability = None for unknown spectra, never scored positive |
| SHINE dose×time interaction (paired) | paired slopes | O | yes | cohort-level pairing (cell-culture), stated; near-null pooled |
| small2023 c100-vs-c00 is a probe-loading effect | characterization | I | yes | flagged characterization-only, not biology |
| diabetes difference exceeds heterogeneity (ratio 1.79) | distance analysis | O | yes | within one cohort; not universal diabetes biology |
| NMF component similarity map = representation, PCA = exploratory | MDS/PCA | I | yes | PCA labelled "not used for inference" |

**Corrections applied this pass:** removed any implication that flat absolute radars
meant no change (now delta-radar default); reframed adenine serum recovery around its
0.4 µM spike; separated "confidence" into atlas support / theme specificity / matrix
recoverability / replicate reliability; SHINE/small2023 given honest cohort framing;
biological pages now lead with effect sizes + heterogeneity, not radar/PCA.
