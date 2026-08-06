# GAIRA V7 — Success Criteria

What V7 must achieve before it replaces the current frozen atlas.

> **STATUS: FROZEN in Phase 00.** These thresholds were provisional during the specification
> pass and were frozen on completion of Phase 00, before any V7 model was fitted. They are not
> adjusted afterwards — not upward if V7 looks strong, and not downward if V7 looks weak
> (principle P-13). The freeze record is in §6; the baseline they are measured against is
> `results/v7_rebuild/phase00/tables/phase00_baseline_metrics.csv`.

---

## 1. The baseline V7 must beat

The V5 frozen atlas `09ed804a40836f4a05a91ba10900cded`, measured under the **same** Phase-00
evaluation harness (not against previously published numbers — the harness must be identical
or the comparison is not like-for-like).

Reference values from `results/v6_rebuild/v63_ontology_revalidation/`, n = 167 analytes:

| Layer | Metric | V5 value |
|---|---|---|
| MSS (17 motifs) | fine top-1 (V6.3 fine ontology) | **0.6707** |
| MSS | fine top-1 (old ontology) | 0.6766 |
| MSS | broad-6 top-1 | **0.8084** |
| MSS | MRR | 0.7594 |
| MSS | macro-F1 | 0.5771 |
| MSS | balanced accuracy | 0.5748 |
| coord (24 components) | fine top-1 | 0.6467 |
| coord | broad-6 top-1 | **0.8204** |
| themes (6) | fine top-1 | 0.6228 |
| themes | broad-6 top-1 | 0.7665 |
| MSS | true representation failures | **31 of 54 (57.4%)** |
| coord | true representation failures | 26 of 57 (45.6%) |
| components | purity ≥ 0.5 | **3 of 24** |
| components | median bootstrap stability | 0.799 |
| basis | explained variance | 0.712 |
| basis | participation ratio | 15.2 (of 24 nominal) |
| motif coverage | analytes uncovered by any v1 motif | 107 of 167 (64.1%) |

Random-ontology control (12 size-matched draws): fine 0.096–0.106, broad ~0.22.

---

## 2. Minimum replacement criteria

### Tier 1 — must all pass

| # | Criterion | Threshold | Rationale |
|---|---|---|---|
| **S-01** | CSM/MSS-equivalent top-1 improvement | **≥ +8 percentage points** over 0.6707 → **≥ 0.7507** | must exceed the ±0.06 CI width of the V5 estimate by a clear margin; anything smaller is not distinguishable from noise at n=167 |
| **S-02** | CSM top-3 | **≥ 90%** | the correct chemistry must be in the top three essentially always for the layer to be usable as evidence |
| **S-03** | Fine-family retrieval improvement | **≥ +8 percentage points** | the fine-resolution ceiling (L-04) is the specific problem V7 exists to solve |
| **S-04** | Broad-superclass retrieval | **maintained or improved** (≥ 0.808 at CSM level) | V5's genuine strength; a fine-resolution gain bought by losing coarse chemistry is not progress |
| **S-05** | True projection failures | **fewer than V5** (< 31 at CSM level) | the single most direct measure of representation quality |
| **S-06** | Deterministic reproduction | byte-identical, twice, two machines | non-negotiable |
| **S-07** | Clean inference from a frozen package | no lab volume, `GAIRA_DATA_ROOT` unset | V5 already achieves this; V7 must not regress it |

### Tier 2 — must all pass, measured qualitatively where noted

| # | Criterion | Threshold |
|---|---|---|
| **S-08** | Diagnostic-band fidelity | improved over V5 at the equivalent layer |
| **S-09** | LSM/CSM stability | ≥ V5's median component stability (0.799) |
| **S-10** | Pathological collisions | fewer than V5 — no CSM whose top-activating family contradicts its chemistry (the `sterol_ring_system` → fatty_acid failure must not recur) |
| **S-11** | Rare-class coverage | improved: sterol, porphyrin, flavin, phosphate, phospholipid, carotenoid, nucleic acid each supported by ≥1 CSM with band-fidelity above threshold |
| **S-12** | Interpretability | not degraded — every CSM nameable, every theme nameable, every BSV axis traceable to chemistry |
| **S-13** | Provenance completeness | 100% of CSMs resolve to LSMs → classes → analytes → sources |
| **S-14** | Calibration | ECE no worse than V5 at the equivalent layer |

### Tier 3 — reported, not gating

| # | Item |
|---|---|
| S-15 | Effective rank of the BSV space vs nominal `K` |
| S-16 | Singleton and anchored CSM counts |
| S-17 | Per-class LSM coverage |
| S-18 | Semantic rescues vs semantic degradations (both reported with equal prominence) |
| S-19 | Reconstruction explained variance |
| S-20 | Live projection latency |

---

## 3. Statistical requirements

A threshold met by a point estimate is not a result.

| Requirement | Specification |
|---|---|
| Significance | McNemar exact + permutation test, paired on the same analytes |
| Confidence intervals | bootstrap over canonical analytes, 95%, reported for every headline number |
| Permutation null | ≥ 12 size-matched random ontologies |
| **Gain beyond mechanical** | every coarse-level comparison reports the gain *beyond* the random-ontology control — V6.3's `gain_beyond_mechanical` column is the template |
| Effect size | Cohen's g and odds ratio |
| Multiple comparisons | correction declared in Phase 00, applied consistently |

**S-01 and S-03 must be significant at α = 0.05 after correction.** An 8-point improvement
with p = 0.3 does not pass.

---

## 4. Where the +8 point threshold comes from

Not arbitrary, and worth stating so it can be argued with:

1. **Noise floor.** The V5 MSS fine estimate is 0.671 with 95% CI [0.599, 0.743] at n=167 —
   a width of ~0.14. An improvement must clearly exceed sampling noise.
2. **The V6.3 lesson.** Ontology cleanup produced changes of −0.012 to +0.018 at coord, MSS,
   and theme levels, none significant. Improvements of that size are indistinguishable from
   noise in this corpus. The threshold must sit well above that band.
3. **Materiality.** +8 points takes MSS-equivalent fine retrieval from 0.671 to ~0.751 — from
   "right two times in three" to "right three times in four". That is a difference a user
   would actually notice.
4. **Cost justification.** V7 is a substantial rebuild with a more complex atlas and a higher
   maintenance burden. A 2-point gain would not justify it even if it were significant.

**If a rebuild of this scope cannot deliver +8 points, the honest conclusion is that the
corpus — not the architecture — is the binding constraint**, and the correct next step is
Phase 09 corpus expansion rather than further architectural work. That is a legitimate and
useful outcome, and the plan should be able to reach it.

---

## 5. Failure handling

If V7 does not meet the Tier-1 criteria:

1. **The V5 atlas is retained.** It stays in production, unmodified.
2. **The negative result is documented in full** — every phase report, every table, every
   figure stays committed. A rebuild that did not clear the bar is evidence about the problem,
   and this project has already benefited from exactly that (V6.3 established that ontology
   cleanup was not the fix, which is why V7 targets representation instead).
3. **Partial adoption is considered explicitly.** V7 may improve some layers without clearing
   the overall bar — for instance better rare-class coverage or better provenance with equal
   retrieval. Partial adoption requires its own written justification and its own criteria; it
   is not a consolation prize awarded by default.
4. **The criteria are not adjusted.** (P-13.)

---

## 6. Freeze record

| Field | Value |
|---|---|
| Status | **FROZEN** |
| Frozen by | Phase 00 — `results/v7_rebuild/phase00/` |
| Frozen at source commit | `b9f78ff5a22366f8d0ae7aab11d635e4ff961e24` |
| Baseline harness | `v7_harness_v1` — the same harness measures V5 and V7 |
| Split manifest | `v7_cv_v1` — 5 folds grouped by `canonical_id`, all leakage checks false |
| Baseline re-measured | ✅ `phase00_baseline_metrics.csv` |
| Quality score | `v7_q_v2`, frozen before Phase 01 |
| Atlas fingerprint at freeze | `09ed804a40836f4a05a91ba10900cded` (rebuilt from raw, max abs diff 0.0) |

### Frozen Tier-1 thresholds, against the measured baseline

| ID | Baseline (V5, `v7_harness_v1`) | Frozen threshold |
|---|---:|---:|
| S-01 CSM/MSS-equivalent fine top-1 | 0.6707 | **≥ 0.7507** |
| S-02 CSM top-3 | — | ≥ 0.90 |
| S-03 fine-family improvement | 0.6707 | **≥ +0.08** |
| S-04 broad-superclass retrieval | 0.8084 | **≥ 0.8084** |
| S-05 true projection failures | 31 of 54 | **< 31** |
| S-09 LSM/CSM stability | 0.799 median | ≥ 0.799 |

S-01 and S-03 must additionally be significant at α = 0.05 after correction (McNemar +
permutation on the frozen folds).

**These numbers are now fixed. No V7 result may be described as passing or failing against any
other threshold.**
