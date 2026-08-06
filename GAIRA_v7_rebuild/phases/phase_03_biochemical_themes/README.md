# Phase 03 — Biochemical theme construction

> ## ARCHIVED — decision A-13
>
> Built and measured as canonical **Phase 03**
> (`results/v7_rebuild/phase03/`). Retired on evidence: class top-1 on unseen molecules fell
> from 0.855 at the CSM layer to **0.405** at the theme layer. All outputs are preserved,
> fingerprinted and reproducible; the theme layer is simply not on the inference path.
> Superseded by the **Chemistry Evidence** layer (Phase 06). See
> `context/GAIRA_V7_ARCHITECTURE_STATUS_AFTER_PHASE05.md` §3 F-01, §5.2 A-13.


**Status:** COMPLETE — `K = 5`, archetypal analysis. 4 themes accepted, 1 rejected.
Outputs at `results/v7_rebuild/phase03/`.

---

## What was produced

| theme | name | CSMs | bootstrap | confidence |
|---|---|---:|---:|---:|
| Theme-01 | carboxyl / ester carbonyl + amide backbone | 16 | 0.69 | 0.76 |
| Theme-02 | aliphatic chain + unsaturated chain | 17 | 0.96 | 0.90 |
| Theme-03 | *(rejected — bootstrap 0.59 < floor 0.60)* | 23 | 0.59 | — |
| Theme-04 | aliphatic chain + polar skeletal backbone | 19 | 0.77 | 0.79 |
| Theme-05 | heterocyclic / conjugated ring + sulfur / thiol | 16 | 0.62 | 0.71 |

25 member CSMs · **15 bridges, left as bridges** · **9 poorly explained, left unplaced**.

## Gate — all passed

- [x] Themes derived FROM the CSMs, never asserted over them (L-05)
- [x] No disease, pathway, process or phenotype name (P-07) — enforced by a registry invariant
- [x] Soft membership retained; no CSM forced to a single parent
- [x] `K` justified on a Pareto frontier with band-based admissibility as a hard veto
- [x] Theme layer's value over the CSM layer measured and reported (+0.082 retrieval, small)
- [x] No chemistry label visible during discovery; revealed once, after `K` was fixed

## What Phase 04 consumes

- `artifacts/theme_membership_v1.npz` — `S` (49 × 5) and the 5 × 676 theme basis
- `artifacts/theme_registry_v1.json` / `.yaml` — names, confidences, counter-evidence,
  gradients, bridge annotations, poorly-explained list
- `artifacts/hierarchy_v1.json` — four inferred levels; the top one is the hydrophobic/polar
  split that Phase 02.5 and PCA also found

**BSV dimension = K = 5.** Four constraints Phase 04 must carry rather than discard are listed
in `results/v7_rebuild/phase03/reports/PHASE_03_REPORT.md` §8.

## Reference documents

- `../../../results/v7_rebuild/phase03/reports/PHASE_03_REPORT.md` — the report
- `../../../results/v7_rebuild/phase03/reports/PHASE_03_SCIENTIFIC_AUDIT.md` — reviewer-style
  audit; **not ready for external submission** without four named experiments
- `../../../results/v7_rebuild/phase03/reports/PHASE_03_FIGURES.pdf` — 13 figures with captions
