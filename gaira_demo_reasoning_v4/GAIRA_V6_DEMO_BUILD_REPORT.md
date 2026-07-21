# GAIRA V6 Demo — Build Report

Completion report for the full public-facing demonstration of the V6 Converged
Reasoning Engine (`gaira_demo_reasoning_v4`). Presentation only; the frozen scientific
stack was measured, never modified. **Nothing was pushed.**

## Pages implemented (all 8)

1. **Overview** — architecture figure + live platform stats + working-hypothesis
   framing.
2. **Reference Atlas** — corpus summary (stage-labelled, not merged); interactive
   reference-family PCA (167 analytes); 24-component explorer (many-to-many, c3
   educational case); MSS atlas + components→MSS→themes Sankey.
3. **How GAIRA Reasons** — the Radar→MSS→Components→Reference drill-down + MSS explorer
   + spectral-collision map.
4. **Calibration** — the signature **reasoning cascade** with a concentration slider;
   adenine (redistribution), ergothioneine (scaling, ρ≈0.96 / R²≈0.95), uricase
   (depletion; oxopurine-specific), and a compare tab (three trajectory classes).
5. **Serum Spike Stress Test** — recoverability tiers from the validated table
   (6/8/39); per-analyte before/after; success-vs-failure; the confidence limitation.
6. **Biological Studies** — genuine-V6 cohorts on one standardized template + registry
   + cross-study generalization.
7. **Future DART** — conceptual only: static→trajectory, data model, an 8-class
   trajectory vocabulary (labelled CONCEPTUAL), falsifiable hypotheses, Au-SERS seam.
8. **Methods & Provenance** — version manifest + fingerprints, data provenance,
   implemented vs conceptual equations, evidence tiers, validation library,
   reproduction commands.

## Architecture used

Frozen `gaira.engine.GAIRAEngine` + `gaira.engine.mss.MSSLayer`, driven by a thin
`demo_core` (engine bridge, theme, figures [matplotlib], interactive [plotly],
per-page analysis modules). Matplotlib figures are auditable headless via
`selfcheck.py`. No science re-implemented.

## Real datasets wired (genuine V6 outputs)

- **COVID serum Raman** — 465 spectra (COVID 159 / Healthy 150 / Suspected 156).
- **HCC serum SERS** — 144 spectra (HCC 72 / control 72).
- **Diabetes plasma-EV SERS** — 63 **patients** (Impact 39 / Strong-D 24),
  patient-level.
- Calibration: adenine / ergothioneine / uricase (committed frozen-atlas projections).
- Serum spike: 53 analytes (committed Spike Validation outputs).

Each biological record was verified to equal the live engine's BSV of its stored
coordinates (genuine V6, not legacy). All sanitized — no demographics, anonymised IDs.

## Cached / committed validated artifacts

- Frozen atlas (`manifold.json` + `manifold_components.npz`) and engine artifacts.
- MSS registry v1 (deterministic, fingerprint-locked).
- Spike Validation tables; biological_artifacts (covid/hcc/diabetes + manifest).

## Main scientific findings displayed

- Calibration: monotonic + saturating dose responses; adenine redistributes,
  ergothioneine scales; uricase depletes the oxopurine motif specifically (which the
  coarse purine theme hides).
- Serum: only strong Ag adsorbers (oxopurines, ergothioneine) are identity-recoverable;
  7/53 exceed the null; confidence does **not** distinguish strong vs weak adsorbers.
- Biological: diabetes Impact-vs-Strong-D shows a robust patient-level purine
  (δ=−0.88) / sulfur (δ=+0.95) contrast (FDR q<0.001); HCC moderate (exploratory);
  COVID near-null (honest negative result, lowest OOD because Raman is nearer domain).

## Tests added

`tests/test_v6_demo_v4.py` — 18 tests (bridge, figures, all-pages-import, Page 4
calibration behaviours, Page 2 frozen-derived reference map + many-to-many graph, Page
5 tiers/failure/confidence-limitation, Page 6 genuine-V6 + patient-level +
no-demographic-leak + unavailable-fabricates-nothing, Page 8 fingerprints, cross-page
link validity). Plus `tests/test_v6_mss_layer.py` (8). **Full V5+V6 suite: 198
passing.** Fresh-checkout simulated via `git archive HEAD` → engine loads, all pages
import, all figures render.

## Fixes beyond the demo (prerequisites for a working fresh checkout)

Two frozen engine files had been silently dropped by broad `.gitignore` rules and
never committed by earlier work — force-added, content unchanged, fingerprint verified:
`biochemical_ontology_v2.yaml` (rule `data/`) and **`manifold_components.npz`** (rule
`*.npz`, the frozen NMF basis itself).

## Known limitations / intentionally deferred

- EV small2023 / SHINE / liver cohorts are marked UNAVAILABLE (ingestion not yet wired
  for V6) — no output fabricated.
- Plotly figures (reference PCA, Sankey) verified to construct but not pixel-audited
  (kaleido absent).
- COVID/HCC are spectrum-level (subject mapping undocumented) → treated as exploratory.
- DART and the Au-SERS observation layer are conceptual; no data or model exists.
- Confidence lacks a matrix-recoverability prior (a recommended, unimplemented upgrade).

## Commit sequence (this build; nothing pushed)

```
4791e54  Track the V6 ontology yaml dropped by the broad data/ ignore rule
565a131  Add the MSS layer + rebuild the V6 scientific-reasoning demo (additive)
60fcc35  Build the definitive V6 Calibration page (Page 4)
8d055d4  Complete Page 2 — Reference Atlas
d69c990  Complete Page 5 — Serum Spike Stress Test
df979bf  Complete Page 6 — Biological Studies on GENUINE V6 outputs
467d56d  Complete Pages 7 (Future DART) and 8 (Methods & Provenance)
2a58524  Cross-page integration — clickable navigation + one narrative
a1000e4  Track the frozen NMF basis dropped by the broad *.npz ignore rule
5f24a11  Expand demo tests — Pages 2/5/6/8 + cross-page + fresh-checkout guards
(+ this commit: audits + documentation)
```

## Definition-of-done checklist

- [x] all eight pages fully implemented
- [x] all real available datasets represented accurately; biological use V6 outputs
- [x] calibration + serum pages show live evidence chains
- [x] every radar uses evidence-backed themes and resolves into MSS + components
- [x] every scientific claim has provenance (SCIENTIFIC_CLAIMS_AUDIT.md)
- [x] every unavailable input fails honestly
- [x] app runs on a fresh checkout (git-archive simulation passes)
- [x] all visualizations rendered + audited (VISUAL_AUDIT.md)
- [x] all old and new tests pass (198)
- [x] frozen scientific assets byte-identical (fingerprint `09ed804a…` verified)
- [x] nothing pushed

---

## Correction & improvement pass

A scientific-correction pass fixed misleading visualizations (not styling) and expanded
the real-data coverage. Root-cause finding: the "identical radars" and "adenine poorly
recoverable" reports were **not code bugs** — they were compositional-closure visual
compression and a 0.4 µM concentration confound, verified by tracing the computation.

**Fixes (all additive; frozen BSV/atlas/NMF/ontology/weights/MSS untouched):**
- Calibration: signed **delta radar** default (shared centred scale); redistribution R
  (adenine 0.46) vs scaling (ergothioneine 0.12) metrics; interpretable-axis MSS
  trajectory; uricase ΔBSV radar; debug table.
- Serum: pure-vs-serum audit — **adenine strong in pure, poor in serum at 0.4 µM**
  (concentration + matrix, not proven poor adsorption); phenylalanine (78 µM) a genuine
  failure; recoverability split into documented terms + ablation.
- Confidence **separated** into atlas support / OOD / theme specificity / matrix
  recoverability / replicate reliability (`metrics.py`); recoverability = None for
  unknown spectra, never scored positive.
- Reference Atlas: **NMF-native** component similarity map (classical MDS) + dendrogram
  now primary; reference-spectrum PCA demoted to a labelled exploratory view.
- Biological rebuilt **effect-size-first** (forest + Cliff's δ + bootstrap + FDR),
  sample BSV/MSS heatmaps, **between-vs-within distance ratio**, SHINE **paired**
  Day0→Day2-per-dose slopes, diabetes exploratory balanced view; radar/PCA demoted to
  expanders.
- Cache keys fingerprinted (atlas/ontology/registry/MSS/BSV); sidebar **developer mode**
  shows inference hashes + BSV/MSS/component vectors.

**New real datasets (genuine V6, committed sanitized):** SHINE EV-SERS (720; D0/D2 ×
APAP dose) and small2023 EV (600; probe titration, characterization-only). Total
biological cohorts now 5.

**Tests:** +19 correction tests (radar-differs, delta-baseline-zero, mechanism,
NMF-distance-deterministic, serum concentration audit, terms/ablation, confidence
separation, biological genuine-V6/patient-level/no-leak). **Full V5+V6 suite: 206
passing.**

**New audit docs:** NMF_VISUALIZATION_AUDIT · CALIBRATION_RADAR_AUDIT ·
SERUM_RECOVERABILITY_AUDIT · BIOLOGICAL_DATA_AUDIT · BIOLOGICAL_ANALYSIS_AUDIT; existing
audits + README + this report updated.

**Correction-pass commits (nothing pushed):**
```
f79a31b  Fix calibration radar (delta-radar default) + redistribution vs scaling
9bebe26  Wire SHINE + small2023 EV (real V6) and rebuild biological effect-size-first
6a9b57e  Audit serum recoverability + separate confidence metrics
70677c5  NMF-native Reference Atlas (component MDS + dendrogram); demote PCA
c9bde60  Cache fingerprint keys + developer debug mode
(+ this docs commit)
```
