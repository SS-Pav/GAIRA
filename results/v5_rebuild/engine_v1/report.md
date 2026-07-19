# GAIRA Converged Engine v1 — Build & Calibration Validation Report

Additive implementation of the V6 reasoning engine on the **frozen** Raman Reference Atlas v0.1. Atlas fingerprint `09ed804a40836f4a05a91ba10900cded` verified on load in every module. No retraining, no NMF regeneration, no preprocessing change, no demo change. Branch `gaira-v5-rebuild-plan`; nothing pushed.

---

## Reproduce

```bash
cd results/v5_rebuild/engine_v1/code
python build_registry.py        # Component Registry v1
python build_theme_weights.py   # Component→Theme mapping v1
python build_reference_norm.py  # reference normalization frame
python run_validation.py        # Part 11 calibration validation
python emit_examples.py         # versions manifest + example outputs
```

---

## What was built

| Deliverable | File |
| --- | --- |
| New rebuild plan | `GAIRA_Rebuild_Plan_vNext.md` |
| Architecture doc (mermaid diagrams) | `GAIRA_Engine_Architecture.md` |
| Component Registry v1 | `artifacts/component_registry_v1.json` (24 objects, provenance per field) |
| Biochemical Ontology v2 | `src/gaira/engine/data/biochemical_ontology_v2.yaml` (13 themes: 11 chemistry + background + unknown) |
| Component→Theme weights v1 | `artifacts/component_theme_weights_v1.json` (many-to-many, 3 evidence lines each) |
| Reference normalization | `artifacts/reference_normalization_v1.json` + `reference_support.npz` |
| Engine package | `src/gaira/engine/` (registry, ontology, normalization, bsv, evidence, domain, radar, pipeline, dart, versioning) |
| Example outputs + versions | `artifacts/example_inferences.json`, `artifacts/versions_manifest.json` |
| Validation summary | `tables/validation_summary.json` |

Registry interpretation confidence: **1 high, 11 moderate, 12 low** — honestly reflecting the Component Audit's finding that most components are chemically enriched but not pure.

---

## Part 11 — calibration validation (no hard-coding)

The BSV theme mapping is derived from evidence weights; there are no analyte-specific rules in the BSV. It nonetheless reproduces the known biochemistry.

### Adenine dose-response → purine theme
The **nucleic_purine** theme rises monotonically with adenine concentration in **6/6** substrate×laser arms (Spearman ρ 0.95–1.00), and is the **single top-rising theme in the 3 colloidal (cAg/cAu) arms**. In the 3 solid-substrate (sAg/sAu) arms purine still rises strongly (ρ ≥ 0.95) but competes with matrix themes — a real substrate effect, reported rather than hidden.

| arm | purine ρ | top-rising theme |
| --- | --- | --- |
| cAg@532 | 0.96 | nucleic_purine ✓ |
| cAg@785 | 1.00 | nucleic_purine ✓ |
| cAu@785 | 1.00 | nucleic_purine ✓ |
| sAg@532 | 0.99 | redox_broad |
| sAg@785 | 0.98 | sterol_membrane |
| sAu@785 | 0.95 | lipid_acyl |

### Ergothioneine dose-response → sulfur theme
The **sulfur_antioxidant** theme is the single strongest theme response (ρ **0.991**) — recovered purely through the evidence weights (ergothioneine's perturbation maps to the sulfur theme). A clean, unforced success.

### Uricase depletion → purine theme drops
Enzymatic urate removal decreases the **nucleic_purine** theme (Δ −0.011, the chemically expected direction). The single most-decreased theme is saccharide (serum-matrix noise), but the purine sign is correct.

### Purine responders (serum spikes)
The purine theme is the top *rising* theme for **hypoxanthine, xanthine, guanine** (3/4). Adenine's spike (0.4 µM, one of the lowest doses) is too weak to move the theme above the serum background — an honest miss consistent with the Spike Validation.

---

## Example inferences (Part 12 output structure)

| input | domain | OOD | conf | top biochemical themes |
| --- | --- | --- | --- | --- |
| pure adenine | buffer | 0.04 | 0.48 | **nucleic_purine 0.32**, saccharide 0.14, protein 0.12 |
| pure glucose | buffer | 0.01 | 0.51 | **saccharide_glycan 0.41**, purine 0.17, lipid 0.07 |
| pure cholesterol | buffer | 0.03 | 0.29 | **lipid_acyl 0.18**, purine 0.15, saccharide 0.14 |
| pure albumin | buffer | 0.09 | 0.25 | lipid 0.14 ≈ **protein 0.14** ≈ purine 0.14 |
| serum hypoxanthine | serum | 0.30 | 0.27 | **nucleic_purine 0.28**, saccharide 0.17 |
| serum phenylalanine | serum | 0.28 | 0.26 | nucleic_purine 0.27 (WRONG), saccharide 0.19 |

Reading: glucose, adenine, cholesterol and the purine serum spikes land on the right theme. Albumin's protein theme is only tied-top (confidence 0.25, honestly low). Phenylalanine in serum lands on purine — **wrong**, but the engine flags it: OOD 0.28 (out of domain), confidence 0.26 (low), and the serum domain caveat states weak Ag adsorbers are not recovered. The failure is surfaced, not hidden.

---

## Where it succeeds / fails / loses confidence

- **Succeeds:** pure-reference themes (glucose, adenine, cholesterol); adenine dose-response; ergothioneine→sulfur; uricase purine drop; purine serum responders.
- **Loses confidence (correctly):** every SERS input carries OOD 0.05–0.30 and attenuated confidence; serum spikes of weak Ag adsorbers show low, low-confidence, sometimes wrong themes with explicit flags.
- **Known limitations:**
  1. `nucleic_purine` appears as a frequent *secondary* theme even for non-purines, because c3 (adenine-encoding, high purine weight) captures shared variance — a property of non-orthogonal atlas components, reported not hidden.
  2. `display`/elevation saturates for pure references (small reference MAD → large z); `composition` is the discriminating radar score and is the default.
  3. Solid-substrate adenine arms show substrate-dependent theme competition.
  4. All validation is on SERS (out of domain); an in-domain Raman dose-response is the top missing measurement (rebuild plan E3).

---

## Determinism & provenance checks

- Atlas fingerprint verified identical on every module load.
- Component→theme weights sum to 1 per component (renormalised distribution).
- Every registry field and every theme weight carries its provenance source.
- No randomness in the pipeline (fixed-dictionary NNLS + closed-form maths).

---

## Next authorized step

Rebuild plan **E2 (ontology refinement)**: re-anchor low-purity component labels on reference loadings + perturbation identity and expand per-theme literature — with no change to the frozen atlas. UI integration (E4) and in-domain Raman validation (E3) remain deferred/data-gated.
