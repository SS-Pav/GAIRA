# GAIRA Raman Reference Atlas v0.1 — Perturbation Response Audit

**Mechanistic audit linking controlled biochemical perturbations to latent Raman motifs.** The atlas (NMF k=24) was frozen throughout: fingerprint `09ed804a40836f4a05a91ba10900cded` verified byte-identical before and after. No retraining, no reprocessing, no ontology change. Branch `gaira-v5-rebuild-plan`; nothing pushed. All projections are the frozen-atlas coordinates already computed in the Spike Validation study.

> **Principal finding: the loop closes at the level of component IDENTITY, not theme label.** When adenine is perturbed, the single strongest-responding latent component is the one that actually encodes adenine in the reference atlas (**c3**) — reproducibly in **4 of 6** substrate×laser arms — even though the Component Audit's coarse label for c3 is *"sterol"*. Perturbation is a sharper probe of what a component *is* than the static purity-based labels, and it recovers a purine sub-classification the atlas was never given.

---

## Reproduce

```bash
cd results/v5_rebuild/perturbation_response/code
python run_response_audit.py     # Parts 1-14  (~1 s; reads cached projections)
python run_mechanistic.py        # Parts 15-16
python make_report.py            # figures + 62-page PDF
```

---

## Vocabulary discipline

A **component** is a mathematical latent Raman motif (an NMF basis vector). Its **theme** is a tentative post-hoc label from the Component Audit. A **response** is the measured change in component activation under perturbation. A response matching a theme *corroborates* it; a mismatch is *reported*, never hidden. No molecular assignment is claimed anywhere.

---

## Observed results

**1. The response is selective, not a global amplitude shift (Part 1).** In the strongest adenine arm (cAg@785) only **7 of 24** components rise while 17 fall. The two largest movers are **c3** and **c13**; c3's single top reference analyte is adenine itself. Dose-response is saturating (Langmuir) in every arm, matching the Spike Validation.

**2. Loop closure at component identity (Parts 3, 15).** For adenine the strongest-responding component is its own encoding component **c3 in 4 of 6** substrate×laser arms — closure that reproduces across substrates and lasers. Across all responsive-component instances the analyte loads its responding component in **7/128**, concentrated in the strongest responders (theme-label agreement is only 15/128, see point 3).

**3. Theme labels are the weak link, not the components (Parts 3, 15).** Low-purity components carry misleading coarse labels: **c3** ("sterol", purity 0.22) actually encodes adenine — its bands 722/1334/1486 cm⁻¹ are purine ring/imidazole modes; **c13** ("pyrimidine") is thymine-dominated. The components are chemically real; only the labels are unreliable.

**4. Purine sub-structure recovered (Part 6).** Response fingerprints split the five purines into two anti-correlated pairs:

| | adenine | hypoxanthine | xanthine | guanine | urate |
| --- | --- | --- | --- | --- | --- |
| adenine | 1.00 | **0.88** | −0.31 | −0.05 | 0.19 |
| hypoxanthine | | 1.00 | −0.21 | 0.00 | −0.13 |
| xanthine | | | 1.00 | **0.90** | −0.22 |
| guanine | | | | 1.00 | −0.26 |

{adenine, hypoxanthine} (6-amino/6-oxo purines) versus {xanthine, guanine} (2,6-dioxo / 2-amino-6-oxo) — a chemically correct sub-classification finer than the single "purine" ontology axis, recovered without the atlas ever being told about it.

**5. Uricase depletion is selective (Part 8).** Enzymatic urate removal (spiked+uricase − spiked) produces a targeted change in a purine-encoding component (**c15 Δ −0.106**, > 2× the median absolute change) rather than a global shift — mechanistically consistent with removing a purine.

**6. Response fingerprints recover chemistry better than raw spectra (Part 10).** Clustering the 53 serum-spike fingerprints recovers chemical family at ARI **0.146** versus **0.078** for clustering the raw spike spectra — the response representation exposes chemical structure that the background-dominated Ag-SERS spectra obscure. Both are modest.

**7. Ergothioneine (Part 7).** Responsive components are dominated by c15/c17/c19; its own encoding component c19 responds but is *not* the strongest, and c19 is a low-purity generic component (top loaders creatinine/tubulin), so its loop closure is **weak** — reported as such rather than forced.

---

## What the data do NOT support

**8. Matrix-invariant signatures (Part 5).** Pure-analyte and serum-spike fingerprints for the same analyte agree only at **median cosine −0.012 ≈ 0**. A component response is **matrix-specific** and must not be reused across matrices as a fixed signature.

**9. Identity recovery for weak adsorbers (Part 9).** Serum responders move *more* (activation norm 0.089 vs 0.029) but not more *focused* (response entropy 0.86 vs 0.85). Most non-purine spikes produce weak, non-specific activation — consistent with prior GAIRA findings that these Ag-SERS spectra are background-dominated.

---

## Component specificity and robustness (Parts 4, 11)

- **Specificity:** components range from specific (high activation Gini, ≤2 activating classes) to generic hubs. The most generic components (high activation across many families) are the background-coupled ones.
- **Anchor candidates for a future BSV** (mathematically stable + meaningfully responsive + high-confidence theme): **c2, c1, c11, c18, c16, c17**. These combine audit stability with perturbation responsiveness and are the defensible axes to anchor BSV interpretation on.

---

## Bipartite network (Part 14)

Component hubs (activated by the most analytes) and analyte hubs (activating the most components) are in `part14_hubs.json`. Hub components correspond to the generic, background-coupled motifs; specific components (e.g. c3 for purines) sit at the periphery with few, chemically-coherent edges — the expected topology if the atlas mixes a few analyte-specific axes with several shared-background axes.

---

## Mechanistic assessment (Part 15)

**Has the loop closed?** *Partially, and instructively.* Perturbation confirms that the atlas axes are **chemically real**, that a perturbed analyte drives its own encoding axis (strongly for adenine, weakly for ergothioneine), and that the atlas encodes chemistry it was never explicitly taught (purine subclasses; selective depletion). It also shows the **theme labels were assigned too coarsely** by a static purity metric — perturbation is the sharper naming tool.

- **Ontology entries that gain confidence:** the components an analyte both drives and encodes (c3 for adenine).
- **Ontology labels that lose confidence:** low-purity components whose coarse label contradicts their perturbation identity (c3 "sterol" → adenine-encoding).
- **Components needing reinterpretation:** re-anchor low-purity labels on reference loadings and perturbation identity, not on the dominant chemical family.

---

## Implications for DART (Part 16, grounded)

A DART electrochemical perturbation produces a time/potential series that, projected into the atlas, is a **trajectory** of the same type built here for chemical dose series. This study provides:
- a **trajectory-fingerprint schema** (path length, straightness, curvature, component turnover, response entropy, OOD evolution) already computed for 7 chemical dose trajectories — the first reference library;
- a concrete **falsifiable prediction**: redox-cycling a single strong Ag adsorber with a known atlas component should move the projected trajectory predominantly along that component and return toward baseline on the reverse sweep.

**Grounded limitations:** everything here is Ag/Au-SERS out of domain; component responses are matrix-specific (consistency ≈ 0), so buffer fingerprints cannot be assumed to transfer; and DART interpretation must anchor on component *identity*, not theme labels. **This study does not demonstrate DART compatibility** — it supplies the trajectory vocabulary and one testable prediction.

---

## Limitations

- All datasets are Ag/Au-SERS projected into a Raman atlas (out of domain); nothing validates in-domain Raman response.
- Pure-vs-serum consistency ≈ 0: fingerprints are matrix-specific.
- Only two chemical dose series (adenine, ergothioneine) have per-level concentration data; the serum spikes are single-concentration, so most "response fingerprints" are single-point contrasts against serum, not dose trajectories.
- The strongest loop closure (adenine → c3) rests on one analyte measured many ways; a second analyte with a high-purity encoding component and a real dose series is needed to generalise.

---

## Outputs

`GAIRA_Raman_Reference_Atlas_v0.1_Perturbation_Response_Audit.pdf` (62 pages: exec summary, methodology, 4 overview figures, **one response-atlas page per analyte ×53**, mechanistic + DART). Tables: `part1_component_dose_response`, `part2_response_fingerprints`, `part3_theme_match`, `part4_component_specificity`, `part5_analyte_consistency`, `part6_purine_*`, `part7_ergothioneine`, `part8_uricase`, `part9_serum_responders`, `part10_response_families` + `part10_clusters`, `part11_component_robustness`, `part14_bipartite_edges` + `part14_hubs`, `part15_mechanistic_assessment`, `part16_dart_implications`. Artifacts: `trajectory_library.json`, `fingerprint_linkage.npz`, `response_audit_manifest.json` (fingerprint verification).
