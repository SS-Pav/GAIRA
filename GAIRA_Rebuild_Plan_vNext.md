# GAIRA Rebuild Plan — vNext (V6: the Converged Reasoning Engine)

**Status:** canonical roadmap. **Supersedes `GAIRA_V5_REBUILD_PLAN.md`** (retained as a historical record, not deleted). Branch `gaira-v5-rebuild-plan`. Nothing pushed. The Raman Reference Atlas, its preprocessing, NMF basis and the completed studies are **frozen**; the demo and historical V1–V5 pipelines are **untouched**. This plan governs the additive `src/gaira/engine/` architecture only.

---

## 1. What GAIRA now is

GAIRA is no longer "a Raman classifier" or "a latent-space embedding". It is a **deterministic biochemical reasoning engine** with a fixed scientific spine:

- the **frozen Raman Reference Atlas v0.1** (NMF k=24, fingerprint `09ed804a…`) is the canonical coordinate system;
- the **Component Registry v1** names the 24 latent Raman motifs with full provenance;
- the **Biochemical Ontology v2** converts motifs into weighted biochemical *themes* (11 chemistry themes + background + unknown) (never a single label per component);
- the **Biochemical State Vector (BSV v2)** is the canonical biochemical representation;
- the **Evidence Engine** supplies scientific provenance for every reading;
- the **radar** is one visualization of the BSV, not the model.

Every interpretation is cautious, deterministic, explainable and grounded in spectroscopy. The only model anywhere in the stack is the frozen non-negative NMF basis, applied by non-negative least squares with the dictionary held fixed.

---

## 2. Completed milestones (frozen scientific evidence — do not modify)

| Milestone | Result | Location |
| --- | --- | --- |
| Phase 1 / 1.5 grounding | 785 nm corpus, 51 matched analytes | `results/v5_rebuild/phase1_5/` |
| Phase 2 Stage A | Direct spectra modality-stratified (Outcome B) | `GAIRA_V5_PHASE2_STAGE_A_*` |
| Phase 2 Stage B | Encoders collapse; no shared rep (Outcome B4) | `GAIRA_V5_PHASE2_STAGE_B_*` |
| Stage B0 preprocessing AutoResearch | Preprocessing does not rescue comparability (P4) | `preprocessing_autoresearch/` |
| Matched-analyte spectral audit | Ag-SERS is background-dominated | `spectral_audit/` |
| **Raman Reference Atlas v0.1** | **NMF k=24 frozen; excitation transfer 0.918** | `foundation/` |
| Component Audit | atlas learns molecular CLASS, not species; stable but low-purity components | `reference_atlas_audit/` |
| Serum Spike Projection Validation | concentration registered; identity not (only strong Ag adsorbers) | `spike_validation/` |
| Perturbation Response Audit | loop closes at component IDENTITY, not theme label; purine substructure recovered | `perturbation_response/` |

**Governing evidence carried into V6:** the atlas axes are chemically real but were *labelled* too coarsely; perturbation is a sharper naming tool; component responses are matrix-specific; every SERS input is out of domain for a Raman atlas.

---

## 3. Architecture evolution (V2 → V6)

| | V2 demo (old) | V6 engine (this plan) |
| --- | --- | --- |
| Axes | 11 hand-curated band-window motifs | 13 biochemical **themes** (11 chemistry + background + unknown) derived from evidence |
| Basis | curated band windows | frozen NMF k=24 latent Raman motifs |
| Normalization | cohort means (circular) | frozen Reference Atlas distribution (Part 10) |
| Interpretation | one label per axis | many-to-many weighted component→theme mapping |
| Provenance | limited | every field + every weight traced (Evidence Engine) |
| Validation | none | perturbation-validated (adenine, ergothioneine, uricase) |
| Confidence | implicit | per-theme confidence × OOD × stability |
| Versioning | none | every layer versioned; ontology evolves without touching coordinates |

The goal was explicitly **not to replace the intuitive radar** but to rebuild it on a validated latent representation: V6 keeps V2's interpretability and adds V5's evidence base.

---

## 4. The converged pipeline (implemented — Part 2)

```
input spectrum (wavenumber, intensity, domain)
  → modality-aware preprocessing        [atlas-native ASLS+SG+L2, frozen]
  → projection into the frozen atlas    [NMF k=24, dictionary held fixed, NNLS]
  → 24 latent Raman motif coordinates
  → ontology mapping                    [component_theme_weights_v1]
  → Biochemical State Vector (BSV v2)    [documented equations]
  → domain-aware interpretation         [serum / EV / buffer / tissue / DART]
  → evidence report                     [components → analytes → perturbation → literature]
  → radar backend                       [themes as axes, confidence, OOD, provenance]
```

Implemented in `src/gaira/engine/` (`pipeline.py` orchestrates). See `GAIRA_Engine_Architecture.md` for module/data-flow diagrams.

---

## 5. Remaining implementation phases

- **E1 (done):** Component Registry v1, Ontology v2, theme weights, BSV v2, evidence engine, domain layer, radar backend, reference normalization, versioning, DART interfaces, calibration validation.
- **E2 — ontology refinement (next authorized):** re-anchor low-purity component labels on reference loadings + perturbation identity (the Perturbation Response Audit named c3 etc.); expand literature citations per theme; add heme/redox evidence when data allow. *Requires no atlas change.*
- **E3 — in-domain Raman validation (data-gated):** acquire/locate an in-domain Raman dose-response to separate "atlas cannot track concentration" from "Ag-SERS is the limiting factor" (the single most informative missing measurement). Only then extend confidence claims beyond SERS.
- **E4 — UI integration (deferred):** wire the radar backend + evidence traces into an interactive front-end; do NOT modify the existing demo until E1–E3 are accepted.
- **E5 — multi-modality (future):** Au-SERS and additional Raman sources as new *observation domains* mapping into the same frozen BSV; the atlas coordinate system stays fixed.

---

## 6. Scientific rationale

The V5 studies proved (a) a shared *spectral* space across modalities is not supported, (b) encoders collapse at this corpus scale, and (c) the Raman-only atlas is stable and excitation-transferable. The defensible move is therefore to **freeze the Raman atlas as the canonical coordinate system and build interpretation, not more representation learning, on top of it.** V6 does exactly that: no new learned model, only versioned, evidence-weighted mapping from frozen coordinates to biochemical themes, with honest uncertainty.

---

## 7. Long-term roadmap

1. **V6 engine (this plan)** — frozen atlas + evidence-weighted BSV + radar. *Done at E1.*
2. **Ontology maturation (E2)** — themes gain literature and perturbation anchoring; low-confidence themes (sulfur, heme, redox) upgraded only as evidence arrives.
3. **In-domain validation (E3)** — Raman dose-response closes the concentration-vs-contrast question.
4. **Modality expansion (E5)** — Au-SERS / DART as observation domains into the same BSV.
5. **BSV maturation** — from a single-spectrum representation to a versioned, comparable biochemical state usable for cohort-level ΔBSV work (only after in-domain validation).

---

## 8. DART integration roadmap (interfaces designed, not implemented — Part 14)

A DART electrochemical perturbation yields a time/potential **series** of spectra → a **trajectory of BSVs** through the frozen atlas, the same object type built for chemical dose series. `engine/dart.py` defines the seam: `DARTAcquisition` → `TrajectoryProjector` (reuses the frozen atlas + `BSVBuilder`) → `TrajectoryComparator` (against the reference trajectory library from the Perturbation Response Audit). One falsifiable prediction is carried forward: *redox-cycling a strong Ag adsorber with a known atlas component should move the trajectory along that component and return on the reverse sweep.* No DART code is implemented until real data exist.

---

## 9. Future foundation-model roadmap

A learned encoder remains **out of scope** until the corpus grows and diversifies (Stage B rejected encoders at this scale; the Component Audit and this engine show a parts-based, interpretable representation is currently the defensible choice). Any future foundation model must (a) beat the frozen NMF atlas on held-out excitation/source transfer, (b) remain projectable into the same BSV theme space, and (c) preserve provenance. Until then the frozen NMF atlas is canonical.

---

## 10. Non-negotiable constraints (all V6 work)

Do not modify the frozen atlas, its NMF, its preprocessing, or the completed studies. Do not touch the demo or historical pipelines. Additive implementation only. Deterministic; provenance on every field; no opaque ML; conservative over optimistic; evidence over performance. Do not push.
