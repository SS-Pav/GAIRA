# V5 Phase 0 (V5.0) — Canonical registries & data contracts

**Question.** Can we admit a well-governed set of molecular-grounding spectra, each with sufficient provenance and acquisition-domain metadata, before any analysis begins?

**Methods.** Built `src/gaira/data/` (schema contract `SpectrumRecord` + one canonical `loader` + admission gate). Loaded every accessible direct-grounding source into a common form and emitted registries (`results/v5_rebuild/phase0/tables/`).

**Results.**
- **295 observations loaded:** RamanBioLib 202 (Raman), metabolite-63 63 (Ag-SERS 633 nm), adenine 6 (Ag-SERS 785 nm), ORC-Ag 24 (peak-only).
- **271 admitted to joint full-spectrum analysis; 24 excluded** (ORC-Ag peak-only, correctly gated out — no reconstructable spectrum).
- **11 acquisition domains** among admitted spectra:
  - Raman: **9 excitation domains** (457.9/488/514.5/532/632.8/633/785/850/1064 nm) — 202 spectra. The Raman corpus is itself multi-instrument/multi-excitation digitized literature.
  - Ag-SERS colloid 633 nm: 63 (metabolite-63). Ag-SERS colloid 785 nm: 6 (adenine, 1 analyte × 6 conc).
- Every admitted spectrum carries: modality, substrate material+geometry, excitation, matrix, replicate status, wavenumber range, point count, intended_role, and representation/eval-only flags.
- Held-out registries (controlled perturbation, biological, physics/peak evidence) are pointer-linked with enforced roles.

**Interpretation.** The governance gate works and immediately surfaces a structural fact: the grounding corpus is **not one acquisition domain** — it is **11**, dominated by multi-excitation spontaneous Raman (RamanBioLib) plus two Ag-SERS colloid domains. "Raman vs Ag-SERS" is a first cut; excitation is a second axis of heterogeneity even within Raman.

**Limitations.**
- amino_acid_raman_grounding (`aa.xlsx`) has an atypical 2-row layout and is not yet in the canonical loader (needs a dedicated parser); deferred.
- Gobbato pure-metabolite spectra (265 Ag-SERS + 153 Raman powders) are inside a zip and not yet loaded; they are the richest matched Raman↔Ag-SERS source and should be added in the next sprint.
- RamanBioLib is already normalized/digitized; metabolite-63 is averaged/bg-subtracted — different upstream preprocessing states.

**Decision.** Phase 0 gate **passed** for the loaded sources: admitted spectra have sufficient provenance + acquisition-domain metadata. Proceed to Phase 1 comparability using RamanBioLib (Raman) + metabolite-63 (Ag-SERS) + adenine (Ag-SERS), while flagging the multi-excitation heterogeneity and the thin matched-analyte overlap as the central risks to test.

**Next action.** Phase 1 preprocessing + comparability experiment.
