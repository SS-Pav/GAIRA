# GAIRA Foundation Model — Scientific Audit

**"What exactly is the frozen biochemical coordinate system that powers GAIRA?"**

A complete, reproduce-from-first-principles audit of the frozen Raman foundation model:
every dataset verified, every spectrum and analyte counted, every preprocessing step
documented, the NMF rebuilt from scratch, every latent component explained, and every
biological interpretation traced back to reference Raman spectra. Nothing was assumed;
nothing inherited blindly. The frozen atlas was **never modified** — reproducing its
fingerprint is one of the results.

Atlas under audit: **NMF k=24**, fingerprint `09ed804a40836f4a05a91ba10900cded`,
375 pure-Raman spectra / 167 analyte labels, 450–1800 cm⁻¹ @ 2 cm⁻¹ (676 bins).

---

## Headline results

- **Byte-for-byte reproducible.** Rebuilding NMF k=24 from the raw Raman corpus reproduces
  the frozen components with max abs difference `0.0` → identical fingerprint. The full
  5-representation × 6-k benchmark reproduces to `1.1e-16`.
- **Raman-only, by construction and by proof.** The representation is fit on three pure-Raman
  sources; SERS is excluded by code assertion and used only for validation.
- **NMF k=24 is re-derived as optimal.** Raw benchmark winner is signed ICA k=32; the
  pre-stated non-negativity (parts-based) constraint selects NMF k=24 — quantitatively
  justified, not defaulted.
- **Stable, non-redundant basis.** Component bootstrap stability 0.65–0.97 (mean 0.81);
  max pairwise basis cosine 0.52 — no duplicate axes.
- **Validates on data it never trained on.** Adenine→purine (ρ=0.996, Langmuir K=0.89 µM),
  ergothioneine→sulfur (ρ=0.93), uricase depletion localises to the oxopurine motif
  (Δ=−0.06); SERS flagged out-of-domain (OOD 3.5× Raman); strong Ag adsorbers recovered,
  weak ones honestly not.

---

## The reports (Parts 1–11)

| Part | Report | What it answers |
|--:|---|---|
| 1 | [GROUNDING_AUDIT.md](reports/GROUNDING_AUDIT.md) | Every data source; training vs validation vs unused |
| 2 | [FOUNDATION_CORPUS_REPORT.md](reports/FOUNDATION_CORPUS_REPORT.md) | The Raman-only corpus; chemical-class balance; gaps; data-quality |
| 3 | [PREPROCESSING_AUDIT.md](reports/PREPROCESSING_AUDIT.md) | Every preprocessing step, audited and justified |
| 4 | [NMF_REBUILD.md](reports/NMF_REBUILD.md) | Recompute the representation; re-determine optimal k |
| 5 | [NMF_EXPLAINED.md](reports/NMF_EXPLAINED.md) | V≈WH mathematically & physically; why NMF over PCA/ICA/AE |
| 6 | [COMPONENT_AUDIT.md](reports/COMPONENT_AUDIT.md) + [components/](components/) | One page per component + global classification |
| 7 | [MSS_AUDIT.md](reports/MSS_AUDIT.md) | How Molecular Spectral Signatures are generated |
| 8 | [BSV_AUDIT.md](reports/BSV_AUDIT.md) | Component→theme→11-axis BSV; radar semantics + limits |
| 9–10 | [VALIDATION_SUMMARY.md](reports/VALIDATION_SUMMARY.md) | Six validation datasets, in order |
| 11 | [FINAL_ASSESSMENT.md](reports/FINAL_ASSESSMENT.md) | Critical scientific appraisal |

Per-component pages: `components/component_c00.md … c23.md`.
Figures: `figures/`. Tables / machine-readable outputs: `tables/`.

---

## Reproduce (deterministic, seed 0)

```bash
# corpus composition + chemical classes + data-quality  (Part 2)
python results/v5_rebuild/foundation_audit/code/corpus_analysis.py
# preprocessing stages + negative-clip quantification    (Part 3)
python results/v5_rebuild/foundation_audit/code/preprocessing_demo.py
# rebuild NMF + reproduce benchmark + k-selection        (Part 4, ~4 min)
python results/v5_rebuild/foundation_audit/code/repro_benchmark.py
python results/v5_rebuild/foundation_audit/code/nmf_selection_fig.py
# per-component audit (24 pages + basis figures)         (Part 6)
python results/v5_rebuild/foundation_audit/code/component_audit.py
# six validation datasets through the frozen atlas        (Parts 9–10)
python results/v5_rebuild/foundation_audit/code/run_validation.py
```

The atlas itself is built by `results/v5_rebuild/foundation/code/{run_c1_benchmark.py,
run_c2_c7.py}` and frozen to `results/v5_rebuild/foundation/artifacts/`. This audit reads
those frozen artifacts and never writes to them.

---

## Scope note

This audit covers the **frozen Raman representation** and its interpretive stack
(components → MSS → BSV) and validation. It does **not** modify the atlas, the engine, or
the demo. Recommended future changes (all versioned, none silent) are listed in
FINAL_ASSESSMENT §"What I would change" — chiefly: document the amino-acid sheet's
provenance, fix ~6 canonicalization duplicates, add pure porphyrin/flavin Raman, and build
the Raman→SERS observation model as a separate layer.
