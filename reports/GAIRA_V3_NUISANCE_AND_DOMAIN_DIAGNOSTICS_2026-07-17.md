# GAIRA V3 — Nuisance & Domain Diagnostics

**Date:** 2026-07-17 · Source: `tools/validate_global_coordinates.py` (η² of nuisance variables on global coordinates). **Diagnostic only — no batch correction is applied in V3.**

η² = fraction of a global axis's variance explained by a categorical nuisance variable (0 = none, 1 = fully confounded).

---

## Dataset identity (serum_liver / ev_diabetes / ergothioneine / adenine)

| Axis | η² (dataset identity) |
| --- | --- |
| Purine-nuc (G01) | 0.92 |
| Purine-met (G02) | 0.86 |
| Aromatic (G07) | 0.85 |
| Sterol (G09) | 0.65 |
| Lipid (G08) | 0.63 |
| Redox (G10) | 0.61 |
| … | … |
| **Mean across 11 axes** | **0.49 (moderate)** |

## Matrix (serum vs EV, restricted to the two SERS biological sets)

| Axis | η² (matrix) |
| --- | --- |
| Aromatic (G07) | 0.77 |
| Lipid (G08) | 0.63 |
| Purine-met (G02) | 0.55 |
| Sterol (G09) | 0.55 |
| Redox (G10) | 0.53 |
| Purine-nuc (G01) | 0.11 |

## Other nuisance axes
- **Raman vs SERS:** cannot be tested within the current fit population — it is **100% Ag-colloid SERS**. This is itself a limitation: the global space has no Raman coverage.
- **Substrate (Ag colloid vs other):** single substrate in the fit population; not separable.
- **Excitation wavelength / instrument / laboratory:** metadata not uniformly available across the projected sets; not tested. (Ergothioneine is 785 nm cAg; adenine bAgNPs; serum/EV per their sources.)

---

## Plain-language verdict

**Dataset identity is a moderate-to-strong separator of the global coordinates** (mean η² = 0.49; several axes > 0.8). The purine and aromatic axes in particular are strongly associated with which dataset a sample came from, and matrix (serum vs EV) explains most of the aromatic/lipid variance. **The V3 global space does NOT achieve full cross-domain invariance.**

This is expected for a prototype: the coordinates come from a transparent band-evidence heuristic over a single-substrate (Ag-SERS) fit population, with no learned batch correction. The value of V3 is that the coordinates are now **frozen and cohort-invariant** (a sample's position no longer depends on its comparison group) — a prerequisite for later cross-domain work — even though absolute cross-dataset comparability is not yet established.

**The global space is a prototype even though full invariance is not achieved.** Disease clustering in the cross-dataset map must NOT be read as biochemical validation, since dataset identity is a substantial driver. A future release should add Raman-regime references and quantify (not correct) batch structure before claiming universal transfer.
