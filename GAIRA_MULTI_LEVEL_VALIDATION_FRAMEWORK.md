# GAIRA Multi-Level Validation Framework

*How GAIRA validates cross-modal and cross-matrix transfer without collapsing distinct
questions into one number. The frozen atlas (`09ed804a40836f4a05a91ba10900cded`) is the fixed
reference; every dataset below is **projected** through it, never used to fit it.*

---

## The problem this framework solves

It is tempting to summarise "how well does a Raman-trained model handle SERS / serum / a
perturbation?" with a single similarity score. That score then gets called *recoverability*,
and it silently conflates four different questions that have four different answers. GAIRA
separates them explicitly.

## The four levels

| Level | Name | Question | Primary metric | Where it lives |
|--:|---|---|---|---|
| **1** | Latent fingerprint preservation | Do the 24 NMF coordinates line up? | `C_component = cos(z_R, z_S)` | pure-Ag-SERS stage |
| **2** | Biochemical theme preservation | Does the broad interpretation survive? | dominant-match; **distinctive** theme cosine vs null; MSS cosine | theme-preservation module |
| **3** | Perturbation sensitivity | Would a controlled change still register? | dose ρ / saturating K; directional motif Δ | adenine, ergothioneine, uricase |
| **4** | Matrix recoverability | Does it survive serum competition? | serum spike displacement + direction | serum spike-in stage |

**These are not tiers of one quantity.** They are correlated (adsorption physics drives all
four) but not interchangeable. The canonical illustration is **adenine**: *weak* at Level 1
(component cosine 0.36), *meaningful* at Level 2 (purine theme dominant, rank #1 preserved),
*strong* at Level 3 (dose ρ = 0.996, Langmuir K = 0.89 µM), *strong* at Level 4 (serum). A
single cosine would have called adenine a failure; the framework shows it is one of the
best-validated analytes GAIRA has.

---

## Level 1 — Latent fingerprint preservation

The similarity of the low-level latent composition — the 24 frozen NMF coordinates. Dominated
by adsorption physics. Median across 51 pure analytes: **0.42**. This is the original
pure-Ag-SERS "coordinate cosine," preserved verbatim; the framework renames it to say exactly
what it measures. Descriptive tiers (Excellent/Good/Moderate/Weak/Poor) are table-reading
thresholds, not learned classes.

## Level 2 — Biochemical theme preservation

The interpretation layer, measured three ways because **no single number is honest alone**:

- **Dominant-theme match** (argmax of the 11-theme composition): preserved for **35%**. But on
  Ag-SERS 50/51 analytes become `nucleic_purine`-dominant (the **purine attractor**), so this
  number mostly reflects which analytes were already purine-dominant in Raman.
- **Raw theme cosine** (median **0.92**): **inflated by a shared compositional baseline** —
  any two analytes sit near 0.9 before preservation is considered. **Never report it alone.**
- **Distinctive theme cosine + null** (baseline-subtracted, self-referenced against every other
  analyte): median **0.11**, separation **+0.014**, self-rank median **25 ≈ chance (26)**. The
  honest signal: identity-specific theme preservation is **real but selective**, strong only for
  oxopurines and a few chemisorbers.
- **MSS motif cosine** (median **0.74**) sits between latent and theme — mid-level structure is
  more robust than exact coordinates but still adsorption-sensitive.

**Rule:** a theme-preservation claim must always carry its null. Raw theme cosine is a
baseline, not a preservation score.

## Level 3 — Perturbation sensitivity

Does a *controlled change* in the analyte move the correct theme in the correct direction?
This is the operationally strongest form of preservation, but it can only be measured where a
perturbation series exists. In GAIRA that is **exactly three analytes**:

- **Adenine** — concentration dose-response; purine theme monotonic (ρ = 0.996), saturating
  (K = 0.89 µM, R² = 0.993).
- **Ergothioneine** — concentration dose-response; sulfur theme monotonic (ρ = 0.927),
  saturating (K = 1.52 µM).
- **Uricase** — **directional** urate depletion (not a dose series): the oxopurine-carbonyl
  motif drops sharply (Δ = −0.060), localising the perturbation at the MSS layer where the
  coarse theme radar smears it.

**Rule:** never fabricate a perturbation score. Every other analyte is *Not tested*.

## Level 4 — Matrix recoverability

Does the analyte survive real serum competition on Ag colloid? From the serum spike-in stage:
**9 strong / 24 moderate / 18 weak** of 51. The strong set is the same oxopurines + adenine +
ergothioneine + creatinine that pass Levels 1–3. Serum adds competition on top of the modality
gap, so Level 4 is a stricter subset of Level 1, not an independent axis.

**Rule:** serum data validate *matrix* recoverability, not pure-analyte theme preservation;
keep the two separate.

---

## Reading the ladder

```
Reference Raman → Level 1 (modality gap, no matrix) → Level 2 (interpretation survival)
                → Level 3 (controlled perturbation)  → Level 4 (serum competition)
                → biological cohorts
```

Each rung isolates one failure mode. The recurring lesson across all four: **the dividing line
is adsorption, not the Raman representation.** The oxopurines transfer, dose-respond, deplete
on cue, and survive serum; weak physisorbers fail everywhere. GAIRA flags the gap honestly
(OOD 0.05 → 0.16) rather than hiding it, and the failures point at a future **observation
model** (surface/instrument physics), not at the frozen atlas.

## Where to look

- Level 1 · `results/v5_rebuild/foundation_audit/reports/PURE_AG_SERS_VALIDATION.md`
- Level 2 · `results/v5_rebuild/pure_ag_sers_theme_preservation/` (spec, report, assessment, figures, cards)
- Levels 3–4 · `results/v5_rebuild/foundation_audit/reports/VALIDATION_SUMMARY.md` +
  `pure_ag_sers_theme_preservation/tables/{perturbation_sensitivity,matrix_recoverability_linkage}.csv`
- Interactive · **Foundation Explorer V2** (`gaira_foundation_explorer_v2/`) — the
  Cross-Modal Validation, Perturbation, and Matrix Recoverability pages walk this framework.

Nothing in this framework refits, retrains, or modifies any frozen asset.
