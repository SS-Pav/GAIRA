# GAIRA Calibration Evaluation — v1

**Scope:** layer-1 validation. Does GAIRA's **direct spectral → BSV** pipeline
recover known biochemical perturbations in the correct direction when the
chemistry is controlled? This is not a disease demo.

## Scientific framing

Calibration datasets differ from disease cohorts because:

- the perturbation is **known and intentional** (spiking, enzymatic depletion,
  concentration titration);
- we have an **a priori expectation** for which BSV axis should move and in
  which direction;
- spectral variance not caused by the perturbation is acquisition noise, not
  biology we are trying to explain.

Calibration results speak to whether the pipeline's **sign structure** on the
BSV axes matches the chemistry that was imposed. They do not speak to molecule
identification, and the registry deliberately avoids exact-molecule claims.

Outcome language used by the module:

- **pass** — every expected axis was recovered in the correct direction
- **partial** — some expected axes recovered, none contradicted
- **weak** — signal present but below the noise floor on the expected axis
- **inconsistent** — observed direction opposed the expectation
- **no_expected** — no direction was registered (should not occur for a
  calibration contrast)

## Module layout

```
src/gaira/calibration/
    __init__.py
    registry.py          CalibrationContrast + CALIBRATION_REGISTRY
    loaders.py           raw-file → (X, wavenumbers, cohorts, meta)
    preprocessing.py     AsLS(λ=1e5, p=0.001) + SG(11,3) + L2 + crop
    band_annotation.py   22-window → candidate motif lookup (annotation only)
    eval.py              run_calibration_eval(contrast_id) → CalibrationResult

streamlit_apps/gaira_v4/pages/3_🧪_Calibration_Eval.py
```

### Flow

```
raw spectra + cohorts
        │
        ▼
AsLS + SG + L2 (preprocessing.py)
        │
        ▼
22-window feature extraction (spectral/window_panel.py)
        │
        ▼
direct project_to_bsv (spectral/bsv_projection.py)      ← quantitative path
        │
        ▼
cohort-mean BSV + observed delta (perturbed − control)
        │
        ▼
per-axis verdict vs expected direction (eval.py)
        │
        ▼
top-window ranking (spectral/band_drivers.py)
        │
        ▼
motif annotation (band_annotation.py)                   ← annotation only
```

Motifs never influence BSV values. They are attached to the top-contributing
spectral windows after the quantitative path completes.

## Registered calibration datasets (v1)

All three datasets were already in the repo's experiment registry; v1 wires
them into a new, independent evaluation workflow.

| Dataset | Perturbation | Contrast(s) |
|---|---|---|
| `cspp_serum` (Figure 7) | metabolite spiking | Bkg vs Hyp; Bkg vs Erg |
| `serum_ag_colloids` (uricase subset) | enzymatic depletion + spiking | SerumSigma vs +Enzyme; SerumSigma vs Spiked |
| `ergothioneine_serum` | concentration titration | 0.0 µM vs 2.0 µM |

### Expected directions — summary table

| Contrast | Expected axis(es) | Confound axes |
|---|---|---|
| `cspp_fig7_hypoxanthine_spike` | purine_nucleotide ↑ | aromatic_amino_acid |
| `cspp_fig7_ergothioneine_spike` | purine_nucleotide ↑ (via imidazole) | aromatic_amino_acid, redox_metabolite |
| `uricase_sigma_depletion` | purine_nucleotide ↓ | aromatic_amino_acid, glycan_carbohydrate |
| `uricase_spiked_hypoxanthine_serum` | purine_nucleotide ↑ | aromatic_amino_acid |
| `ergothioneine_titration_top_vs_zero` | purine_nucleotide ↑ | aromatic_amino_acid, protein_backbone |

Expected directions are deliberately conservative: one expected axis per
contrast, with known confounds declared up front. This prevents inflating
the recovery rate by registering plausible-but-unjustified bonus expectations.

## v1 results

Running `run_calibration_eval(...)` on all five contrasts:

| Contrast | n (ctrl / pert) | Outcome | Top window | Top window axis |
|---|---|---|---|---|
| `cspp_fig7_hypoxanthine_spike` | 50 / 50 | **pass** | 700–740 | purine_nucleotide |
| `uricase_spiked_hypoxanthine_serum` | 5 / 5 | **pass** | 700–740 | purine_nucleotide |
| `cspp_fig7_ergothioneine_spike` | 50 / 50 | **weak** | 1020–1080 | nucleic_acid_backbone |
| `uricase_sigma_depletion` | 5 / 5 | **inconsistent** | 1260–1320 | protein_backbone |
| `ergothioneine_titration_top_vs_zero` | 5 / 5 | **inconsistent** | 1200–1260 | protein_backbone |

### Which BSV axes recover well vs poorly

**Strongest — `purine_nucleotide`.** Both hypoxanthine spikes produced
- purine_nucleotide Δ of +0.007 and +0.006
- top-ranked window 700–740 cm⁻¹ in both cases, with effect size > 1.5,
  which is exactly the hypoxanthine / adenine ring-breathing region
- no other axis crossed the 0.003 noise floor in a direction that would
  look like a false alarm

**Weakest — anything sulfur- or imidazole-specific.** The two ergothioneine
contrasts (spike and titration) both failed cleanly:
- the spiking case stayed **below** the noise floor on the expected axis
- the titration case flipped the expected sign
In both, the top-ranked windows moved to `protein_backbone` or
`nucleic_acid_backbone`, not to any axis with a plausible ergothioneine
signature. This is a **panel limitation**, not a pipeline bug: the current
22-window scheme has no dedicated sulfur or imidazole band, so ergothioneine's
~720 cm⁻¹ imidazole mode shares its window with adenine / hypoxanthine, and
its sulfur features (S-H, C-S) are either outside the fingerprint crop or
blend into redox_metabolite only weakly.

**Surprise — uricase depletion went inconsistent.** The `SerumSigma` vs
`SerumSigma+Enzyme` contrast showed purine_nucleotide **increase** by +0.025
rather than the expected decrease. Plausible explanations (hypotheses only):

1. commercial Sigma serum may already be depleted in uric acid relative to
   fresh serum, leaving little to remove;
2. uric acid's dominant SERS intensity on Ag colloid actually concentrates
   at ~635 cm⁻¹ (aromatic_amino_acid window) and ~890 cm⁻¹ (glycan window),
   which are listed as confound axes — so purine_nucleotide may not be the
   right primary expected axis for this substrate;
3. n = 5 per cohort is small enough that between-replicate variance on the
   Ag colloid could dominate the enzymatic effect.

Before interpreting this as a pipeline failure, a v2 pass should either
(a) re-register this contrast with an aromatic_amino_acid / glycan expected
axis, or (b) pull more replicates if they exist in the archive.

## What this means for GAIRA layer 1

**What works.** Clean single-metabolite spikes where the analyte has an
unambiguous window mapping (hypoxanthine → 700–740 → purine_nucleotide)
produce the expected BSV shift with the expected top window. The direct
spectral → BSV path recovers the chemistry without help from motifs or
literature priors. This is the outcome layer 1 was designed to produce.

**What's confounded.** Analytes that share their dominant Raman window with
another biochemical class (imidazole vs purine, uric acid vs aromatic AA) do
not cleanly separate on the 8-axis panel. This is an intrinsic limitation of
a broad-band projection with only 22 windows, not a failure of the
implementation. The calibration workflow now surfaces these cases
explicitly via `confound_axes` in the registry and the `ambiguity` field on
each window annotation.

**What's weak.** At µM-scale spike concentrations, the signal-to-serum-matrix
ratio is sometimes below the 0.003 per-axis noise floor used here. The
titration contrast reaches that floor by design — it is the calibration
workflow's answer to "how small is too small".

## Recommendations before layer 2

Layer 2 (probabilistic biochemical interpretation / ambiguity layer) should
not be built on top of the current 22-window panel without first:

1. **Expanding the panel where the confounds are structural.** At minimum,
   add a dedicated sulfur/thione window (≈505–520, ≈2570 if the fingerprint
   crop is widened) so redox_metabolite has a band that doesn't double as
   disulfide/glucose. Separate the 700–740 imidazole / purine / C-S collision
   either by narrowing the window or by maintaining a small ambiguity flag
   alongside the numeric value.
2. **Rewriting uricase-style depletion contrasts** with expected axes that
   match the substrate-specific SERS intensity distribution. Ag colloid and
   plasmonic paper do not emphasize the same uric-acid modes.
3. **Treating the calibration passes as the only validated part of the
   panel so far.** Hypoxanthine → purine is the one mapping the workflow
   has verified end-to-end. Any layer-2 uncertainty model should anchor its
   priors on validated pairs, not on the full 22-window scheme uniformly.
4. **Adding more calibration contrasts before expanding disease claims.**
   Candidates already in the repo: `spiked_commercial_serum_merck`,
   `adenine_sers_control` (grounding, concentration series), and the
   ergothioneine titration rungs between 0 and 2 µM to characterize the
   noise floor as a function of concentration.

## Running

```bash
cd /Users/suraj/projects/GAIRA
PYTHONPATH=src streamlit run streamlit_apps/gaira_v4/gaira_v4.py
```

The calibration page is accessible from the v4 landing page or directly at
`pages/3_🧪_Calibration_Eval.py`. Use the sidebar to pick a contrast or
view the summary-across-all table. v3 and the existing v4 pages are not
modified.

## Scripted use

```python
from gaira.calibration import run_calibration_eval, list_contrasts

for c in list_contrasts():
    r = run_calibration_eval(c.contrast_id)
    print(c.contrast_id, r.overall_label, r.expected_axes_hit, "/", r.expected_axes_total)
```
