# NMF_REBUILD
### Recomputing the representation from scratch — is NMF k=24 still the answer?

*Part 4 of the GAIRA Foundation Model audit. The instruction was explicit: do not assume
previous outputs; recompute the representation, redetermine the optimal component number
with the same benchmark methodology, and — if 24 remains optimal — justify it
quantitatively. Reproduced by `foundation_audit/code/repro_benchmark.py` and
`nmf_selection_fig.py`. Nothing here re-freezes or alters the deployed atlas.*

---

## 0. The two headline results

1. **The frozen atlas is byte-for-byte reproducible.** Rebuilding NMF k=24 from the raw
   Raman corpus (seed 0, `init="nndsvda"`, `max_iter=1500`) yields components **identical
   to the frozen `manifold_components.npz`** (max abs diff `0.0`), so the SHA-256
   fingerprint reproduces exactly:
   `09ed804a40836f4a05a91ba10900cded`. Explained variance 0.712, matching the build log.

2. **The full 30-cell benchmark reproduces to floating-point identity.** Re-running all 5
   representations × 6 latent sizes on the analyte-grouped protocol reproduces the
   committed benchmark with **max abs difference 1.1 × 10⁻¹⁶** and an **identical full
   ranking**. The selection is deterministic and re-derives NMF k=24.

The frozen biochemical coordinate system is therefore not a lucky artifact of one run; it
is a deterministic function of the Raman corpus + a documented, seeded pipeline.

---

## 1. The benchmark methodology (unchanged, re-run)

Every candidate representation is scored on the SAME analyte-grouped, held-out protocol
(`GroupKFold` by analyte → no analyte appears in both train and test), across
k ∈ {4, 8, 12, 16, 24, 32}, seed 0. Six criteria, deliberately **not** reconstruction-led:

| Criterion | Weight | What it protects |
|---|---:|---|
| Neighbourhood preservation | 0.25 | analyte structure survives into the latent space |
| Replicate robustness | 0.25 | replicates of one analyte stay together |
| Component stability | 0.20 | components reproduce under analyte bootstrap |
| Interpretability (sparsity + band localisation) | 0.15 | loadings look like spectra, not noise |
| Reconstruction | 0.10 | fidelity — intentionally NOT dominant |
| Nuisance control | 0.05 | excitation/source not encoded |

Reconstruction is only 10 % on purpose: a coordinate system for *biological
interpretation* must preserve chemical structure and be readable, not merely compress.

---

## 2. What the benchmark says (reproduced numbers)

Best cell per representation (full table:
`tables/c1_representation_benchmark_repro.csv`; figure
`figures/nmf_selection_score_vs_k.png`):

| Rank | Representation | k | Score | recon err | stability | sparsity | non-neg? |
|---:|---|---:|---:|---:|---:|---:|:--:|
| 1 (raw) | **ICA** | 32 | **0.7866** | 0.486 | 0.773 | 0.503 | ✗ |
| 2 | **NMF** | 24 | **0.7864** | 0.550 | **0.812** | **0.819** | ✓ |
| 3 | ICA | 24 | 0.775 | 0.530 | 0.758 | 0.495 | ✗ |
| 4 | ICA | 16 | 0.774 | 0.579 | 0.728 | 0.498 | ✗ |
| 5 | NMF | 32 | 0.767 | 0.523 | 0.805 | 0.846 | ✓ |
| — | PCA (best) | 24 | 0.739 | 0.530 | 0.532 | 0.475 | ✗ |
| — | Autoencoder (best) | 32 | 0.581 | 0.481 | 0.162 | 0.382 | ✗ |
| — | SparseDict (best) | 24 | 0.429 | 0.832 | 0.796 | 0.620 | ✗ |

**The top 5 cells sit within 0.02 of each other — a statistical tie.** ICA k=32 edges
NMF k=24 by 0.0002 (0.03 %), which is far inside the noise of a 4-fold CV estimate.

---

## 3. Breaking the tie — the pre-stated, physical rule

The tie-break is not chosen after seeing the winner; it is fixed in code
(`benchmark.select_with_tiebreak`) and motivated by the scientific objective:

> **A Raman spectrum of a mixture is a non-negative sum of its molecular components.**
> A biochemical *coordinate* is only meaningful if it reads as "how much of this theme is
> present" — which requires a **parts-based, non-negative** decomposition. A signed
> component (ICA/PCA) would imply *negative* biochemical content and cannot be a
> proportion.

So within the tie band the admissible set is restricted to non-negative decompositions —
**only NMF qualifies** — then ranked by score, then by smaller k. → **NMF k=24**.

Why NMF wins on the substance, not just the constraint (figure
`figures/nmf_selection_criteria.png`):

- **Component stability 0.812** — the highest of any representation (ICA 0.773, PCA
  0.532, Autoencoder 0.156). Its parts reproduce under resampling.
- **Loading sparsity 0.819 / band localisation 0.054 (best)** — its components look like
  *spectra with a few bands*, which is what makes the downstream MSS/BSV layers
  interpretable. ICA (0.503) and PCA (0.475) loadings are delocalised.
- **Near-zero nuisance leakage** (excitation 0.019, source 0.063) — it does not encode
  which instrument/excitation produced a spectrum.

ICA's advantages (better reconstruction 0.486; higher replicate robustness 0.834) are
real but serve goals we deliberately down-weight, and come with signed, delocalised,
un-interpretable components. PCA is a distant 3rd; the autoencoder is 4th and
catastrophically unstable (0.156 — a different local optimum every run); sparse dictionary
learning reconstructs worst.

---

## 4. Why k = 24 specifically

- **Intrinsic dimensionality of the corpus** (from the fitted latent covariance):
  participation ratio **15.2**, 90 % of latent variance in **16** components,
  entropy-effective rank ≈ 16. The true dimensionality is ~15–16.
- k=24 sits deliberately **above** that (≈1.5×). This is intentional over-completeness:
  extra components let chemically distinct-but-correlated motifs separate (e.g. purine vs
  pyrimidine, acyl vs sterol) instead of being forced to share an axis, at the cost of a
  few low-variance / redundant components (catalogued in Part 6). Within the NMF tie
  (k=24 at 0.786 vs k=32 at 0.767), the smaller k=24 is preferred by the simplicity rule
  and scores higher.
- Explained variance at k=24 is **0.712** — modest by design: L2-normalised,
  baseline-removed Raman shape is high-rank, and the benchmark does not chase
  reconstruction.

---

## 5. Verdict

**k=24 remains optimal under the same methodology — retained, and quantitatively
justified.** The recomputation changes nothing: the raw benchmark winner is a signed ICA
that the pre-stated non-negativity constraint correctly excludes from serving as a
*biochemical proportion* coordinate; among admissible parts-based decompositions NMF k=24
is the best and reproduces bit-for-bit. The representation is deterministic, stable, and
interpretable. **No change is made or recommended to the frozen atlas.**

Residual honesty: the ICA-vs-NMF gap being 0.0002 means the *score* does not by itself
prefer NMF — the **physics** (non-negativity) does. That dependence is explicit and
defensible, and is the single most important design decision in the model. Reproduce:

```
python results/v5_rebuild/foundation_audit/code/repro_benchmark.py   # ~4 min, seed 0
python results/v5_rebuild/foundation_audit/code/nmf_selection_fig.py
```
