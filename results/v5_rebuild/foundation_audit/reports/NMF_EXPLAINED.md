# NMF_EXPLAINED
### What the factorization is, mathematically and physically

*Part 5 of the GAIRA Foundation Model audit. A self-contained explanation of the frozen
decomposition: the input matrix, W and H, the equation, why Raman mixtures suit NMF, and
why NMF was chosen over PCA, ICA, sparse coding, dictionary learning and autoencoders —
with the reproduced benchmark numbers (Part 4) as evidence.*

---

## 1. The input matrix V

The corpus is stacked into one matrix **V** (called `X` in code):

```
V  ∈  ℝ^(375 × 676),   V ≥ 0
```

- **Rows (375)** = one preprocessed Raman spectrum each (a pure analyte at one
  excitation). Every row is crop→ASLS→Savitzky-Golay→resample→L2→clip≥0 (Part 3), so all
  rows live on the identical 450–1800 cm⁻¹, 2 cm⁻¹ axis and are non-negative unit vectors.
- **Columns (676)** = wavenumber bins (450, 452, …, 1800 cm⁻¹). Column *b* holds the
  intensity of every spectrum at that Raman shift.

`V[i, b]` = "how much Raman intensity analyte *i* has at wavenumber *b*."

---

## 2. The factorization  V ≈ W H

NMF finds two non-negative matrices whose product reconstructs V:

```
V   ≈   W · H
(375×676) (375×24)(24×676)          with   W ≥ 0 ,  H ≥ 0
```

- **H  ∈ ℝ^(24 × 676) — the basis spectra (the atlas).** Each of the 24 rows is a
  full-length spectral loading over the 676 bins: a *learned Raman motif*. These are the
  frozen components — the coordinate axes of the biochemical state space. (In sklearn,
  `H = model.components_`; this is exactly what is hashed to the fingerprint.)
- **W  ∈ ℝ^(375 × 24) — the activations (coordinates).** Row *i* says how strongly
  analyte *i* uses each of the 24 motifs. After L1 normalization, `W[i]` becomes a
  *proportion* vector — the raw material of the Biochemical State Vector (Part 8).

Reconstruction of one spectrum:

```
V[i]  ≈  Σ_{j=1..24}  W[i, j] · H[j]
```

i.e. **every spectrum is rebuilt as a non-negative weighted sum of the 24 motifs.** That
sum is the physical claim: a measured spectrum is an additive mixture of molecular
contributions.

The fit minimizes the Frobenius reconstruction error under non-negativity:

```
min_{W,H ≥ 0}  ‖ V − W H ‖²_F ,   solved by coordinate descent, init = NNDSVD-a, seed 0
```

**Projection of a new spectrum** (serum, SERS, biological) holds **H fixed** and solves
only for its coordinates — a non-negative least squares fit onto the frozen dictionary
(`non_negative_factorization(..., update_H=False)`). The atlas can never be altered by a
query; new data only *receives* coordinates.

Measured properties of the frozen fit (Part 4): explained variance 0.712; 24 components
carry per-component variance 0.018–0.119; **every component's bootstrap stability ≥ 0.65**
(mean 0.812) — no unstable axis.

---

## 3. Why Raman mixtures suit NMF specifically

Three physical facts about Raman spectroscopy make a non-negative, parts-based, linear
model the *natural* one:

1. **Additivity.** To first order the Raman spectrum of a mixture is the concentration-
   weighted **sum** of its components' spectra. That is literally `V[i] = Σ_j W[i,j] H[j]`
   — the NMF model *is* the physics of a dilute mixture.
2. **Non-negativity.** Raman intensity is non-negative, and a molecule cannot be present
   in *negative* amount. A decomposition whose parts and weights are both ≥ 0 keeps every
   quantity physically meaningful; a coordinate reads as "how much of this motif is
   present," never a cancellation of two signed abstractions.
3. **Sparsity / locality.** A molecule has a few characteristic bands. Non-negativity
   pushes NMF toward loadings that are *localised* bundles of bands (measured: sparsity
   0.819, band-localisation 0.054 — best of all candidates), so a component looks like a
   spectrum you can assign to chemistry, which is the entire premise of the MSS/BSV layers.

The consequence: NMF's basis vectors are not abstract directions of variance — they are
**spectral basis functions** (motifs) that can be read band-by-band and traced back to the
reference analytes that activate them (Part 6).

---

## 4. Why NMF, not the alternatives

All five were benchmarked identically (Part 4). The decision is evidence-based, not
default:

| Method | Parts-based / ≥0 | Interpretable loadings | Stability | Verdict |
|---|:--:|:--:|:--:|---|
| **NMF** | **✓** | **✓ (sparse, localised 0.82)** | **✓ 0.81** | **selected** |
| PCA | ✗ (signed) | ✗ (delocalised 0.48; orthogonality is a mathematical, not chemical, constraint) | ✗ 0.53 | 3rd (0.739) |
| ICA | ✗ (signed) | ✗ (0.50) | ✓ 0.77 | raw-top by score, excluded by non-negativity |
| Sparse coding / dict. learning | ✗ | partly (0.62) | ✓ 0.80 | worst reconstruction (0.83); last (0.429) |
| Autoencoder | ✗ | ✗ | ✗✗ **0.16** | 4th; a different solution every run |

- **PCA** maximises variance with orthogonal, signed axes. Orthogonality is a
  *mathematical* convenience with no biochemical meaning; the axes mix "more lipid / less
  protein" into a single signed direction, and half of PC loadings are negative — you
  cannot say "38 % purine" from a PCA score. Components are also unstable (0.53).
- **ICA** seeks statistically independent, signed sources. It scored *highest raw*
  (0.7866) — but its independence assumption is not the mixture physics, its sources are
  signed (no "amount present"), and its loadings are delocalised (0.50). It is excluded by
  the non-negativity constraint precisely because a biochemical proportion cannot be
  negative.
- **Sparse coding / dictionary learning** gives sparse codes but here reconstructs
  poorly (rel. error 0.83) and offers no additivity guarantee on the codes.
- **Autoencoder** is the most flexible but the least suitable: non-linear, signed latent,
  and **catastrophically unstable** (component stability 0.16 — it finds a different basis
  on every bootstrap). A foundation coordinate system must be reproducible; this is the
  opposite. It is included only as an honest comparator, never as a foundation claim.

**Bottom line.** NMF is the only candidate that is simultaneously (a) faithful to Raman
mixture physics (additive, non-negative), (b) interpretable (sparse, band-localised,
traceable to analytes), and (c) reproducible (stable components, deterministic seed). The
benchmark makes the first two measurable and the tie-break makes the physics decisive.
That is why the coordinate system is an NMF, and why its axes are *spectral basis
functions* rather than abstract latent directions.
