# GAIRA V7 Phase 02 — Pre-registration

**Committed before any Phase 02 sweep was run (P-12).** Everything below is a rule, not a
result. If a rule turns out to be defective it may be changed only by demonstrating the defect,
recording the before/after, and re-running — the Phase 01 `k_c` correction is the template.

Governing specification: `plan/VALIDATION_AND_DECISION_RULES.md` §3,
`architecture/LEARNING_MODE_ARCHITECTURE.md` Stages 2–4, contracts C-06 and C-07.

---

## 1. What is being tested

**H0 (null).** Local Spectral Motifs learned independently within chemical classes do *not*
converge: apparent cross-class similarity is explained by shared broad Raman structure (CH
stretches, skeletal modes) and no pair describes the same biochemical phenomenon.

**H1.** Some LSMs from independent class-local fits describe one shared spectral phenomenon,
recoverable as a Consensus Spectral Motif.

**Every candidate merge is a falsifiable hypothesis.** The default is *not merged*. Leaving
motifs separate is an acceptable, publishable outcome.

---

## 2. Edge features — seven, all recorded on every edge

Contract C-06 requires six; the brief adds reconstruction substitutability, which is a
strictly stronger falsification test than any of the six. All seven are computed for all
`50 × 49 / 2 = 1225` pairs.

| # | Feature | Definition | Range |
|---|---|---|---|
| 1 | `spectral_cosine` | `⟨h_i,h_j⟩ / (‖h_i‖‖h_j‖)` on the full 676-bin grid | [0,1] |
| 2 | `band_overlap` | cosine restricted to the union of both motifs' diagnostic band windows (±8 cm⁻¹ around each dominant band), each masked vector renormalised | [0,1] |
| 3 | `peak_agreement` | `2·|matched| / (|B_i| + |B_j|)`, matched = dominant bands within ±10 cm⁻¹; **positions only, intensity-free** | [0,1] |
| 4 | `bootstrap_cooccurrence` | `(fraction of R resamples in which both motifs are recovered) × (mean resampled pair cosine over those resamples)` | [0,1] |
| 5 | `activation_cooccurrence` | Spearman correlation of the two motifs' activation profiles across the 154 canonical molecules, clipped at 0 | [0,1] |
| 6 | `provenance_overlap` | Jaccard of activated-molecule supports, **null-discounted**: `max(0, (J − E[J])/(1 − E[J]))` with `E[J]` the expectation under random supports drawn from the pair's own molecule pool | [0,1] |
| 7 | `substitutability` | `min(s_{i→j}, s_{j→i})`, where `s_{i→j}` is the mean retained explained variance when `h_j` replaces `h_i` in `i`'s class dictionary, over the molecules `i` supports | [0,1] |

### Two construction decisions that must be stated in advance

**Activations are computed by independent non-negative scalar projection, not joint NNLS.**
Joint NNLS over 50 non-orthogonal motifs makes near-duplicate motifs *split* each other's mass
and appear anticorrelated — which would penalise precisely the pairs Phase 02 exists to find.
The joint-NNLS activation matrix is computed anyway and reported as a sensitivity check.

**Provenance is measured on projected support, not class-local support.** Class-local support
is disjoint across classes by construction, so a class-local provenance feature would be
identically zero for every cross-class pair — degenerate exactly where it is needed. Projected
support (which molecules activate the motif when it is projected against the whole reference
set) is defined across classes. The within-class null discount required by risk R-01 is applied
on top, so the feature cannot simply re-encode the class partition.

**Feature independence is a claim, so it is measured.** The 7×7 inter-feature correlation
matrix over all 1225 pairs is reported. Features that turn out to be near-duplicates of each
other are not six independent lines of evidence and will be reported as such.

---

## 3. Edge weight

$$w_{ij} = \prod_{f} \big(\text{feature}_f\big)^{\alpha_f}, \qquad \sum_f \alpha_f = 1$$

| feature | `α` | why this exponent |
|---|---:|---|
| `band_overlap` | 0.25 | agreement *where it matters*; the single most discriminating channel |
| `spectral_cosine` | 0.20 | overall shape, deliberately **not** dominant |
| `peak_agreement` | 0.15 | excitation-invariant, intensity-free |
| `bootstrap_cooccurrence` | 0.15 | is the relationship a property of the data or of one fit |
| `substitutability` | 0.10 | can one actually do the other's job |
| `activation_cooccurrence` | 0.10 | do they respond to the same molecules |
| `provenance_overlap` | 0.05 | partly circular (R-01), so deliberately small |

**A geometric mean, not an arithmetic one.** Under an arithmetic mean a cosine of 0.95 can
carry an edge whose every other channel is near zero. Under a geometric mean any single
near-zero channel drives the weight to zero. This is the operational form of "never merge
motifs solely because spectral cosine is high", and it is the reason no per-feature hard floor
is imposed: floors would add six unswept arbitrary cuts, whereas the geometric mean enforces
the same requirement continuously and leaves exactly one cut — the edge threshold — which *is*
swept.

---

## 4. Threshold sweep (rule 3c)

- Sweep `τ ∈ {0.05, 0.10, …, 0.90}`.
- At each `τ`: build `G_τ = (V, {e : w_e ≥ τ})`, run Louvain over 12 seeds, record the
  consensus partition, `n_communities`, and **community stability** = mean adjusted Rand index
  over 25 perturbations (10% of edges removed at random) against the unperturbed partition.
- **Stable region** = a maximal contiguous run of `τ` values with stability ≥ 0.95 × (max
  stability over the sweep) *and* constant `n_communities`.
- **Selection:** the midpoint of the **widest** stable region of length ≥ 3 consecutive `τ`.
- **If no stable region of length ≥ 3 exists, the graph construction is inadequate and the
  Phase 02 gate FAILS** (risk R-07). That is a finding, not a nuisance to be tuned away.

---

## 5. Integration-method comparison (rule 3a)

All five candidates are run and the **full table is published regardless of the winner**.

1. Louvain graph communities (modularity; no `M` chosen in advance)
2. hierarchical consensus clustering (average linkage on `1 − w`, consensus matrix over 50 resamples)
3. spectral clustering (Laplacian eigen-decomposition; needs `M`)
4. sparse non-negative meta-factorisation of the activation matrix `A ≈ UV` (needs `M`)
5. hybrid: communities as initialisation for a constrained non-negative refit

Composite (each criterion min–max normalised across candidates, direction applied):

| criterion | dir | weight |
|---|---|---:|
| consensus stability across resamples | ↑ | 0.20 |
| within-CSM spectral cohesion | ↑ | 0.15 |
| between-CSM separation | ↑ | 0.15 |
| chemical coherence | ↑ | 0.10 |
| retained LSM information (reconstruction of the LSM set from the CSM set) | ↑ | 0.15 |
| held-out reconstruction of molecules | ↑ | 0.10 |
| hyperparameter sensitivity | ↓ | 0.05 |
| singleton fraction | ↓ | 0.05 |
| between-CSM redundancy | ↓ | 0.05 |

**No method is presumed.** The stated prior is that graph or hybrid routes look more promising
because meta-NMF sees only one of seven evidence channels — a hypothesis, not a decision.
**If meta-NMF wins, molecule-discriminating LSM survival must be verified explicitly (R-06).**

## 6. `M` selection (rule 3b)

For methods that require `M`: sweep `M ∈ [2, 25]`, score on the same composite with the
singleton and redundancy penalties applied, and take the **smallest `M` on the Pareto plateau**
(within 0.02 composite units of the maximum, contiguous run containing the maximum — the
contiguity requirement is carried over from the Phase 01 `k_c` correction).

## 7. Consensus operator Π

Three candidates, selected on within-CSM cohesion and reconstruction, all three reported:
stability-weighted mean · non-negative medoid · leading non-negative direction (NMF rank-1).
Output renormalised to unit L2 on the canonical grid; non-negativity is an invariant, not a
preference.

## 8. Merge acceptance — a CSM is *accepted* only if all hold

1. every contributing edge survives at the selected threshold;
2. `bootstrap_cooccurrence ≥ 0.50` for the contributing edges (majority of resamples);
3. within-CSM cohesion ≥ the between-CSM separation for that CSM;
4. reconstruction of its supporting molecules does not degrade by more than **0.05 absolute
   explained variance** versus the contributing LSMs;
5. it survives leave-one-class-out;
6. its dominant bands admit a stated spectroscopic assignment consistent with the contributing
   chemistry (class-conditioned assignment, as corrected in the Phase 01 investigation).

A group failing any of these is **reported as a rejected consensus motif with the reason**, and
its LSMs stay separate. Singletons (`n_lsms == 1`) are not failures — they are LSMs no other
class-local decomposition independently confirmed, and are flagged, counted and kept visible.

## 9. Pre-declared false-merge suspects

These four cross-class pairs are named in advance and investigated whatever the graph says:
`peptide_protein ↔ polysaccharide` · `acylglycerol ↔ fatty_acid` ·
`phospholipid_sphingolipid ↔ sterol_steroid` · `purine ↔ sulfur_thiol_cofactor`.

For each, the verdict must be one of: **genuine shared chemistry · overlapping skeletal
vibrations · glycoprotein biology · lipid superfamily convergence · artefact** — with the
evidence that discriminates them.

**Null model.** Merge confidence is also computed for a size-matched null: 200 random
band-position permutations of each motif, preserving band count and intensity distribution. An
observed edge weight is only meaningful relative to what this null produces.

## 10. Frozen constants

```
R_BOOTSTRAP        = 24     analyte-level resamples for feature 4
BAND_TOL_CM        = 10.0   peak matching tolerance (feature 3)
BAND_HALFWIDTH_CM  =  8.0   diagnostic mask half-width (feature 2)
MIN_ACTIVATION     =  0.05  a molecule "activates" a motif above this normalised share
TAU_GRID           = 0.05 … 0.90 step 0.05
LOUVAIN_SEEDS      = 12
PERTURB_REPEATS    = 25     edge-removal repeats for community stability
PERTURB_FRACTION   = 0.10
M_SWEEP            = 2 … 25
PLATEAU_TOLERANCE  = 0.02   composite units
EV_DEGRADE_MAX     = 0.05   absolute explained-variance loss allowed on merge
NULL_PERMUTATIONS  = 200
BASE_SEED          = 0      the run is deterministic given this schedule
```
