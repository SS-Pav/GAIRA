# GAIRA V7 — Inference Mode Architecture

The live path: one spectrum in, one BSV plus evidence and QC out. Deterministic, batch-
independent, and free of any fitting.

---

## 1. The canonical inference path

```
INPUT   wavenumbers ℝ^n, intensities ℝ^n, metadata {excitation, instrument, domain?}
   │
   │  ── STEP 1 ──  canonical preprocessing            [deterministic, per-spectrum]
   │     crop to 450–1800 cm⁻¹
   │     resample onto the frozen 676-bin grid (2.0 cm⁻¹)
   │     asls baseline → savgol smoothing → L2 normalisation
   │     QC: grid coverage, saturation, spike flags
   ▼
   x ∈ ℝ₊^676
   │
   │  ── STEP 2 ──  fixed-dictionary non-negative projection
   │                against the frozen Consensus Spectral Motif (CSM) basis
   │     c(x) = argmin_{c ≥ 0} ‖x − cᵀ·CSM‖²          (NNLS, frozen CSM ∈ ℝ₊^{M×676})
   ▼
   c(x) ∈ ℝ₊^M                                        CSM activations
   │
   │  ── STEP 3 ──  evidence expansion                 [optional, for explanation only]
   │     per-CSM: dominant bands, supporting analytes, supporting classes,
   │              n_lsms, is_singleton, is_anchored
   │     per-LSM: contribution within each activated CSM
   ▼
   evidence record
   │
   │  ── STEP 4 ──  soft biochemical themes
   │     t(x) = Sᵀ c(x)                                (frozen S ∈ ℝ₊^{M×K}, row-normalised)
   ▼
   t(x) ∈ ℝ₊^K
   │
   │  ── STEP 5 ──  absolute BSV
   │     BSV(x) = t(x)                                 ABSOLUTE. Not a delta. Not a label.
   ▼
   BSV(x) ∈ ℝ₊^K
   │
   │  ── STEP 6 ──  reference comparison, QC, uncertainty
   │     elevation:      z_k = (t_k − μ_k) / σ_k       (frozen reference frame)
   │     OOD:            score against the frozen reference support
   │     reconstruction: residual ‖x − c(x)ᵀ·CSM‖ / ‖x‖
   │     band fidelity:  agreement on each activated CSM's diagnostic bands
   │     uncertainty:    propagated projection → S → BSV
   │     confidence tier derived from the above
   ▼
   BSV + elevation + OOD + residual + uncertainty + confidence tier
   │
   │  ── STEP 7 ──  domain-context interpretation      [DOWNSTREAM ONLY]
   │     serum / EV / plasma / tissue / pathogen priors
   │     multi-assignment, ambiguity, confidence tiers
   ▼
OUTPUT  interpretation + full evidence + provenance + atlas version & fingerprint
```

---

## 2. Hard prohibitions

Inference must **never**:

| Prohibited | Why |
|---|---|
| fit NMF | would move the coordinate axes |
| fit PCA | would make the view batch-dependent |
| fit UMAP / t-SNE / any manifold learning | no stable out-of-sample transform; batch-dependent |
| cluster | output would depend on batch-mates |
| run graph community detection | offline structure discovery |
| optimise the ontology | offline curation |
| tune a threshold on the incoming data | leaks the test set into its own scoring |
| normalise across the batch | output would depend on batch-mates |
| let domain context influence any pre-BSV step | breaks universality of the representation |

**The general principle, of which every row above is an instance:**

> The output for a spectrum must be identical whether it is processed alone or as part of a
> batch of ten thousand.

This is directly testable and is a Phase-05 gate.

---

## 3. Permitted operations

| Permitted | Note |
|---|---|
| canonical preprocessing | deterministic, per-spectrum, no cross-spectrum statistics |
| NNLS against the frozen CSM dictionary | the core projection |
| matrix multiply by frozen `S` | theme mapping |
| affine transform by frozen `(μ, σ)` | elevation |
| **application** of a frozen PCA `(P, μ_P)` | visualisation only; `P` fitted offline |
| distances against frozen reference objects | OOD, nearest-reference evidence |
| uncertainty propagation through frozen linear maps | analytic, deterministic |
| appending a BSV to a trajectory | DART; see `LIVE_DART_COMPATIBILITY.md` |

### The PCA distinction, stated once more because it is easy to get wrong

- **Offline (learning mode):** `P, μ_P = fit_pca(reference BSVs)` → frozen into the atlas.
- **Live (inference mode):** `y = Pᵀ(BSV − μ_P)` → a picture.

`y` is **visualisation output**. It is never the canonical BSV, never enters interpretation,
never used for retrieval, scoring, or any decision. No V7 document, figure caption, or
docstring may describe PCA or UMAP as inference.

---

## 4. Uncertainty propagation

Uncertainty enters at three points and is carried to the output rather than being recomputed
at the end:

| Source | Nature | Propagation |
|---|---|---|
| **Measurement** | noise, baseline residual, grid resampling | into the projection residual and the activation covariance |
| **Representation** | CSM uncertainty (spread of contributing LSMs), singleton/anchor status, narrow analyte support | attached per CSM; widens the theme-level interval through `S` |
| **Mapping** | membership entropy of `S` rows | a CSM spread across many themes contributes diffuse, low-confidence theme mass |

**Non-negotiable:** an axis whose supporting CSMs are singletons or anchors must report
**wider** uncertainty than an axis supported by broad cross-class consensus. The V5 failure
was that a motif with 1.2% corpus coverage produced an output indistinguishable in form from
one with 7.2% coverage. In V7, support breadth must be visible in the number, not only in a
registry someone might consult.

---

## 5. QC and out-of-distribution handling

| Check | Signal | Response |
|---|---|---|
| Grid coverage | fraction of 450–1800 cm⁻¹ actually measured | flag; refuse below a floor |
| Saturation / spikes | detector artefacts | flag |
| Reconstruction residual | `‖x − ĉᵀCSM‖ / ‖x‖` | high ⇒ chemistry outside the atlas span |
| Band fidelity | agreement on activated CSMs' diagnostic bands | low ⇒ activation is diffuse mass, not a real band match |
| OOD score | distance to the frozen reference support | high ⇒ outside the validated domain |
| Domain mismatch | e.g. a SERS spectrum submitted to a Raman atlas | flag prominently |

**The engine reports; it does not silently refuse.** A high-OOD spectrum still yields a BSV —
with an explicit warning that it lies outside the atlas's validated boundary. Silently
returning a confident-looking number for out-of-domain input is the worse failure. The atlas
declares its validated boundary (`ARTIFACT_AND_MANIFEST_SPEC.md`) and QC states where the
input sits relative to it.

---

## 6. Output contract

Every inference returns:

```
atlas_version, atlas_fingerprint
preprocessing_config_hash
csm_activations        ℝ₊^M   + per-CSM evidence & provenance flags
theme_activations      ℝ₊^K
bsv                    ℝ₊^K   [ABSOLUTE]
bsv_elevation          ℝ^K    [derived: z-scored vs reference frame]
uncertainty            per-axis intervals
qc                     coverage, residual, band fidelity, OOD, flags
confidence_tier
provenance             CSM → LSM → class → analyte → source, resolvable
```

Schema and versioning rules: `DATA_CONTRACTS.md`.

**Naming discipline in the output.** `bsv` is absolute. `bsv_elevation` is derived and signed.
A ΔBSV, if computed, is a separate field produced by a separate call over two BSVs — the
inference path never returns a delta under the name `bsv`.

---

## 7. Determinism and portability guarantees

| Guarantee | Verified in Phase 05 by |
|---|---|
| Identical input → byte-identical output | repeat run comparison |
| Batch-independent | single-spectrum vs batch-of-N comparison |
| No RNG in the inference path | static check for RNG use / seeding |
| No fitting in the inference path | static check for `fit`, `fit_transform`, `partial_fit` |
| Runs on a clean clone with no lab volume | fresh-clone smoke test with `GAIRA_DATA_ROOT` unset |
| Atlas integrity | fingerprint verified against the manifest on every load |
| Cross-machine reproducibility | same output on a second platform |

The V5 engine already achieves the clean-clone property — `assets/foundation/` is
self-contained and needs no raw data and no SSD. **V7 must not regress it.** This is a gate,
not an aspiration.
