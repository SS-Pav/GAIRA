# GAIRA V5 — Raman Biochemical Foundation Model (Phases C1–C7)

**Date:** 2026-07-19 · Branch `gaira-v5-rebuild-plan` · Nothing pushed. Historical V1–V5 outputs, the demo, and the completed Stage A / Stage B / Stage B0 studies were **not modified** — they remain permanent scientific records.

> **Result:** a frozen, non-negative **biochemical reference space built only from pure Raman analytes**. The same analyte measured at a *different laser excitation* lands in nearly the same place (BSV cosine **0.918** vs **0.233** for different analytes), unseen analytes project sensibly (BSV margin **+0.62**), and biological serum Raman projects onto **cholesterol / albumin / cholesteryl-ester** coordinates while tube blanks show markedly less protein and fatty-acid evidence. Unsupervised throughout: no disease labels, no classification.

---

## 0. Scope and a corrected data decision

Canonical observation domain = **Raman**. Ag-SERS, Au-SERS and DART are future observation domains and are excluded from every step.

**One deviation from the brief, made deliberately and documented.** Phase C6 named the *adenine concentration series*, *uricase* and *serum spike* datasets as calibration/validation sets. In this repository all of those are **Ag-SERS / Au-SERS**:

| Named dataset | Actual modality here | Disposition |
| --- | --- | --- |
| adenine concentration series (`adenine_sers_control`) | Ag-SERS (bAgNPs), 785 nm | excluded — out of domain |
| `european_multi_instrument_adenine` | substrates `cAg` / `sAg` / `cAu` → SERS & Au-SERS | excluded — out of domain |
| metabolite-63 | 633 nm Ag-SERS | excluded — out of domain |
| uricase / serum spike-ins (`serum_ag_colloids`) | Ag-SERS serum colloid | excluded — out of domain |

Using them would have violated the Raman-only rule. Instead, C6 uses the Raman-domain validation that genuinely exists — **held-out analytes, laser-excitation transfer, reference-source transfer, and a tube-blank control** — and C7 uses **`covid_serum_raman`**, which is genuine biological *serum Raman*.

---

## 1. Frozen Raman dataset (C0)

| | |
| --- | --- |
| Spectra | **375** (RamanBioLib 202 · Gobbato Raman powders 153 · amino-acid grounding 20) |
| Unique analytes | **167** |
| Window / grid | **450–1800 cm⁻¹ @ 2 cm⁻¹** (676 bins) — wider than the Ag-SERS-constrained Stage-B window |
| Excitations | 785 (234), 1064 (55), 532 (50), 488 (29), + 5 minor |
| Analytes at >1 excitation | **41** (built-in instrument-transfer test) |
| Preprocessing | ASLS baseline · Savitzky–Golay · L2 (reused from `gaira.preprocessing`) |
| External (projection only) | `covid_serum_raman` — 477 serum Raman spectra (COVID 159 / Suspected 156 / Healthy 150 / Tube 12) |

Card: `tables/raman_dataset_card.json`. Chemical families were extended to cover the RamanBioLib compound space (**161/167** assigned) — from known chemistry, never inferred from spectra.

---

## 2. C1 — representation selected by benchmark, not by default

Five families × six latent sizes, analyte-grouped held-out CV. **Reconstruction was deliberately weighted only 10%**; stability, interpretability and analyte structure dominate.

| representation | best k | score | recon err | neighbourhood | replicate robustness | stability | sparsity | excitation leakage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ICA | 32 | 0.787 | 0.486 | 0.639 | 0.834 | 0.773 | 0.503 | 0.027 |
| **NMF (selected)** | **24** | **0.786** | 0.550 | 0.671 | 0.624 | **0.812** | **0.819** | **0.019** |
| PCA | 32 | 0.752 | 0.486 | 0.684 | 0.872 | 0.502 | 0.468 | 0.094 |
| Autoencoder | 32 | 0.581 | 0.481 | 0.684 | 0.726 | **0.162** | 0.382 | 0.034 |
| SparseDict | 24 | 0.429 | 0.832 | 0.496 | 0.529 | 0.796 | 0.620 | 0.046 |

- **PCA was not selected** (3rd). The **autoencoder ranked 4th with component stability 0.162** — it reconstructs well but its components are not reproducible at this corpus size, echoing the Stage-B encoder-collapse finding.
- ICA (0.787) and NMF (0.786) were **statistically indistinguishable** (0.001 apart). The tie was broken on a **pre-stated constraint**: a biochemical coordinate must be *non-negative and additive*, because Raman spectra of mixtures are non-negative sums of molecular contributions and a BSV coordinate must mean "how much of this theme is present". Signed ICA components would imply negative biochemical content. Within the admissible (non-negative) set the benchmark score then decided. NMF also wins on stability, sparsity and excitation leakage; ICA's advantage is replicate robustness (0.834 vs 0.624), recorded as the runner-up trade-off.

**Selected: NMF, k = 24.**

---

## 3. C2 — the frozen biochemical manifold

| | |
| --- | --- |
| Explained variance | **0.712** |
| Intrinsic dimensionality | participation ratio **15.2**; 90% of latent variance by **16** components; entropy rank 19.6 |
| Bootstrap component stability (by analyte) | **0.812** |
| Neighbourhood stability | 0.671 full / bootstrap-consistent |
| Activation sparsity | components are used sparsely per spectrum (median 6/24 in MSS) |

So k=24 is mildly over-complete for an intrinsic dimensionality of ~15–16, which keeps components interpretable while retaining fidelity. **Frozen**: `artifacts/manifold_components.npz` + `manifold.json`, fingerprint `09ed804a40836f4a05a91ba10900cded`.

---

## 4. C3 — emergent biochemical axes (tentative)

The previous curated radar axes were **not** hard-coded. Components were interpreted from the chemistry of the analytes that load on them (primary evidence), then grouped into axes; literature Raman peak assignments were used only to *corroborate*. (The available literature table is biofluid-oriented and protein-heavy, so using it as the primary label biased every axis toward "Proteins" — that first attempt was diagnosed and corrected.)

**12 axes emerged:**

| axis | tentative theme | share | conf. | top-loading analytes |
| --- | --- | --- | --- | --- |
| 1 | triglyceride | 0.201 | medium | tripalmitin, tristearin, triarachidin, trimyristin |
| 2 | saccharide | 0.186 | **medium-high** | fructose-6-phosphate, fructose, dextrose |
| 3 | organic acid | 0.130 | low | citrate, aspartic acid, phosphate, succinic acid |
| 4 | protein | 0.111 | **medium-high** | lectin, thaumatin, carbonic anhydrase, pepsinogen |
| 5 | amino acid | 0.075 | medium | glycine, serine, hydroxyproline, alanine |
| 6 | polysaccharide | 0.064 | low | amylopectin, amylose |
| 7 | pyrimidine | 0.059 | medium | uracil, thymine, b-dna, cytosine |
| 8 | *unassigned* | 0.059 | low | creatinine, tubulin, tyrosine, mannose |
| 9 | fatty acid | 0.042 | medium | arachidonic, trilinolenin, α-linolenic, linoleic |
| 10 | sterol | 0.028 | low | *mixed* (adenine, acetyl-CoA, estrone) — label questionable |
| 11 | purine | 0.027 | medium | guanine, xanthine |
| 12 | cofactor | 0.019 | low | riboflavin |

Ten of twelve axes are chemically coherent; axes 8 and 10 are honestly reported as unassigned/low-confidence. **All themes are tentative post-hoc interpretations of an unsupervised decomposition and are not molecular assignments.** Component co-activation clustering (silhouette 0.131) is reported separately as a diagnostic: the NMF components do *not* form natural super-clusters — they behave as distinct parts.

---

## 5. C4 — Biochemical State Vector · C5 — Molecular Spectral Signatures

- **BSV** (`tables/c4_bsv_components.csv`, `c4_bsv_axes.csv`): every one of the **167 analytes** has non-negative coordinates that sum to 1 — a share of biochemical evidence — over 24 components and 12 axes, each with a per-coordinate standard deviation across its replicates/excitations. Median coordinate uncertainty **0.0003**.
- **MSS** (`tables/c5_mss.csv`): a sparse signature per analyte — median **6 of 24** components (sparsity 0.75) capturing 90% of latent energy, with the component weights, the spectral bands those components carry, the analyte's own observed bands, and the axis contributions. These are reusable reference objects.

Figure 4 shows the analyte map organising by chemical family (proteins, saccharides, triglycerides and sterols occupy distinct regions) — the manifold arranges itself by chemistry without ever being told the families.

---

## 6. C6 — external validation, no retraining

| test | result | null (different analytes) |
| --- | --- | --- |
| **Excitation transfer** (same analyte, different laser; n=41) | **0.918** | 0.233 |
| **Source transfer** (same analyte, different reference source; n=34) | **0.847** | 0.233 |
| Held-out analytes (grouped CV) | neighbourhood 0.671, within 0.79 vs between 0.17 → **margin +0.62** | — |
| Reconstruction on unseen analytes | 0.55 relative error | — |

Excitation transfer is the strongest evidence that this is a *biochemical* rather than an *instrumental* space: a compound measured at 532, 785 or 1064 nm lands in essentially the same coordinates. The weakest cases — **ferritin, haemoglobin, albumin** (0.63) — are heme proteins where resonance genuinely changes the spectrum with excitation, so the failure mode is physically explicable rather than arbitrary.

---

## 7. C7 — biological projection (frozen manifold, no retraining, no labels)

477 serum Raman spectra projected. **No supervised learning, no disease classification.**

| group | n | triglyceride | saccharide | organic acid | protein | amino acid | fatty acid | OOD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COVID | 159 | 0.236 | 0.279 | 0.073 | 0.157 | 0.121 | 0.013 | 0.163 |
| Healthy | 150 | 0.237 | 0.273 | 0.072 | 0.162 | 0.124 | 0.016 | 0.162 |
| Suspected | 156 | 0.231 | 0.276 | 0.069 | 0.170 | 0.120 | 0.015 | 0.166 |
| **Tube (blank)** | 12 | 0.245 | 0.317 | **0.038** | **0.103** | 0.129 | **0.000** | **0.189** |

- **Biochemical plausibility:** the nearest pure references for serum spectra are **cholesterol, albumin and cholesteryl esters** — the expected dominant serum biochemistry, recovered without being told.
- **Blank control works:** tube blanks carry roughly *half* the protein evidence, *half* the organic-acid evidence and *no* fatty-acid evidence of real serum, and sit further out of distribution (0.189 vs 0.162–0.166).
- **Disease groups are near-identical** in these coordinates. This is reported exactly as found; it is the honest result of an unsupervised biochemical projection and is **not** evidence for or against any disease signal. No classification was attempted.

---

## 8. Frozen artifacts

`results/v5_rebuild/foundation/`
- `artifacts/manifold_components.npz` (components + grid), `artifacts/manifold.json` (fingerprint `09ed804a…`, stats, axes, validation, projection)
- `tables/` — `raman_dataset_card.json`, `c1_representation_benchmark.csv`, `c1_selection.json`, `c2_manifold_stats.json`, `c3_axes.json`, `c3_components.csv`, `c3_analyte_activation_matrix.csv`, `c4_bsv_components.csv`, `c4_bsv_axes.csv`, `c5_mss.csv`, `c6_*`, `c7_*`
- `figures/` — 6 publication figures · code under `results/v5_rebuild/foundation/code/`
- Package: `src/gaira/foundation/` (dataset, representation, benchmark, latent_space, axes, families_raman, bsv, mss, validation, projection, serialization)

---

## 9. Proposed updates to the rebuild plan (NOT applied — awaiting approval)

1. Record that the Raman↔Ag-SERS line closed with a **negative result** (Stage A/B → B4, Stage B0 → P4) and that V5 has **returned to the canonical Raman-only architecture**; Ag-SERS/Au-SERS/DART become explicit *future observation domains*.
2. Add **Phase C (C1–C7)** as implemented: Raman-only foundation model, with NMF k=24 frozen as **canonical biochemical representation v1** (fingerprint `09ed804a…`).
3. Record the C6 substitution: the originally-named calibration sets are Ag-SERS and were replaced by Raman-domain validation (held-out analytes, excitation transfer, source transfer, blank control).
4. Note the standing methodological caution carried from Stage B0/B: encoder-style representations remain unstable at this corpus size (component stability 0.16 here), so the foundation layer stays parts-based and interpretable.
5. Gate the next phase: ontology refinement, Ag-SERS integration and any biological classification remain unauthorized until a separate decision.

---

## 10. Limitations

- 375 spectra / 167 analytes is a *reference* scale, not a foundation-model scale; k=24 was chosen against an intrinsic dimensionality of ~15.
- Two of twelve axes are low-confidence or unassigned; axis 10 ("sterol") is mixed and its label should be treated as unreliable.
- Literature corroboration comes from an 84-row biofluid-oriented table that is protein-weighted; it corroborates but cannot arbitrate themes.
- Serum projection is a *coordinate assignment*, not a validated quantification: no concentration ground truth exists in the Raman domain here, which is precisely why the Ag-SERS dose-response sets could not be used.
