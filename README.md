# GAIRA

**A frozen, reproducible biochemical coordinate system for Raman spectroscopy.**

GAIRA learns a universal biochemical reference space **once**, from pure-compound Raman
spectra, freezes it to a cryptographic fingerprint, and projects every future spectrum —
serum, SERS, EV, a dose series, a clinical sample — into that same space. The result is
not a per-dataset classifier but a *coordinate system*: coordinates that mean the same
thing across instruments, substrates and studies.

- **Atlas:** NMF, k = 24, fingerprint `09ed804a40836f4a05a91ba10900cded`, learned from 375
  pure-Raman spectra / 167 analytes.
- **Reproducible to the bit:** rebuilding the NMF from the corpus reproduces the frozen
  basis exactly. See `results/v5_rebuild/foundation_audit/`.
- **Portable:** clone → install → run. No SSD_Rad, no absolute paths, no lab data required.

---

## This repository is GAIRA_Core

GAIRA is organized as two conceptual repositories:

- **GAIRA_Core (this repo)** — the permanent, frozen **scientific product**: the inference
  engine, the frozen foundation model, the ontology/MSS layers, the Streamlit demos, and
  the complete reproducible audit (reports, figures, tables). Everything needed to clone,
  understand, run and inspect GAIRA — with no access to lab infrastructure.
- **GAIRA_Lab (separate; not in this repo)** — active research: raw spectra, large
  datasets, experimental notebooks, training runs, intermediate outputs, future model
  development. The lab volume (`/Volumes/SSD_Rad/GAIRA_DATA/`) remains the source of truth
  for laboratory work. Core never depends on it at runtime.

The build/regeneration scripts under `results/v5_rebuild/*/code/` and the corpus loaders in
`src/gaira/foundation/` reference the lab volume — but only to *regenerate* the frozen
assets. Running the engine and the demos requires none of that; they read only the
committed, in-repo assets.

---

## Repository structure

```
GAIRA/
├── assets/
│   └── foundation/            ← the FROZEN biochemical reference model (self-contained)
│                                NMF basis, ontology, MSS, weights, normalization, MANIFEST
├── src/gaira/                 ← the Python package (engine, foundation, layers)
│   └── engine/                  inference engine — loads assets/foundation/, never the lab
├── gaira_foundation_explorer/ ← Streamlit app: interactive review of the frozen model
├── gaira_demo_reasoning_v4/   ← Streamlit app: the scientific-reasoning demo
├── results/v5_rebuild/        ← the reproducible build + audit (provenance, not runtime)
│   ├── foundation/              frozen NMF build artifacts (source of assets/foundation)
│   ├── engine_v1/               registry, ontology weights, normalization frame
│   └── foundation_audit/        the complete Foundation Model audit
│       ├── reports/  figures/  tables/  components/  code/
├── demo_data/                 ← where each demo's precomputed inputs live (see its README)
├── requirements.txt           ← runtime dependencies
└── pyproject.toml             ← `pip install -e .` to install the `gaira` package
```

The **frozen model lives in `assets/foundation/`**. Each demo's **precomputed inputs**
live under `results/v5_rebuild/foundation_audit/` (Explorer) and the demo's own committed
artifacts (Reasoning demo) — documented in `demo_data/README.md`. **Laboratory datasets
belong in GAIRA_Lab** (the SSD volume), never in this repo.

---

## Install

```bash
git clone https://github.com/SS-Pav/GAIRA.git
cd GAIRA
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional: make the package importable from anywhere
pip install -e .
```

Runtime dependencies are lightweight: `numpy`, `scipy`, `scikit-learn`, `pandas`,
`pyyaml`, `streamlit`, `plotly`, `matplotlib`. (`torch` is needed only to *re-run* the
representation benchmark, not to run inference or the demos.)

## Launch

**Foundation Explorer** — an interactive review article documenting and validating the
frozen model:

```bash
streamlit run gaira_foundation_explorer/app.py
```

**Reasoning demo** — the scientific-reasoning demonstration:

```bash
streamlit run gaira_demo_reasoning_v4/app.py
```

**Foundation Explorer V2** — the cross-modal transfer study: a four-level validation
framework (latent fingerprint → biochemical theme → perturbation → matrix) for the
Raman → Ag-SERS jump. Additive; the original Explorer is unchanged.

```bash
streamlit run gaira_foundation_explorer_v2/app.py
```

**Foundation Explorer V3** — the **Representation Hierarchy**: the transfer story reorganised
into five levels (latent → MSS motif → biochemical theme → perturbation → matrix), with new
rank-preservation and top-k metrics, the purine attractor quantified (ΔPurine), and honest
null controls throughout. Additive; V1 and V2 are unchanged.

```bash
streamlit run gaira_foundation_explorer_v3/app.py
```

**Foundation Explorer V4** — **null-calibrated hierarchical recovery**: every metric
is calibrated against an analyte-mismatched null, so "recovery" means statistically specific
(retrieval rank-1 + stable), never a raw cosine threshold. Establishes that analyte-specific
cross-modal recovery is rare at every level (latent 7/51, MSS 3/51, theme 4/51), that **MSS is not
the primary metric**, and that the purine attractor is present in the unspiked-serum blank before
any analyte. Additive; V1–V3 unchanged; reproduces V3 bit-for-bit.

```bash
streamlit run gaira_foundation_explorer_v4/app.py
```

**Foundation Explorer V5** — **abstraction recovery**: when exact identity is lost,
does the correct *broader* chemistry survive? The expected motif/theme is often *present* in the
Ag-SERS top-3 (MSS 40%, theme 49%) but rarely analyte-*specific* (≤2/48); cross-modal
subclass/family classification is at chance. A Raman→Raman control proves the taxonomy is separable
and that abstraction helps *within* Raman (0.23→0.42) — the Ag-SERS modality gap collapses it.
**Presence ≠ recovery.** Molecular subclass is an evaluation overlay, not a learned layer. Additive;
V1–V4 unchanged; reproduces V4 identity.

```bash
streamlit run gaira_foundation_explorer_v5/app.py
```

**Foundation Explorer V6** *(current)* — a **detection gate** before recovery. V5 scored all analytes
equally, conflating *measurement* failure (invisible on silver) with *representation* failure. V6
adds a Stage-0 detection gate (deterministic, no-ML, validated before freezing): **22/51 analytes are
detectable, 29 undetectable.** Among detectable analytes, exact identity ~doubles (14%→23%) and
presence rises, but analyte-specific recovery stays low — the residual is representational. A learned
Raman→SERS transfer model is justified for **~11 detectable, representation-limited analytes**; the
rest need a better substrate. Additive; V1–V5 unchanged; reuses V5 recovery flags.

```bash
streamlit run gaira_foundation_explorer_v6/app.py
```

All apps read only committed assets + precomputed outputs. None requires SSD_Rad,
raw spectra, or any recomputation.

## Use the engine directly

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from gaira.engine import GAIRAEngine

eng = GAIRAEngine()                       # loads assets/foundation/ ; verifies fingerprint
out = eng.infer(coordinates=np.full(24, 1/24), domain="serum")
print(out.bsv.composition)                # 11 biochemical themes + confidence + OOD
```

---

## Reproduce & audit

The complete, first-principles audit of the frozen model — grounding corpus,
preprocessing, NMF rebuild, per-component analysis, MSS/BSV, and six validation datasets —
is in **`results/v5_rebuild/foundation_audit/`** (start with its `README.md`). It rebuilds
the atlas from scratch and reproduces the fingerprint byte-for-byte. The Foundation
Explorer app is an interactive front-end to that audit.

## Scientific principles

Spectra are mixtures, not fingerprints. Peak ≠ molecule. GAIRA prefers biochemical
themes/motifs over exact molecule claims, tracks uncertainty (confidence + OOD), and keeps
a clean separation between the **representation** (pure-Raman biochemistry) and the
**observation** (how a given surface/instrument sees it). SERS is used to *validate* the
model, never to fit it.

Validation follows a **ladder** that separates each failure mode:
**reference Raman → pure Ag-SERS (modality gap) → controlled perturbation (concentration) →
serum matrix (competition) → biological cohorts.** The pure-Ag-SERS rung is the bridge —
it shows transfer is already adsorption-selective *before* serum, so the serum results are
the expected continuation. See `results/v5_rebuild/foundation_audit/reports/PURE_AG_SERS_VALIDATION.md`.

That rung is analysed in depth by the **multi-level transfer framework** — latent fingerprint
preservation, biochemical theme preservation (with a null control), perturbation sensitivity,
and matrix recoverability — in `GAIRA_MULTI_LEVEL_VALIDATION_FRAMEWORK.md` and
`results/v5_rebuild/pure_ag_sers_theme_preservation/`, with an interactive front-end in
`gaira_foundation_explorer_v2/`. Its central lesson: theme-level and fingerprint-level
preservation are distinct, and a high *raw* theme cosine is a compositional-baseline artifact —
identity-specific theme preservation is selective, tracking adsorption, because Ag-SERS
homogenises most analytes toward a purine attractor.

The **V4 null-calibrated recovery analysis**
(`results/v5_rebuild/hierarchical_recoverability_v4/`, Explorer V4) sharpens this into
statistics: each representation level is measured against an analyte-mismatched null, so a metric
maps to a purpose only if it carries analyte identity above chance. The metric-purpose mapping:
**latent cosine** = substrate/fingerprint fidelity (and the best cross-modal identity cosine);
**MSS** = intermediate motif candidate (rejected as primary by the null); **raw theme BSV** =
broad biochemical interpretation, not identity; **residual/null-adjusted theme metrics** =
analyte-specific diagnostic; **perturbation** = functional validation (strongest evidence);
**matrix recovery** = mixture visibility (separate property). "Detectable/recoverable" is never
assigned from a raw cosine threshold.

---

## GAIRA V7 — the frozen runtime

The V7 scientific architecture is **frozen after Phase 09**, and the runtime platform around it is
frozen after Phase 10. V7 is a Raman-only biochemical inference engine: a spectrum is projected
into a frozen motif atlas, matched against grounded reference evidence, and summarised as a
16-axis Chemistry Evidence profile with calibrated confidence and complete provenance.

**Run it from a clean clone — no external volume, no cloud account, no LLM:**

```bash
export PYTHONPATH=$PWD/src
python -m gaira.v7.cli info                   # engine identity, performance, limitations
python -m gaira.v7.cli infer spectrum.csv     # inference; --report out.pdf
python -m gaira.v7.api                        # HTTP service   → :8000/docs
python -m gaira.v7.mcp                        # MCP tool server (stdio)
streamlit run streamlit_apps/gaira_v7_console.py
```

The frozen atlas is committed (ten files, ~10 MB, all git-tracked), so inference works
immediately after cloning.

### Atlas identity — two hashes, two names

| term | value |
|---|---|
| **Scientific Atlas Fingerprint** | `09ed804a40836f4a05a91ba10900cded` |
| **Frozen Runtime Content Hash** | `2e43ddcca7d3be41c5f9da016fb8277f` |

Never call both "the atlas fingerprint". See
`results/v7_rebuild/phase10/reports/PHASE_10_ARCHITECTURE.md` §4.

### Validated performance

Molecule retrieval top-1 **0.6053** · top-5 0.7947 · MRR 0.6870.
Chemistry Evidence on **unseen molecules** top-1 **0.8507** · top-3 0.9760.
Cross-surface parity: 60 comparisons over 6 surfaces, **max |Δ| = 0.0**.
Single-spectrum inference 2.34 ms; API overhead 1.61 ms.

### Scope — read before using any output

- **Chemistry Evidence is RELATIVE.** Not a concentration, not an abundance, not a mixture
  fraction. L2 normalisation removes absolute scale in the first preprocessing stage.
- **Retrieved molecules are reference ANALOGUES**, not identifications.
- **There is no validated open-set detection.** The engine cannot determine that the true
  molecule is absent from its 154-molecule bank.
- **Pure Raman reference spectra only.** SERS, serum, plasma, EV, bacteria and tissue are
  extension points with defined contracts and no validation.
- **DART is not a modality.** It is a dynamic perturbation protocol built on Raman/SERS
  measurements — `I(wavenumber, potential, time)` is still a vibrational measurement — and it
  attaches at the **trajectory layer**, downstream of the frozen representation. See
  `PHASE_10_PLUGIN_ARCHITECTURE.md`.

Full documentation: `results/v7_rebuild/phase10/reports/`, including
`PHASE_10_GAIRA_RUNTIME_PLATFORM_REPORT.pdf`.
