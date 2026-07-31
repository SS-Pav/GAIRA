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

Both apps read only committed assets + precomputed outputs. Neither requires SSD_Rad,
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
