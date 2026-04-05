# GAIRA Project Context

You are working on GAIRA (GenAI Raman Analysis), a domain-aware AI reasoning and representation-learning system for Raman/SERS spectra of biological samples.

## Core mission

GAIRA is not just a classifier. Its purpose is to:
1. interpret Raman/SERS spectra of biological samples,
2. ground interpretations in spectral references and literature,
3. apply domain-specific biological context,
4. produce cautious, explainable biochemical interpretations,
5. evolve toward dynamic biochemical inference for DART-Met.

Always prioritize:
- scientific defensibility,
- domain-aware interpretation,
- provenance,
- modular architecture,
- reproducible pipelines.

Never overclaim exact molecular identities from noisy biological Raman/SERS data.

## Repo and storage locations

Local repo:
- /Users/suraj/projects/GAIRA

Primary SSD data root:
- /Volumes/SSD_Rad/GAIRA_DATA

Important SSD subtrees:
- raw datasets: /Volumes/SSD_Rad/GAIRA_DATA/raw
- processed outputs: /Volumes/SSD_Rad/GAIRA_DATA/processed
- interim DB if needed: /Volumes/SSD_Rad/GAIRA_DATA/interim

Common project locations in repo:
- src/
- scripts/
- config/
- data/registry/
- reports/

## Current architectural state

### Global v1
Previous shared/global encoder work already exists and should be treated as Global v1.
Global v1 learned:
- sample type,
- dataset identity,
- within-dataset structure,
but did not convincingly learn shared cross-dataset biology.

Do not discard Global v1; use it as the baseline comparison for Global v2.

### Autoresearch and pilot learnings
We ran a major autoresearch sprint and multiple pilots.

Key lessons:
- current handcrafted BSV/family layers are useful for interpretation but often too coarse for subtle biological discrimination,
- spectral baselines often capture task signal that current BSV geometry does not,
- serum tasks frequently show real biological overlap plus strong adsorption/background structure,
- pilot-specific rescue tactics reached diminishing returns,
- the bottleneck is now representation learning and richer structured evidence.

### Serum doctrine learned from pilots
For serum:
- per-spectrum spectral baselines remain the sanity anchor,
- patient/sample-level aggregation is often better for interpretation than for classification,
- subtype geometry may remain intrinsically weak,
- GAIRA’s current strength in serum is interpretation more than geometry rescue.

## Current active Global v2 dataset state

Removed/deferred from active registry:
- stroke_urine_sers
- coeliac_faecal_sers

These remain on disk for provenance but are not active in the current Global v2 core training corpus.

Final active Global v2 core corpus:
- mycoplasma_na_sers
- ovarian_plasma_raman_sers
- single_vesicle_ev_raman
- ucla_saliva_sev_gc

Potential later augmentation datasets:
- stemcell_diff_mito_sers
- tumor_purine_secretome_sers

Support/grounding-only datasets should not be silently mixed into core training.

## Historical preprocessing doctrine

The best-supported historical canonical GAIRA embedding preprocessing lane is:
- v2_*_poly3_vector

This means:
- dataset-specific crop/interp,
- poly3 baseline correction,
- no smoothing,
- vector_l2 normalization.

AsLS / airPLS were comparison-only and are not the canonical shared training lane unless explicitly re-decided later.

When working with Global v2 processed data, prefer:
- /Volumes/SSD_Rad/GAIRA_DATA/processed/gaira_autoresearch/gaira_autoresearch_v1/global_v2_preprocessed_canonical

Do not introduce a conflicting preprocessing doctrine unless explicitly asked.

## Global v2 representation plan

Global v2 should be a shared spectral encoder build, not a return to more pilot squeezing.

Planned design:
- shared spectral backbone,
- likely 1D CNN / residual 1D CNN first,
- domain/sample-type adapters rather than fully separate encoders,
- compare against Global v1 and paper-specific baselines,
- preserve interpretability via grounding and BSV-like theme projection layers on top.

Why 1D CNN first:
- spectra are 1D structured signals,
- faster and simpler to train/debug than a transformer,
- suitable for local peak/band motif learning,
- good first serious backbone before more complex multimodal modeling.

Do not jump straight to a giant transformer or vague “complex AI.”
Build the disciplined backbone first.

## GPU / remote training context

Known prior good GPU context:
- Google Cloud GPU VM
- Tesla T4
- zone: us-east4-c
- driver: 535.288.01
- CUDA: 12.2

Treat this as prior known-good context, but verify before reuse.

If future runs use Colab or a new VM, require an explicit environment note rather than assuming compatibility.

Do not assume personal account details.
Use placeholders where needed.

## Literature / RAG / evidence-layer direction

A second major branch is the literature-backed spectral structural evidence layer.

Plan:
1. consolidate manuscript + SI + source-data corpus,
2. build a structured spectral evidence registry,
3. evaluate evidence retrieval on known spectral regions,
4. later consider learned spectral–text alignment,
5. only then consider an LLM synthesis layer.

Do not confuse the literature corpus with the shared encoder.
They are parallel but connected systems.

### Spectral structural evidence layer concept
The long-term goal is not a naive PDF RAG.
It is a structured evidence system where literature contributes:
- peak/band ranges,
- biochemical themes,
- sample-matrix context,
- Raman vs SERS context,
- substrate context,
- disease/perturbation context,
- uncertainty and competing interpretations.

Never treat literature peak assignments as ground truth.
Treat them as contextual evidence with provenance.

## Working style for this repo

Always:
- inspect current files before modifying,
- preserve provenance,
- keep additions modular,
- write verification artifacts,
- update registries carefully,
- distinguish active vs legacy vs deferred assets,
- prefer explicit documentation over silent assumptions.

Do not:
- break existing pipelines,
- silently change routing logic,
- silently mix support datasets into training lanes,
- overclaim biological certainty,
- add new datasets without showing why they improve the architecture.

## Preferred execution style

When starting a new task:
1. restate the goal in repo-specific terms,
2. inspect current relevant files and outputs,
3. propose a narrow execution plan,
4. make changes,
5. write verification artifacts,
6. summarize exactly what changed and what remains uncertain.

If a task touches datasets, registries, preprocessing, or training inputs, be especially careful and evidence-based.