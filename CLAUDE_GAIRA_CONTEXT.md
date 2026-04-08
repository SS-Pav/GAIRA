# GAIRA / Structured_Evidence_v2 — Claude Code Project Context

## Purpose
GAIRA is being built as a domain-aware, scientifically grounded Raman/SERS evidence and interpretation engine.

This repository/workflow is **not** a generic ML pipeline and **not** a PDF scraping project. The current focus is building a robust structured evidence layer that can support later probabilistic biochemical interpretation.

## Scientific operating principles
1. **Spectra are mixtures, not fingerprints.**
2. **Peak ≠ molecule.** Do not make exact molecule claims from one peak.
3. Prefer **biochemical themes / subfamilies / motifs** over definitive molecule assignments.
4. Use **domain-aware weighting** (serum vs EV vs plasma vs tissue vs pathogen).
5. Enforce **uncertainty-aware interpretation**:
   - region-based mapping, not exact-peak matching
   - multi-assignment support
   - ambiguity tracking
   - confidence tiers
6. Literature assignments are **not ground truth**. Many papers overclaim by matching nearby wavenumbers to expected biology.
7. Source-backed, assignment-grade evidence outranks vague mentions.

## Current architecture
### Core layers
- **Candidate / acquisition layer**
  - discovered papers
  - asset status (OA-ready, blocked, fetchable, needs resolution)
  - rescue packet metadata
- **Staging extraction layer**
  - OA full-text harvests
  - candidate assignments extracted by harvesters or future MCP agents
  - must not directly write final truth into evidence
- **Final evidence layer**
  - validated assignment-grade evidence only
  - neighborhoods
  - motifs
  - condition links

### Interpretation philosophy
GAIRA should evolve into a **probabilistic biochemical interpretation engine** by enforcing:
- region-based mapping
- multi-assignment support
- domain-aware weighting
- uncertainty in outputs
- assignment ambiguity layer

## Current status snapshot
### Literature / acquisition
- Large candidate pool exists (~165 current candidates in the Streamlit console at time of writing)
- OA text-first acquisition pipeline is working
- Manual rescue shortlist exists for blocked high-value papers
- Broad raw search pools are noisy; selected OA pools are the right rerun target

### Evidence / extraction
- Current evidence contribution page is working and useful
- Extraction yield improved significantly after caption/table/body-aware upgrade
- OA text-first is viable as the primary acquisition lane
- OA text-only is **not sufficient as the sole extraction path**; some papers still require figure/SI follow-up

### Working interpretation of current source classes
For ingested sources, treat them as one of:
- `high_value_ingested`
- `moderate_value_ingested`
- `low_value_context_only`
- `partial_ingest_followup_needed`

For candidates, treat them as one of:
- `OA READY`
- `OA FETCH`
- `BLOCKED`
- `NEEDS RESOLUTION`
- `LOW VALUE`
- `DISCARDED`

## Clean working cadence
All new work should happen under a clean workspace rooted at:

`/Volumes/SSD_Rad/GAIRA_DATA/structured_evidence_v2/`

Recommended directory layout:

```text
structured_evidence_v2/
  config/
  staging/
    candidate_lake/
    oa_text_corpus/
    mcp_harvest/
    rescue_packet/
  processed/
    extraction_runs/
    rerun_reports/
    figure_followup/
  registry/
    candidate_registry.csv
    ingested_registry.csv
    blocked_high_value.csv
  exports/
    summaries/
    manual_rescue_packets/
```

## Current near-term roadmap
### Phase A
Partition the current candidate lake and low-impact ingested set into actionable queues:
- ready now
- oa fetchable
- blocked high-value
- needs resolution
- low value

### Phase B
Reclassify low-impact ingested sources:
- keep
- recoverable
- follow-up-needed
- context-only

### Phase C
Build / populate `Structured_Evidence_v2` staging layer

### Phase D
Add the **assignment ambiguity / probabilistic evidence layer**:
- region-based mapping
- multi-assignment support
- uncertainty tiering
- domain-aware weighting hooks

### Phase D.5
Configure Claude Code MCPs for acquisition/staging only

### Phase E
Use Claude/MCP on the **existing candidate lake first**, not on a brand-new raw search pool

### Phase F
Launch next search-axis expansion runs only after the current lake has been partitioned and worked

## Phase D.5 — MCP setup policy
MCP should be used as an **upstream acquisition + staging assistant**, not a final truth writer.

### Allowed uses for MCP / Claude Code
- search PubMed / Europe PMC / bioRxiv / OA sources
- fetch OA text / metadata / captions / table text
- summarize likely assignment richness
- identify supplement / figure / rescue opportunities
- populate staging-layer candidate evidence
- document provenance for later GAIRA validation

### Not allowed
- direct writes into final evidence truth without GAIRA QC
- exact molecule truth claims from single peaks
- bypassing ambiguity handling

### Recommended MCP categories
Prioritize connectors/tools for:
1. PubMed / Europe PMC / PMC search
2. bioRxiv / preprint retrieval
3. web fetch / HTML-to-text conversion
4. optional citation / DOI metadata helpers
5. optional local filesystem / structured staging writers

### MCP operational rule
All MCP-derived outputs must land in **staging** first.
Final evidence layer ingestion must still run through the current GAIRA evidence extraction/QC engine.

## Claude Code operating instructions
### Important
Claude Code will **not automatically know the full context of prior ChatGPT/Codex conversations**.
You must rely on:
- this project context file
- additional rules / docs in-repo
- explicit prompts for each phase

### Working style
When modifying this project:
- reason from spectroscopy first, then data architecture, then ML
- preserve provenance
- do not loosen QC casually
- do not ingest vague literature mentions as truth
- do not overclaim molecule specificity
- prefer modular additions over broad refactors

### What to optimize for
- evidence quality
- extraction yield from assignment-grade content
- clean staging vs final evidence separation
- practical operator workflows
- scalability of acquisition without sacrificing scientific defensibility

### What to avoid
- broad uncontrolled ingestion
- review-heavy clutter
- classifier-only papers with no biochemical interpretation
- exact-peak exact-molecule logic
- hidden ambiguity

## Current tactical priorities
1. Partition current candidates and low-impact ingested sources
2. Tighten / recover extraction where justified
3. Build ambiguity-aware evidence schema
4. Add MCP-assisted acquisition/staging
5. Only then expand search axes further

## Search-axis strategy for later runs
Later search runs should be systematic and assignment-oriented, not random. Examples:
- serum + SERS/Raman + assignment
- serum + Raman + cancer
- plasma + Raman + metabolites
- EV + SERS + biomarker
- tissue + Raman + spectral assignment
- disease labels + Raman/SERS + peak assignment

Each future run may collect ~1000 raw candidates, but aggressive filtering is expected and desirable.

## Deliverable philosophy
Every meaningful run should end with:
1. evidence delta
2. paper classification table
3. follow-up list
4. rescue shortlist if needed

If a run does not produce those, it is probably not ready to merge into the working cadence.
