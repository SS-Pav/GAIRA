# GAIRA LFM v1 — Phase 3: Real Evidence Retrieval

## What Phase 3 Adds

Phase 2 used a hardcoded mock evidence packet. Phase 3 replaces it with a real local retrieval layer that finds relevant evidence from curated GAIRA documents based on the user's query.

The pipeline is now:

```
user query → keyword retrieval over curated docs → evidence packet assembly → Gemini → structured answer
```

## Architecture

```
TextQueryRetriever
  ├── loads curated MD docs from repo + SSD reports
  ├── splits into ~117 markdown sections
  ├── scores sections by IDF-weighted keyword overlap
  └── returns top-k RetrievedItem objects

EvidencePacketBuilder
  ├── deduplicates near-identical evidence
  ├── assembles evidence/provenance/caveats/domain_context
  └── returns dict compatible with prompt_builder

Streamlit App
  ├── calls retriever on each query
  ├── shows retrieved evidence in sidebar
  ├── builds prompt from real packet
  └── renders Gemini response
```

## Retrieval Sources

### Included (15 documents, ~117 sections)

**Repo docs (11):**
| File | Content |
|---|---|
| docs/bsv_v1_component_definitions.md | 8 BSV component definitions |
| docs/bsv_v2_component_revision_notes.md | Scientific rationale for BSV design |
| docs/bsv_v1_scoring_logic.md | Motif-to-component contribution rules |
| docs/gaira_landscape_v4_summary.md | Condition-specific BSV deltas |
| docs/gaira_landscape_v5_summary.md | 45-subcomponent signal decomposition |
| docs/gaira_spectral_query_v1_hcc_summary.md | HCC holdout spectral alignment |
| docs/gaira_serum_context.md | Serum SERS interpretive context |
| docs/gaira_ev_context.md | EV SERS interpretive context |
| docs/d1_contrast_schema.md | Directional peak change schema |
| docs/d1_peak_matching_rules.md | Peak tolerance and matching logic |
| docs/d1_1_liver_contrast_summary.md | Liver contrast evidence findings |

**SSD reports (4):**
| File | Content |
|---|---|
| gaira_spectral_query_v3_cca_summary.md | CCA multi-condition BSV analysis |
| gaira_spectral_query_v3_2_cca_summary.md | CCA refined band query |
| gaira_spectral_query_v3_3_cca_summary.md | CCA robustness/randomization test |
| gaira_spectral_query_v2_1b_substrate_sensitivity.md | Au vs AgNP substrate sensitivity |

### Excluded (for now)
- Raw code files (no interpretive content)
- Phase-internal pipeline reports (phaseB, phaseG, etc. — operational, not evidence)
- Blocked/candidate registry CSVs (acquisition tracking, not biochemical evidence)
- Raw spectral query CSV outputs (numerical, not text-interpretable)

## Retrieval Method

**Keyword scoring with IDF weighting.** For each query:

1. Tokenize query into lowercase words
2. For each document section, compute: `score = Σ(IDF(token))` for tokens present
3. Title matches get 2x bonus (titles are more topically concentrated)
4. Return top-k sections sorted by score
5. Long sections truncated to 1500 chars

This is deterministic, inspectable, and easy to upgrade later.

## How to Run

### Retrieval test
```bash
cd /Users/suraj/projects/GAIRA
PYTHONPATH=src python scripts/test_gaira_retrieval.py
```

### Streamlit app
```bash
cd /Users/suraj/projects/GAIRA
PYTHONPATH=src streamlit run streamlit_apps/gaira_lfm_v1_text_query.py
```

## Known Limitations

- Lexical/keyword only — no semantic similarity or embeddings
- Fixed source list — not dynamically discovering new evidence files
- No graph traversal or multi-hop reasoning
- IDF computed over ~117 sections — small corpus, weighting is coarse
- Truncation at 1500 chars may lose detail from longer sections
- No relevance feedback or query expansion

## What Remains for Phase 4

- Trust graph visualization
- Confidence calibration on retrieved evidence
- Potentially: lightweight embedding retrieval upgrade
- Potentially: evidence quality scoring
