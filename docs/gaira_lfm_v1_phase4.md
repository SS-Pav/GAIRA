# GAIRA LFM v1 — Phase 4: Evidence-Tiered Retrieval

## Problem with Phase 3

Phase 3 retrieval was dominated by meta-summaries — "what changed" notes, verdicts, and project planning text ranked as highly as atomic evidence. This made the evidence packet self-referential: Gemini was reasoning over GAIRA's commentary about itself rather than grounded biochemical facts.

## What Phase 4 Changes

### Source typing

Every retrieval source is now assigned an evidence tier:

| Tier | Weight | Description |
|---|---|---|
| grounded_evidence | 1.5x | Atomic facts, BSV definitions, scoring rules, contrast schema |
| domain_context | 1.3x | Serum/EV context, liver contrast findings, measurement caveats |
| benchmark_summary | 1.0x | Condition-level analysis results, spectral window data |
| spectral_query | 0.8x | Prior GAIRA analysis outputs (descriptive, not prescriptive) |
| meta_summary | 0.4x | Verdicts, "what changed", roadmap notes |

### Section-level demotion/promotion

Within any document, individual sections are further adjusted:

- **Demoted** (0.3x): titles containing "what changed", "verdict", "output files", "what should come next", "overall viability"
- **Promoted** (1.4x in non-grounded docs): titles containing "delta vs healthy", "biochemical shifts", "spectral windows", "substrate sensitivity", BSV component names

### Scoring formula

```
final_score = raw_lexical_score × tier_weight × section_adjustment
```

### Diversity cap

At most 3 items from the same source document. This prevents one large report from monopolizing the evidence packet.

### Packet assembly

Evidence items are reordered by tier priority before being sent to Gemini:
1. grounded_evidence first
2. domain_context second
3. benchmark_summary third
4. spectral_query last

This ensures Gemini sees the most atomic evidence first.

## Source Registry

Defined in `src/gaira/retrieval/source_registry.py`:

**Grounded Evidence (5 docs):**
- BSV v1 component definitions
- BSV v2 revision notes
- BSV scoring logic
- Contrast evidence schema
- Peak matching rules

**Domain Context (3 docs):**
- Serum SERS context
- EV SERS context
- Liver contrast summary

**Benchmark Summary (3 docs):**
- Landscape v4 summary
- Landscape v5 summary
- Spectral query v1 HCC summary

**Spectral Query (4 docs, SSD):**
- CCA v3, v3.2, v3.3 summaries
- Substrate sensitivity v2.1b

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

- Retrieval is still lexical (keyword + IDF), not semantic
- Source typing is hand-curated — new documents need manual registration
- Section-level demotion patterns are string-match heuristics
- No graph traversal or multi-hop reasoning
- Trust graph visualization is deferred to Phase 5

## What Remains for Phase 5

- Trust graph or evidence-chain visualization
- Confidence calibration on evidence quality
- Potential embedding-based retrieval upgrade
