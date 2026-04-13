# GAIRA LFM v1 — Full Structured Evidence Corpus Expansion

## What Was Done

Expanded the retrieval corpus from 28 documents / 216 sections to **36 documents / 279 sections** by adding scientifically useful sources while preserving source typing and retrieval quality.

## Sources Added

| Tier | New docs | Examples |
|---|---|---|
| Grounding Component | +2 | BSV output format, BSV table format |
| Evidence Rules | +1 | Contrast schema proposal (deferred) |
| Context Source | +2 | Contrast integration notes, domain context template |
| Benchmark Summary | +3 | Landscape v3 BSV bugfix, evidence ontology (Phase C), extraction pilot (Phase G) |

## Final Corpus Coverage

| Tier | Weight | Docs | Sections |
|---|---|---|---|
| Grounding Component | 1.5x | 7 | 34 |
| Evidence Rules | 1.3x | 6 | 29 |
| Context Source | 1.3x | 8 | 43 |
| Benchmark Summary | 1.0x | 10 | 99 |
| Analysis Summary | 0.8x | 5 | 74 |
| **Total** | | **36** | **279** |

No meta_summary sources included. Zero planning/roadmap/operational docs in the index.

## What Was Excluded

- **Phase operational reports** (phaseB, phaseD, phaseE, phaseF, etc.) — process-tracking, not evidence
- **LFM phase docs** (gaira_lfm_v1_phase*.md) — app development notes
- **Internal ops console docs** — UI/tooling docs
- **Raw ingestion templates** (biosample, knowledge core) — schema templates only
- **OA selection rules** — acquisition policy, not biochemical evidence
- **Recovery/rescue reports** — operational triage

## Retrieval Audit Results

8 representative queries tested:

| Query | Grounding | Context | Benchmark | Meta-dominated | Diversity |
|---|---|---|---|---|---|
| Q1: HCC + serum SERS | 3 items | 4 items | 3 items | No | 7 |
| Q2: CCA vs HCC | 1 | 6 | 1 | No | 9 |
| Q3: purine + liver | 4 | 5 | 1 | No | 8 |
| Q4: glycan themes | 4 | 5 | 0 | No | 8 |
| Q5: aromatic AA + cancer | 4 | 2 | 1 | No | 7 |
| Q6: nucleic acid + EV | 4 | 1 | 1 | No | 6 |
| Q7: healthy vs liver disease | 0 | 6 | 2 | No | 6 |
| Q8: serum caveats | 2 | 5 | 3 | No | 8 |

**Summary:**
- Grounding present: 7/8 queries
- Context present: 8/8 queries
- Meta-dominated: 0/8 queries
- Average source diversity: 7.4

**Verdict: Retrieval quality is good.**

Key observations:
- Q5 ("aromatic amino acid + cancer"): `aromatic_amino_acid` definition ranks #1 with score 46.2 — highest single-item score
- Q3 ("purine + liver"): `purine_nucleotide` definition ranks #1 with score 35.7
- Q7 ("healthy vs liver"): no grounding components surface because the query is about comparison, not component definitions — appropriate behavior
- New Phase C ontology and Phase G extraction docs surface when relevant (Q1, Q2, Q8) without dominating

## Known Remaining Limitations

- No direct primary paper retrieval — all sources are GAIRA-curated artifacts
- No embedding-based semantic retrieval — still keyword/IDF
- Some benchmark summary sections overlap across landscape versions (v2/v3/v4/v5)
- The 3-per-source diversity cap may occasionally exclude relevant deep sections from large documents

## How to Run

```bash
# Retrieval audit
PYTHONPATH=src python scripts/test_gaira_full_corpus_retrieval.py

# App
PYTHONPATH=src streamlit run streamlit_apps/gaira_lfm_v1_text_query.py
```
