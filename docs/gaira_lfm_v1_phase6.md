# GAIRA LFM v1 — Phase 6: Section Linking + Confidence + Graph Polish

## What Phase 6 Adds

Three upgrades to make GAIRA_LFM_v1 answers more auditable:

1. **Section-to-evidence linking** — each answer section shows which evidence items support it
2. **Confidence composition** — structured assessment of how well grounded the answer is
3. **Trust graph polish** — cleaner colors, section sub-nodes, dashed back-links, improved layout

## Section-to-Evidence Linking

### How it works

For each parsed response section (Summary, Themes, Strongest Evidence, etc.), the linker:

1. Tokenizes the section text (with stopword removal)
2. Tokenizes each retrieved evidence item
3. Computes coverage score: fraction of section tokens found in each evidence item
4. Returns top-3 evidence items per section with scores

This is **lexical overlap**, not semantic attribution. It answers "which evidence items share the most vocabulary with this section" — a transparent heuristic, not causal proof.

### What it shows in the UI

Under each response section, a compact "Supported by" line lists the top evidence items with:
- Evidence tier badge
- Title (truncated)
- Support score
- Source filename

### Limitations

- Overlap-based: may miss paraphrased content
- Top-3 cutoff: some sections may have broader support
- Sections with generic language may match many items weakly

## Confidence Composition

### How it works

The confidence composer examines:

| Factor | What it measures |
|---|---|
| Tier mix | Fraction of grounded vs context vs summary items |
| Source diversity | Number of distinct source documents |
| Contradiction signals | Whether evidence texts contain conflicting terms (enriched/depleted, etc.) |
| Section sparsity | Which sections have weak or no evidence support |

### Output labels

| Label | Condition |
|---|---|
| strongly grounded | >= 50% grounded_evidence + 3+ sources |
| well grounded | >= 50% grounded + context |
| benchmark-supported | >= 50% benchmark/spectral |
| partially grounded | At least 1 grounded item |
| weakly grounded | No grounded evidence |

### What it is NOT

This is **not a calibrated probability**. It describes evidence composition, not answer correctness. "Strongly grounded" means the evidence base is rich and atomic — not that the answer is certainly right.

## Trust Graph Improvements

### Visual changes from Phase 5

| Aspect | Phase 5 | Phase 6 |
|---|---|---|
| Color palette | Basic colors | Refined palette with better contrast |
| Node shapes | dot/box only | box (structural), dot (evidence), diamond (sections) |
| Section nodes | Not present | Added: Summary, Themes, Strongest, etc. |
| Section back-links | Not present | Dashed lines from evidence to linked sections |
| Edge styling | Uniform | Hierarchical: thicker main flow, thinner links |
| Legend | Expandable only | Inline compact legend above graph |
| Hover tooltips | Basic | Rich HTML with tier color badges |
| Font sizes | Uniform | Scaled by node importance |
| Layout | Basic hierarchical | Tuned spacing (200px level, 60px node) |

### Graph structure

```
[Query] → [Evidence₁] → [Packet] → [Response] → [Summary]
         → [Evidence₂] ↗                       → [Themes]
         → [Evidence₃] ↗                       → [Strongest]
              ↓ ↓ (dashed)                      → [Caveats]
           [Summary] [Themes]                   → [Confidence]
```

Dashed edges show which evidence items support which response sections (top-2 links per section).

## How to Run

```bash
cd /Users/suraj/projects/GAIRA
PYTHONPATH=src streamlit run streamlit_apps/gaira_lfm_v1_text_query.py
```

## What Is Deferred

- Sentence-level attribution (per-sentence, not per-section)
- Embedding-based linking (semantic instead of lexical)
- Calibrated confidence scores
- Full corpus graph (beyond query-local)
- Spectral query integration
- Graph database backend
