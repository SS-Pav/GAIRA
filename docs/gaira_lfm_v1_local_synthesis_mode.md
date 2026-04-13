# GAIRA LFM v1 — Local Synthesis Mode

## Why

Gemini free-tier quotas (20 requests/day for some models) block development iteration. Local synthesis mode generates the same structured GAIRA response without any external LLM call, so all pipeline components — retrieval, motif/theme mapping, BSV profiling, radar plots, trust graph — can be tested continuously.

## What Inputs It Uses

The local synthesizer takes the same pipeline outputs that would normally feed the Gemini prompt:

| Input | Source |
|---|---|
| query | User's text question |
| retrieved_items | TextQueryRetriever output |
| motifs_themes_bsv | motif_theme_mapper output |
| literature_bsv_profile | literature_bsv_builder output |
| evidence_packet | evidence_packet_builder output |

## How It Differs from Gemini Mode

| Aspect | Local Synthesis | Gemini Mode |
|---|---|---|
| External API call | None | Gemini API (with fallback) |
| Response generation | Rule-based templates | LLM generation |
| Scientific synthesis quality | Structured but mechanical | More naturally written |
| Latency | Instant | 10-15s |
| Rate limits | None | Free tier limits apply |
| Cost | Free | API usage |

## What It Generates

The same 6 canonical sections:

1. **Summary** — bottom line from detected conditions, top themes, support level
2. **Biochemical Themes** — from detected motifs/themes, ordered by evidence count
3. **Strongest Evidence** — top motifs with peak regions, BSV profile highlights
4. **Supporting Evidence** — context/benchmark sources, secondary themes, weaker BSV axes
5. **Caveats** — max 3, text-query appropriate (no spectral validation language)
6. **Confidence Notes** — evidence convergence assessment based on tier mix and theme coverage

## Mode Control

The sidebar has a radio button:
- **Local synthesis** (default) — no API call
- **Gemini** — calls API with model fallback

If Gemini is selected but fails, the app falls back to local synthesis automatically.

## How to Run

```bash
cd /Users/suraj/projects/GAIRA
PYTHONPATH=src streamlit run streamlit_apps/gaira_lfm_v1_text_query.py
```

Default mode is "Local synthesis" — no API key or network access needed.

## Limitations

- Responses are template-based, not naturally synthesized prose
- No cross-evidence reasoning or nuanced contradiction handling
- Summary and confidence notes use fixed sentence patterns
- Section support linking works but is based on the synthesized text, not LLM reasoning

This mode is for development and testing. Gemini mode produces higher-quality scientific prose when API access is available.
