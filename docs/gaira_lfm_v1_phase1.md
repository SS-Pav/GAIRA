# GAIRA LFM v1 — Phase 1: Gemini Backend Integration

## What is GAIRA_LFM_v1?

GAIRA_LFM_v1 is the text query layer of GAIRA — a reasoning engine that takes user questions about Raman/SERS biochemistry, retrieves relevant evidence from the GAIRA evidence corpus, and uses a large foundation model (Gemini) to produce structured, uncertainty-aware biochemical interpretations.

This is **not** a classifier, predictor, or spectral analysis tool. It is a text-based reasoning engine grounded in GAIRA's evidence base.

## What Phase 1 Implements

Phase 1 is the minimal backend vertical slice:

```
text query → GAIRA evidence/context → Gemini → structured response
```

### Components

| File | Purpose |
|---|---|
| `src/gaira/llm/gemini_client.py` | Thin Gemini API wrapper. Reads `GEMINI_API_KEY` from env. |
| `src/gaira/llm/prompt_builder.py` | Builds GAIRA-philosophy-enforcing prompts with evidence, provenance, and caveats. |
| `src/gaira/llm/response_schema.py` | `GAIRAResponse` dataclass — parses structured sections from raw LLM output. |
| `scripts/test_gaira_lfm_v1.py` | Smoke test with mock evidence. |

### How Gemini Is Used

- Model: `gemini-2.0-flash` (fast, cost-effective)
- Temperature: 0.3 (deterministic-leaning)
- Single synchronous call per query
- No conversation history, no tool use, no function calling
- API key via `GEMINI_API_KEY` environment variable

### GAIRA Reasoning Philosophy (Enforced in Prompt)

The prompt instructs the model to:
- Interpret in biochemical **themes**, not exact molecules
- Separate strongest evidence from supporting/contextual
- State caveats and uncertainty explicitly
- Prefer region-based reasoning over exact-wavenumber matching
- Acknowledge that literature assignments are not ground truth

## How to Run

```bash
export GEMINI_API_KEY=your-key-here
cd /Users/suraj/projects/GAIRA
python scripts/test_gaira_lfm_v1.py
```

## What Is NOT Included

- No spectral query (BSV projection, condition deltas)
- No graph UI / Streamlit
- No evidence retrieval from database (mock evidence only)
- No conversation memory
- No streaming responses
- No async

## Next Phases

- **Phase 2**: Connect to real GAIRA evidence retrieval (DuckDB / registry)
- **Phase 3**: Streamlit UI integration
- **Phase 4**: Evidence quality scoring and confidence calibration
