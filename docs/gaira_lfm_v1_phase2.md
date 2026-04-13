# GAIRA LFM v1 — Phase 2: Minimal Streamlit Text Query App

## What Phase 2 Adds

Phase 1 delivered the backend: Gemini client, prompt builder, response parser, and a CLI smoke test.

Phase 2 adds a minimal Streamlit interface on top of that same backend:

- Text input for user queries
- Example query selector
- Evidence/context preview in sidebar
- Gemini call with spinner + error handling
- Parsed response rendered as clean markdown sections
- Session state preserves last query/response across reruns

## How to Run

```bash
cd /Users/suraj/projects/GAIRA
PYTHONPATH=src streamlit run streamlit_apps/gaira_lfm_v1_text_query.py
```

Requires `GEMINI_API_KEY` in environment (set in `~/.zshrc` or exported before launch).

## What Is Mocked vs Real

| Component | Status |
|---|---|
| Gemini API call | **Real** — calls `gemini-2.5-flash` |
| Prompt builder | **Real** — uses `src/gaira/llm/prompt_builder.py` |
| Response parser | **Real** — uses `src/gaira/llm/response_schema.py` |
| Evidence packet | **Mock** — hardcoded HCC/SERS evidence snippets |
| Provenance | **Mock** — hardcoded source list |
| Evidence retrieval | **Not implemented** — Phase 3 |

## What Is Intentionally Deferred

- **Phase 3**: Real evidence retrieval from GAIRA's evidence corpus (DuckDB / registry)
- **Phase 4**: Trust graph visualization, confidence calibration
- **Future**: Spectral query integration, conversation memory, streaming responses

## v1 Scope Reminder

GAIRA_LFM_v1 is **text query only**:
- No spectral query (BSV projection, condition deltas)
- No classifiers or predictors
- No disease priors
- No graph animation or trust visualization yet
