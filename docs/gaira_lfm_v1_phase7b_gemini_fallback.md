# GAIRA LFM v1 — Phase 7B: Gemini Model Fallback + Response Normalization

## Why

The free-tier Gemini API has strict daily quotas (e.g. 20 requests/day for gemini-2.5-flash). When exhausted, the app crashed with 429 RESOURCE_EXHAUSTED. Different Gemini models have separate quotas, so falling back to another model can keep the app working.

## Configured Model Order

| Priority | Model | Notes |
|---|---|---|
| 1 | gemini-2.5-flash | Primary — best quality |
| 2 | gemini-2.0-flash | Fallback — good quality, separate quota |
| 3 | gemini-2.0-flash-lite | Last resort — fastest, lowest quality |

## What Counts as Retryable

Fallback triggers on these error patterns only:

- `429` / `RESOURCE_EXHAUSTED` — quota or rate limit
- `quota` / `rate limit` — explicit limit messages
- `503` / `temporarily unavailable` — transient server issues
- `deadline exceeded` — timeout

Fallback does NOT trigger on:
- Invalid request format (400)
- Authentication errors (401/403)
- Bad prompt content
- Empty response (treated as model-specific, retried within same model)

## Fallback Behavior

1. Try primary model with up to 2 retries (5s, 10s backoff)
2. If retryable error persists, move to next model
3. Repeat for each model in the chain
4. If all models fail, raise a clean RuntimeError summarizing all attempts

The `generate_text()` function now returns a `GeminiResult` dataclass:
```python
@dataclass
class GeminiResult:
    text: str              # The response text
    model_used: str        # Which model succeeded
    fallback_used: bool    # True if primary model failed
    attempts: list[dict]   # History of all attempts
```

## Response Normalization

Different Gemini models produce slightly different formatting:
- `## Summary` vs `### Summary` vs `**Summary**` vs `**Summary:**`
- Varying bullet styles, spacing, section completeness

The normalizer (`normalize_response_text()`) in `response_schema.py`:
1. Regex-matches all heading variants for each of the 6 canonical sections
2. Replaces them with consistent `### Header` format
3. Does not alter body content or scientific meaning
4. Runs before section parsing

This ensures the parser works reliably regardless of which Gemini model responded.

## UI Transparency

After each query, the app shows:
- `Model: gemini-2.5-flash` — if primary succeeded
- `Model: gemini-2.0-flash (fallback — primary model quota exceeded)` — if fallback was used

Shown as a compact caption below the "GAIRA Response" header.

## How to Run

```bash
cd /Users/suraj/projects/GAIRA
PYTHONPATH=src streamlit run streamlit_apps/gaira_lfm_v1_text_query.py
```

## What This Does NOT Include

- No cross-provider fallback (OpenAI, Anthropic)
- No gemini-2.5-pro (not on free tier)
- No automatic quota monitoring or preemptive model selection
- No response quality comparison between models
