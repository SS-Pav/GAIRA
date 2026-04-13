"""
Gemini API wrapper for GAIRA text queries with model fallback.

Tries multiple Gemini models in priority order when quota/rate
limits are hit. Returns both the response text and which model
succeeded.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from google import genai
from google.genai import types


FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

# Error substrings that indicate a retryable provider issue (quota/rate/transient).
# These trigger fallback to the next model.
_RETRYABLE_PATTERNS = [
    "429",
    "RESOURCE_EXHAUSTED",
    "quota",
    "rate limit",
    "rate_limit",
    "temporarily unavailable",
    "503",
    "deadline exceeded",
]


def _is_retryable(error: Exception) -> bool:
    """Return True if the error is a retryable provider issue."""
    msg = str(error).lower()
    return any(p.lower() in msg for p in _RETRYABLE_PATTERNS)


def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Export it before running:\n"
            "  export GEMINI_API_KEY=your-key-here"
        )
    return genai.Client(api_key=api_key)


@dataclass
class GeminiResult:
    """Result from a Gemini call, including model metadata."""
    text: str
    model_used: str
    fallback_used: bool = False
    attempts: list[dict] = field(default_factory=list)


def generate_text(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.3,
    max_output_tokens: int = 4096,
    retries_per_model: int = 2,
) -> GeminiResult:
    """Send a prompt to Gemini with automatic model fallback.

    Tries models in FALLBACK_MODELS order (or starting from `model`
    if specified). Falls back on retryable errors only.

    Args:
        prompt: The full prompt string.
        model: Starting model (default: first in FALLBACK_MODELS).
        temperature: Sampling temperature.
        max_output_tokens: Response length cap.
        retries_per_model: Retries within each model before falling back.

    Returns:
        GeminiResult with text, model_used, and attempt history.

    Raises:
        RuntimeError: If all models fail.
    """
    client = _get_client()

    # Build model list starting from the requested model
    if model and model in FALLBACK_MODELS:
        start = FALLBACK_MODELS.index(model)
        models = FALLBACK_MODELS[start:]
    elif model:
        models = [model] + FALLBACK_MODELS
    else:
        models = list(FALLBACK_MODELS)

    attempts: list[dict] = []

    for mi, current_model in enumerate(models):
        for retry in range(retries_per_model + 1):
            try:
                response = client.models.generate_content(
                    model=current_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    ),
                )
                text = response.text
                if not text:
                    raise RuntimeError(
                        f"Empty response from {current_model}. "
                        f"finish_reason={getattr(response.candidates[0], 'finish_reason', 'unknown') if response.candidates else 'no_candidates'}"
                    )

                attempts.append({"model": current_model, "status": "success"})
                return GeminiResult(
                    text=text,
                    model_used=current_model,
                    fallback_used=(mi > 0),
                    attempts=attempts,
                )

            except Exception as e:
                is_retry = _is_retryable(e)
                attempts.append({
                    "model": current_model,
                    "status": "retryable" if is_retry else "fatal",
                    "error": str(e)[:200],
                })

                if not is_retry:
                    # Non-retryable error — don't try other models
                    raise RuntimeError(
                        f"Non-retryable error from {current_model}: {e}"
                    ) from e

                if retry < retries_per_model:
                    wait = 5 * (retry + 1)
                    time.sleep(wait)
                # else: fall through to next model

    # All models exhausted
    model_summary = ", ".join(models)
    raise RuntimeError(
        f"All Gemini models exhausted ({model_summary}). "
        f"{len(attempts)} attempts total. "
        "Check API quota at https://ai.dev/rate-limit"
    )
