"""GAIRA V7 — Phase 10: FastAPI dependencies.

The engine loads once at startup and is shared read-only. It holds no mutable state and draws no
random numbers, so no request-level locking is needed on the science.
"""
from __future__ import annotations

from gaira.v7.runtime.service import GAIRAService

MAX_UPLOAD_BYTES = 32 * 1024 * 1024


def get_service() -> GAIRAService:
    return GAIRAService.instance()
